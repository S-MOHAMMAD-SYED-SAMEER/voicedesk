"""The build files, read rather than built.

A Docker build needs a daemon, a registry and a few minutes; these need
none of that. They are static checks on the three files a deployment is
shipped through, and they catch the things that are cheap to get wrong and
expensive to discover: a root container, a shell-form command that swallows
SIGTERM, a credential baked into an image, migrations run from startup.

Building the image is a separate gate, run by hand, and the README says what
it proved and on what.
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
COMPOSE = ROOT / "docker-compose.yml"
REQUIREMENTS = ROOT / "requirements.txt"


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return DOCKERFILE.read_text()


@pytest.fixture(scope="module")
def compose() -> str:
    return COMPOSE.read_text()


def _instructions(text: str) -> list[tuple[str, str]]:
    """Every Docker instruction, with line continuations joined up."""
    joined = re.sub(r"\\\n", " ", text)
    found = []
    for line in joined.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        keyword, _, rest = stripped.partition(" ")
        found.append((keyword.upper(), rest.strip()))
    return found


# --- the files exist -------------------------------------------------------


@pytest.mark.parametrize(
    "path", [DOCKERFILE, DOCKERIGNORE, COMPOSE, REQUIREMENTS]
)
def test_the_build_file_is_there(path: pathlib.Path) -> None:
    assert path.is_file()


# --- who the container runs as ---------------------------------------------


def test_the_container_does_not_run_as_root(dockerfile: str) -> None:
    """A root process in a container is one escape away from root outside it."""
    users = [rest for keyword, rest in _instructions(dockerfile) if keyword == "USER"]

    assert users
    assert users[-1] != "root"


def test_the_user_is_created_before_it_is_switched_to(dockerfile: str) -> None:
    instructions = _instructions(dockerfile)
    created = next(
        index
        for index, (keyword, rest) in enumerate(instructions)
        if keyword == "RUN" and "useradd" in rest
    )
    switched = next(
        index for index, (keyword, _) in enumerate(instructions) if keyword == "USER"
    )

    assert created < switched


def test_the_user_has_a_high_fixed_uid(dockerfile: str) -> None:
    """Fixed so a mounted volume's ownership is predictable; high so it
    cannot collide with a host account."""
    assert "--uid 10001" in dockerfile


def test_nothing_installs_after_the_user_is_switched(dockerfile: str) -> None:
    """An unprivileged `pip install` in a later layer would simply fail."""
    instructions = _instructions(dockerfile)
    switched = next(
        index for index, (keyword, _) in enumerate(instructions) if keyword == "USER"
    )

    for keyword, rest in instructions[switched + 1 :]:
        assert not (keyword == "RUN" and "pip install" in rest)


# --- what it runs ----------------------------------------------------------


def test_the_command_is_exec_form(dockerfile: str) -> None:
    """Shell form puts `/bin/sh` at PID 1, which does not pass on SIGTERM —
    so the application never drains and the orchestrator kills it."""
    command = next(rest for keyword, rest in _instructions(dockerfile) if keyword == "CMD")

    assert command.startswith("[")
    assert command.endswith("]")


def test_the_command_serves_the_application(dockerfile: str) -> None:
    command = next(rest for keyword, rest in _instructions(dockerfile) if keyword == "CMD")

    assert "uvicorn" in command
    assert "app.main:app" in command
    assert "0.0.0.0" in command


def test_the_command_does_not_reload(dockerfile: str) -> None:
    """`--reload` watches the filesystem and restarts mid-call."""
    assert "--reload" not in dockerfile


def test_no_migration_runs_from_the_image_command(dockerfile: str) -> None:
    """Replicas would race the same DDL, and a failed migration would take a
    running service down with it. It is a deployment step."""
    command = next(rest for keyword, rest in _instructions(dockerfile) if keyword == "CMD")

    assert "alembic" not in command
    for keyword, rest in _instructions(dockerfile):
        assert not (keyword == "ENTRYPOINT" and "alembic" in rest)


def test_the_health_check_uses_liveness_not_readiness(dockerfile: str) -> None:
    """A container must not be killed because PostgreSQL blinked."""
    check = next(
        rest for keyword, rest in _instructions(dockerfile) if keyword == "HEALTHCHECK"
    )

    assert "/health" in check
    assert "/ready" not in check


def test_the_port_is_declared_once_and_matches(dockerfile: str) -> None:
    exposed = [rest for keyword, rest in _instructions(dockerfile) if keyword == "EXPOSE"]

    assert exposed == ["8000"]
    assert "--port 8000" in dockerfile.replace('", "', " ")


# --- what is not in the image ----------------------------------------------

CREDENTIAL_SHAPES = (
    "sk-ant-",
    "AC0",
    "password=",
    "PASSWORD=",
    "SECRET=",
)


@pytest.mark.parametrize("shape", CREDENTIAL_SHAPES)
def test_no_credential_is_baked_into_the_image(dockerfile: str, shape: str) -> None:
    assert shape not in dockerfile


def test_no_secret_setting_is_given_a_value_in_the_image(dockerfile: str) -> None:
    """`ENV VOICEDESK_ANTHROPIC_API_KEY=...` would put a key in a layer,
    where it survives every later `unset` and ships with the image."""
    for keyword, rest in _instructions(dockerfile):
        if keyword != "ENV":
            continue
        for secret in ("API_KEY", "AUTH_TOKEN", "DATABASE_URL", "PASSWORD"):
            assert secret not in rest.upper()


def test_the_environment_is_not_set_to_production_in_the_image(
    dockerfile: str,
) -> None:
    """Which environment an artefact runs in is the deployment's decision.

    Checked against the instructions rather than the prose: the comment
    above `CMD` names the variable on purpose, to say what must be supplied
    at run time.
    """
    for keyword, rest in _instructions(dockerfile):
        assert "VOICEDESK_ENVIRONMENT" not in rest


def test_the_local_environment_file_is_excluded() -> None:
    """The one file most likely to hold a real key."""
    ignored = DOCKERIGNORE.read_text()

    assert ".env\n" in ignored
    assert ".env.*" in ignored


def test_the_example_environment_file_is_kept() -> None:
    assert "!.env.example" in DOCKERIGNORE.read_text()


@pytest.mark.parametrize("excluded", [".git/", ".venv/", "tests/", "__pycache__/"])
def test_local_state_is_excluded(excluded: str) -> None:
    assert excluded in DOCKERIGNORE.read_text()


def test_the_application_and_its_migrations_are_copied(dockerfile: str) -> None:
    copied = " ".join(rest for keyword, rest in _instructions(dockerfile) if keyword == "COPY")

    assert "app" in copied
    assert "alembic" in copied
    assert "alembic.ini" in copied


# --- the pinned dependency set ---------------------------------------------


def test_every_requirement_is_pinned_exactly() -> None:
    """A range here would mean a rebuild installs something the suite never
    ran on."""
    for line in REQUIREMENTS.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        assert "==" in line, line


def test_the_image_installs_from_the_pinned_set(dockerfile: str) -> None:
    assert "-r requirements.txt" in dockerfile


def test_the_application_is_installed_without_resolving_anything(
    dockerfile: str,
) -> None:
    """So installing the package cannot quietly pull a different version in
    over the pinned set."""
    assert "pip install --no-deps ." in dockerfile


def test_every_declared_dependency_is_in_the_pinned_set() -> None:
    import tomllib

    pinned = {
        line.split("==")[0].lower().replace("_", "-")
        for line in REQUIREMENTS.read_text().splitlines()
        if "==" in line and not line.startswith("#")
    }
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]

    for requirement in declared:
        name = re.split(r"[><=\[]", requirement)[0].strip().lower()
        assert name in pinned, name


def test_no_vendor_sdk_arrived_that_the_application_does_not_import() -> None:
    """Milestone 6 and 8 both decided against one; nothing since changed it."""
    pinned = REQUIREMENTS.read_text().lower()

    assert "twilio==" not in pinned
    assert "deepgram" not in pinned
    assert "elevenlabs" not in pinned


# --- the local stack -------------------------------------------------------


def test_the_stack_runs_migrations_as_their_own_step(compose: str) -> None:
    assert "alembic" in compose
    assert "service_completed_successfully" in compose


def test_the_application_waits_for_the_database_to_be_ready(compose: str) -> None:
    assert "service_healthy" in compose


def test_the_local_stack_is_not_production(compose: str) -> None:
    """Its credentials are local defaults, so it must not claim otherwise —
    production would refuse to start on them anyway."""
    assert "VOICEDESK_ENVIRONMENT: production" not in compose


def test_the_stack_needs_no_model_key_to_come_up(compose: str) -> None:
    """A fresh clone runs the whole thing with no account anywhere."""
    assert "${VOICEDESK_ANTHROPIC_API_KEY:-}" in compose

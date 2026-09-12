"""What must be true before this process answers a real telephone.

Every check here is about a *deployment mistake*, not a runtime condition.
They run once, at startup, and they run only when `environment` is
`production` — because the whole of the rest of this repository is built to
work with no credential at all, and breaking that would take the offline
harness, the test suite and the evaluation suite with it.

The principle is fail-fast over fail-quiet. A misconfigured deployment
currently discovers its problem when the first caller rings: a missing public
base URL surfaces as a 503 to the carrier, a missing model key as a failed
turn the caller hears an apology for. Those are expensive ways to learn
something a process could have said at boot.

Every problem is collected before any is raised, so an operator fixing a
deployment sees the whole list at once instead of one variable per restart.
"""

from dataclasses import dataclass

from app.config import DEFAULT_DATABASE_URL, Settings, get_settings


class ConfigurationError(RuntimeError):
    """Production configuration that cannot safely serve a call."""


@dataclass(frozen=True)
class Problem:
    """One thing wrong, and the variable that fixes it."""

    setting: str
    detail: str

    def __str__(self) -> str:
        return f"{self.setting}: {self.detail}"


def problems(settings: Settings) -> list[Problem]:
    """Everything that would stop this configuration serving a real call.

    Empty for any environment but production: a developer running against
    offline providers with no key is not misconfigured, they are developing.
    """
    if not settings.is_production:
        return []

    found: list[Problem] = []
    found += _database(settings)
    found += _model(settings)
    found += _telephony(settings)
    found += _surface(settings)
    return found


def check(settings: Settings | None = None) -> None:
    """Raise unless this process is fit to answer a telephone."""
    resolved = settings or get_settings()
    found = problems(resolved)
    if not found:
        return

    lines = "\n".join(f"  - {problem}" for problem in found)
    raise ConfigurationError(
        f"VOICEDESK_ENVIRONMENT is production, but this configuration cannot "
        f"safely serve a call:\n{lines}\n"
        "Set these and start again, or set VOICEDESK_ENVIRONMENT to something "
        "other than 'production' for development."
    )


# --- what each area needs --------------------------------------------------


def _database(settings: Settings) -> list[Problem]:
    if settings.database_url.strip() == DEFAULT_DATABASE_URL:
        return [
            Problem(
                "VOICEDESK_DATABASE_URL",
                "is still the local development default, which names a "
                "well-known password on localhost. Set it explicitly.",
            )
        ]
    if not settings.database_url.strip():
        return [Problem("VOICEDESK_DATABASE_URL", "is empty.")]
    return []


def _model(settings: Settings) -> list[Problem]:
    """The dialogue layer is the one provider with no offline substitute.

    Speech has one — the offline transcriber and tone are real code — so a
    deployment may legitimately run without Deepgram or ElevenLabs. There is
    no offline model, and a receptionist that cannot think cannot answer.
    """
    if settings.anthropic_api_key.strip():
        return []
    return [
        Problem(
            "VOICEDESK_ANTHROPIC_API_KEY",
            "is empty, and there is no offline language model to fall back "
            "to. Set it, or use the SDK's own ANTHROPIC_API_KEY.",
        )
    ]


def _telephony(settings: Settings) -> list[Problem]:
    """Only what the carrier path actually reads.

    `twilio_account_sid` is deliberately not required: nothing in this
    application uses it. Requiring a variable the code ignores teaches
    operators that the list is decoration.
    """
    if not settings.telephony_enabled:
        return []

    found: list[Problem] = []
    if not settings.twilio_auth_token.strip():
        found.append(
            Problem(
                "VOICEDESK_TWILIO_AUTH_TOKEN",
                "is empty, so no webhook signature can be checked and no "
                "stream URL can be signed.",
            )
        )
    if not settings.public_base_url.strip():
        found.append(
            Problem(
                "VOICEDESK_PUBLIC_BASE_URL",
                "is empty, so there is no address to tell the carrier to "
                "stream to. Calls would be answered with a 503.",
            )
        )
    if not settings.validate_twilio_signature:
        found.append(
            Problem(
                "VOICEDESK_VALIDATE_TWILIO_SIGNATURE",
                "is off. An unsigned webhook is an open door to this "
                "database and this account's model budget; it may be turned "
                "off for replaying captured requests locally and nowhere "
                "else.",
            )
        )
    return found


def _surface(settings: Settings) -> list[Problem]:
    """Development surfaces that must not be reachable from the internet."""
    found: list[Problem] = []
    if settings.debug:
        found.append(
            Problem(
                "VOICEDESK_DEBUG",
                "is on, which echoes every SQL statement — including caller "
                "names, numbers and transcripts — into the logs.",
            )
        )
    return found


__all__ = ["ConfigurationError", "Problem", "check", "problems"]

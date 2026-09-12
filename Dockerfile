# Minimal production image: the FastAPI app, its migrations, and the
# evaluation suite so a release gate can run inside the artefact it gates.
#
# Override PYTHON_IMAGE to build from a registry mirror or an internal
# registry, e.g.
#   docker build --build-arg PYTHON_IMAGE=mirror.gcr.io/library/python:3.13-slim .
ARG PYTHON_IMAGE=python:3.13-slim
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, from the pinned set the test suite actually ran on, so
# a rebuild installs the same versions rather than whatever is newest. Then
# the application itself with --no-deps, so installing it cannot quietly
# resolve something different.
COPY requirements.txt ./
RUN pip install --require-hashes=false -r requirements.txt

# Every runtime dependency ships a manylinux wheel — psycopg[binary] included
# — so no compiler and no system library is needed here.
COPY pyproject.toml README.md ./
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./

RUN pip install --no-deps . && rm -rf build voicedesk.egg-info

# Nothing is written to disk at run time: no uploads, no rendered pages, no
# audio kept. Everything durable is in PostgreSQL, so there is no volume and
# the filesystem can stay read-only.
RUN useradd --system --create-home --uid 10001 voicedesk \
    && chown -R voicedesk:voicedesk /app
USER voicedesk

EXPOSE 8000

# Liveness, deliberately — it touches nothing, so a container is not killed
# because PostgreSQL blinked. Readiness is `/ready`, which an orchestrator
# should use to decide whether to send this container a call.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# No secret is baked in. VOICEDESK_DATABASE_URL and VOICEDESK_ANTHROPIC_API_KEY
# must be supplied at run time, and with VOICEDESK_ENVIRONMENT=production the
# process refuses to start without them. See .env.example and the README.
#
# Migrations are NOT run from here. `alembic upgrade head` is a deployment
# step, run once before the new version starts, rather than something every
# replica races the others to do.
#
# Exec form, so Uvicorn is PID 1 and receives SIGTERM directly — which is
# what starts the application's own draining.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

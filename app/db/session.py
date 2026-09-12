"""Engine and session management.

The engine is created lazily and cached, so importing this module never opens
a connection.
"""

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        settings.database_url,
        # Recycle connections the server closed underneath us rather than
        # surfacing a stale-connection error on the first query.
        pool_pre_ping=True,
        # A database that is not answering should say so in seconds. Without
        # this, opening a connection waits on the operating system's TCP
        # timeout — minutes — with a caller on the line.
        connect_args={"connect_timeout": settings.database_connect_timeout_seconds},
        # SQL echoed here carries caller names, numbers and transcripts, so
        # it follows `debug`, which production refuses to have on.
        echo=settings.debug,
    )


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a session that is closed after the request."""
    with get_sessionmaker()() as session:
        yield session


def reset_engine() -> None:
    """Drop the cached engine and sessionmaker (used when settings change)."""
    if get_engine.cache_info().currsize:
        get_engine().dispose()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()

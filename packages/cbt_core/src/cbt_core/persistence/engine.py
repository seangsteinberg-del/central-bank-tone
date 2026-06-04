"""Engine and session-factory construction (CLAUDE.md sections 2 and 11).

The database URL comes from :class:`cbt_core.settings.Settings`; this module never reads the
environment itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from cbt_core.persistence.base import Base
from cbt_core.settings import Settings

if TYPE_CHECKING:
    from sqlalchemy import Table


def create_engine_from_settings(settings: Settings) -> Engine:
    """Create a SQLAlchemy engine from settings.

    Args:
        settings: The application settings holding the database URL.

    Returns:
        A configured engine. ``pool_pre_ping`` guards against stale connections.
    """
    return create_engine(settings.database_url, pool_pre_ping=True, future=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build a session factory bound to ``engine``.

    Args:
        engine: The engine sessions are bound to.

    Returns:
        A ``sessionmaker``. ``expire_on_commit`` is off so committed rows can still be mapped
        to domain models before the session closes.
    """
    return sessionmaker(bind=engine, expire_on_commit=False)


def make_demo_engine(path: str | None = None) -> Engine:
    """Create a SQLite engine for the keyless, Docker-less demo.

    Args:
        path: A file path for the SQLite database, or ``None`` for a shared in-memory database
            (a single pooled connection so every session sees the same data, suitable for tests).

    Returns:
        A SQLite engine. This is the demo and test backend; production uses PostgreSQL via
        :func:`create_engine_from_settings`.
    """
    if path is None:
        return create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            future=True,
        )
    return create_engine(f"sqlite+pysqlite:///{path}", future=True)


def create_demo_schema(engine: Engine) -> None:
    """Create the non-vector tables for a SQLite demo: speaker, speech, tone_observation.

    Deliberately skips the ``speech_chunk`` table, whose pgvector column requires PostgreSQL; the
    demo retrieves with an :class:`~cbt_core.persistence.memory.InMemoryChunkRetriever` instead. No
    append-only triggers are installed (those are PostgreSQL DDL in the migrations); the demo is a
    single-operator, throwaway database.

    Args:
        engine: The SQLite engine to create the tables on.
    """
    # Local import keeps the ORM row classes encapsulated in the persistence layer.
    from cbt_core.persistence.rows import SpeakerRow, SpeechRow, ToneObservationRow

    tables = cast(
        "list[Table]",
        [SpeakerRow.__table__, SpeechRow.__table__, ToneObservationRow.__table__],
    )
    Base.metadata.create_all(engine, tables=tables)

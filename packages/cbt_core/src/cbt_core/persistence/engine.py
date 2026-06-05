"""Engine and session-factory construction (CLAUDE.md sections 2 and 11).

The database URL comes from :class:`cbt_core.settings.Settings`; this module never reads the
environment itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from cbt_core.persistence.base import Base
from cbt_core.settings import Settings

# The append-only trigger DDL, mirroring migrations 0001 (tone_observation) and 0002 (speech). The
# alembic migrations are the source of truth for the production schema; this constant lets a setup
# that builds its schema with create_all (the no-pgvector live database, ADR 0018) install the same
# database-level immutability guarantee (CLAUDE.md section 4). Keep the two in sync. PostgreSQL only.
_IMMUTABILITY_DDL: tuple[str, ...] = (
    """
    CREATE OR REPLACE FUNCTION cbt_block_mutation() RETURNS trigger AS $$
    BEGIN
        RAISE EXCEPTION '% is append-only; % is not permitted', TG_TABLE_NAME, TG_OP;
    END;
    $$ LANGUAGE plpgsql;
    """,
    "DROP TRIGGER IF EXISTS tone_observation_append_only ON tone_observation;",
    "CREATE TRIGGER tone_observation_append_only BEFORE UPDATE OR DELETE ON tone_observation "
    "FOR EACH ROW EXECUTE FUNCTION cbt_block_mutation();",
    "DROP TRIGGER IF EXISTS speech_append_only ON speech;",
    "CREATE TRIGGER speech_append_only BEFORE UPDATE OR DELETE ON speech "
    "FOR EACH ROW EXECUTE FUNCTION cbt_block_mutation();",
)

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
    """Create the non-vector tables for a SQLite demo: speaker, speech, speech_stance, tone_observation.

    Deliberately skips the ``speech_chunk`` table, whose pgvector column requires PostgreSQL; the
    demo retrieves with an :class:`~cbt_core.persistence.memory.InMemoryChunkRetriever` instead. No
    append-only triggers are installed (those are PostgreSQL DDL in the migrations); the demo is a
    single-operator, throwaway database.

    Args:
        engine: The SQLite engine to create the tables on.
    """
    # Local import keeps the ORM row classes encapsulated in the persistence layer.
    from cbt_core.persistence.rows import (
        SpeakerRow,
        SpeechRow,
        SpeechStanceRow,
        ToneObservationRow,
    )

    tables = cast(
        "list[Table]",
        [
            SpeakerRow.__table__,
            SpeechRow.__table__,
            SpeechStanceRow.__table__,
            ToneObservationRow.__table__,
        ],
    )
    Base.metadata.create_all(engine, tables=tables)


def create_immutability_triggers(engine: Engine) -> None:
    """Install the append-only triggers on ``speech`` and ``tone_observation`` (PostgreSQL).

    For the no-pgvector live database (ADR 0018), which builds its schema with
    :func:`create_demo_schema` rather than the alembic migrations, this installs the same
    database-level append-only guarantee the migrations would (CLAUDE.md section 4): a trigger
    rejects any UPDATE or DELETE on either table. Idempotent (the triggers are dropped first).

    Args:
        engine: A PostgreSQL engine whose ``speech`` and ``tone_observation`` tables already exist.
            Not for SQLite, whose dialect has no PL/pgSQL; the keyless SQLite demo relies on the
            repositories exposing no update or delete instead.
    """
    with engine.begin() as connection:
        for statement in _IMMUTABILITY_DDL:
            connection.execute(text(statement))

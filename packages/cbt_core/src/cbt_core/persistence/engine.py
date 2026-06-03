"""Engine and session-factory construction (CLAUDE.md sections 2 and 11).

The database URL comes from :class:`cbt_core.settings.Settings`; this module never reads the
environment itself.
"""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from cbt_core.settings import Settings


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

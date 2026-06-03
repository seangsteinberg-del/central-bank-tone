"""Run Alembic migrations to head against the configured database.

The database URL comes from ``cbt_core.settings`` (``CBT_DATABASE_URL``); this script sets only
the migration script location and lets ``migrations/env.py`` resolve the URL, so configuration
stays in one place (CLAUDE.md section 11).

Usage::

    uv run python scripts/migrate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from alembic import command
from alembic.config import Config

import cbt_core.persistence as _persistence

_MIGRATIONS = Path(_persistence.__file__).resolve().parent / "migrations"


def main() -> int:
    """Upgrade the database to the latest migration."""
    config = Config()
    config.set_main_option("script_location", str(_MIGRATIONS))
    command.upgrade(config, "head")
    print("migrations applied: database is at head")
    return 0


if __name__ == "__main__":
    sys.exit(main())

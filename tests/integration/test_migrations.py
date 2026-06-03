"""The migration round trip must work on real Postgres (CLAUDE.md section 6)."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

import cbt_core.persistence as _persistence

_MIGRATIONS_DIR = Path(_persistence.__file__).resolve().parent / "migrations"


def _config(url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_MIGRATIONS_DIR))
    config.set_main_option("sqlalchemy.url", url)
    return config


@pytest.mark.integration
def test_upgrade_downgrade_upgrade_round_trip(fresh_postgres_url: str) -> None:
    config = _config(fresh_postgres_url)

    command.upgrade(config, "head")
    command.downgrade(config, "base")
    command.upgrade(config, "head")

    engine = create_engine(fresh_postgres_url, future=True)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    assert {"speaker", "tone_observation"}.issubset(tables)


@pytest.mark.integration
def test_downgrade_to_base_drops_all_tables(fresh_postgres_url: str) -> None:
    config = _config(fresh_postgres_url)

    command.upgrade(config, "head")
    command.downgrade(config, "base")

    engine = create_engine(fresh_postgres_url, future=True)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    assert "speaker" not in tables
    assert "tone_observation" not in tables

"""Integration fixtures: a Postgres testcontainer, skipped when Docker is unavailable.

A session-scoped container is migrated once; per-test isolation comes from a connection-level
transaction that is rolled back. The migration round-trip test gets its own throwaway container
so its downgrade-to-base does not disturb the shared schema (CLAUDE.md section 5: hermetic).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import Connection, Engine, create_engine

import cbt_core.persistence as _persistence

if TYPE_CHECKING:
    from alembic.config import Config
    from testcontainers.postgres import PostgresContainer

MIGRATIONS_DIR = Path(_persistence.__file__).resolve().parent / "migrations"
# pgvector image (a superset of postgres) so the speech_chunk vector table and similarity
# search run in integration tests.
_POSTGRES_IMAGE = "pgvector/pgvector:pg16"


def _start_postgres() -> PostgresContainer:
    from testcontainers.postgres import PostgresContainer

    container = PostgresContainer(_POSTGRES_IMAGE, driver="psycopg")
    container.start()
    return container


def alembic_config(url: str) -> Config:
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.set_main_option("sqlalchemy.url", url)
    return config


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    try:
        container = _start_postgres()
    except Exception as exc:  # Docker missing or daemon unreachable
        pytest.skip(f"Docker not available for integration tests: {exc}")
    try:
        yield container.get_connection_url()
    finally:
        container.stop()


@pytest.fixture(scope="session")
def migrated_engine(postgres_url: str) -> Iterator[Engine]:
    from alembic import command

    command.upgrade(alembic_config(postgres_url), "head")
    engine = create_engine(postgres_url, future=True)
    yield engine
    engine.dispose()


@pytest.fixture
def pg_connection(migrated_engine: Engine) -> Iterator[Connection]:
    connection = migrated_engine.connect()
    transaction = connection.begin()
    try:
        yield connection
    finally:
        transaction.rollback()
        connection.close()


@pytest.fixture
def fresh_postgres_url() -> Iterator[str]:
    try:
        container = _start_postgres()
    except Exception as exc:  # Docker missing or daemon unreachable
        pytest.skip(f"Docker not available for integration tests: {exc}")
    try:
        yield container.get_connection_url()
    finally:
        container.stop()

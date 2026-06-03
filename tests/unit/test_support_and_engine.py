"""Tests for service support helpers and engine construction (CLAUDE.md sections 5, 11)."""

from __future__ import annotations

from uuid import UUID

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from cbt_core import Settings, create_engine_from_settings, make_session_factory
from cbt_core.services._support import default_id_factory, utc_now


@pytest.mark.unit
def test_default_id_factory_returns_unique_uuids() -> None:
    first, second = default_id_factory(), default_id_factory()
    assert isinstance(first, UUID)
    assert first != second


@pytest.mark.unit
def test_utc_now_is_timezone_aware() -> None:
    now = utc_now()
    assert now.tzinfo is not None
    assert now.utcoffset() is not None


@pytest.mark.unit
def test_create_engine_from_settings_uses_the_configured_url(dummy_settings: Settings) -> None:
    engine = create_engine_from_settings(dummy_settings)
    assert isinstance(engine, Engine)
    assert engine.url.get_backend_name() == "sqlite"
    engine.dispose()


@pytest.mark.unit
def test_make_session_factory_binds_to_the_engine(dummy_settings: Settings) -> None:
    engine = create_engine_from_settings(dummy_settings)
    factory = make_session_factory(engine)
    assert isinstance(factory, sessionmaker)
    with factory() as session:
        assert session.bind is engine
    engine.dispose()

"""Shared test fixtures (CLAUDE.md section 5).

Fixtures are hermetic: an in-process SQLite engine with the schema created fresh per test,
deterministic id and clock factories, and a TestClient wired to SQLite-backed services. No
fixture touches the network or a live database; Postgres-only invariants live in
``tests/integration`` behind the ``integration`` marker.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import ConnectionPoolEntry, StaticPool
from tests._stubs import StubLlmClient

from cbt_api.dependencies import Services
from cbt_core import (
    AnalysisService,
    Settings,
    SpeakerService,
    ToneAnalysis,
    ToneLabel,
    ToneService,
)
from cbt_core.persistence import Base
from cbt_core.services._support import Clock, IdFactory
from cbt_core.settings import Environment

FROZEN_TIME = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def dummy_settings() -> Settings:
    """Development settings with a dummy secret and an in-process SQLite URL."""
    return Settings(
        environment=Environment.DEVELOPMENT,
        database_url="sqlite://",
        secret_key=SecretStr("test-secret-not-a-real-key"),
    )


@pytest.fixture
def sqlite_engine() -> Iterator[Engine]:
    """A shared in-memory SQLite engine with the full schema and foreign keys enforced."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(
        dbapi_connection: DBAPIConnection, _record: ConnectionPoolEntry
    ) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session_factory(sqlite_engine: Engine) -> sessionmaker[Session]:
    """A session factory bound to the in-memory SQLite engine."""
    return sessionmaker(bind=sqlite_engine, expire_on_commit=False)


@pytest.fixture
def id_factory() -> IdFactory:
    """A deterministic id factory yielding UUID(int=1), UUID(int=2), ... ."""
    counter = itertools.count(1)
    return lambda: UUID(int=next(counter))


@pytest.fixture
def frozen_clock() -> Clock:
    """A clock that always returns a fixed, timezone-aware instant."""
    return lambda: FROZEN_TIME


@pytest.fixture
def speaker_service(
    session_factory: sessionmaker[Session], id_factory: IdFactory
) -> SpeakerService:
    """A speaker service with a deterministic id factory."""
    return SpeakerService(session_factory, id_factory=id_factory)


@pytest.fixture
def tone_service(
    session_factory: sessionmaker[Session], id_factory: IdFactory, frozen_clock: Clock
) -> ToneService:
    """A tone service with deterministic id and clock factories."""
    return ToneService(session_factory, id_factory=id_factory, clock=frozen_clock)


@pytest.fixture
def stub_tone_analysis() -> ToneAnalysis:
    """A fixed tone analysis the stub LLM client returns."""
    return ToneAnalysis(
        summary="The speaker signalled a readiness to keep policy tight.",
        tone=ToneLabel.HAWKISH,
        score=0.6,
        rationale="Repeated emphasis on persistent inflation risk.",
    )


@pytest.fixture
def stub_llm_client(stub_tone_analysis: ToneAnalysis) -> StubLlmClient:
    """A deterministic stub LLM client."""
    return StubLlmClient(stub_tone_analysis)


@pytest.fixture
def analysis_service(
    stub_llm_client: StubLlmClient,
    speaker_service: SpeakerService,
    tone_service: ToneService,
) -> AnalysisService:
    """An analysis service wired to the stub LLM client and SQLite-backed services."""
    return AnalysisService(stub_llm_client, speaker_service, tone_service)


@pytest.fixture
def services(
    dummy_settings: Settings,
    sqlite_engine: Engine,
    speaker_service: SpeakerService,
    tone_service: ToneService,
) -> Services:
    """A SQLite-backed service container for wiring into the API."""
    return Services(
        settings=dummy_settings,
        engine=sqlite_engine,
        speaker_service=speaker_service,
        tone_service=tone_service,
    )


@pytest.fixture
def client(services: Services) -> Iterator[TestClient]:
    """A TestClient whose app state is the SQLite-backed service container."""
    from cbt_api.app import create_app

    app = create_app(services.settings)
    app.state.services = services
    with TestClient(app) as test_client:
        yield test_client

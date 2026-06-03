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
from tests._stubs import StubChunkRetriever, StubLlmClient

from cbt_api.dependencies import Services
from cbt_core import (
    IndexingService,
    IngestionService,
    QaService,
    Settings,
    SpeakerService,
    ToneAnalysis,
    ToneLabel,
    ToneService,
)
from cbt_core.domain.qa import RetrievedChunk
from cbt_core.persistence import Base
from cbt_core.services._support import Clock, IdFactory
from cbt_core.settings import Environment

FROZEN_TIME = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def dummy_settings() -> Settings:
    """Development settings with dummy secrets and an in-process SQLite URL."""
    return Settings(
        environment=Environment.DEVELOPMENT,
        database_url="sqlite://",
        secret_key=SecretStr("test-secret-not-a-real-key"),
        gemini_api_key=SecretStr("test-gemini-key-not-real"),
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
def ingestion_service(
    session_factory: sessionmaker[Session],
    stub_llm_client: StubLlmClient,
    id_factory: IdFactory,
) -> IngestionService:
    """An ingestion service wired to the stub LLM client and the SQLite engine."""
    return IngestionService(
        session_factory, stub_llm_client, id_factory=id_factory, model_id="gemini-test"
    )


@pytest.fixture
def indexing_service(
    session_factory: sessionmaker[Session],
    stub_llm_client: StubLlmClient,
    id_factory: IdFactory,
) -> IndexingService:
    """An indexing service wired to the stub LLM client and the SQLite engine."""
    return IndexingService(
        session_factory, stub_llm_client, max_chars=200, overlap=20, id_factory=id_factory
    )


@pytest.fixture
def qa_service(stub_llm_client: StubLlmClient, speaker_service: SpeakerService) -> QaService:
    """A Q&A service whose retriever returns a fixed chunk (no pgvector needed)."""
    chunk = RetrievedChunk(
        speech_id=UUID(int=900),
        chunk_index=0,
        text="an excerpt",
        title="A speech",
        url="https://example.org/s/900",
        distance=0.1,
    )
    return QaService(stub_llm_client, StubChunkRetriever([chunk]), speaker_service)


@pytest.fixture
def services(
    dummy_settings: Settings,
    sqlite_engine: Engine,
    speaker_service: SpeakerService,
    tone_service: ToneService,
    ingestion_service: IngestionService,
    indexing_service: IndexingService,
    qa_service: QaService,
) -> Services:
    """A SQLite-backed service container for wiring into the API."""
    return Services(
        settings=dummy_settings,
        engine=sqlite_engine,
        speaker_service=speaker_service,
        tone_service=tone_service,
        ingestion_service=ingestion_service,
        indexing_service=indexing_service,
        qa_service=qa_service,
    )


@pytest.fixture
def client(services: Services) -> Iterator[TestClient]:
    """A TestClient whose app state is the SQLite-backed service container."""
    from cbt_api.app import create_app

    app = create_app(services.settings)
    app.state.services = services
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def web_client(
    dummy_settings: Settings,
    sqlite_engine: Engine,
    speaker_service: SpeakerService,
    tone_service: ToneService,
    ingestion_service: IngestionService,
    indexing_service: IndexingService,
    qa_service: QaService,
) -> Iterator[TestClient]:
    """A TestClient for the web UI app, backed by the same SQLite-backed services."""
    from cbt_web.app import create_app
    from cbt_web.dependencies import Services as WebServices

    app = create_app(dummy_settings)
    app.state.services = WebServices(
        settings=dummy_settings,
        engine=sqlite_engine,
        speaker_service=speaker_service,
        tone_service=tone_service,
        ingestion_service=ingestion_service,
        indexing_service=indexing_service,
        qa_service=qa_service,
    )
    with TestClient(app) as test_client:
        yield test_client

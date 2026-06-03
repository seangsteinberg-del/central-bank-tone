"""Application wiring and FastAPI dependencies for the web UI (CLAUDE.md section 2).

Builds the engine, session factory, and ``cbt_core`` services once per process, stores them on
the application state, and hands services to views via typed dependencies. Views depend on a
service, never on the engine, a repository, or the logger. This adapter is independent of the
API adapter: it imports ``cbt_core`` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, cast

from fastapi import Depends, Request
from sqlalchemy import Engine

from cbt_core import (
    IndexingService,
    IngestionService,
    QaService,
    Settings,
    SpeakerService,
    SpeechRetriever,
    ToneService,
    build_gemini_client,
    create_engine_from_settings,
    make_session_factory,
)


@dataclass(frozen=True)
class Services:
    """The process-wide service container, built once and stored on ``app.state``."""

    settings: Settings
    engine: Engine
    speaker_service: SpeakerService
    tone_service: ToneService
    ingestion_service: IngestionService
    indexing_service: IndexingService
    qa_service: QaService


def build_services(settings: Settings) -> Services:
    """Construct the engine, session factory, and services from settings.

    Args:
        settings: The application settings.

    Returns:
        A fully wired :class:`Services` container.
    """
    engine = create_engine_from_settings(settings)
    session_factory = make_session_factory(engine)
    llm = build_gemini_client(settings)
    speaker_service = SpeakerService(session_factory)
    return Services(
        settings=settings,
        engine=engine,
        speaker_service=speaker_service,
        tone_service=ToneService(session_factory),
        ingestion_service=IngestionService(session_factory, llm),
        indexing_service=IndexingService(session_factory, llm),
        qa_service=QaService(llm, SpeechRetriever(session_factory), speaker_service),
    )


def get_services(request: Request) -> Services:
    """Return the service container stored on the application state."""
    return cast(Services, request.app.state.services)


ServicesDep = Annotated[Services, Depends(get_services)]


def get_speaker_service(services: ServicesDep) -> SpeakerService:
    """Return the speaker service."""
    return services.speaker_service


def get_tone_service(services: ServicesDep) -> ToneService:
    """Return the tone service."""
    return services.tone_service


def get_ingestion_service(services: ServicesDep) -> IngestionService:
    """Return the ingestion (speech) service."""
    return services.ingestion_service


def get_indexing_service(services: ServicesDep) -> IndexingService:
    """Return the indexing service."""
    return services.indexing_service


def get_qa_service(services: ServicesDep) -> QaService:
    """Return the question-answering service."""
    return services.qa_service


SpeakerServiceDep = Annotated[SpeakerService, Depends(get_speaker_service)]
ToneServiceDep = Annotated[ToneService, Depends(get_tone_service)]
IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
IndexingServiceDep = Annotated[IndexingService, Depends(get_indexing_service)]
QaServiceDep = Annotated[QaService, Depends(get_qa_service)]

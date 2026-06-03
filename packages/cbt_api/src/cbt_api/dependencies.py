"""Application wiring and FastAPI dependencies (CLAUDE.md section 2).

Builds the settings, engine, session factory, and services once, stores them on the app state,
and hands services to routes via typed dependencies. Routes depend on a service, never on the
engine, a repository, or the logger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, cast
from uuid import UUID

from fastapi import Depends, Request
from sqlalchemy import Engine

from cbt_core import (
    Settings,
    SpeakerService,
    ToneService,
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


def build_services(settings: Settings) -> Services:
    """Construct the engine, session factory, and services from settings.

    Args:
        settings: The application settings.

    Returns:
        A fully wired :class:`Services` container.
    """
    engine = create_engine_from_settings(settings)
    session_factory = make_session_factory(engine)
    return Services(
        settings=settings,
        engine=engine,
        speaker_service=SpeakerService(session_factory),
        tone_service=ToneService(session_factory),
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


def get_correlation_id(request: Request) -> UUID:
    """Return the correlation id for the current request.

    The correlation-id middleware parses or mints it and stores it on ``request.state``.
    """
    return cast(UUID, request.state.correlation_id)


SpeakerServiceDep = Annotated[SpeakerService, Depends(get_speaker_service)]
ToneServiceDep = Annotated[ToneService, Depends(get_tone_service)]
CorrelationIdDep = Annotated[UUID, Depends(get_correlation_id)]

"""Speaker and tone-observation routes.

Each route validates its request body against a schema, calls a service with typed values, and
returns a response model. Routes never touch a repository, the engine, or the logger directly.
Core exceptions propagate to the handlers in ``errors.py``.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, status

from cbt_api.dependencies import (
    CorrelationIdDep,
    IndexingServiceDep,
    IngestionServiceDep,
    QaServiceDep,
    SpeakerServiceDep,
    ToneServiceDep,
)
from cbt_api.schemas import (
    AnswerResponse,
    AskRequest,
    SpeakerCreate,
    SpeakerResponse,
    SpeechIngestRequest,
    SpeechResponse,
    ToneObservationCreate,
    ToneObservationResponse,
)

router = APIRouter(prefix="/speakers", tags=["speakers"])


@router.post("", status_code=status.HTTP_201_CREATED)
def create_speaker(
    body: SpeakerCreate, service: SpeakerServiceDep, correlation_id: CorrelationIdDep
) -> SpeakerResponse:
    """Register a new speaker."""
    speaker = service.register_speaker(
        name=body.name,
        central_bank=body.central_bank,
        role=body.role,
        correlation_id=correlation_id,
    )
    return SpeakerResponse.from_domain(speaker)


@router.get("")
def list_speakers(
    service: SpeakerServiceDep, correlation_id: CorrelationIdDep
) -> list[SpeakerResponse]:
    """List every speaker, ordered by name."""
    speakers = service.list_speakers(correlation_id=correlation_id)
    return [SpeakerResponse.from_domain(speaker) for speaker in speakers]


@router.get("/{speaker_id}")
def get_speaker(
    speaker_id: UUID, service: SpeakerServiceDep, correlation_id: CorrelationIdDep
) -> SpeakerResponse:
    """Fetch a single speaker by id."""
    speaker = service.get_speaker(speaker_id, correlation_id=correlation_id)
    return SpeakerResponse.from_domain(speaker)


@router.post("/{speaker_id}/tone-observations", status_code=status.HTTP_201_CREATED)
def record_observation(
    speaker_id: UUID,
    body: ToneObservationCreate,
    service: ToneServiceDep,
    correlation_id: CorrelationIdDep,
) -> ToneObservationResponse:
    """Record an append-only tone observation for a speaker."""
    observation = service.record_observation(
        speaker_id=speaker_id,
        tone=body.tone,
        score=body.score,
        source_text=body.source_text,
        observed_at=body.observed_at,
        correlation_id=correlation_id,
    )
    return ToneObservationResponse.from_domain(observation)


@router.get("/{speaker_id}/tone-observations")
def list_observations(
    speaker_id: UUID, service: ToneServiceDep, correlation_id: CorrelationIdDep
) -> list[ToneObservationResponse]:
    """List a speaker's tone observations, oldest first."""
    observations = service.observations_for(speaker_id, correlation_id=correlation_id)
    return [ToneObservationResponse.from_domain(observation) for observation in observations]


@router.post("/{speaker_id}/speeches", status_code=status.HTTP_201_CREATED)
def ingest_speech(
    speaker_id: UUID,
    body: SpeechIngestRequest,
    ingestion: IngestionServiceDep,
    indexing: IndexingServiceDep,
    correlation_id: CorrelationIdDep,
) -> SpeechResponse:
    """Ingest, analyze, and index a single speech for a speaker."""
    speech = ingestion.ingest_speech(
        speaker_id=speaker_id,
        title=body.title,
        url=body.url,
        delivered_at=body.delivered_at,
        text=body.text,
        language=body.language,
        correlation_id=correlation_id,
    )
    indexing.index_speech(speech.id, correlation_id=correlation_id)
    return SpeechResponse.from_domain(speech)


@router.get("/{speaker_id}/speeches")
def list_speeches(
    speaker_id: UUID, ingestion: IngestionServiceDep, correlation_id: CorrelationIdDep
) -> list[SpeechResponse]:
    """List a speaker's analyzed speeches, most recent first."""
    speeches = ingestion.list_speeches(speaker_id, correlation_id=correlation_id)
    return [SpeechResponse.from_domain(speech) for speech in speeches]


@router.post("/{speaker_id}/ask")
def ask(
    speaker_id: UUID,
    body: AskRequest,
    service: QaServiceDep,
    correlation_id: CorrelationIdDep,
) -> AnswerResponse:
    """Answer a question about a speaker, grounded in their speeches."""
    answer = service.answer(
        speaker_id=speaker_id, question=body.question, correlation_id=correlation_id
    )
    return AnswerResponse.from_domain(answer)

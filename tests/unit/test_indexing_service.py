"""Tests for the speech indexing service against SQLite (CLAUDE.md section 5).

These cover chunking, embedding, and storing; the pgvector similarity search itself is covered
by the Postgres integration tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

import pytest
from sqlalchemy.orm import Session, sessionmaker
from tests._stubs import StubLlmClient

from cbt_core import IndexingService, IngestionService, SpeakerService
from cbt_core.domain.registry import CentralBank
from cbt_core.exceptions import EntityNotFoundError, LlmError
from cbt_core.persistence.repositories import SpeechChunkRepository

_LONG_TEXT = " ".join(f"word{i}" for i in range(300))


def _ingest(ingestion: IngestionService, speaker_service: SpeakerService) -> UUID:
    speaker = speaker_service.register_speaker(
        name="Powell", central_bank=CentralBank.FEDERAL_RESERVE, role="Chair"
    )
    speech = ingestion.ingest_speech(
        speaker_id=speaker.id,
        title="Outlook",
        url="https://example.org/s/1",
        delivered_at=datetime(2026, 1, 1, tzinfo=UTC),
        text=_LONG_TEXT,
    )
    return speech.id


@pytest.mark.unit
def test_index_speech_stores_chunks(
    ingestion_service: IngestionService,
    indexing_service: IndexingService,
    speaker_service: SpeakerService,
    session_factory: object,
) -> None:
    speech_id = _ingest(ingestion_service, speaker_service)
    count = indexing_service.index_speech(speech_id)
    assert count > 1  # the long text splits into several chunks
    with session_factory() as session:  # type: ignore[attr-defined]  # fixture typed as object
        assert SpeechChunkRepository(session).has_chunks(speech_id) is True


@pytest.mark.unit
def test_index_speech_is_idempotent(
    ingestion_service: IngestionService,
    indexing_service: IndexingService,
    speaker_service: SpeakerService,
) -> None:
    speech_id = _ingest(ingestion_service, speaker_service)
    indexing_service.index_speech(speech_id)
    assert indexing_service.index_speech(speech_id) == 0  # already indexed, no re-embed


@pytest.mark.unit
def test_index_unknown_speech_raises_not_found(
    indexing_service: IndexingService, stub_llm_client: StubLlmClient
) -> None:
    with pytest.raises(EntityNotFoundError):
        indexing_service.index_speech(UUID(int=404))


class _WrongCountEmbedder:
    """An LLM stub whose embed returns the wrong number of vectors (a contract violation)."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return zero vectors regardless of the chunk count, to trip the length guard."""
        return []


@pytest.mark.unit
def test_index_speech_raises_llm_error_on_embedding_count_mismatch(
    ingestion_service: IngestionService,
    speaker_service: SpeakerService,
    session_factory: sessionmaker[Session],
) -> None:
    # A wrong embedding count is an internal contract violation; it must raise the documented
    # LlmError (a CbtError), not a bare ValueError from a strict zip that bypasses the handlers.
    speech_id = _ingest(ingestion_service, speaker_service)
    service = IndexingService(session_factory, _WrongCountEmbedder())  # type: ignore[arg-type]
    with pytest.raises(LlmError):
        service.index_speech(speech_id)

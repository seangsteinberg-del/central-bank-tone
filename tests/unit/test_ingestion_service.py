"""Tests for the speech ingestion service (CLAUDE.md sections 2 and 5)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from tests._stubs import StubLlmClient

from cbt_core import IngestionService, SpeakerService, ToneAnalysis, ToneService
from cbt_core.domain.registry import CentralBank
from cbt_core.exceptions import EntityNotFoundError

_HAWKISH_TEXT = "The committee will tighten policy and hike rates given inflationary pressure."


def _register(service: SpeakerService) -> UUID:
    speaker = service.register_speaker(
        name="Jerome Powell", central_bank=CentralBank.FEDERAL_RESERVE, role="Chair"
    )
    return speaker.id


def _ingest(service: IngestionService, speaker_id: UUID, *, text: str = _HAWKISH_TEXT) -> object:
    return service.ingest_speech(
        speaker_id=speaker_id,
        title="On the outlook",
        url="https://example.org/speech/1",
        delivered_at=datetime(2026, 1, 15, tzinfo=UTC),
        text=text,
    )


@pytest.mark.unit
def test_ingest_persists_model_tone_and_lexicon_baseline(
    ingestion_service: IngestionService,
    speaker_service: SpeakerService,
    stub_tone_analysis: ToneAnalysis,
) -> None:
    speaker_id = _register(speaker_service)
    speech = _ingest(ingestion_service, speaker_id)
    assert speech.tone == stub_tone_analysis.tone
    assert speech.score == stub_tone_analysis.score
    assert speech.summary == stub_tone_analysis.summary
    assert speech.central_bank is CentralBank.FEDERAL_RESERVE
    assert speech.lexicon_score > 0  # the source text is hawkish


@pytest.mark.unit
def test_ingest_is_idempotent_by_source_hash(
    ingestion_service: IngestionService,
    speaker_service: SpeakerService,
    stub_llm_client: StubLlmClient,
) -> None:
    speaker_id = _register(speaker_service)
    first = _ingest(ingestion_service, speaker_id)
    second = _ingest(ingestion_service, speaker_id)
    assert second.id == first.id
    assert len(stub_llm_client.calls) == 1  # the second ingest spent no model call


@pytest.mark.unit
def test_ingest_appends_a_tone_observation(
    ingestion_service: IngestionService,
    speaker_service: SpeakerService,
    tone_service: ToneService,
) -> None:
    speaker_id = _register(speaker_service)
    speech = _ingest(ingestion_service, speaker_id)
    observations = tone_service.observations_for(speaker_id)
    assert len(observations) == 1
    assert observations[0].source_sha256 == speech.source_sha256
    assert observations[0].tone == speech.tone
    assert observations[0].lexicon_score == speech.lexicon_score


@pytest.mark.unit
def test_ingest_agreeing_tone_is_not_flagged_for_review(
    ingestion_service: IngestionService, speaker_service: SpeakerService
) -> None:
    # The stub model returns a hawkish score; hawkish source text agrees with the lexicon.
    speaker_id = _register(speaker_service)
    speech = _ingest(ingestion_service, speaker_id, text=_HAWKISH_TEXT)
    assert speech.needs_review is False


@pytest.mark.unit
def test_ingest_flags_review_when_model_and_lexicon_disagree(
    ingestion_service: IngestionService,
    speaker_service: SpeakerService,
    tone_service: ToneService,
) -> None:
    # The stub model is hawkish (+0.6); a clearly dovish text makes the lexicon disagree.
    speaker_id = _register(speaker_service)
    dovish = "We will ease policy, cut rates, and stay accommodative and patient amid headwinds."
    speech = _ingest(ingestion_service, speaker_id, text=dovish)
    assert speech.lexicon_score < 0
    assert speech.needs_review is True
    assert tone_service.observations_for(speaker_id)[0].needs_review is True


@pytest.mark.unit
def test_ingest_unknown_speaker_raises_and_skips_the_model(
    ingestion_service: IngestionService, stub_llm_client: StubLlmClient
) -> None:
    with pytest.raises(EntityNotFoundError):
        _ingest(ingestion_service, UUID(int=404))
    assert stub_llm_client.calls == []

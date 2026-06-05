"""Tests for the speech ingestion service (CLAUDE.md sections 2 and 5)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

import pytest
from tests._stubs import StubLlmClient

from cbt_core import IngestionService, SpeakerService, SpeechStance, ToneAnalysis, ToneService
from cbt_core.domain.registry import CentralBank
from cbt_core.exceptions import EntityNotFoundError
from cbt_core.persistence.repositories import SpeechStanceRepository

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

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
    assert speech.model_id == "gemini-test"  # the configured model is recorded on the speech


@pytest.mark.unit
def test_ingest_persists_the_structured_pipeline_fields(
    ingestion_service: IngestionService,
    speaker_service: SpeakerService,
    session_factory: sessionmaker[Session],
) -> None:
    # The structured pipeline (ADR 0021) runs on every ingest and persists its decomposition into
    # the derived speech_stance table, not onto the immutable speech.
    speaker_id = _register(speaker_service)
    speech = _ingest(ingestion_service, speaker_id, text=_HAWKISH_TEXT)
    with session_factory() as session:
        stance = SpeechStanceRepository(session).get(speech.id)
    assert stance is not None
    assert -1.0 <= stance.rate_path <= 1.0
    assert 0.0 <= stance.uncertainty <= 1.0
    # The hawkish source is about inflation, so that aspect appears in the breakdown.
    assert "inflation" in stance.aspect_scores


@pytest.mark.unit
def test_get_stance_and_stances_by_speech_expose_the_decomposition(
    ingestion_service: IngestionService, speaker_service: SpeakerService
) -> None:
    speaker_id = _register(speaker_service)
    speech = _ingest(ingestion_service, speaker_id)
    stance = ingestion_service.get_stance(speech.id)
    assert stance is not None
    assert stance.speech_id == speech.id
    assert ingestion_service.get_stance(UUID(int=12345)) is None  # an unscored speech
    assert speech.id in ingestion_service.stances_by_speech()


@pytest.mark.unit
def test_speech_stance_repository_get_replace_and_list(
    ingestion_service: IngestionService,
    speaker_service: SpeakerService,
    session_factory: sessionmaker[Session],
) -> None:
    speaker_id = _register(speaker_service)
    speech = _ingest(ingestion_service, speaker_id)
    with session_factory() as session:
        repo = SpeechStanceRepository(session)
        assert repo.get(speech.id) is not None
        assert repo.get(UUID(int=999)) is None  # an unscored speech has no decomposition
        assert set(repo.all_by_speech()) == {speech.id}
        # Re-derivation replaces the existing row (the decomposition is recomputable, not appended).
        repo.upsert(
            SpeechStance(
                speech_id=speech.id,
                rate_path=-0.2,
                uncertainty=0.5,
                structured_net=-0.1,
                classifier_net=0.0,
                lexicon_net=0.0,
                needs_review=True,
                aspect_scores={"growth": -0.2},
                model_id="re-scored",
            )
        )
        session.commit()
    with session_factory() as session:
        replaced = SpeechStanceRepository(session).get(speech.id)
    assert replaced is not None
    assert replaced.rate_path == -0.2
    assert replaced.needs_review is True


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

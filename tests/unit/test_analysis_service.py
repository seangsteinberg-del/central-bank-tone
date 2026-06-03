"""Tests for the speech analysis service (CLAUDE.md sections 2 and 5)."""

from __future__ import annotations

from uuid import UUID

import pytest
from tests._stubs import StubLlmClient

from cbt_core import AnalysisService, SpeakerService
from cbt_core.domain.registry import CentralBank
from cbt_core.exceptions import EntityNotFoundError


def _register(service: SpeakerService) -> UUID:
    speaker = service.register_speaker(
        name="Jerome Powell", central_bank=CentralBank.FEDERAL_RESERVE, role="Chair"
    )
    return speaker.id


@pytest.mark.unit
def test_analyze_speech_records_the_model_tone(
    analysis_service: AnalysisService,
    speaker_service: SpeakerService,
    stub_tone_analysis: object,
) -> None:
    speaker_id = _register(speaker_service)
    result = analysis_service.analyze_speech(
        speaker_id=speaker_id, source_text="we will keep policy restrictive"
    )
    assert result.analysis == stub_tone_analysis
    assert result.observation.tone == result.analysis.tone
    assert result.observation.score == result.analysis.score
    assert result.observation.speaker_id == speaker_id


@pytest.mark.unit
def test_analyze_speech_persists_an_observation(
    analysis_service: AnalysisService, speaker_service: SpeakerService, tone_service: object
) -> None:
    speaker_id = _register(speaker_service)
    analysis_service.analyze_speech(speaker_id=speaker_id, source_text="text")
    # The tone_service shares the SQLite engine, so the observation is readable back.
    observations = tone_service.observations_for(speaker_id)  # type: ignore[attr-defined]  # fixture typed as object
    assert len(observations) == 1


@pytest.mark.unit
def test_analyze_speech_for_unknown_speaker_does_not_call_the_model(
    analysis_service: AnalysisService, stub_llm_client: StubLlmClient
) -> None:
    with pytest.raises(EntityNotFoundError):
        analysis_service.analyze_speech(speaker_id=UUID(int=404), source_text="text")
    assert stub_llm_client.calls == []

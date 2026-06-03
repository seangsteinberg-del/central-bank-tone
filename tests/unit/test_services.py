"""Service tests against SQLite with injected clock and id factory (CLAUDE.md sections 2, 5)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from uuid import UUID

import pytest

from cbt_core import SpeakerService, ToneService
from cbt_core.domain.registry import CentralBank
from cbt_core.domain.tone import ToneLabel
from cbt_core.exceptions import EntityNotFoundError, InvalidInputError
from cbt_core.services._support import Clock


def _register(service: SpeakerService) -> UUID:
    speaker = service.register_speaker(
        name="Jerome Powell", central_bank=CentralBank.FEDERAL_RESERVE, role="Chair"
    )
    return speaker.id


@pytest.mark.unit
def test_register_speaker_persists_and_is_readable(speaker_service: SpeakerService) -> None:
    speaker = speaker_service.register_speaker(
        name="Christine Lagarde", central_bank=CentralBank.ECB, role="President"
    )
    assert speaker.id == UUID(int=1)
    assert speaker_service.get_speaker(speaker.id) == speaker


@pytest.mark.unit
def test_register_speaker_rejects_blank_name(speaker_service: SpeakerService) -> None:
    with pytest.raises(InvalidInputError):
        speaker_service.register_speaker(name="", central_bank=CentralBank.ECB, role="President")


@pytest.mark.unit
def test_get_unknown_speaker_raises_not_found(speaker_service: SpeakerService) -> None:
    with pytest.raises(EntityNotFoundError):
        speaker_service.get_speaker(UUID(int=404))


@pytest.mark.unit
def test_list_speakers_returns_all_registered(speaker_service: SpeakerService) -> None:
    speaker_service.register_speaker(
        name="Powell", central_bank=CentralBank.FEDERAL_RESERVE, role="Chair"
    )
    speaker_service.register_speaker(
        name="Bailey", central_bank=CentralBank.BANK_OF_ENGLAND, role="Governor"
    )
    assert {speaker.name for speaker in speaker_service.list_speakers()} == {"Powell", "Bailey"}


@pytest.mark.unit
def test_record_observation_defaults_to_the_injected_clock(
    speaker_service: SpeakerService, tone_service: ToneService, frozen_clock: Clock
) -> None:
    speaker_id = _register(speaker_service)
    observation = tone_service.record_observation(
        speaker_id=speaker_id, tone=ToneLabel.HAWKISH, score=0.6, source_text="we will hold rates"
    )
    assert observation.observed_at == frozen_clock()


@pytest.mark.unit
def test_record_observation_hashes_source_text_and_does_not_keep_it(
    speaker_service: SpeakerService, tone_service: ToneService
) -> None:
    speaker_id = _register(speaker_service)
    text = "inflation remains elevated"
    observation = tone_service.record_observation(
        speaker_id=speaker_id, tone=ToneLabel.HAWKISH, score=0.7, source_text=text
    )
    assert observation.source_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert text not in observation.model_dump_json()


@pytest.mark.unit
def test_record_observation_with_explicit_timestamp(
    speaker_service: SpeakerService, tone_service: ToneService
) -> None:
    speaker_id = _register(speaker_service)
    when = datetime(2025, 12, 1, 8, 0, tzinfo=UTC)
    observation = tone_service.record_observation(
        speaker_id=speaker_id,
        tone=ToneLabel.DOVISH,
        score=-0.3,
        source_text="growth is slowing",
        observed_at=when,
    )
    assert observation.observed_at == when


@pytest.mark.unit
def test_record_observation_for_unknown_speaker_raises_not_found(
    tone_service: ToneService,
) -> None:
    with pytest.raises(EntityNotFoundError):
        tone_service.record_observation(
            speaker_id=UUID(int=404),
            tone=ToneLabel.NEUTRAL,
            score=0.0,
            source_text="text",
        )


@pytest.mark.unit
def test_record_observation_rejects_out_of_range_score(
    speaker_service: SpeakerService, tone_service: ToneService
) -> None:
    speaker_id = _register(speaker_service)
    with pytest.raises(InvalidInputError):
        tone_service.record_observation(
            speaker_id=speaker_id, tone=ToneLabel.HAWKISH, score=2.0, source_text="text"
        )


@pytest.mark.unit
def test_observations_for_unknown_speaker_raises_not_found(tone_service: ToneService) -> None:
    with pytest.raises(EntityNotFoundError):
        tone_service.observations_for(UUID(int=404))


@pytest.mark.unit
def test_observations_for_returns_recorded_observations(
    speaker_service: SpeakerService, tone_service: ToneService
) -> None:
    speaker_id = _register(speaker_service)
    tone_service.record_observation(
        speaker_id=speaker_id, tone=ToneLabel.HAWKISH, score=0.5, source_text="a"
    )
    observations = tone_service.observations_for(speaker_id)
    assert len(observations) == 1
    assert observations[0].speaker_id == speaker_id

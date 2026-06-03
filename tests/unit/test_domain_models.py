"""Tests for the domain models and their validation (CLAUDE.md sections 3 and 5)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from cbt_core.domain.models import Speaker, ToneObservation
from cbt_core.domain.registry import CentralBank
from cbt_core.domain.tone import ToneLabel

_SHA = "a" * 64


def _speaker() -> Speaker:
    return Speaker(
        id=UUID(int=1), name="Jerome Powell", central_bank=CentralBank.FEDERAL_RESERVE, role="Chair"
    )


@pytest.mark.unit
def test_speaker_is_immutable() -> None:
    speaker = _speaker()
    with pytest.raises(ValidationError):
        speaker.name = "someone else"  # type: ignore[misc]  # assigning to a frozen field on purpose, to assert runtime immutability


@pytest.mark.unit
def test_speaker_rejects_empty_name() -> None:
    with pytest.raises(ValidationError):
        Speaker(id=UUID(int=1), name="", central_bank=CentralBank.ECB, role="President")


@pytest.mark.unit
def test_speaker_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Speaker(
            id=UUID(int=1),
            name="x",
            central_bank=CentralBank.ECB,
            role="President",
            unexpected="nope",  # type: ignore[call-arg]  # passing an unknown kwarg on purpose, to assert extra="forbid" rejects it
        )


@pytest.mark.unit
def test_observation_requires_timezone_aware_datetime() -> None:
    with pytest.raises(ValidationError):
        ToneObservation(
            id=UUID(int=2),
            speaker_id=UUID(int=1),
            observed_at=datetime(2026, 1, 1, 12, 0, 0),
            tone=ToneLabel.HAWKISH,
            score=0.5,
            source_sha256=_SHA,
        )


@pytest.mark.unit
@pytest.mark.parametrize("score", [-1.5, 1.5, 2.0])
def test_observation_rejects_out_of_range_score(score: float) -> None:
    with pytest.raises(ValidationError):
        ToneObservation(
            id=UUID(int=2),
            speaker_id=UUID(int=1),
            observed_at=datetime(2026, 1, 1, tzinfo=UTC),
            tone=ToneLabel.NEUTRAL,
            score=score,
            source_sha256=_SHA,
        )


@pytest.mark.unit
@pytest.mark.parametrize("bad_sha", ["", "xyz", "A" * 64, "a" * 63])
def test_observation_rejects_malformed_sha256(bad_sha: str) -> None:
    with pytest.raises(ValidationError):
        ToneObservation(
            id=UUID(int=2),
            speaker_id=UUID(int=1),
            observed_at=datetime(2026, 1, 1, tzinfo=UTC),
            tone=ToneLabel.DOVISH,
            score=0.0,
            source_sha256=bad_sha,
        )


@pytest.mark.unit
def test_observation_accepts_valid_values() -> None:
    observation = ToneObservation(
        id=UUID(int=2),
        speaker_id=UUID(int=1),
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
        tone=ToneLabel.MIXED,
        score=-0.25,
        source_sha256=_SHA,
    )
    assert observation.tone is ToneLabel.MIXED
    assert observation.score == -0.25

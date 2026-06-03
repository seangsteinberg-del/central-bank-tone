"""Tests for domain/ORM mappers (CLAUDE.md section 2)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from cbt_core.domain.models import Speaker, ToneObservation
from cbt_core.domain.registry import CentralBank
from cbt_core.domain.tone import ToneLabel
from cbt_core.persistence.mappers import (
    observation_to_row,
    row_to_observation,
    row_to_speaker,
    speaker_to_row,
)
from cbt_core.persistence.rows import ToneObservationRow

_SHA = "b" * 64


@pytest.mark.unit
def test_speaker_round_trips_through_a_row() -> None:
    speaker = Speaker(
        id=UUID(int=1), name="Christine Lagarde", central_bank=CentralBank.ECB, role="President"
    )
    assert row_to_speaker(speaker_to_row(speaker)) == speaker


@pytest.mark.unit
def test_observation_round_trips_through_a_row() -> None:
    observation = ToneObservation(
        id=UUID(int=2),
        speaker_id=UUID(int=1),
        observed_at=datetime(2026, 3, 1, 9, 30, tzinfo=UTC),
        tone=ToneLabel.HAWKISH,
        score=0.8,
        source_sha256=_SHA,
    )
    assert row_to_observation(observation_to_row(observation)) == observation


@pytest.mark.unit
def test_naive_stored_timestamp_is_read_back_as_utc() -> None:
    # Simulates the SQLite path, where the dialect cannot persist the offset.
    row = ToneObservationRow(
        id=UUID(int=3),
        speaker_id=UUID(int=1),
        observed_at=datetime(2026, 3, 1, 9, 30),
        tone=ToneLabel.DOVISH,
        score=-0.4,
        source_sha256=_SHA,
    )
    observation = row_to_observation(row)
    assert observation.observed_at.tzinfo is not None
    assert observation.observed_at == datetime(2026, 3, 1, 9, 30, tzinfo=UTC)

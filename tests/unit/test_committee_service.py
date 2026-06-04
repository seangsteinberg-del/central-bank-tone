"""Unit tests for the committee tone-movement read model (CLAUDE.md section 5).

These pin the point-in-time semantics that make the speech page honest: a member counts only once
they have spoken, a member's "current" reading is their latest one on or before the speech (never a
later one), the overall move is the mean of members' individual shifts, and members are ordered by
how much they moved.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session, sessionmaker

from cbt_core import (
    CentralBank,
    CommitteeService,
    IngestionService,
    SpeakerService,
    ToneAnalysis,
    ToneLabel,
)
from cbt_core.domain.qa import RetrievedChunk
from cbt_core.exceptions import EntityNotFoundError
from cbt_core.services._support import IdFactory


class _ScriptedLlm:
    """An LlmClient whose tone for a speech is looked up by its text, so a test can script scores."""

    def __init__(self, by_text: dict[str, tuple[float, ToneLabel]]) -> None:
        self._by_text = by_text

    def analyze_tone(self, speech_text: str) -> ToneAnalysis:
        score, tone = self._by_text[speech_text]
        return ToneAnalysis(
            summary=f"Summary of: {speech_text}",
            tone=tone,
            score=score,
            rationale="scripted for the test",
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0, 0.0] for _ in texts]

    def answer(self, question: str, chunks: Sequence[RetrievedChunk]) -> str:
        return "not used"


def _tone_for(score: float) -> ToneLabel:
    """A coherent label for a scripted score (the movement only depends on the score)."""
    if score > 0.05:
        return ToneLabel.HAWKISH
    if score < -0.05:
        return ToneLabel.DOVISH
    return ToneLabel.NEUTRAL


# (speaker name, central bank, role, year, score). One speech per row, delivered on March 1.
_FED = CentralBank.FEDERAL_RESERVE
_SCENARIO = [
    ("Ben Bernanke", _FED, "Chair", 2006, -0.10),
    ("Janet Yellen", _FED, "Chair", 2014, 0.00),
    ("Janet Yellen", _FED, "Chair", 2016, -0.20),
    ("Jerome Powell", _FED, "Chair", 2017, 0.10),
    ("Jerome Powell", _FED, "Chair", 2018, 0.40),  # the subject speech
    ("Jerome Powell", _FED, "Chair", 2020, 0.90),  # after the subject; must be ignored as-of 2018
    ("Future Person", _FED, "Governor", 2020, 0.50),  # had not spoken as of 2018; excluded
    ("Christine Lagarde", CentralBank.ECB, "President", 2018, 0.30),  # different committee
]


def _seed(
    session_factory: sessionmaker[Session], id_factory: IdFactory
) -> dict[tuple[str, int], UUID]:
    """Seed the scenario and return a map from (speaker name, year) to the ingested speech id."""
    by_text = {
        f"text-{name}-{year}": (score, _tone_for(score)) for name, _, _, year, score in _SCENARIO
    }
    speakers = SpeakerService(session_factory, id_factory=id_factory)
    ingestion = IngestionService(session_factory, _ScriptedLlm(by_text), id_factory=id_factory)
    speech_ids: dict[tuple[str, int], UUID] = {}
    for name, bank, role, year, _score in _SCENARIO:
        speaker = speakers.ensure_speaker(name=name, central_bank=bank, role=role)
        speech = ingestion.ingest_speech(
            speaker_id=speaker.id,
            title=f"{name} {year}",
            url=f"https://example.org/{name}-{year}",
            delivered_at=datetime(year, 3, 1, tzinfo=UTC),
            text=f"text-{name}-{year}",
        )
        speech_ids[(name, year)] = speech.id
    return speech_ids


def test_movement_for_speech_reports_each_member_and_the_overall_shift(
    session_factory: sessionmaker[Session], id_factory: IdFactory
) -> None:
    speech_ids = _seed(session_factory, id_factory)
    movement = CommitteeService(session_factory).movement_for_speech(
        speech_ids[("Jerome Powell", 2018)]
    )

    assert movement.central_bank is _FED
    assert movement.as_of == datetime(2018, 3, 1, tzinfo=UTC)
    # The committee as of 2018 is exactly the three Fed members who had spoken by then.
    assert [m.speaker.name for m in movement.members] == [
        "Jerome Powell",  # moved +0.30, the largest shift
        "Janet Yellen",  # moved -0.20
        "Ben Bernanke",  # single reading, no prior shift -> sorted last
    ]
    # Lagarde (different committee) and the future Fed governor (no reading yet) are excluded.
    assert "Christine Lagarde" not in {m.speaker.name for m in movement.members}
    assert "Future Person" not in {m.speaker.name for m in movement.members}


def test_subject_uses_the_clicked_speech_not_a_later_one(
    session_factory: sessionmaker[Session], id_factory: IdFactory
) -> None:
    speech_ids = _seed(session_factory, id_factory)
    movement = CommitteeService(session_factory).movement_for_speech(
        speech_ids[("Jerome Powell", 2018)]
    )

    subject = movement.subject
    assert subject.speaker.name == "Jerome Powell"
    assert subject.is_subject is True
    # Point-in-time: the 2018 reading is current and the 2017 one is prior; the 2020 reading
    # (which exists in the corpus) must not leak in.
    assert subject.current_score == pytest.approx(0.40)
    assert subject.previous_score == pytest.approx(0.10)
    assert subject.delta == pytest.approx(0.30)
    assert subject.observation_count == 2


def test_member_without_a_prior_reading_has_no_delta(
    session_factory: sessionmaker[Session], id_factory: IdFactory
) -> None:
    speech_ids = _seed(session_factory, id_factory)
    movement = CommitteeService(session_factory).movement_for_speech(
        speech_ids[("Jerome Powell", 2018)]
    )

    bernanke = next(m for m in movement.members if m.speaker.name == "Ben Bernanke")
    assert bernanke.observation_count == 1
    assert bernanke.previous_score is None
    assert bernanke.delta is None


def test_committee_tone_and_overall_delta_are_means_over_the_right_members(
    session_factory: sessionmaker[Session], id_factory: IdFactory
) -> None:
    speech_ids = _seed(session_factory, id_factory)
    movement = CommitteeService(session_factory).movement_for_speech(
        speech_ids[("Jerome Powell", 2018)]
    )

    # Standing tone is the mean of the three members' current scores: (0.40 - 0.20 - 0.10) / 3.
    assert movement.committee_tone == pytest.approx((0.40 - 0.20 - 0.10) / 3)
    # The overall move averages only the members that have a prior reading: (+0.30, -0.20) / 2.
    assert movement.overall_delta == pytest.approx((0.30 - 0.20) / 2)
    assert movement.movers == 2


def test_single_reading_committee_has_no_overall_delta(
    session_factory: sessionmaker[Session], id_factory: IdFactory
) -> None:
    speakers = SpeakerService(session_factory, id_factory=id_factory)
    by_text = {"only-speech": (0.25, ToneLabel.HAWKISH)}
    ingestion = IngestionService(session_factory, _ScriptedLlm(by_text), id_factory=id_factory)
    speaker = speakers.register_speaker(name="Solo Governor", central_bank=_FED, role="Governor")
    speech = ingestion.ingest_speech(
        speaker_id=speaker.id,
        title="The only speech",
        url="https://example.org/solo",
        delivered_at=datetime(2025, 6, 1, tzinfo=UTC),
        text="only-speech",
    )

    movement = CommitteeService(session_factory).movement_for_speech(speech.id)
    assert [m.speaker.name for m in movement.members] == ["Solo Governor"]
    assert movement.subject.delta is None
    assert movement.overall_delta is None
    assert movement.movers == 0
    assert movement.committee_tone == pytest.approx(0.25)


def test_unknown_speech_raises_not_found(session_factory: sessionmaker[Session]) -> None:
    with pytest.raises(EntityNotFoundError):
        CommitteeService(session_factory).movement_for_speech(uuid4())

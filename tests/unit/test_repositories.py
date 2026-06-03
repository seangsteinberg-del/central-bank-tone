"""Repository tests against an in-process SQLite engine (CLAUDE.md section 5)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from sqlalchemy.orm import Session, sessionmaker

from cbt_core.domain.models import Speaker, ToneObservation
from cbt_core.domain.registry import CentralBank
from cbt_core.domain.tone import ToneLabel
from cbt_core.exceptions import EntityNotFoundError
from cbt_core.persistence.repositories import SpeakerRepository, ToneObservationRepository

_SHA = "c" * 64


def _make_speaker(int_id: int, name: str) -> Speaker:
    return Speaker(
        id=UUID(int=int_id), name=name, central_bank=CentralBank.FEDERAL_RESERVE, role="Member"
    )


@pytest.mark.unit
def test_add_then_get_returns_the_speaker(session_factory: sessionmaker[Session]) -> None:
    speaker = _make_speaker(1, "Powell")
    with session_factory() as session:
        SpeakerRepository(session).add(speaker)
        session.commit()
    with session_factory() as session:
        assert SpeakerRepository(session).get(speaker.id) == speaker


@pytest.mark.unit
def test_get_unknown_speaker_raises_not_found(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session, pytest.raises(EntityNotFoundError):
        SpeakerRepository(session).get(UUID(int=999))


@pytest.mark.unit
def test_exists_reflects_presence(session_factory: sessionmaker[Session]) -> None:
    speaker = _make_speaker(1, "Powell")
    with session_factory() as session:
        repo = SpeakerRepository(session)
        assert repo.exists(speaker.id) is False
        repo.add(speaker)
        session.commit()
        assert repo.exists(speaker.id) is True


@pytest.mark.unit
def test_list_speakers_is_ordered_by_name(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        repo = SpeakerRepository(session)
        repo.add(_make_speaker(1, "Zeta"))
        repo.add(_make_speaker(2, "Alpha"))
        session.commit()
    with session_factory() as session:
        names = [speaker.name for speaker in SpeakerRepository(session).list_all()]
    assert names == ["Alpha", "Zeta"]


@pytest.mark.unit
def test_append_then_list_returns_observations_oldest_first(
    session_factory: sessionmaker[Session],
) -> None:
    speaker = _make_speaker(1, "Powell")
    older = ToneObservation(
        id=UUID(int=10),
        speaker_id=speaker.id,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
        tone=ToneLabel.DOVISH,
        score=-0.5,
        source_sha256=_SHA,
    )
    newer = ToneObservation(
        id=UUID(int=11),
        speaker_id=speaker.id,
        observed_at=datetime(2026, 2, 1, tzinfo=UTC),
        tone=ToneLabel.HAWKISH,
        score=0.5,
        source_sha256=_SHA,
    )
    with session_factory() as session:
        SpeakerRepository(session).add(speaker)
        observations = ToneObservationRepository(session)
        observations.append(newer)
        observations.append(older)
        session.commit()
    with session_factory() as session:
        listed = ToneObservationRepository(session).list_for_speaker(speaker.id)
    assert [observation.id for observation in listed] == [older.id, newer.id]


@pytest.mark.unit
def test_list_for_unknown_speaker_is_empty(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        assert ToneObservationRepository(session).list_for_speaker(UUID(int=404)) == []

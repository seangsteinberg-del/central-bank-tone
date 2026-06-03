"""Tests for the speech repository against in-process SQLite (CLAUDE.md section 5)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session, sessionmaker

from cbt_core.domain.models import Speaker
from cbt_core.domain.registry import CentralBank
from cbt_core.domain.speech import Speech
from cbt_core.domain.tone import ToneLabel
from cbt_core.exceptions import EntityNotFoundError
from cbt_core.persistence.repositories import SpeakerRepository, SpeechRepository

_SPEAKER_ID = UUID(int=1)
_SHA = "d" * 64


def _speaker() -> Speaker:
    return Speaker(
        id=_SPEAKER_ID, name="Powell", central_bank=CentralBank.FEDERAL_RESERVE, role="Chair"
    )


def _speech(speech_id: int, *, sha: str = _SHA, delivered: datetime | None = None) -> Speech:
    return Speech(
        id=UUID(int=speech_id),
        speaker_id=_SPEAKER_ID,
        central_bank=CentralBank.FEDERAL_RESERVE,
        title="On the economic outlook",
        url=f"https://example.org/speech/{speech_id}",
        delivered_at=delivered or datetime(2026, 1, 1, tzinfo=UTC),
        language="en",
        text="full speech text",
        source_sha256=sha,
        summary="A short summary.",
        tone=ToneLabel.HAWKISH,
        score=0.5,
        lexicon_score=0.3,
        rationale="Hawkish emphasis.",
    )


@pytest.fixture
def _with_speaker(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        SpeakerRepository(session).add(_speaker())
        session.commit()


@pytest.mark.unit
@pytest.mark.usefixtures("_with_speaker")
def test_add_then_get_returns_the_speech(session_factory: sessionmaker[Session]) -> None:
    speech = _speech(10)
    with session_factory() as session:
        SpeechRepository(session).add(speech)
        session.commit()
    with session_factory() as session:
        assert SpeechRepository(session).get(speech.id) == speech


@pytest.mark.unit
def test_get_unknown_speech_raises_not_found(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session, pytest.raises(EntityNotFoundError):
        SpeechRepository(session).get(UUID(int=999))


@pytest.mark.unit
def test_speech_requires_timezone_aware_delivered_at() -> None:
    with pytest.raises(ValidationError):
        _speech(10, delivered=datetime(2026, 1, 1, 12, 0))


@pytest.mark.unit
@pytest.mark.usefixtures("_with_speaker")
def test_find_by_source_sha256(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        repo = SpeechRepository(session)
        assert repo.find_by_source_sha256(_SHA) is None
        repo.add(_speech(10))
        session.commit()
        assert repo.find_by_source_sha256(_SHA) is not None


@pytest.mark.unit
@pytest.mark.usefixtures("_with_speaker")
def test_list_for_speaker_is_most_recent_first(session_factory: sessionmaker[Session]) -> None:
    older = _speech(10, sha="a" * 64, delivered=datetime(2025, 1, 1, tzinfo=UTC))
    newer = _speech(11, sha="b" * 64, delivered=datetime(2026, 6, 1, tzinfo=UTC))
    with session_factory() as session:
        repo = SpeechRepository(session)
        repo.add(older)
        repo.add(newer)
        session.commit()
    with session_factory() as session:
        listed = SpeechRepository(session).list_for_speaker(_SPEAKER_ID)
    assert [speech.id for speech in listed] == [newer.id, older.id]

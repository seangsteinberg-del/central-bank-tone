"""The speech table is append-only at the database level (CLAUDE.md sections 4, 5, 13)."""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import Connection, text
from sqlalchemy.exc import DBAPIError

_INSERT_SPEAKER = text(
    "INSERT INTO speaker (id, name, central_bank, role) VALUES (:id, :name, :cb, :role)"
)
_INSERT_SPEECH = text(
    "INSERT INTO speech "
    "(id, speaker_id, central_bank, title, url, delivered_at, language, body, "
    "source_sha256, summary, tone, score, lexicon_score, rationale) "
    "VALUES (:id, :speaker_id, :cb, :title, :url, now(), :lang, :body, :sha, :summary, "
    ":tone, :score, :lex, :rationale)"
)
_SHA = "e" * 64


def _insert_speaker(connection: Connection) -> str:
    speaker_id = uuid4()
    connection.execute(
        _INSERT_SPEAKER,
        {"id": speaker_id, "name": "Powell", "cb": "federal_reserve", "role": "Chair"},
    )
    return str(speaker_id)


def _insert_speech(connection: Connection, speaker_id: str, *, sha: str = _SHA) -> str:
    speech_id = uuid4()
    connection.execute(
        _INSERT_SPEECH,
        {
            "id": speech_id,
            "speaker_id": speaker_id,
            "cb": "federal_reserve",
            "title": "Outlook",
            "url": f"https://example.org/s/{speech_id}",
            "lang": "en",
            "body": "the speech text",
            "sha": sha,
            "summary": "a summary",
            "tone": "hawkish",
            "score": 0.5,
            "lex": 0.3,
            "rationale": "hawkish emphasis",
        },
    )
    return str(speech_id)


def _seed_speech(connection: Connection) -> str:
    return _insert_speech(connection, _insert_speaker(connection))


@pytest.mark.integration
def test_speech_rejects_mutation_of_historical_row(pg_connection: Connection) -> None:
    speech_id = _seed_speech(pg_connection)
    with pytest.raises(DBAPIError):
        pg_connection.execute(text("UPDATE speech SET score = 0 WHERE id = :id"), {"id": speech_id})


@pytest.mark.integration
def test_speech_rejects_deletion_of_historical_row(pg_connection: Connection) -> None:
    speech_id = _seed_speech(pg_connection)
    with pytest.raises(DBAPIError):
        pg_connection.execute(text("DELETE FROM speech WHERE id = :id"), {"id": speech_id})


@pytest.mark.integration
def test_speech_source_sha256_is_unique(pg_connection: Connection) -> None:
    speaker_id = _insert_speaker(pg_connection)
    _insert_speech(pg_connection, speaker_id, sha=_SHA)
    with pytest.raises(DBAPIError):
        # A second speech with the same source hash violates the unique constraint.
        _insert_speech(pg_connection, speaker_id, sha=_SHA)

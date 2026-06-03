"""Foreign-key semantics on real Postgres (CLAUDE.md section 6)."""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import Connection, text
from sqlalchemy.exc import DBAPIError

_INSERT_SPEAKER = text(
    "INSERT INTO speaker (id, name, central_bank, role) VALUES (:id, :name, :cb, :role)"
)
_INSERT_OBSERVATION = text(
    "INSERT INTO tone_observation "
    "(id, speaker_id, observed_at, tone, score, source_sha256) "
    "VALUES (:id, :speaker_id, now(), :tone, :score, :sha)"
)
_SHA = "a" * 64


@pytest.mark.integration
def test_deleting_a_speaker_with_observations_is_restricted(pg_connection: Connection) -> None:
    speaker_id = uuid4()
    pg_connection.execute(
        _INSERT_SPEAKER,
        {"id": speaker_id, "name": "Powell", "cb": "federal_reserve", "role": "Chair"},
    )
    pg_connection.execute(
        _INSERT_OBSERVATION,
        {"id": uuid4(), "speaker_id": speaker_id, "tone": "hawkish", "score": 0.5, "sha": _SHA},
    )
    with pytest.raises(DBAPIError):
        pg_connection.execute(text("DELETE FROM speaker WHERE id = :id"), {"id": speaker_id})


@pytest.mark.integration
def test_observation_requires_an_existing_speaker(pg_connection: Connection) -> None:
    with pytest.raises(DBAPIError):
        pg_connection.execute(
            _INSERT_OBSERVATION,
            {"id": uuid4(), "speaker_id": uuid4(), "tone": "dovish", "score": 0.0, "sha": _SHA},
        )


@pytest.mark.integration
def test_deleting_a_speaker_without_observations_succeeds(pg_connection: Connection) -> None:
    speaker_id = uuid4()
    pg_connection.execute(
        _INSERT_SPEAKER,
        {"id": speaker_id, "name": "Solo", "cb": "ecb", "role": "President"},
    )
    pg_connection.execute(text("DELETE FROM speaker WHERE id = :id"), {"id": speaker_id})
    remaining = pg_connection.execute(
        text("SELECT count(*) FROM speaker WHERE id = :id"), {"id": speaker_id}
    ).scalar_one()
    assert remaining == 0

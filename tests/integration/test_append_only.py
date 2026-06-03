"""The append-only guarantee is enforced by the database, not just the app.

These are security-sensitive tests (CLAUDE.md sections 4, 5, 13): the trigger on
``tone_observation`` must reject any UPDATE or DELETE of a historical row.
"""

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


def _seed_observation(connection: Connection) -> str:
    speaker_id = uuid4()
    observation_id = uuid4()
    connection.execute(
        _INSERT_SPEAKER,
        {"id": speaker_id, "name": "Powell", "cb": "federal_reserve", "role": "Chair"},
    )
    connection.execute(
        _INSERT_OBSERVATION,
        {
            "id": observation_id,
            "speaker_id": speaker_id,
            "tone": "hawkish",
            "score": 0.5,
            "sha": _SHA,
        },
    )
    return str(observation_id)


@pytest.mark.integration
def test_tone_observation_rejects_mutation_of_historical_row(pg_connection: Connection) -> None:
    observation_id = _seed_observation(pg_connection)
    with pytest.raises(DBAPIError):
        pg_connection.execute(
            text("UPDATE tone_observation SET score = 0 WHERE id = :id"), {"id": observation_id}
        )


@pytest.mark.integration
def test_tone_observation_rejects_deletion_of_historical_row(pg_connection: Connection) -> None:
    observation_id = _seed_observation(pg_connection)
    with pytest.raises(DBAPIError):
        pg_connection.execute(
            text("DELETE FROM tone_observation WHERE id = :id"), {"id": observation_id}
        )

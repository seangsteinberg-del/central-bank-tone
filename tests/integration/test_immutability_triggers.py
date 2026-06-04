"""``create_immutability_triggers`` installs the append-only guarantee on a create_all schema.

The no-pgvector live setup (ADR 0018) builds its schema with ``create_demo_schema`` rather than the
alembic migrations, so ``create_immutability_triggers`` must install the same database-level block
the migrations do. Security-sensitive (CLAUDE.md sections 4, 5): a trigger must reject any UPDATE or
DELETE of a historical row. Skipped when Docker is unavailable.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError

from cbt_core import create_demo_schema, create_immutability_triggers

# create_demo_schema builds the enums from the Python member names, so the labels are upper-case
# (unlike the migrations, which declare lower-case values).
_INSERT_SPEAKER = text(
    "INSERT INTO speaker (id, name, central_bank, role) "
    "VALUES (:id, 'Powell', 'FEDERAL_RESERVE', 'Chair')"
)
_INSERT_OBSERVATION = text(
    "INSERT INTO tone_observation (id, speaker_id, observed_at, tone, score, source_sha256) "
    "VALUES (:id, :speaker_id, now(), 'HAWKISH', 0.5, :sha)"
)


@pytest.mark.integration
def test_create_immutability_triggers_blocks_mutation(fresh_postgres_url: str) -> None:
    engine = create_engine(fresh_postgres_url, future=True)
    try:
        create_demo_schema(engine)
        speaker_id, observation_id = uuid4(), uuid4()
        with engine.begin() as connection:
            connection.execute(_INSERT_SPEAKER, {"id": speaker_id})
            connection.execute(
                _INSERT_OBSERVATION,
                {"id": observation_id, "speaker_id": speaker_id, "sha": "a" * 64},
            )

        # Before the triggers are installed, an update is permitted.
        with engine.begin() as connection:
            connection.execute(
                text("UPDATE tone_observation SET score = 0.1 WHERE id = :id"),
                {"id": observation_id},
            )

        create_immutability_triggers(engine)

        # After, the database itself rejects an update of the historical row.
        with pytest.raises(DBAPIError), engine.begin() as connection:
            connection.execute(
                text("UPDATE tone_observation SET score = 0.2 WHERE id = :id"),
                {"id": observation_id},
            )
    finally:
        engine.dispose()

"""create_all and the alembic migrations must persist enum labels identically (one schema spine).

Regression guard for a real, latent divergence: SQLAlchemy stores a Python enum by member NAME by
default, while the migrations freeze the lowercase member VALUES, so a database built by
``create_all`` (the SQLite demo and the no-pgvector live DB, ADR 0018) stored ``FEDERAL_RESERVE``
where a migrated database stored ``federal_reserve`` (CLAUDE.md section 2). rows.py now pins
``values_callable`` so the two build paths agree and the ORM round-trips against a migrated
database; this test fails if that pin is dropped.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session

from cbt_core import create_demo_schema, make_demo_engine
from cbt_core.domain.registry import CentralBank
from cbt_core.domain.tone import ToneLabel
from cbt_core.persistence.rows import SpeakerRow, ToneObservationRow


def test_create_all_persists_enum_values_not_member_names() -> None:
    engine = make_demo_engine(None)
    create_demo_schema(engine)
    speaker_id = uuid4()
    with Session(engine) as session:
        session.add(
            SpeakerRow(
                id=speaker_id,
                name="Powell",
                central_bank=CentralBank.FEDERAL_RESERVE,
                role="Chair",
            )
        )
        session.add(
            ToneObservationRow(
                id=uuid4(),
                speaker_id=speaker_id,
                observed_at=datetime(2025, 1, 1, tzinfo=UTC),
                tone=ToneLabel.HAWKISH,
                score=0.5,
                source_sha256="a" * 64,
            )
        )
        session.commit()
    with engine.connect() as connection:
        bank = connection.execute(text("SELECT central_bank FROM speaker")).scalar_one()
        tone = connection.execute(text("SELECT tone FROM tone_observation")).scalar_one()
    # The stored label is the lowercase migration value, not the uppercase Python member name.
    assert bank == CentralBank.FEDERAL_RESERVE.value == "federal_reserve"
    assert tone == ToneLabel.HAWKISH.value == "hawkish"
    assert bank != CentralBank.FEDERAL_RESERVE.name
    assert tone != ToneLabel.HAWKISH.name

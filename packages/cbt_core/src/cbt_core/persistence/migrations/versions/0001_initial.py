"""initial schema: speaker and append-only tone_observation

Revision ID: 0001
Revises:
Create Date: 2026-06-03

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = None
branch_labels = None
depends_on = None

# Enum values are frozen as literals here so this migration is a stable snapshot, independent
# of later additions to the cbt_core schema spine (CLAUDE.md section 6).
_CENTRAL_BANK = sa.Enum(
    "federal_reserve",
    "ecb",
    "bank_of_england",
    "bank_of_japan",
    "bank_of_canada",
    "reserve_bank_of_australia",
    "swiss_national_bank",
    "peoples_bank_of_china",
    name="central_bank",
)
_TONE_LABEL = sa.Enum("hawkish", "dovish", "neutral", "mixed", name="tone_label")


def upgrade() -> None:
    op.create_table(
        "speaker",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("central_bank", _CENTRAL_BANK, nullable=False),
        sa.Column("role", sa.String(length=200), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_speaker"),
    )
    op.create_table(
        "tone_observation",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("speaker_id", sa.Uuid(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("tone", _TONE_LABEL, nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.CheckConstraint(
            "score >= -1.0 AND score <= 1.0", name="ck_tone_observation_score_in_range"
        ),
        sa.CheckConstraint("length(source_sha256) = 64", name="ck_tone_observation_sha256_length"),
        sa.ForeignKeyConstraint(
            ["speaker_id"],
            ["speaker.id"],
            name="fk_tone_observation_speaker_id_speaker",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_tone_observation"),
    )
    op.create_index("ix_tone_observation_speaker_id", "tone_observation", ["speaker_id"])

    # Enforce the append-only guarantee at the database level (CLAUDE.md section 4): a trigger
    # rejects any UPDATE or DELETE on a historical observation.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION cbt_block_mutation() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'tone_observation is append-only; % is not permitted', TG_OP;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER tone_observation_append_only
        BEFORE UPDATE OR DELETE ON tone_observation
        FOR EACH ROW EXECUTE FUNCTION cbt_block_mutation();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS tone_observation_append_only ON tone_observation;")
    op.execute("DROP FUNCTION IF EXISTS cbt_block_mutation();")
    op.drop_index("ix_tone_observation_speaker_id", table_name="tone_observation")
    op.drop_table("tone_observation")
    op.drop_table("speaker")
    _TONE_LABEL.drop(op.get_bind())
    _CENTRAL_BANK.drop(op.get_bind())

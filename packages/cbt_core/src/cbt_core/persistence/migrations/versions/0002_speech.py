"""add the append-only speech table

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-03

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | None = "0001"
branch_labels = None
depends_on = None

# Reference the enum types created by migration 0001; do not recreate them.
_CENTRAL_BANK = postgresql.ENUM(name="central_bank", create_type=False)
_TONE_LABEL = postgresql.ENUM(name="tone_label", create_type=False)


def upgrade() -> None:
    op.create_table(
        "speech",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("speaker_id", sa.Uuid(), nullable=False),
        sa.Column("central_bank", _CENTRAL_BANK, nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("url", sa.String(length=2000), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("language", sa.String(length=12), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("tone", _TONE_LABEL, nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("lexicon_score", sa.Float(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.CheckConstraint("score >= -1.0 AND score <= 1.0", name="ck_speech_score_in_range"),
        sa.CheckConstraint(
            "lexicon_score >= -1.0 AND lexicon_score <= 1.0",
            name="ck_speech_lexicon_score_in_range",
        ),
        sa.CheckConstraint("length(source_sha256) = 64", name="ck_speech_sha256_length"),
        sa.ForeignKeyConstraint(
            ["speaker_id"],
            ["speaker.id"],
            name="fk_speech_speaker_id_speaker",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_speech"),
        sa.UniqueConstraint("source_sha256", name="uq_speech_source_sha256"),
    )
    op.create_index("ix_speech_speaker_id", "speech", ["speaker_id"])

    # Reuse the append-only trigger function from migration 0001.
    op.execute(
        """
        CREATE TRIGGER speech_append_only
        BEFORE UPDATE OR DELETE ON speech
        FOR EACH ROW EXECUTE FUNCTION cbt_block_mutation();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS speech_append_only ON speech;")
    op.drop_index("ix_speech_speaker_id", table_name="speech")
    op.drop_table("speech")

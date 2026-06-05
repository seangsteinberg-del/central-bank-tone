"""add the derived speech_stance table (structured-pipeline decomposition, ADR 0021)

Revision ID: 0006
Revises: 0005
Create Date: 2026-06-05

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: str | None = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The structured stance decomposition (ADR 0021) is derived, recomputable data, like the speech
    # chunks: a forward-looking rate-path measure, a directional uncertainty band, the per-aspect net
    # map, and the three cross-check nets. It lives in its own table, NOT on the append-only speech
    # row, so it can be (re)computed for any speech as the method improves without touching the
    # immutable tone record. One row per speech; the row's presence means the speech has been scored.
    op.create_table(
        "speech_stance",
        sa.Column(
            "speech_id",
            sa.Uuid(),
            sa.ForeignKey("speech.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("rate_path", sa.Float(), nullable=False),
        sa.Column("uncertainty", sa.Float(), nullable=False),
        sa.Column("structured_net", sa.Float(), nullable=False),
        sa.Column("classifier_net", sa.Float(), nullable=False),
        sa.Column("lexicon_net", sa.Float(), nullable=False),
        sa.Column("needs_review", sa.Boolean(), nullable=False),
        sa.Column("aspect_scores", sa.JSON(), nullable=False),
        sa.Column("model_id", sa.String(length=100), nullable=False, server_default="unknown"),
        sa.CheckConstraint("rate_path >= -1.0 AND rate_path <= 1.0", name="rate_path_in_range"),
        sa.CheckConstraint(
            "uncertainty >= 0.0 AND uncertainty <= 1.0", name="uncertainty_in_range"
        ),
    )


def downgrade() -> None:
    op.drop_table("speech_stance")

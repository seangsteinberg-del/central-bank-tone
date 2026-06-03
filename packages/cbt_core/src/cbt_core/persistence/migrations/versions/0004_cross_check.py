"""add lexicon cross-check columns (lexicon_score, needs_review)

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-03

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: str | None = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The model/lexicon cross-check, persisted so the tone series carries a confidence marker.
    op.add_column(
        "tone_observation",
        sa.Column("lexicon_score", sa.Float(), nullable=False, server_default="0.0"),
    )
    op.add_column(
        "tone_observation",
        sa.Column("needs_review", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "speech",
        sa.Column("needs_review", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_check_constraint(
        "lexicon_score_in_range",
        "tone_observation",
        "lexicon_score >= -1.0 AND lexicon_score <= 1.0",
    )


def downgrade() -> None:
    op.drop_constraint("lexicon_score_in_range", "tone_observation", type_="check")
    op.drop_column("speech", "needs_review")
    op.drop_column("tone_observation", "needs_review")
    op.drop_column("tone_observation", "lexicon_score")

"""add the structured-pipeline fields to speech (rate_path, uncertainty, aspect_scores)

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
    # The structured stance pipeline (ADR 0021) adds a forward-looking rate-path measure, a
    # directional uncertainty band, and a per-aspect net-hawkishness map. All are nullable: a speech
    # scored before the pipeline existed simply has no structured detail until it is re-scored.
    op.add_column("speech", sa.Column("rate_path", sa.Float(), nullable=True))
    op.add_column("speech", sa.Column("uncertainty", sa.Float(), nullable=True))
    op.add_column("speech", sa.Column("aspect_scores", sa.JSON(), nullable=True))
    op.create_check_constraint(
        "rate_path_in_range",
        "speech",
        "rate_path IS NULL OR (rate_path >= -1.0 AND rate_path <= 1.0)",
    )
    op.create_check_constraint(
        "uncertainty_in_range",
        "speech",
        "uncertainty IS NULL OR (uncertainty >= 0.0 AND uncertainty <= 1.0)",
    )


def downgrade() -> None:
    op.drop_constraint("uncertainty_in_range", "speech", type_="check")
    op.drop_constraint("rate_path_in_range", "speech", type_="check")
    op.drop_column("speech", "aspect_scores")
    op.drop_column("speech", "uncertainty")
    op.drop_column("speech", "rate_path")

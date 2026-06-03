"""record the model that produced each speech analysis (speech.model_id)

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-03

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | None = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Pin the model per speech so the tone time series stays comparable as the model changes.
    op.add_column(
        "speech",
        sa.Column("model_id", sa.String(length=100), nullable=False, server_default="unknown"),
    )


def downgrade() -> None:
    op.drop_column("speech", "model_id")

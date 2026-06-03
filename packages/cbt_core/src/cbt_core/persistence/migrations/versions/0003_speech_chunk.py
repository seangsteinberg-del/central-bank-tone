"""add the pgvector speech_chunk table for retrieval

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-03

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | None = "0002"
branch_labels = None
depends_on = None

_EMBEDDING_DIM = 768


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.create_table(
        "speech_chunk",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("speech_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(_EMBEDDING_DIM), nullable=False),
        sa.ForeignKeyConstraint(
            ["speech_id"],
            ["speech.id"],
            name="fk_speech_chunk_speech_id_speech",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_speech_chunk"),
        sa.UniqueConstraint(
            "speech_id", "chunk_index", name="uq_speech_chunk_speech_id_chunk_index"
        ),
    )
    op.create_index("ix_speech_chunk_speech_id", "speech_chunk", ["speech_id"])
    # Approximate-nearest-neighbour index for cosine similarity search.
    op.execute(
        "CREATE INDEX ix_speech_chunk_embedding ON speech_chunk "
        "USING hnsw (embedding vector_cosine_ops);"
    )


def downgrade() -> None:
    op.drop_index("ix_speech_chunk_embedding", table_name="speech_chunk")
    op.drop_index("ix_speech_chunk_speech_id", table_name="speech_chunk")
    op.drop_table("speech_chunk")
    # The vector extension is left installed; other objects may depend on it.

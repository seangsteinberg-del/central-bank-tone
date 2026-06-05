"""ORM row definitions (CLAUDE.md section 2).

These are the persistence representation of the domain. They are a distinct type from the
Pydantic domain models and never leave the persistence layer; mappers translate. The mapped
enum columns reference the schema spine directly, so the registry stays the single source of
truth.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from cbt_core.domain.qa import EMBEDDING_DIM
from cbt_core.domain.registry import CentralBank
from cbt_core.domain.tone import ToneLabel
from cbt_core.persistence.base import Base


def _enum_values(enum_type: type[StrEnum]) -> list[str]:
    """Persist a Python enum by its lowercase member values, matching the migrations.

    SQLAlchemy stores a Python enum by member NAME by default (``FEDERAL_RESERVE``); the alembic
    migrations, the production schema's source of truth (CLAUDE.md sections 2 and 6), define the
    lowercase member VALUES (``federal_reserve``). Passing this as ``values_callable`` makes the
    ORM, and the ``create_all`` schema build used by the SQLite demo and the no-pgvector live
    database (ADR 0018), emit the same labels as the migrations, so the two schema-build paths
    cannot drift and the ORM round-trips against a migrated database.
    """
    return [member.value for member in enum_type]


class SpeakerRow(Base):
    """ORM row for a speaker."""

    __tablename__ = "speaker"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    central_bank: Mapped[CentralBank] = mapped_column(
        Enum(CentralBank, name="central_bank", native_enum=True, values_callable=_enum_values),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(200), nullable=False)


class ToneObservationRow(Base):
    """ORM row for an append-only tone observation.

    The append-only guarantee is enforced at the database level by a trigger created in the
    migration (CLAUDE.md section 4); this class deliberately exposes no mutation path.
    """

    __tablename__ = "tone_observation"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    speaker_id: Mapped[UUID] = mapped_column(
        # The speaker owns its observations' identity but not their lifetime: keep history even
        # if a speaker row is removed, so deletes are RESTRICTed (CLAUDE.md section 6).
        ForeignKey("speaker.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    tone: Mapped[ToneLabel] = mapped_column(
        Enum(ToneLabel, name="tone_label", native_enum=True, values_callable=_enum_values),
        nullable=False,
    )
    score: Mapped[float] = mapped_column(Float, nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    lexicon_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    needs_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        CheckConstraint("score >= -1.0 AND score <= 1.0", name="score_in_range"),
        CheckConstraint(
            "lexicon_score >= -1.0 AND lexicon_score <= 1.0", name="lexicon_score_in_range"
        ),
        CheckConstraint("length(source_sha256) = 64", name="sha256_length"),
    )


class SpeechRow(Base):
    """ORM row for an analyzed speech.

    Immutable once ingested; the append-only guarantee is enforced at the database level by a
    trigger created in the migration (CLAUDE.md section 4).
    """

    __tablename__ = "speech"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    speaker_id: Mapped[UUID] = mapped_column(
        ForeignKey("speaker.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    central_bank: Mapped[CentralBank] = mapped_column(
        Enum(CentralBank, name="central_bank", native_enum=True, values_callable=_enum_values),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    url: Mapped[str] = mapped_column(String(2000), nullable=False)
    delivered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    language: Mapped[str] = mapped_column(String(12), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    tone: Mapped[ToneLabel] = mapped_column(
        Enum(ToneLabel, name="tone_label", native_enum=True, values_callable=_enum_values),
        nullable=False,
    )
    score: Mapped[float] = mapped_column(Float, nullable=False)
    lexicon_score: Mapped[float] = mapped_column(Float, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    needs_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    model_id: Mapped[str] = mapped_column(String(100), nullable=False, default="unknown")

    __table_args__ = (
        UniqueConstraint("source_sha256", name="source_sha256"),
        CheckConstraint("score >= -1.0 AND score <= 1.0", name="score_in_range"),
        CheckConstraint(
            "lexicon_score >= -1.0 AND lexicon_score <= 1.0", name="lexicon_score_in_range"
        ),
        CheckConstraint("length(source_sha256) = 64", name="sha256_length"),
    )


class SpeechStanceRow(Base):
    """ORM row for a speech's structured stance decomposition (ADR 0021).

    Derived, recomputable data (one row per speech), like :class:`SpeechChunkRow`: it is computed
    from the immutable speech text and may be recomputed as the pipeline improves, so this table is
    deliberately NOT append-only. ``CASCADE`` on the foreign key because the decomposition's lifetime
    is owned entirely by its speech.
    """

    __tablename__ = "speech_stance"

    speech_id: Mapped[UUID] = mapped_column(
        ForeignKey("speech.id", ondelete="CASCADE"), primary_key=True
    )
    rate_path: Mapped[float] = mapped_column(Float, nullable=False)
    uncertainty: Mapped[float] = mapped_column(Float, nullable=False)
    structured_net: Mapped[float] = mapped_column(Float, nullable=False)
    classifier_net: Mapped[float] = mapped_column(Float, nullable=False)
    lexicon_net: Mapped[float] = mapped_column(Float, nullable=False)
    needs_review: Mapped[bool] = mapped_column(Boolean, nullable=False)
    aspect_scores: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    model_id: Mapped[str] = mapped_column(String(100), nullable=False, default="unknown")

    __table_args__ = (
        CheckConstraint("rate_path >= -1.0 AND rate_path <= 1.0", name="rate_path_in_range"),
        CheckConstraint("uncertainty >= 0.0 AND uncertainty <= 1.0", name="uncertainty_in_range"),
    )


class SpeechChunkRow(Base):
    """ORM row for an embedded chunk of a speech, used for retrieval (RAG).

    Derived index data: chunks are recomputed deterministically from the immutable speech text,
    so this table is not append-only. The ``embedding`` column is a pgvector vector and is only
    queryable with the similarity operators on PostgreSQL.
    """

    __tablename__ = "speech_chunk"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    speech_id: Mapped[UUID] = mapped_column(
        ForeignKey("speech.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)

    __table_args__ = (
        UniqueConstraint("speech_id", "chunk_index", name="speech_chunk_speech_id_chunk_index"),
    )

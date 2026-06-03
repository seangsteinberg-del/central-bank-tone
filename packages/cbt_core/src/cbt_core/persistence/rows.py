"""ORM row definitions (CLAUDE.md section 2).

These are the persistence representation of the domain. They are a distinct type from the
Pydantic domain models and never leave the persistence layer; mappers translate. The mapped
enum columns reference the schema spine directly, so the registry stays the single source of
truth.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, Enum, Float, ForeignKey, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from cbt_core.domain.registry import CentralBank
from cbt_core.domain.tone import ToneLabel
from cbt_core.persistence.base import Base


class SpeakerRow(Base):
    """ORM row for a speaker."""

    __tablename__ = "speaker"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    central_bank: Mapped[CentralBank] = mapped_column(
        Enum(CentralBank, name="central_bank", native_enum=True), nullable=False
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
        Enum(ToneLabel, name="tone_label", native_enum=True), nullable=False
    )
    score: Mapped[float] = mapped_column(Float, nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        CheckConstraint("score >= -1.0 AND score <= 1.0", name="score_in_range"),
        CheckConstraint("length(source_sha256) = 64", name="sha256_length"),
    )

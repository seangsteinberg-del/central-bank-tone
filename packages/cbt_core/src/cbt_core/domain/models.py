"""Pydantic domain models (CLAUDE.md sections 2 and 3).

Domain models are immutable (``frozen``) and validated on construction. They are distinct from
the ORM rows in ``persistence/``; mappers translate between the two. Pass these types around,
not primitives or dicts.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cbt_core.domain.registry import CentralBank
from cbt_core.domain.tone import ToneLabel

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class Speaker(BaseModel):
    """A central bank official whose speeches the system analyzes.

    Attributes:
        id: Stable identifier for the speaker.
        name: The speaker's full name.
        central_bank: The institution the speaker belongs to (schema spine).
        role: The speaker's role, for example ``"Governor"`` or ``"Board Member"``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    name: str = Field(min_length=1, max_length=200)
    central_bank: CentralBank
    role: str = Field(min_length=1, max_length=200)


class ToneObservation(BaseModel):
    """An append-only record of the tone assigned to a speaker at a point in time.

    Observations are never updated or deleted (CLAUDE.md section 13); a correction is a new
    observation. The database enforces this with a trigger.

    Attributes:
        id: Stable identifier for this observation.
        speaker_id: The speaker this observation is about.
        observed_at: When the observation was made. Must be timezone-aware.
        tone: The discrete tone label (schema spine).
        score: Continuous tone in ``[-1.0, 1.0]`` (negative dovish, positive hawkish).
        source_sha256: Lowercase hex sha256 of the source speech text. The text itself is
            never stored on the observation (CLAUDE.md section 7).
        lexicon_score: The deterministic lexicon's score for the same text, the auditable
            cross-check on ``score``. Defaults to ``0.0`` for manually recorded observations
            that have no lexicon context.
        needs_review: True when ``score`` and ``lexicon_score`` disagreed enough to flag for
            review (ADR 0008). A PM can filter the tone series on this confidence marker.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    speaker_id: UUID
    observed_at: datetime
    tone: ToneLabel
    score: float = Field(ge=-1.0, le=1.0)
    source_sha256: str = Field(pattern=_SHA256_PATTERN)
    lexicon_score: float = Field(ge=-1.0, le=1.0, default=0.0)
    needs_review: bool = False

    @field_validator("observed_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        """Reject naive datetimes so every stored timestamp is unambiguous.

        Raises:
            ValueError: If ``observed_at`` has no timezone information.
        """
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        return value

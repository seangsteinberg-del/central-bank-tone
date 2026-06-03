"""HTTP request and response schemas (CLAUDE.md section 3).

These Pydantic models are the API boundary contract. They are deliberately separate from the
core domain models: the wire format can evolve without changing the domain, and every request
body is validated against one of these before any business logic runs.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from cbt_core import CentralBank, Speaker, ToneLabel, ToneObservation

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class SpeakerCreate(BaseModel):
    """Request body to register a speaker."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    central_bank: CentralBank
    role: str = Field(min_length=1, max_length=200)


class SpeakerResponse(BaseModel):
    """Response body describing a speaker."""

    id: UUID
    name: str
    central_bank: CentralBank
    role: str

    @classmethod
    def from_domain(cls, speaker: Speaker) -> SpeakerResponse:
        """Build the response from a domain :class:`Speaker`."""
        return cls(
            id=speaker.id,
            name=speaker.name,
            central_bank=speaker.central_bank,
            role=speaker.role,
        )


class ToneObservationCreate(BaseModel):
    """Request body to record a tone observation for a speaker."""

    model_config = ConfigDict(extra="forbid")

    tone: ToneLabel
    score: float = Field(ge=-1.0, le=1.0)
    source_text: str = Field(min_length=1)
    observed_at: datetime | None = None


class ToneObservationResponse(BaseModel):
    """Response body describing a recorded tone observation."""

    id: UUID
    speaker_id: UUID
    observed_at: datetime
    tone: ToneLabel
    score: float
    source_sha256: str = Field(pattern=_SHA256_PATTERN)

    @classmethod
    def from_domain(cls, observation: ToneObservation) -> ToneObservationResponse:
        """Build the response from a domain :class:`ToneObservation`."""
        return cls(
            id=observation.id,
            speaker_id=observation.speaker_id,
            observed_at=observation.observed_at,
            tone=observation.tone,
            score=observation.score,
            source_sha256=observation.source_sha256,
        )

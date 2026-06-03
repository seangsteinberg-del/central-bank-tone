"""HTTP request and response schemas (CLAUDE.md section 3).

These Pydantic models are the API boundary contract. They are deliberately separate from the
core domain models: the wire format can evolve without changing the domain, and every request
body is validated against one of these before any business logic runs.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from cbt_core import Answer, CentralBank, Speaker, Speech, ToneLabel, ToneObservation

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


class SpeechIngestRequest(BaseModel):
    """Request body to ingest and analyze a single speech."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=500)
    url: str = Field(min_length=1, max_length=2000)
    delivered_at: AwareDatetime
    text: str = Field(min_length=1)
    language: str = Field(min_length=2, max_length=12, default="en")


class SpeechResponse(BaseModel):
    """Response body describing an analyzed speech (without the full source text)."""

    id: UUID
    speaker_id: UUID
    central_bank: CentralBank
    title: str
    url: str
    delivered_at: datetime
    language: str
    summary: str
    tone: ToneLabel
    score: float
    lexicon_score: float
    rationale: str
    source_sha256: str = Field(pattern=_SHA256_PATTERN)

    @classmethod
    def from_domain(cls, speech: Speech) -> SpeechResponse:
        """Build the response from a domain :class:`Speech`."""
        return cls(
            id=speech.id,
            speaker_id=speech.speaker_id,
            central_bank=speech.central_bank,
            title=speech.title,
            url=speech.url,
            delivered_at=speech.delivered_at,
            language=speech.language,
            summary=speech.summary,
            tone=speech.tone,
            score=speech.score,
            lexicon_score=speech.lexicon_score,
            rationale=speech.rationale,
            source_sha256=speech.source_sha256,
        )


class AskRequest(BaseModel):
    """Request body to ask a question about a speaker."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)


class CitationResponse(BaseModel):
    """A citation in an answer."""

    speech_id: UUID
    chunk_index: int
    title: str
    url: str


class AnswerResponse(BaseModel):
    """Response body for a question about a speaker."""

    text: str
    abstained: bool
    citations: list[CitationResponse]

    @classmethod
    def from_domain(cls, answer: Answer) -> AnswerResponse:
        """Build the response from a domain :class:`Answer`."""
        return cls(
            text=answer.text,
            abstained=answer.abstained,
            citations=[
                CitationResponse(
                    speech_id=citation.speech_id,
                    chunk_index=citation.chunk_index,
                    title=citation.title,
                    url=citation.url,
                )
                for citation in answer.citations
            ],
        )

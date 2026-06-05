"""The Speech domain model (CLAUDE.md sections 2 and 3).

A speech is an immutable analyzed unit: the source text and metadata plus its analysis (a Gemini
summary and tone, and the deterministic lexicon cross-check). Fields align with the BIS central
bankers' speeches schema (institution, speaker, title, date, url). It is frozen and validated on
construction; the persistence layer stores it append-only.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cbt_core.domain.registry import CentralBank
from cbt_core.domain.tone import ToneLabel

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class Speech(BaseModel):
    """A central bank speech and its tone analysis.

    Attributes:
        id: Stable identifier for the speech.
        speaker_id: The speaker who gave it.
        central_bank: The institution (schema spine).
        title: The speech title.
        url: The source URL the speech was ingested from.
        delivered_at: When the speech was delivered. Must be timezone-aware.
        language: ISO-ish language code of the source text, for example ``en``.
        text: The full speech text.
        source_sha256: Lowercase hex sha256 of ``text``; the natural key for deduplication.
        summary: The model's summary of the speech.
        tone: The model's discrete tone label (schema spine).
        score: The model's continuous tone in ``[-1.0, 1.0]`` (negative dovish, positive
            hawkish).
        lexicon_score: The deterministic lexicon baseline tone in ``[-1.0, 1.0]``.
        rationale: The model's one-line justification for the tone.
        needs_review: True when the headline tone and the lexicon baseline disagreed enough at
            ingestion to flag for review (ADR 0008). The richer directional review signal from the
            structured ensemble (ADR 0021) lives on the derived :class:`SpeechStance`, which can be
            recomputed; this field is the immutable ingestion-time flag.
        model_id: The model that produced the analysis (for example ``gemini-2.5-flash``),
            recorded so the tone series stays comparable as the model changes over time.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    speaker_id: UUID
    central_bank: CentralBank
    title: str = Field(min_length=1, max_length=500)
    url: str = Field(min_length=1, max_length=2000)
    delivered_at: datetime
    language: str = Field(min_length=2, max_length=12, default="en")
    text: str = Field(min_length=1)
    source_sha256: str = Field(pattern=_SHA256_PATTERN)
    summary: str = Field(min_length=1, max_length=4000)
    tone: ToneLabel
    score: float = Field(ge=-1.0, le=1.0)
    lexicon_score: float = Field(ge=-1.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=2000)
    needs_review: bool = False
    model_id: str = Field(default="unknown", min_length=1, max_length=100)

    @field_validator("delivered_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        """Reject naive datetimes so every stored timestamp is unambiguous.

        Raises:
            ValueError: If ``delivered_at`` has no timezone information.
        """
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("delivered_at must be timezone-aware")
        return value


class SpeechStance(BaseModel):
    """The structured stance decomposition of a speech (ADR 0021): derived, recomputable data.

    Kept separate from the immutable :class:`Speech` because it is derived from the speech text and
    can be recomputed as the pipeline improves, exactly like the retrieval chunks, without touching
    the append-only tone record. The holistic headline (``Speech.score`` and ``Speech.tone``) stays
    the production tone; this carries the decomposition a macro reader uses and the directional
    cross-check.

    Attributes:
        speech_id: The speech this decomposition belongs to (the primary key, one row per speech).
        rate_path: The forward-looking net-hawkishness (policy intent, separated from
            backward-looking description), in ``[-1.0, 1.0]``.
        uncertainty: The share of applicable cross-checks that disagree with the headline's
            direction, in ``[0.0, 1.0]``.
        structured_net: The structured sentence-level net-hawkishness cross-check.
        classifier_net: The supervised classifier's net-hawkishness cross-check (zero where the
            classifier does not apply to the institution).
        lexicon_net: The deterministic lexicon's net-hawkishness cross-check.
        needs_review: Whether a majority of the applicable cross-checks contradict the headline.
        aspect_scores: The net-hawkishness within each policy aspect that appeared, keyed by aspect
            name (inflation, growth, employment, balance sheet, financial stability, guidance).
        model_id: The model that produced the sentence classification, recorded for comparability.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    speech_id: UUID
    rate_path: float = Field(ge=-1.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    structured_net: float = Field(ge=-1.0, le=1.0)
    classifier_net: float = Field(ge=-1.0, le=1.0)
    lexicon_net: float = Field(ge=-1.0, le=1.0)
    needs_review: bool
    aspect_scores: dict[str, float]
    model_id: str = Field(default="unknown", min_length=1, max_length=100)

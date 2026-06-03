"""The model's analysis of a speech (CLAUDE.md sections 2 and 3).

``ToneAnalysis`` is the structured result the LLM produces for a speech: a short summary, a
discrete tone label from the schema spine, a continuous score, and a one-line rationale. It is a
domain value object; the LLM boundary returns it and the analysis service records a
``ToneObservation`` from it. The model is used directly as the Gemini structured-output schema,
so a malformed model response fails validation rather than producing a fabricated result.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from cbt_core.domain.tone import ToneLabel


class ToneAnalysis(BaseModel):
    """A summary and tone judgement of a single speech.

    Attributes:
        summary: A concise summary of the speech.
        tone: The discrete tone label (schema spine).
        score: Continuous tone in ``[-1.0, 1.0]`` (negative dovish, positive hawkish).
        rationale: A one-line justification for the tone judgement.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    summary: str = Field(min_length=1, max_length=4000)
    tone: ToneLabel
    score: float = Field(ge=-1.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=2000)

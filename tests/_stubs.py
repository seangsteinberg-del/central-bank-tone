"""Reusable test doubles shared across test modules."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from cbt_core.domain.analysis import ToneAnalysis
from cbt_core.domain.qa import EMBEDDING_DIM, RetrievedChunk


def _deterministic_embedding(text: str) -> list[float]:
    """A deterministic, non-zero unit-ish vector derived from the text (no model needed)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[index % len(digest)] / 255.0 + 0.001 for index in range(EMBEDDING_DIM)]


class StubLlmClient:
    """A deterministic in-memory LlmClient; records the texts it was asked to analyze."""

    def __init__(self, analysis: ToneAnalysis) -> None:
        self._analysis = analysis
        self.calls: list[str] = []
        self.answers = 0

    def analyze_tone(self, speech_text: str) -> ToneAnalysis:
        self.calls.append(speech_text)
        return self._analysis

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_deterministic_embedding(text) for text in texts]

    def answer(self, question: str, chunks: Sequence[RetrievedChunk]) -> str:
        self.answers += 1
        return f"Based on {len(chunks)} excerpt(s): a grounded answer to {question!r}."

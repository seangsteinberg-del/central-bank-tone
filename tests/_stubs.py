"""Reusable test doubles shared across test modules."""

from __future__ import annotations

from cbt_core.domain.analysis import ToneAnalysis


class StubLlmClient:
    """A deterministic in-memory LlmClient; records the texts it was asked to analyze."""

    def __init__(self, analysis: ToneAnalysis) -> None:
        self._analysis = analysis
        self.calls: list[str] = []

    def analyze_tone(self, speech_text: str) -> ToneAnalysis:
        self.calls.append(speech_text)
        return self._analysis

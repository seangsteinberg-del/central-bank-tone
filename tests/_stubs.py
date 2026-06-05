"""Reusable test doubles shared across test modules."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from cbt_core.analysis import ClassifiedSentence, StanceLabel, infer_aspect, infer_horizon
from cbt_core.domain.analysis import ToneAnalysis
from cbt_core.domain.qa import EMBEDDING_DIM, RetrievedChunk
from cbt_core.domain.tone import ToneLabel

_HAWKISH_CUES = ("hike", "tighten", "raise", "restrictive")
_DOVISH_CUES = ("cut", "ease", "easing", "accommodat", "stimulus")


def _deterministic_embedding(text: str) -> list[float]:
    """A deterministic, non-zero unit-ish vector derived from the text (no model needed)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[index % len(digest)] / 255.0 + 0.001 for index in range(EMBEDDING_DIM)]


def _stub_stance(sentence: str) -> StanceLabel:
    """A deterministic keyword stance for the stub classifier."""
    lowered = sentence.lower()
    if any(cue in lowered for cue in _HAWKISH_CUES):
        return StanceLabel.HAWKISH
    if any(cue in lowered for cue in _DOVISH_CUES):
        return StanceLabel.DOVISH
    return StanceLabel.NEUTRAL


def _stub_classified_sentences(sentences: Sequence[str]) -> list[ClassifiedSentence]:
    """Classify each sentence deterministically (shared by the stub and scripted clients)."""
    return [
        ClassifiedSentence(
            text=sentence,
            label=_stub_stance(sentence),
            aspect=infer_aspect(sentence),
            horizon=infer_horizon(sentence),
        )
        for sentence in sentences
    ]


class StubLlmClient:
    """A deterministic in-memory LlmClient; records the texts it was asked to analyze."""

    def __init__(self, analysis: ToneAnalysis) -> None:
        self._analysis = analysis
        self.calls: list[str] = []
        self.classified = 0
        self.answers = 0

    def analyze_tone(self, speech_text: str) -> ToneAnalysis:
        self.calls.append(speech_text)
        return self._analysis

    def classify_sentences(self, sentences: Sequence[str]) -> list[ClassifiedSentence]:
        self.classified += 1
        return _stub_classified_sentences(sentences)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_deterministic_embedding(text) for text in texts]

    def answer(self, question: str, chunks: Sequence[RetrievedChunk]) -> str:
        self.answers += 1
        return f"Based on {len(chunks)} excerpt(s): a grounded answer to {question!r}."


class ScriptedLlmClient:
    """An LlmClient whose tone is looked up by speech text, so a test can script per-speech scores.

    The non-scripted methods reuse the deterministic stub behaviour, so this is a full
    :class:`~cbt_core.llm.LlmClient`; the protocol-conformance test asserts it stays in step.
    """

    def __init__(self, by_text: dict[str, tuple[float, ToneLabel]]) -> None:
        self._by_text = by_text

    def analyze_tone(self, speech_text: str) -> ToneAnalysis:
        score, tone = self._by_text[speech_text]
        return ToneAnalysis(
            summary=f"Summary of: {speech_text}",
            tone=tone,
            score=score,
            rationale="scripted for the test",
        )

    def classify_sentences(self, sentences: Sequence[str]) -> list[ClassifiedSentence]:
        return _stub_classified_sentences(sentences)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_deterministic_embedding(text) for text in texts]

    def answer(self, question: str, chunks: Sequence[RetrievedChunk]) -> str:
        return "not used"


class StubChunkRetriever:
    """A retriever that returns fixed chunks, for testing the Q&A path without pgvector."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def search(
        self, speaker_id: object, query_embedding: object, top_k: int
    ) -> list[RetrievedChunk]:
        return self._chunks[:top_k]

    def search_all(self, query_embedding: object, top_k: int) -> list[RetrievedChunk]:
        return self._chunks[:top_k]

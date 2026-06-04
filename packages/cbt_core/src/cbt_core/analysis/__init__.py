"""Deterministic analysis helpers that complement the LLM (ADR 0008, ADR 0013)."""

from __future__ import annotations

from cbt_core.analysis.chunking import chunk_text
from cbt_core.analysis.classifier import (
    TONE_CLASSES,
    ClassifierScore,
    ToneClassifier,
    ToneModelError,
)
from cbt_core.analysis.lexicon import (
    DISAGREEMENT_THRESHOLD,
    HawkishDovishLexicon,
    LexiconScore,
    disagrees,
)

__all__ = [
    "DISAGREEMENT_THRESHOLD",
    "TONE_CLASSES",
    "ClassifierScore",
    "HawkishDovishLexicon",
    "LexiconScore",
    "ToneClassifier",
    "ToneModelError",
    "chunk_text",
    "disagrees",
]

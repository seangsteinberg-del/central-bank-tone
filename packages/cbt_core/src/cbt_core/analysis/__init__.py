"""Deterministic analysis helpers that complement the LLM (ADR 0008)."""

from __future__ import annotations

from cbt_core.analysis.chunking import chunk_text
from cbt_core.analysis.lexicon import (
    DISAGREEMENT_THRESHOLD,
    HawkishDovishLexicon,
    LexiconScore,
    disagrees,
)

__all__ = [
    "DISAGREEMENT_THRESHOLD",
    "HawkishDovishLexicon",
    "LexiconScore",
    "chunk_text",
    "disagrees",
]

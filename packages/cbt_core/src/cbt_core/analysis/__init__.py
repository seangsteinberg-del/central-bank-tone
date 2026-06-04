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
from cbt_core.analysis.stance import (
    DEFAULT_POLICY_TERMS,
    Aspect,
    ClassifiedSentence,
    Horizon,
    PolicyRelevanceFilter,
    StanceAggregate,
    StanceAssessment,
    StanceLabel,
    aggregate_stances,
    combine_signals,
    infer_aspect,
    infer_horizon,
    net_hawkishness,
    split_sentences,
    to_tone_label,
)

__all__ = [
    "DEFAULT_POLICY_TERMS",
    "DISAGREEMENT_THRESHOLD",
    "TONE_CLASSES",
    "Aspect",
    "ClassifiedSentence",
    "ClassifierScore",
    "HawkishDovishLexicon",
    "Horizon",
    "LexiconScore",
    "PolicyRelevanceFilter",
    "StanceAggregate",
    "StanceAssessment",
    "StanceLabel",
    "ToneClassifier",
    "ToneModelError",
    "aggregate_stances",
    "chunk_text",
    "combine_signals",
    "disagrees",
    "infer_aspect",
    "infer_horizon",
    "net_hawkishness",
    "split_sentences",
    "to_tone_label",
]

"""Deterministic hawkish/dovish lexicon tone baseline (ADR 0008).

A license-clean baseline implementing the Apel and Blix Grimaldi net-hawkishness method with our
own word lists (we do not copy the licensed Loughran-McDonald or non-commercial FOMC
dictionaries). Pure Python, no network, fully deterministic: it is a transparent, auditable
cross-check on the Gemini tone score, and it abstains (score 0.0) when no signal words are found
rather than guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Our own hawkish / dovish term lists. Kept small and documented on purpose (ADR 0008).
_HAWKISH_TERMS: tuple[str, ...] = (
    "tighten",
    "tightening",
    "raise rates",
    "rate hike",
    "hike",
    "hikes",
    "restrictive",
    "firming",
    "overheating",
    "inflationary pressure",
    "upside risk to inflation",
    "withdraw accommodation",
    "vigilant",
    "vigilance",
    "higher inflation",
    "robust growth",
)
_DOVISH_TERMS: tuple[str, ...] = (
    "accommodative",
    "accommodation",
    "ease",
    "easing",
    "rate cut",
    "cut rates",
    "stimulus",
    "downside risk",
    "downside risks",
    "economic slack",
    "weak demand",
    "support growth",
    "patient",
    "gradual",
    "subdued inflation",
    "lower growth",
    "headwinds",
)


@dataclass(frozen=True)
class LexiconScore:
    """A deterministic net-hawkishness score and the term hit counts behind it.

    Attributes:
        score: Net hawkishness in ``[-1.0, 1.0]`` (positive hawkish, negative dovish), or
            ``0.0`` when no signal words were found.
        hawkish_hits: Number of hawkish term occurrences.
        dovish_hits: Number of dovish term occurrences.
    """

    score: float
    hawkish_hits: int
    dovish_hits: int


def _count_terms(text: str, terms: tuple[str, ...]) -> int:
    """Count whole-word occurrences of each term in ``text`` (already lowercased)."""
    return sum(len(re.findall(rf"\b{re.escape(term)}\b", text)) for term in terms)


class HawkishDovishLexicon:
    """Scores monetary-policy tone by counting hawkish versus dovish terms."""

    def __init__(
        self,
        hawkish: tuple[str, ...] = _HAWKISH_TERMS,
        dovish: tuple[str, ...] = _DOVISH_TERMS,
    ) -> None:
        """Build the lexicon.

        Args:
            hawkish: Hawkish terms to count.
            dovish: Dovish terms to count.
        """
        self._hawkish = hawkish
        self._dovish = dovish

    def score(self, text: str) -> LexiconScore:
        """Score the net hawkishness of a text.

        Args:
            text: The text to score.

        Returns:
            A :class:`LexiconScore`. The ``score`` is ``(hawkish - dovish) / total`` hits, or
            ``0.0`` when there are no hits.
        """
        lowered = text.lower()
        hawkish_hits = _count_terms(lowered, self._hawkish)
        dovish_hits = _count_terms(lowered, self._dovish)
        total = hawkish_hits + dovish_hits
        score = 0.0 if total == 0 else (hawkish_hits - dovish_hits) / total
        return LexiconScore(score=score, hawkish_hits=hawkish_hits, dovish_hits=dovish_hits)

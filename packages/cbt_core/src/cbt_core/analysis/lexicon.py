"""Deterministic hawkish/dovish lexicon tone baseline (ADR 0008).

A license-clean, transparent baseline: our own curated hawkish and dovish term lists (we do not
copy the licensed Loughran-McDonald or the non-commercial FOMC dictionaries) scored by a
net-hawkishness ratio in the spirit of Apel and Blix Grimaldi (2012). It is pure and
deterministic (no network). To be defensible rather than naive it:

- matches multi-word phrases longest-first and consumes the matched span, so a phrase is never
  double-counted by one of its own substrings ("rate hike" is one hawkish hit, not also "hike"),
  which also resolves contradictions like "withdraw accommodation" (hawkish) versus the substring
  "accommodation" (dovish);
- applies a short negation window, so "not accommodative" reads hawkish and "no further
  tightening" reads dovish; and
- abstains (score 0.0) when no signal terms are found rather than guessing.

Its accuracy is not asserted but measured: ``scripts/eval_tone.py`` scores it against the
annotated FOMC benchmark and reports the numbers (see ``docs/research/tone-evaluation.md``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Polarity-tagged term lists. Curated, documented, and license-clean (ADR 0008). Multi-word
# phrases are matched before their substrings, so "rate hike" counts once as hawkish, not also as
# "hike", and "withdraw accommodation" counts hawkish without its substring "accommodation" also
# scoring dovish.
_HAWKISH_TERMS: tuple[str, ...] = (
    "withdraw accommodation",
    "remove accommodation",
    "upside risk to inflation",
    "upside risks to inflation",
    "inflationary pressure",
    "inflationary pressures",
    "price pressures",
    "overheating",
    "raise rates",
    "raise the policy rate",
    "rate hike",
    "rate hikes",
    "rate increase",
    "rate increases",
    "tighten",
    "tightening",
    "tighter",
    "hike",
    "hikes",
    "hiking",
    "restrictive",
    "firming",
    "firmer",
    "vigilant",
    "vigilance",
    "elevated inflation",
    "higher inflation",
    "persistent inflation",
    "robust growth",
    "strong demand",
)
_DOVISH_TERMS: tuple[str, ...] = (
    "rate cut",
    "rate cuts",
    "cut rates",
    "lower rates",
    "lower the policy rate",
    "ease policy",
    "policy easing",
    "accommodative",
    "accommodation",
    "easing",
    "stimulus",
    "stimulate",
    "downside risk",
    "downside risks",
    "economic slack",
    "labor market slack",
    "weak demand",
    "soft demand",
    "support growth",
    "supportive",
    "patient",
    "gradual",
    "subdued inflation",
    "weak inflation",
    "low inflation",
    "disinflation",
    "slowing growth",
    "headwinds",
    "recession",
)

# A negator within this many words before a term flips that term's polarity ("not accommodative"
# reads hawkish). A short window keeps the heuristic local and avoids crossing most clauses.
_NEGATION_WINDOW = 3
_NEGATORS: frozenset[str] = frozenset(
    {
        "not",
        "no",
        "never",
        "without",
        "cannot",
        "nor",
        "none",
        "neither",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "won't",
        "don't",
        "doesn't",
        "didn't",
        "can't",
        "couldn't",
        "wouldn't",
        "shouldn't",
    }
)
_WORD_RE = re.compile(r"[a-z']+")


# A model-vs-lexicon gap at least this large (or any opposite-sign disagreement) flags a speech
# for review. ADR 0008: a large disagreement is surfaced, not silently averaged away.
DISAGREEMENT_THRESHOLD = 0.5


@dataclass(frozen=True)
class LexiconScore:
    """A deterministic net-hawkishness score and the term hit counts behind it.

    Attributes:
        score: Net hawkishness in ``[-1.0, 1.0]`` (positive hawkish, negative dovish), or
            ``0.0`` when no signal words were found.
        hawkish_hits: Number of hawkish term occurrences, after negation flips.
        dovish_hits: Number of dovish term occurrences, after negation flips.
    """

    score: float
    hawkish_hits: int
    dovish_hits: int

    @property
    def fired(self) -> bool:
        """Whether any lexicon term matched (so the score carries signal, not abstention)."""
        return self.hawkish_hits + self.dovish_hits > 0


def disagrees(
    model_score: float, lexicon: LexiconScore, *, threshold: float = DISAGREEMENT_THRESHOLD
) -> bool:
    """Whether a model tone score and the lexicon score diverge enough to flag for review.

    Returns ``False`` when the lexicon abstained (no signal terms), since an abstention cannot
    disagree. Otherwise flags an opposite-sign disagreement or a magnitude gap of at least
    ``threshold`` (ADR 0008: a large disagreement is a signal to review).

    Args:
        model_score: The model's tone score in ``[-1.0, 1.0]``.
        lexicon: The lexicon's score for the same text.
        threshold: The minimum absolute gap that counts as a disagreement.

    Returns:
        True if the two scores diverge enough to warrant review.
    """
    if not lexicon.fired:
        return False
    opposite_sign = model_score * lexicon.score < 0
    return opposite_sign or abs(model_score - lexicon.score) >= threshold


def _is_negated(text: str, match_start: int) -> bool:
    """Return whether a negator appears within the preceding ``_NEGATION_WINDOW`` words."""
    preceding = _WORD_RE.findall(text[:match_start])
    return any(word in _NEGATORS for word in preceding[-_NEGATION_WINDOW:])


class HawkishDovishLexicon:
    """Scores monetary-policy tone by counting hawkish versus dovish terms.

    Phrases are matched longest-first and the matched character span is consumed, so no phrase is
    double-counted by one of its substrings. A negator within a short window before a term flips
    that term's polarity.
    """

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
        # Longest first so a multi-word phrase is matched before any of its substrings.
        self._terms: tuple[tuple[str, int], ...] = tuple(
            sorted(
                [(term, 1) for term in hawkish] + [(term, -1) for term in dovish],
                key=lambda pair: len(pair[0]),
                reverse=True,
            )
        )

    def score(self, text: str) -> LexiconScore:
        """Score the net hawkishness of a text.

        Args:
            text: The text to score.

        Returns:
            A :class:`LexiconScore`. The ``score`` is ``(hawkish - dovish) / total`` hits after
            negation flips, or ``0.0`` when there are no hits.
        """
        lowered = text.lower()
        consumed = bytearray(len(lowered))
        hawkish_hits = 0
        dovish_hits = 0
        for term, polarity in self._terms:
            for match in re.finditer(rf"\b{re.escape(term)}\b", lowered):
                start, end = match.start(), match.end()
                if any(consumed[start:end]):
                    continue
                consumed[start:end] = b"\x01" * (end - start)
                effective = -polarity if _is_negated(lowered, start) else polarity
                if effective > 0:
                    hawkish_hits += 1
                else:
                    dovish_hits += 1
        total = hawkish_hits + dovish_hits
        score = 0.0 if total == 0 else (hawkish_hits - dovish_hits) / total
        return LexiconScore(score=score, hawkish_hits=hawkish_hits, dovish_hits=dovish_hits)

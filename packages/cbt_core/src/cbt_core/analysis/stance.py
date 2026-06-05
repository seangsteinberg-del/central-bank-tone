"""Sentence-level monetary-policy stance aggregation (ADR 0021).

The verified state of the art for turning central-bank text into a tone signal does not feed a
whole speech to a model and ask for one number (a single greedy call scores near a majority-class
baseline). It does three things in order: split into sentences, keep only the policy-relevant ones,
classify each as hawkish / dovish / neutral, then aggregate the labels into a normalized
net-hawkishness measure. This is the "Trillion Dollar Words" recipe (Shah, Paturi and Chava, ACL
2023): ``Measure = (#Hawkish - #Dovish) / #Total`` over the dictionary-filtered sentences (neutrals
counted in the denominator). See ``docs/research/tone-sota-blueprint.md``.

This module is the model-agnostic heart of that pipeline. It owns the parts that must be exactly
reproducible and so must not live inside a model call:

- the sentence split and the policy-relevance filter (a Gorodnichenko-style keyword gate that drops
  the neutral filler which otherwise dilutes the average);
- the aggregation maths (the net-hawkishness measure, the same measure restricted to
  forward-looking sentences so rate-PATH intent is separated from backward-looking description, and
  a per-aspect breakdown); and
- the mapping from a continuous measure to a discrete :class:`~cbt_core.domain.tone.ToneLabel`,
  including an honest ``MIXED`` only when both sides are materially present and an honest abstention
  (``NEUTRAL``) when no policy-relevant sentence was found.

Who assigns each sentence its :class:`StanceLabel`, :class:`Aspect` and :class:`Horizon` is left to
the caller: the Gemini classifier in production (one batched structured call), the supervised
classifier offline, or a stub in tests. That separation keeps the irreproducible model judgement
out of this module and the reproducible accounting in it, where it can be tested without a network
or a GPU.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from cbt_core.domain.tone import ToneLabel

# Default thresholds for turning the continuous measure into a label. Both are deliberately small:
# the measure is already normalized to the count of policy-relevant sentences, so a net of a few
# tenths is a real lean. They are arguments on the mapping function, not module constants, so a
# caller can tune them without editing this module.
_DEFAULT_NEUTRAL_BAND = 0.05
_DEFAULT_MIXED_MIN_SHARE = 0.25

# A spread among the (fired) ensemble signals at least this large flags a speech for review, as does
# any opposite-sign disagreement. Matches the lexicon cross-check threshold (ADR 0008): a large
# disagreement is surfaced as uncertainty, not averaged away.
_DEFAULT_REVIEW_THRESHOLD = 0.5

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
_WORD_RE = re.compile(r"[a-z']+")


class StanceLabel(StrEnum):
    """The directional monetary-policy stance of a single sentence.

    A three-way scheme matching the annotated FOMC benchmark: ``HAWKISH`` indicates future
    tightening, ``DOVISH`` indicates future easing, ``NEUTRAL`` is mixed, no-change, or not
    policy-directional. There is deliberately no ``MIXED`` at the sentence level; two-sidedness is a
    property of a whole speech (many opposing sentences), recorded by :class:`StanceAggregate` and
    the document-level :class:`~cbt_core.domain.tone.ToneLabel`, not of one sentence.
    """

    HAWKISH = "hawkish"
    DOVISH = "dovish"
    NEUTRAL = "neutral"


class Aspect(StrEnum):
    """The policy aspect a sentence speaks to, so tone can be read per topic, not just in aggregate.

    A macro reader cares whether the hawkishness is about inflation or about the balance sheet.
    ``OTHER`` is the honest bucket for a policy-relevant sentence that does not fit a named aspect;
    it is not a dumping ground for off-topic text, which the relevance filter drops before
    classification.
    """

    INFLATION = "inflation"
    GROWTH = "growth"
    EMPLOYMENT = "employment"
    BALANCE_SHEET = "balance_sheet"
    FINANCIAL_STABILITY = "financial_stability"
    GUIDANCE = "guidance"
    OTHER = "other"


class Horizon(StrEnum):
    """Whether a sentence looks forward (policy intent) or backward (description of what happened).

    Separating these is the single most decision-relevant axis: a forward-looking hawkish sentence
    ("we will need to raise rates further") is a rate-PATH signal, while a backward-looking one
    ("inflation rose last year") merely describes the environment. ``UNSPECIFIED`` is used when a
    sentence carries no clear temporal direction.
    """

    FORWARD = "forward"
    BACKWARD = "backward"
    UNSPECIFIED = "unspecified"


# A Gorodnichenko-style relevance gate: a sentence is scored only if it mentions monetary policy,
# its instruments, or its objectives. This drops the procedural and ceremonial filler that, left in,
# pulls every speech toward neutral. Relevance is about topic, not polarity, so the set is broad and
# includes neutral policy nouns ("policy rate", "balance sheet") that carry no direction themselves.
DEFAULT_POLICY_TERMS: frozenset[str] = frozenset(
    {
        "inflation",
        "inflationary",
        "disinflation",
        "deflation",
        "price stability",
        "prices",
        "cpi",
        "interest rate",
        "interest rates",
        "policy rate",
        "bank rate",
        "rate",
        "rates",
        "hike",
        "cut",
        "tightening",
        "easing",
        "accommodation",
        "accommodative",
        "restrictive",
        "monetary policy",
        "monetary",
        "policy stance",
        "stance",
        "balance sheet",
        "asset purchases",
        "quantitative",
        "liquidity",
        "employment",
        "labour market",
        "labor market",
        "unemployment",
        "wages",
        "growth",
        "demand",
        "output",
        "gdp",
        "economic activity",
        "recovery",
        "recession",
        "slowdown",
        "outlook",
        "forecast",
        "projections",
        "guidance",
    }
)


@dataclass(frozen=True)
class ClassifiedSentence:
    """One policy-relevant sentence with the stance, aspect and horizon a classifier assigned it.

    Attributes:
        text: The sentence as it appeared in the speech (trimmed).
        label: The directional stance the classifier assigned.
        aspect: The policy aspect the sentence speaks to.
        horizon: Whether the sentence is forward-looking (intent) or backward-looking (description).
    """

    text: str
    label: StanceLabel
    aspect: Aspect = Aspect.OTHER
    horizon: Horizon = Horizon.UNSPECIFIED


@dataclass(frozen=True)
class StanceAggregate:
    """The document-level result of aggregating classified sentences (ADR 0021).

    The headline ``net_hawkishness`` is the Trillion Dollar Words measure: the hawkish minus dovish
    sentence count over the policy-relevant sentence count, in ``[-1.0, 1.0]``. ``forward_net`` is
    the same measure restricted to forward-looking sentences, the rate-PATH intent signal a reader
    trades on. ``by_aspect`` is the measure computed within each aspect that appeared.

    Attributes:
        net_hawkishness: ``(hawkish - dovish) / relevant`` in ``[-1.0, 1.0]``; ``0.0`` when no
            policy-relevant sentence was found (an honest abstention, not a confident neutral).
        tone: The discrete document label this measure maps to.
        hawkish: Count of hawkish policy-relevant sentences.
        dovish: Count of dovish policy-relevant sentences.
        neutral: Count of neutral policy-relevant sentences.
        relevant: Count of policy-relevant sentences (the measure's denominator).
        total: Count of all sentences considered, relevant or not.
        forward_net: The net-hawkishness measure over forward-looking sentences only, or ``0.0``
            when there were none.
        forward_relevant: Count of forward-looking policy-relevant sentences.
        by_aspect: The net-hawkishness measure within each aspect that appeared, keyed by aspect.
    """

    net_hawkishness: float
    tone: ToneLabel
    hawkish: int
    dovish: int
    neutral: int
    relevant: int
    total: int
    forward_net: float
    forward_relevant: int
    by_aspect: Mapping[Aspect, float]

    @property
    def fired(self) -> bool:
        """Whether any policy-relevant sentence was found (so the measure is signal, not abstention)."""
        return self.relevant > 0

    @property
    def directional_share(self) -> float:
        """The share of policy-relevant sentences that took a side (hawkish or dovish)."""
        if self.relevant == 0:
            return 0.0
        return (self.hawkish + self.dovish) / self.relevant


def split_sentences(text: str) -> list[str]:
    """Split text into trimmed, non-empty sentences.

    A deliberately simple, deterministic splitter on sentence-final punctuation. It is shared by
    production and tests so the sentence boundaries can never drift between them.

    Args:
        text: The text to split.

    Returns:
        The sentences in order, trimmed, with empties removed.
    """
    return [match.group().strip() for match in _SENTENCE_RE.finditer(text) if match.group().strip()]


class PolicyRelevanceFilter:
    """Keeps only sentences that mention monetary policy, its instruments, or its objectives.

    A Gorodnichenko-style keyword gate (ADR 0021). It decides topic relevance, not polarity, so it
    is intentionally broad and includes neutral policy nouns. Dropping irrelevant sentences before
    aggregation is what stops procedural filler from diluting the tone measure toward neutral.
    """

    def __init__(self, terms: Iterable[str] = DEFAULT_POLICY_TERMS) -> None:
        """Build the filter.

        Args:
            terms: The relevance keywords and phrases; a sentence is relevant if it contains any.
        """
        self._terms: frozenset[str] = frozenset(term.lower() for term in terms)
        # Single-word terms are matched against the tokenized sentence (so "rate" does not fire on
        # "accelerate"); multi-word terms are matched as substrings of the lowered sentence.
        self._single: frozenset[str] = frozenset(t for t in self._terms if " " not in t)
        self._phrases: tuple[str, ...] = tuple(t for t in self._terms if " " in t)

    def is_relevant(self, sentence: str) -> bool:
        """Whether a sentence is policy-relevant.

        Args:
            sentence: The sentence to test.

        Returns:
            True if the sentence contains any relevance keyword (word) or phrase (substring).
        """
        lowered = sentence.lower()
        if any(phrase in lowered for phrase in self._phrases):
            return True
        words = set(_WORD_RE.findall(lowered))
        return not words.isdisjoint(self._single)

    def filter(self, sentences: Iterable[str]) -> list[str]:
        """Return only the policy-relevant sentences, in order.

        Args:
            sentences: The sentences to filter.

        Returns:
            The subset that is policy-relevant.
        """
        return [sentence for sentence in sentences if self.is_relevant(sentence)]


# Cue terms for the deterministic aspect and horizon heuristics. These are a coarse, transparent
# fallback used by the offline path and where a model does not supply the axes; the production Gemini
# classifier judges aspect and horizon itself. Aspects are tested in order, so the first with the
# most cue hits wins ties, which is why the more specific aspects come before GROWTH.
_ASPECT_CUES: tuple[tuple[Aspect, frozenset[str]], ...] = (
    (
        Aspect.INFLATION,
        frozenset({"inflation", "inflationary", "disinflation", "deflation", "price", "cpi"}),
    ),
    (
        Aspect.EMPLOYMENT,
        frozenset({"employment", "unemployment", "jobs", "labour", "labor", "wage"}),
    ),
    (
        Aspect.BALANCE_SHEET,
        frozenset({"balance sheet", "asset purchases", "quantitative", "reserves", "liquidity"}),
    ),
    (
        Aspect.FINANCIAL_STABILITY,
        frozenset({"financial stability", "financial system", "systemic", "leverage", "credit"}),
    ),
    (
        Aspect.GROWTH,
        frozenset({"growth", "gdp", "output", "demand", "recession", "recovery", "activity"}),
    ),
    (Aspect.GUIDANCE, frozenset({"guidance", "outlook", "projection", "forecast"})),
)
_FORWARD_CUES: frozenset[str] = frozenset(
    {
        "will",
        "expect",
        "anticipate",
        "going forward",
        "ahead",
        "future",
        "coming",
        "intend",
        "outlook",
        "projection",
        "forecast",
        "likely",
        "to come",
    }
)
_BACKWARD_CUES: frozenset[str] = frozenset(
    {
        "was",
        "were",
        "has been",
        "had",
        "last year",
        "previously",
        "over the past",
        "rose",
        "fell",
        "declined",
        "slowed",
    }
)


def _cue_hits(lowered: str, words: set[str], cues: frozenset[str]) -> int:
    """Count how many cues appear, matching single words on tokens and phrases as substrings."""
    return sum(1 for cue in cues if (cue in words if " " not in cue else cue in lowered))


def infer_aspect(sentence: str) -> Aspect:
    """Guess the policy aspect of a sentence from keyword cues (a deterministic fallback).

    Args:
        sentence: The sentence to classify.

    Returns:
        The aspect with the most cue hits, ties broken by the order in :data:`_ASPECT_CUES`, or
        :attr:`Aspect.OTHER` when no cue matches.
    """
    lowered = sentence.lower()
    words = set(_WORD_RE.findall(lowered))
    best_aspect = Aspect.OTHER
    best_hits = 0
    for aspect, cues in _ASPECT_CUES:
        hits = _cue_hits(lowered, words, cues)
        if hits > best_hits:
            best_aspect, best_hits = aspect, hits
    return best_aspect


def infer_horizon(sentence: str) -> Horizon:
    """Guess whether a sentence is forward- or backward-looking from cues (a deterministic fallback).

    Args:
        sentence: The sentence to classify.

    Returns:
        :attr:`Horizon.FORWARD` or :attr:`Horizon.BACKWARD` when one side has strictly more cue
        hits, otherwise :attr:`Horizon.UNSPECIFIED`.
    """
    lowered = sentence.lower()
    words = set(_WORD_RE.findall(lowered))
    forward = _cue_hits(lowered, words, _FORWARD_CUES)
    backward = _cue_hits(lowered, words, _BACKWARD_CUES)
    if forward > backward:
        return Horizon.FORWARD
    if backward > forward:
        return Horizon.BACKWARD
    return Horizon.UNSPECIFIED


def net_hawkishness(hawkish: int, dovish: int, relevant: int) -> float:
    """Compute the normalized net-hawkishness measure.

    Args:
        hawkish: Count of hawkish sentences.
        dovish: Count of dovish sentences.
        relevant: Count of policy-relevant sentences (the denominator; neutrals included).

    Returns:
        ``(hawkish - dovish) / relevant`` in ``[-1.0, 1.0]``, or ``0.0`` when ``relevant`` is zero
        (an honest abstention rather than a divide-by-zero or a fabricated neutral).
    """
    if relevant <= 0:
        return 0.0
    return (hawkish - dovish) / relevant


def to_tone_label(
    aggregate_counts: tuple[int, int, int],
    *,
    neutral_band: float = _DEFAULT_NEUTRAL_BAND,
    mixed_min_share: float = _DEFAULT_MIXED_MIN_SHARE,
) -> ToneLabel:
    """Map hawkish/dovish/relevant counts to a discrete document tone.

    The order of checks matters and encodes the policy: abstain first, then call a genuinely
    two-sided speech ``MIXED``, then apply the neutral band, then take the sign.

    Args:
        aggregate_counts: A ``(hawkish, dovish, relevant)`` triple.
        neutral_band: A speech whose absolute net hawkishness is below this is ``NEUTRAL``.
        mixed_min_share: A speech is ``MIXED`` when both the hawkish and the dovish share of
            relevant sentences are at least this large (strong arguments in both directions).

    Returns:
        The document :class:`~cbt_core.domain.tone.ToneLabel`. ``NEUTRAL`` is returned for an
        abstention (no relevant sentences), never as a silent fallback for uncertainty.
    """
    hawkish, dovish, relevant = aggregate_counts
    if relevant <= 0:
        return ToneLabel.NEUTRAL
    hawkish_share = hawkish / relevant
    dovish_share = dovish / relevant
    if hawkish_share >= mixed_min_share and dovish_share >= mixed_min_share:
        return ToneLabel.MIXED
    net = (hawkish - dovish) / relevant
    if abs(net) < neutral_band:
        return ToneLabel.NEUTRAL
    return ToneLabel.HAWKISH if net > 0 else ToneLabel.DOVISH


def aggregate_stances(
    sentences: Sequence[ClassifiedSentence],
    *,
    total: int | None = None,
    neutral_band: float = _DEFAULT_NEUTRAL_BAND,
    mixed_min_share: float = _DEFAULT_MIXED_MIN_SHARE,
) -> StanceAggregate:
    """Aggregate classified policy-relevant sentences into a document stance.

    ``sentences`` are the policy-relevant sentences only (the relevance filter has already run).
    Pass ``total`` to record how many sentences the speech had before filtering; it does not affect
    the measure, which is normalized to the relevant count per the Trillion Dollar Words recipe.

    Args:
        sentences: The classified, policy-relevant sentences.
        total: The count of all sentences before the relevance filter; defaults to the number of
            relevant sentences when not given.
        neutral_band: Forwarded to :func:`to_tone_label`.
        mixed_min_share: Forwarded to :func:`to_tone_label`.

    Returns:
        The :class:`StanceAggregate`. With no sentences it is an honest abstention: a ``0.0``
        measure, a ``NEUTRAL`` tone, and ``relevant == 0`` so :attr:`StanceAggregate.fired` is
        ``False``.
    """
    relevant = len(sentences)
    hawkish = sum(1 for s in sentences if s.label is StanceLabel.HAWKISH)
    dovish = sum(1 for s in sentences if s.label is StanceLabel.DOVISH)
    neutral = relevant - hawkish - dovish

    forward = [s for s in sentences if s.horizon is Horizon.FORWARD]
    forward_hawkish = sum(1 for s in forward if s.label is StanceLabel.HAWKISH)
    forward_dovish = sum(1 for s in forward if s.label is StanceLabel.DOVISH)

    by_aspect: dict[Aspect, float] = {}
    seen_aspects = {s.aspect for s in sentences}
    for aspect in seen_aspects:
        in_aspect = [s for s in sentences if s.aspect is aspect]
        a_hawkish = sum(1 for s in in_aspect if s.label is StanceLabel.HAWKISH)
        a_dovish = sum(1 for s in in_aspect if s.label is StanceLabel.DOVISH)
        by_aspect[aspect] = net_hawkishness(a_hawkish, a_dovish, len(in_aspect))

    return StanceAggregate(
        net_hawkishness=net_hawkishness(hawkish, dovish, relevant),
        tone=to_tone_label(
            (hawkish, dovish, relevant),
            neutral_band=neutral_band,
            mixed_min_share=mixed_min_share,
        ),
        hawkish=hawkish,
        dovish=dovish,
        neutral=neutral,
        relevant=relevant,
        total=relevant if total is None else total,
        forward_net=net_hawkishness(forward_hawkish, forward_dovish, len(forward)),
        forward_relevant=len(forward),
        by_aspect=by_aspect,
    )


@dataclass(frozen=True)
class StanceAssessment:
    """The ensembled stance verdict for a speech (ADR 0021): one headline score with its context.

    The headline ``score`` and ``tone`` are the model's holistic judgement of the whole speech (its
    dynamic range is what surfaces moves and divergence for a macro reader). The structured
    sentence-level pipeline supplies the decision-relevant decomposition: ``rate_path`` (forward
    intent, separated from backward-looking description) and the per-aspect breakdown in
    ``aggregate``. Three independent signals (the structured net, the classifier, the lexicon) are
    not averaged into the headline (ADR 0008) but compared against it; their spread is the
    ``uncertainty`` band, and a wide spread or an opposite-sign disagreement sets ``needs_review``.

    Attributes:
        score: The headline net-hawkishness (the model's holistic score), in ``[-1.0, 1.0]``.
        tone: The headline discrete tone.
        rate_path: The forward-looking net-hawkishness (policy intent), from the structured pipeline.
        aggregate: The structured :class:`StanceAggregate` (counts, forward sub-measure, per-aspect).
        structured_net: The structured sentence-level net-hawkishness cross-check.
        classifier_net: The supervised classifier's net-hawkishness cross-check over the sentences.
        lexicon_score: The deterministic lexicon's net-hawkishness cross-check.
        uncertainty: The spread (max minus min) among the headline and the fired cross-checks, in
            ``[0.0, 2.0]``.
        needs_review: Whether the signals disagree enough to warrant review (a spread at or above
            the threshold, or any opposite-sign disagreement).
    """

    score: float
    tone: ToneLabel
    rate_path: float
    aggregate: StanceAggregate
    structured_net: float
    classifier_net: float
    lexicon_score: float
    uncertainty: float
    needs_review: bool


def combine_signals(
    headline_score: float,
    headline_tone: ToneLabel,
    aggregate: StanceAggregate,
    classifier_net: float,
    lexicon_score: float,
    *,
    lexicon_fired: bool,
    review_threshold: float = _DEFAULT_REVIEW_THRESHOLD,
) -> StanceAssessment:
    """Combine the holistic headline with the structured, classifier, and lexicon cross-checks.

    The holistic score is the headline (ADR 0021); the structured net, the classifier, and the
    lexicon are not averaged into it (ADR 0008) but compared against it to quantify uncertainty. The
    lexicon is included only when it fired, since an abstention is not a disagreement.

    Args:
        headline_score: The model's holistic net-hawkishness for the whole speech (the headline).
        headline_tone: The model's holistic discrete tone (the headline).
        aggregate: The structured :class:`StanceAggregate` for the speech.
        classifier_net: The supervised classifier's net-hawkishness over the same sentences.
        lexicon_score: The deterministic lexicon's net-hawkishness for the speech.
        lexicon_fired: Whether the lexicon matched any term (so its score carries signal).
        review_threshold: The minimum spread that flags the speech for review.

    Returns:
        The :class:`StanceAssessment` carrying the headline, the rate-path, the cross-checks, the
        spread, and the review flag.
    """
    signals = [headline_score, aggregate.net_hawkishness, classifier_net]
    if lexicon_fired:
        signals.append(lexicon_score)
    spread = max(signals) - min(signals)
    opposite_sign = any(a * b < 0 for a, b in itertools.combinations(signals, 2))
    needs_review = spread >= review_threshold or opposite_sign
    return StanceAssessment(
        score=headline_score,
        tone=headline_tone,
        rate_path=aggregate.forward_net,
        aggregate=aggregate,
        structured_net=aggregate.net_hawkishness,
        classifier_net=classifier_net,
        lexicon_score=lexicon_score,
        uncertainty=spread,
        needs_review=needs_review,
    )

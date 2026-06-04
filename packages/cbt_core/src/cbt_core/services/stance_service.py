"""Stance assessment service (ADR 0021).

Orchestrates the structured tone pipeline: split a speech into sentences, keep the policy-relevant
ones, have the model classify each (one batched call), aggregate the labels into a net-hawkishness
measure with a forward-looking sub-measure and a per-aspect breakdown, and cross-check that against
the supervised classifier and the deterministic lexicon. The model aggregate is the production
score; the cross-checks are not averaged into it (ADR 0008) but compared, so a wide disagreement is
surfaced as uncertainty, never hidden.

This service does no persistence and holds no transaction; it is the analysis brain that
``IngestionService`` calls before it writes a speech and its tone observation. The classifier and
lexicon are independent local cross-checks (they run with no network), not a fallback for the model.
"""

from __future__ import annotations

from cbt_core.analysis import (
    DISAGREEMENT_THRESHOLD,
    ClassifiedSentence,
    HawkishDovishLexicon,
    PolicyRelevanceFilter,
    StanceAssessment,
    StanceLabel,
    ToneClassifier,
    aggregate_stances,
    combine_signals,
    split_sentences,
)
from cbt_core.llm.client import LlmClient


class StanceService:
    """Assess a speech's tone with the structured pipeline and two independent cross-checks."""

    def __init__(
        self,
        llm_client: LlmClient,
        *,
        classifier: ToneClassifier | None = None,
        lexicon: HawkishDovishLexicon | None = None,
        relevance_filter: PolicyRelevanceFilter | None = None,
        review_threshold: float = DISAGREEMENT_THRESHOLD,
    ) -> None:
        """Build the service.

        Args:
            llm_client: The LLM boundary; its :meth:`classify_sentences` supplies the model labels.
            classifier: The supervised cross-check classifier; the bundled model is loaded if not
                supplied.
            lexicon: The deterministic lexicon cross-check; a default one is used if not supplied.
            relevance_filter: The policy-relevance filter; a default one is used if not supplied.
            review_threshold: The minimum spread among the signals that flags a speech for review.
        """
        self._llm = llm_client
        self._classifier = classifier if classifier is not None else ToneClassifier.load_default()
        self._lexicon = lexicon if lexicon is not None else HawkishDovishLexicon()
        self._filter = relevance_filter if relevance_filter is not None else PolicyRelevanceFilter()
        self._review_threshold = review_threshold

    def assess(self, text: str) -> StanceAssessment:
        """Score a speech's tone from its policy-relevant sentences, with an uncertainty band.

        Args:
            text: The full speech text.

        Returns:
            The :class:`~cbt_core.analysis.StanceAssessment`: the model's net-hawkishness measure
            and tone (the production score), the classifier and lexicon cross-checks, the spread
            between them as uncertainty, and whether they disagree enough to warrant review. With no
            policy-relevant sentence the model aggregate is an honest abstention (neutral, zero).
        """
        sentences = split_sentences(text)
        relevant = self._filter.filter(sentences)
        model_aggregate = aggregate_stances(
            self._llm.classify_sentences(relevant), total=len(sentences)
        )
        classifier_aggregate = aggregate_stances(
            [
                ClassifiedSentence(
                    text=sentence, label=StanceLabel(self._classifier.score(sentence).label)
                )
                for sentence in relevant
            ],
            total=len(sentences),
        )
        lexicon_result = self._lexicon.score(text)
        return combine_signals(
            model_aggregate,
            classifier_aggregate.net_hawkishness,
            lexicon_result.score,
            lexicon_fired=lexicon_result.fired,
            review_threshold=self._review_threshold,
        )

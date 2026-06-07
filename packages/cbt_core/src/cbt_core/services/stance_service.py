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

from uuid import UUID

from cbt_core.analysis import (
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
from cbt_core.domain.registry import CentralBank
from cbt_core.domain.speech import SpeechStance
from cbt_core.domain.tone import ToneLabel
from cbt_core.exceptions import LlmError
from cbt_core.llm.client import LlmClient
from cbt_core.logging import get_logger

_logger = get_logger(__name__)


def build_stance(
    speech_id: UUID, assessment: StanceAssessment, *, model_id: str = "unknown"
) -> SpeechStance:
    """Build the derived :class:`~cbt_core.domain.speech.SpeechStance` from a stance assessment.

    Shared by ingestion and the re-score path so the persisted decomposition is constructed one way.

    Args:
        speech_id: The speech the decomposition belongs to.
        assessment: The ensemble assessment to persist.
        model_id: The model that produced the sentence classification.

    Returns:
        The :class:`~cbt_core.domain.speech.SpeechStance` ready to upsert.
    """
    return SpeechStance(
        speech_id=speech_id,
        rate_path=assessment.rate_path,
        uncertainty=assessment.uncertainty,
        structured_net=assessment.structured_net,
        classifier_net=assessment.classifier_net,
        lexicon_net=assessment.lexicon_score,
        needs_review=assessment.needs_review,
        aspect_scores={aspect.value: net for aspect, net in assessment.aggregate.by_aspect.items()},
        model_id=model_id,
    )


# The supervised classifier is trained on the FOMC benchmark and measured not to transfer out of
# distribution (ADR 0013; 59.9% on FOMC vs 32.1% on op-fed). It is therefore only a valid cross-check
# for the institution it was trained on; for every other bank it would inject noise and inflate the
# uncertainty band, so it is excluded from their ensemble.
CLASSIFIER_VALID_BANKS: frozenset[CentralBank] = frozenset({CentralBank.FEDERAL_RESERVE})


class StanceService:
    """Assess a speech's tone with the structured pipeline and independent cross-checks."""

    def __init__(
        self,
        llm_client: LlmClient,
        *,
        classifier: ToneClassifier | None = None,
        lexicon: HawkishDovishLexicon | None = None,
        relevance_filter: PolicyRelevanceFilter | None = None,
    ) -> None:
        """Build the service.

        Args:
            llm_client: The LLM boundary; its :meth:`classify_sentences` supplies the model labels.
            classifier: The supervised cross-check classifier; the bundled model is loaded if not
                supplied.
            lexicon: The deterministic lexicon cross-check; a default one is used if not supplied.
            relevance_filter: The policy-relevance filter; a default one is used if not supplied.
        """
        self._llm = llm_client
        self._classifier = classifier if classifier is not None else ToneClassifier.load_default()
        self._lexicon = lexicon if lexicon is not None else HawkishDovishLexicon()
        self._filter = relevance_filter if relevance_filter is not None else PolicyRelevanceFilter()

    def assess(
        self,
        text: str,
        *,
        headline_score: float,
        headline_tone: ToneLabel,
        central_bank: CentralBank,
    ) -> StanceAssessment:
        """Decompose and cross-check a speech around its holistic headline score.

        The caller supplies the headline (the model's holistic whole-speech judgement, the dynamic
        index a macro reader tracks). This service runs the structured sentence-level pipeline to
        add the rate-path and per-aspect decomposition, and cross-checks the headline by direction
        against the structured net, the supervised classifier (only where it is valid), and the
        lexicon, reporting the share that disagree as an uncertainty band. The headline is never
        overwritten by the cross-checks (ADR 0008).

        Args:
            text: The full speech text.
            headline_score: The model's holistic net-hawkishness for the speech (the headline).
            headline_tone: The model's holistic discrete tone (the headline).
            central_bank: The institution; decides whether the FOMC-trained classifier is a valid
                cross-check for this speech.

        Returns:
            The :class:`~cbt_core.analysis.StanceAssessment`: the headline, the rate-path (forward)
            measure, the per-aspect breakdown, the cross-checks, the directional uncertainty, and the
            review flag. With no policy-relevant sentence, or if the model's sentence classification
            fails, the structured part is an honest abstention but the headline is preserved.
        """
        sentences = split_sentences(text)
        relevant = self._filter.filter(sentences)
        classified: list[ClassifiedSentence]
        try:
            classified = self._llm.classify_sentences(relevant)
        except LlmError:
            # The structured sentence pipeline is the supplementary decomposition, not the headline
            # (ADR 0021). When the model cannot return a usable per-sentence classification - it can
            # miscount on a very long speech - degrade the structured part to an honest abstention and
            # keep the headline and the local cross-checks, rather than dropping the whole speech.
            # Logged as a WARNING, never silent (CLAUDE.md section 3); the decomposition is derived
            # and can be re-scored later as the method improves.
            _logger.warning("stance_classification_degraded", relevant_sentences=len(relevant))
            classified = []
        model_aggregate = aggregate_stances(classified, total=len(sentences))
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
            headline_score,
            headline_tone,
            model_aggregate,
            classifier_aggregate.net_hawkishness,
            lexicon_result.score,
            lexicon_fired=lexicon_result.fired,
            classifier_applies=central_bank in CLASSIFIER_VALID_BANKS,
        )

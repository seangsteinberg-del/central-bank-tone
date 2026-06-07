"""Tests for the stance assessment service (ADR 0021).

The service is exercised with the deterministic stub model (keyword stance) plus the real bundled
classifier and lexicon, so the assessment is fully reproducible without a network.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tests._stubs import StubLlmClient

from cbt_core import CentralBank, LlmError, StanceAssessment, StanceService, ToneLabel

_FED = CentralBank.FEDERAL_RESERVE


class _ClassificationFails:
    """A model whose per-sentence classification always fails (Gemini miscount on a long speech)."""

    def classify_sentences(self, sentences: object) -> list[object]:
        raise LlmError("Gemini returned an unparseable sentence classification")


class _FixedClassifier:
    """A stand-in classifier that assigns every sentence the same label (to control the cross-check)."""

    def __init__(self, label: str) -> None:
        self._label = label

    def score(self, _text: str) -> SimpleNamespace:
        return SimpleNamespace(label=self._label, score=0.0, probabilities={}, confidence=1.0)


@pytest.mark.unit
def test_assess_keeps_the_headline_and_runs_the_pipeline(stub_llm_client: StubLlmClient) -> None:
    service = StanceService(stub_llm_client)
    result = service.assess(
        "The committee will raise interest rates and tighten policy. "
        "Inflation pressures call for a restrictive stance. "
        "Thank you for the kind introduction.",
        headline_score=0.42,
        headline_tone=ToneLabel.HAWKISH,
        central_bank=_FED,
    )
    assert isinstance(result, StanceAssessment)
    # The headline is preserved; the cross-checks never overwrite it.
    assert result.score == 0.42
    assert result.tone is ToneLabel.HAWKISH
    assert stub_llm_client.classified == 1


@pytest.mark.unit
def test_assess_drops_filler_with_the_relevance_filter(stub_llm_client: StubLlmClient) -> None:
    result = StanceService(stub_llm_client).assess(
        "Inflation remains too high and we will raise rates. Thank you all for coming today.",
        headline_score=0.3,
        headline_tone=ToneLabel.HAWKISH,
        central_bank=_FED,
    )
    # The greeting is not policy-relevant, so the relevant count is below the total sentence count.
    assert result.aggregate.total == 2
    assert result.aggregate.relevant == 1


@pytest.mark.unit
def test_assess_preserves_headline_when_structured_part_abstains(
    stub_llm_client: StubLlmClient,
) -> None:
    result = StanceService(stub_llm_client).assess(
        "Thank you all. It is a pleasure to be here with you today.",
        headline_score=0.0,
        headline_tone=ToneLabel.NEUTRAL,
        central_bank=_FED,
    )
    assert result.aggregate.fired is False
    assert result.structured_net == 0.0
    assert result.score == 0.0
    assert result.needs_review is False


@pytest.mark.unit
def test_assess_populates_rate_path_and_both_cross_checks(stub_llm_client: StubLlmClient) -> None:
    result = StanceService(stub_llm_client).assess(
        "We will cut rates and ease policy to support growth as inflation falls.",
        headline_score=-0.3,
        headline_tone=ToneLabel.DOVISH,
        central_bank=_FED,
    )
    assert result.score == -0.3
    assert result.rate_path == result.aggregate.forward_net
    assert -1.0 <= result.structured_net <= 1.0
    assert -1.0 <= result.classifier_net <= 1.0
    assert 0.0 <= result.uncertainty <= 1.0


@pytest.mark.unit
def test_assess_excludes_the_classifier_for_non_fed_banks(stub_llm_client: StubLlmClient) -> None:
    # A classifier that always dissents (dovish) against a hawkish headline.
    service = StanceService(stub_llm_client, classifier=_FixedClassifier("dovish"))
    text = "We will raise interest rates and tighten policy as inflation runs too high."
    fed = service.assess(
        text, headline_score=0.5, headline_tone=ToneLabel.HAWKISH, central_bank=_FED
    )
    ecb = service.assess(
        text, headline_score=0.5, headline_tone=ToneLabel.HAWKISH, central_bank=CentralBank.ECB
    )
    # For the Fed the dovish classifier is a valid cross-check that dissents; for the ECB it is
    # excluded (it is FOMC-trained and does not transfer), so the ECB shows less disagreement.
    assert fed.uncertainty > ecb.uncertainty


@pytest.mark.unit
def test_assess_degrades_to_abstention_when_classification_fails() -> None:
    # A failed sentence classification must not lose the speech: the headline stands and the
    # structured part abstains (ADR 0021; CLAUDE.md section 3, honest degradation not a silent drop).
    result = StanceService(_ClassificationFails()).assess(  # type: ignore[arg-type]
        "We will raise interest rates and tighten policy as inflation runs too high.",
        headline_score=0.5,
        headline_tone=ToneLabel.HAWKISH,
        central_bank=_FED,
    )
    assert result.score == 0.5
    assert result.tone is ToneLabel.HAWKISH
    assert result.structured_net == 0.0
    assert result.aggregate.fired is False

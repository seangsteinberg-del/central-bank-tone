"""Tests for the stance assessment service (ADR 0021).

The service is exercised with the deterministic stub model (keyword stance) plus the real bundled
classifier and lexicon, so the assessment is fully reproducible without a network.
"""

from __future__ import annotations

import pytest
from tests._stubs import StubLlmClient

from cbt_core import StanceAssessment, StanceService, ToneLabel


@pytest.mark.unit
def test_assess_scores_hawkish_text_hawkish(stub_llm_client: StubLlmClient) -> None:
    service = StanceService(stub_llm_client)
    result = service.assess(
        "The committee will raise interest rates and tighten policy. "
        "Inflation pressures call for a restrictive stance. "
        "Thank you for the kind introduction."
    )
    assert isinstance(result, StanceAssessment)
    assert result.score > 0
    assert result.tone is ToneLabel.HAWKISH
    assert stub_llm_client.classified == 1


@pytest.mark.unit
def test_assess_drops_filler_with_the_relevance_filter(stub_llm_client: StubLlmClient) -> None:
    result = StanceService(stub_llm_client).assess(
        "Inflation remains too high and we will raise rates. Thank you all for coming today."
    )
    # The greeting is not policy-relevant, so the relevant count is below the total sentence count.
    assert result.aggregate.total == 2
    assert result.aggregate.relevant == 1


@pytest.mark.unit
def test_assess_abstains_when_no_policy_sentence_is_present(stub_llm_client: StubLlmClient) -> None:
    result = StanceService(stub_llm_client).assess(
        "Thank you all. It is a pleasure to be here with you today."
    )
    assert result.score == 0.0
    assert result.tone is ToneLabel.NEUTRAL
    assert result.aggregate.fired is False
    assert result.needs_review is False


@pytest.mark.unit
def test_assess_populates_both_cross_checks(stub_llm_client: StubLlmClient) -> None:
    result = StanceService(stub_llm_client).assess(
        "We will cut rates and ease policy to support growth as inflation falls."
    )
    # The production score is the model aggregate; the classifier is an independent cross-check.
    assert result.score == result.aggregate.net_hawkishness
    assert -1.0 <= result.classifier_net <= 1.0
    assert result.uncertainty >= 0.0

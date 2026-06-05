"""Tests for the stance assessment service (ADR 0021).

The service is exercised with the deterministic stub model (keyword stance) plus the real bundled
classifier and lexicon, so the assessment is fully reproducible without a network.
"""

from __future__ import annotations

import pytest
from tests._stubs import StubLlmClient

from cbt_core import StanceAssessment, StanceService, ToneLabel


@pytest.mark.unit
def test_assess_keeps_the_headline_and_runs_the_pipeline(stub_llm_client: StubLlmClient) -> None:
    service = StanceService(stub_llm_client)
    result = service.assess(
        "The committee will raise interest rates and tighten policy. "
        "Inflation pressures call for a restrictive stance. "
        "Thank you for the kind introduction.",
        headline_score=0.42,
        headline_tone=ToneLabel.HAWKISH,
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
    )
    assert result.score == -0.3
    assert result.rate_path == result.aggregate.forward_net
    assert -1.0 <= result.structured_net <= 1.0
    assert -1.0 <= result.classifier_net <= 1.0
    assert result.uncertainty >= 0.0

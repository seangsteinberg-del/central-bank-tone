"""Tests for the sentence-level stance aggregation engine (ADR 0021).

Covers the sentence splitter, the policy-relevance filter (word-boundary and phrase matching), the
normalized net-hawkishness measure and its abstention guard, the continuous-to-discrete tone
mapping (abstention, mixed, neutral band, and the directional cases), and the full aggregation
including the forward-looking sub-measure and the per-aspect breakdown.
"""

from __future__ import annotations

import pytest

from cbt_core import ToneLabel
from cbt_core.analysis import (
    Aspect,
    ClassifiedSentence,
    Horizon,
    PolicyRelevanceFilter,
    StanceAggregate,
    StanceLabel,
    aggregate_stances,
    combine_signals,
    infer_aspect,
    infer_horizon,
    net_hawkishness,
    split_sentences,
    to_tone_label,
)


def _aggregate(net: float, tone: ToneLabel = ToneLabel.NEUTRAL) -> StanceAggregate:
    """A minimal aggregate with a chosen net measure, for testing the signal combiner."""
    return StanceAggregate(
        net_hawkishness=net,
        tone=tone,
        hawkish=0,
        dovish=0,
        neutral=0,
        relevant=1,
        total=1,
        forward_net=0.0,
        forward_relevant=0,
        by_aspect={},
    )


def _sentence(
    label: StanceLabel,
    *,
    aspect: Aspect = Aspect.OTHER,
    horizon: Horizon = Horizon.UNSPECIFIED,
    text: str = "x",
) -> ClassifiedSentence:
    """Build a classified sentence; only the fields a test asserts on need to be set."""
    return ClassifiedSentence(text=text, label=label, aspect=aspect, horizon=horizon)


@pytest.mark.unit
def test_split_sentences_trims_and_drops_empties() -> None:
    assert split_sentences("  Rates rise.  Inflation falls!  ") == [
        "Rates rise.",
        "Inflation falls!",
    ]


@pytest.mark.unit
def test_split_sentences_keeps_a_final_unterminated_clause() -> None:
    assert split_sentences("we will hike") == ["we will hike"]


@pytest.mark.unit
def test_split_sentences_empty_text_is_no_sentences() -> None:
    assert split_sentences("   ") == []


@pytest.mark.unit
def test_relevance_filter_matches_a_single_policy_word() -> None:
    assert PolicyRelevanceFilter().is_relevant("The inflation outlook has worsened.")


@pytest.mark.unit
def test_relevance_filter_matches_a_multiword_phrase() -> None:
    assert PolicyRelevanceFilter().is_relevant("We discussed the balance sheet at length.")


@pytest.mark.unit
def test_relevance_filter_rejects_off_topic_filler() -> None:
    assert not PolicyRelevanceFilter().is_relevant("Thank you for the kind introduction.")


@pytest.mark.unit
def test_relevance_filter_word_match_respects_token_boundaries() -> None:
    # "rate" is a policy word, but it must not fire on "accelerate" via substring matching.
    assert not PolicyRelevanceFilter().is_relevant("Investment continued to accelerate sharply.")


@pytest.mark.unit
def test_relevance_filter_filter_keeps_only_relevant_sentences_in_order() -> None:
    sentences = [
        "Good afternoon everyone.",
        "Inflation remains too high.",
        "It is a pleasure to be here.",
        "We may need to raise rates.",
    ]
    assert PolicyRelevanceFilter().filter(sentences) == [
        "Inflation remains too high.",
        "We may need to raise rates.",
    ]


@pytest.mark.unit
def test_net_hawkishness_normalizes_by_relevant_count() -> None:
    # 3 hawkish, 1 dovish, 6 relevant -> (3 - 1) / 6.
    assert net_hawkishness(3, 1, 6) == pytest.approx(2 / 6)


@pytest.mark.unit
def test_net_hawkishness_is_plus_one_when_all_hawkish() -> None:
    assert net_hawkishness(4, 0, 4) == pytest.approx(1.0)


@pytest.mark.unit
def test_net_hawkishness_is_minus_one_when_all_dovish() -> None:
    assert net_hawkishness(0, 4, 4) == pytest.approx(-1.0)


@pytest.mark.unit
def test_net_hawkishness_abstains_to_zero_when_no_relevant_sentences() -> None:
    assert net_hawkishness(0, 0, 0) == 0.0


@pytest.mark.unit
def test_to_tone_label_abstains_to_neutral_with_no_relevant_sentences() -> None:
    assert to_tone_label((0, 0, 0)) is ToneLabel.NEUTRAL


@pytest.mark.unit
def test_to_tone_label_is_hawkish_when_net_clears_the_band() -> None:
    assert to_tone_label((5, 1, 6)) is ToneLabel.HAWKISH


@pytest.mark.unit
def test_to_tone_label_is_dovish_when_net_is_negative() -> None:
    assert to_tone_label((1, 5, 6)) is ToneLabel.DOVISH


@pytest.mark.unit
def test_to_tone_label_is_neutral_inside_the_band() -> None:
    # 5 hawkish, 5 dovish, 100 relevant -> net 0.0, but both shares (0.05) below the mixed minimum.
    assert to_tone_label((5, 5, 100)) is ToneLabel.NEUTRAL


@pytest.mark.unit
def test_to_tone_label_is_mixed_when_both_sides_are_strongly_present() -> None:
    # 4 hawkish, 4 dovish, 10 relevant -> each share 0.4 >= 0.25, so genuinely two-sided.
    assert to_tone_label((4, 4, 10)) is ToneLabel.MIXED


@pytest.mark.unit
def test_to_tone_label_mixed_takes_precedence_over_a_directional_lean() -> None:
    # Strongly two-sided but net leans hawkish; MIXED is the honest label, not HAWKISH.
    assert to_tone_label((5, 3, 10)) is ToneLabel.MIXED


@pytest.mark.unit
def test_aggregate_stances_empty_input_is_an_honest_abstention() -> None:
    result = aggregate_stances([])
    assert result.net_hawkishness == 0.0
    assert result.tone is ToneLabel.NEUTRAL
    assert result.relevant == 0
    assert result.fired is False
    assert result.directional_share == 0.0


@pytest.mark.unit
def test_aggregate_stances_counts_and_measure_match_the_labels() -> None:
    # 4 hawkish, 1 dovish, 3 neutral: the dovish share (1/8) is below the mixed minimum, so this is
    # a hawkish lean, not MIXED.
    sentences = [
        *[_sentence(StanceLabel.HAWKISH) for _ in range(4)],
        _sentence(StanceLabel.DOVISH),
        *[_sentence(StanceLabel.NEUTRAL) for _ in range(3)],
    ]
    result = aggregate_stances(sentences)
    assert (result.hawkish, result.dovish, result.neutral, result.relevant) == (4, 1, 3, 8)
    assert result.net_hawkishness == pytest.approx((4 - 1) / 8)
    assert result.tone is ToneLabel.HAWKISH
    assert result.fired is True
    assert result.directional_share == pytest.approx(5 / 8)


@pytest.mark.unit
def test_aggregate_stances_forward_measure_uses_only_forward_sentences() -> None:
    sentences = [
        _sentence(StanceLabel.HAWKISH, horizon=Horizon.FORWARD),
        _sentence(StanceLabel.HAWKISH, horizon=Horizon.FORWARD),
        # A backward-looking dovish sentence: it lowers net_hawkishness but not forward_net.
        _sentence(StanceLabel.DOVISH, horizon=Horizon.BACKWARD),
    ]
    result = aggregate_stances(sentences)
    assert result.forward_relevant == 2
    assert result.forward_net == pytest.approx(1.0)
    assert result.net_hawkishness == pytest.approx((2 - 1) / 3)


@pytest.mark.unit
def test_aggregate_stances_forward_measure_abstains_with_no_forward_sentences() -> None:
    result = aggregate_stances([_sentence(StanceLabel.HAWKISH, horizon=Horizon.BACKWARD)])
    assert result.forward_relevant == 0
    assert result.forward_net == 0.0


@pytest.mark.unit
def test_aggregate_stances_breaks_the_measure_down_by_aspect() -> None:
    sentences = [
        _sentence(StanceLabel.HAWKISH, aspect=Aspect.INFLATION),
        _sentence(StanceLabel.HAWKISH, aspect=Aspect.INFLATION),
        _sentence(StanceLabel.DOVISH, aspect=Aspect.GROWTH),
    ]
    result = aggregate_stances(sentences)
    assert result.by_aspect[Aspect.INFLATION] == pytest.approx(1.0)
    assert result.by_aspect[Aspect.GROWTH] == pytest.approx(-1.0)
    assert set(result.by_aspect) == {Aspect.INFLATION, Aspect.GROWTH}


@pytest.mark.unit
def test_aggregate_stances_records_pre_filter_total_when_given() -> None:
    result = aggregate_stances([_sentence(StanceLabel.HAWKISH)], total=10)
    assert result.total == 10
    assert result.relevant == 1


@pytest.mark.unit
def test_aggregate_stances_total_defaults_to_relevant_count() -> None:
    result = aggregate_stances([_sentence(StanceLabel.HAWKISH), _sentence(StanceLabel.DOVISH)])
    assert result.total == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("Inflation remains elevated above our target.", Aspect.INFLATION),
        ("The labour market is tight and wage pressures persist.", Aspect.EMPLOYMENT),
        ("We will continue to shrink the balance sheet.", Aspect.BALANCE_SHEET),
        ("Risks to financial stability have built up.", Aspect.FINANCIAL_STABILITY),
        ("Output growth and demand have softened.", Aspect.GROWTH),
        ("Our projection and forward guidance are unchanged.", Aspect.GUIDANCE),
        ("Thank you all for the warm welcome.", Aspect.OTHER),
    ],
)
def test_infer_aspect_picks_the_dominant_topic(sentence: str, expected: Aspect) -> None:
    assert infer_aspect(sentence) is expected


@pytest.mark.unit
def test_infer_aspect_breaks_ties_by_declared_order() -> None:
    # One inflation cue and one growth cue: inflation is declared first, so it wins the tie.
    assert infer_aspect("Inflation and growth were both discussed.") is Aspect.INFLATION


@pytest.mark.unit
@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("We will raise rates further.", Horizon.FORWARD),
        ("Inflation rose sharply last year.", Horizon.BACKWARD),
        ("The committee met in Basel.", Horizon.UNSPECIFIED),
        ("Rates rose and will rise again.", Horizon.UNSPECIFIED),
    ],
)
def test_infer_horizon_reads_temporal_cues(sentence: str, expected: Horizon) -> None:
    assert infer_horizon(sentence) is expected


@pytest.mark.unit
def test_combine_signals_takes_score_and_tone_from_the_model_aggregate() -> None:
    result = combine_signals(
        _aggregate(0.6, ToneLabel.HAWKISH),
        classifier_net=0.5,
        lexicon_score=0.0,
        lexicon_fired=False,
    )
    assert result.score == 0.6
    assert result.tone is ToneLabel.HAWKISH
    assert result.classifier_net == 0.5


@pytest.mark.unit
def test_combine_signals_agreement_is_low_uncertainty_and_no_review() -> None:
    result = combine_signals(
        _aggregate(0.3), classifier_net=0.35, lexicon_score=0.4, lexicon_fired=True
    )
    assert result.uncertainty == pytest.approx(0.1)
    assert result.needs_review is False


@pytest.mark.unit
def test_combine_signals_wide_spread_flags_review() -> None:
    result = combine_signals(
        _aggregate(0.8), classifier_net=0.0, lexicon_score=0.0, lexicon_fired=False
    )
    assert result.uncertainty == pytest.approx(0.8)
    assert result.needs_review is True


@pytest.mark.unit
def test_combine_signals_opposite_sign_flags_review_even_within_the_band() -> None:
    # A small spread (0.2) but the model and classifier disagree on direction.
    result = combine_signals(
        _aggregate(0.1), classifier_net=-0.1, lexicon_score=0.0, lexicon_fired=False
    )
    assert result.uncertainty == pytest.approx(0.2)
    assert result.needs_review is True


@pytest.mark.unit
def test_combine_signals_ignores_an_abstaining_lexicon() -> None:
    # The lexicon score is extreme but it abstained, so it must not count toward the spread.
    result = combine_signals(
        _aggregate(0.3), classifier_net=0.3, lexicon_score=-1.0, lexicon_fired=False
    )
    assert result.uncertainty == pytest.approx(0.0)
    assert result.needs_review is False


@pytest.mark.unit
def test_combine_signals_counts_a_fired_lexicon_toward_the_spread() -> None:
    result = combine_signals(
        _aggregate(0.3), classifier_net=0.3, lexicon_score=0.9, lexicon_fired=True
    )
    assert result.uncertainty == pytest.approx(0.6)
    assert result.needs_review is True

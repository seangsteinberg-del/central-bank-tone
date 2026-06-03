"""Tests for the deterministic hawkish/dovish lexicon baseline (ADR 0008)."""

from __future__ import annotations

import pytest

from cbt_core.analysis.lexicon import HawkishDovishLexicon, LexiconScore, disagrees


@pytest.mark.unit
def test_hawkish_text_scores_positive() -> None:
    score = HawkishDovishLexicon().score(
        "The committee will tighten policy and hike rates given inflationary pressure."
    )
    assert score.score > 0
    assert score.hawkish_hits > 0


@pytest.mark.unit
def test_dovish_text_scores_negative() -> None:
    score = HawkishDovishLexicon().score(
        "We keep an accommodative stance, support growth, and stay patient amid headwinds."
    )
    assert score.score < 0
    assert score.dovish_hits > 0


@pytest.mark.unit
def test_text_without_signal_words_abstains_at_zero() -> None:
    score = HawkishDovishLexicon().score("The weather today is pleasant and the room was full.")
    assert score.score == 0.0
    assert score.hawkish_hits == 0
    assert score.dovish_hits == 0


@pytest.mark.unit
def test_score_is_bounded_to_unit_interval() -> None:
    score = HawkishDovishLexicon().score("hike hike hike tighten tightening")
    assert score.score == 1.0
    assert -1.0 <= score.score <= 1.0


@pytest.mark.unit
def test_custom_word_lists_are_used() -> None:
    lexicon = HawkishDovishLexicon(hawkish=("up",), dovish=("down",))
    score = lexicon.score("up up down")
    assert score.hawkish_hits == 2
    assert score.dovish_hits == 1
    assert score.score == pytest.approx(1 / 3)


@pytest.mark.unit
def test_phrase_is_not_double_counted_by_its_substring() -> None:
    # "rate hike" must count once as hawkish, not also as the substring "hike".
    score = HawkishDovishLexicon().score("the committee discussed a rate hike")
    assert score.hawkish_hits == 1
    assert score.dovish_hits == 0


@pytest.mark.unit
def test_hawkish_phrase_is_not_cancelled_by_a_dovish_substring() -> None:
    # "withdraw accommodation" is hawkish; its substring "accommodation" must not also score dovish.
    score = HawkishDovishLexicon().score("the committee will withdraw accommodation")
    assert score.hawkish_hits == 1
    assert score.dovish_hits == 0
    assert score.score == 1.0


@pytest.mark.unit
def test_negation_flips_a_dovish_term_to_hawkish() -> None:
    score = HawkishDovishLexicon().score("policy is not accommodative")
    assert score.hawkish_hits == 1
    assert score.dovish_hits == 0
    assert score.score == 1.0


@pytest.mark.unit
def test_negation_flips_a_hawkish_term_to_dovish() -> None:
    score = HawkishDovishLexicon().score("there will be no further tightening")
    assert score.dovish_hits == 1
    assert score.hawkish_hits == 0
    assert score.score == -1.0


@pytest.mark.unit
def test_plural_phrase_counts_once() -> None:
    score = HawkishDovishLexicon().score("downside risks remain elevated")
    assert score.dovish_hits == 1


@pytest.mark.unit
def test_disagrees_is_false_when_lexicon_abstains() -> None:
    # No terms fired: the lexicon cannot disagree, whatever the model said.
    abstained = LexiconScore(score=0.0, hawkish_hits=0, dovish_hits=0)
    assert disagrees(0.9, abstained) is False


@pytest.mark.unit
def test_disagrees_on_opposite_sign() -> None:
    dovish = LexiconScore(score=-0.8, hawkish_hits=0, dovish_hits=2)
    assert disagrees(0.7, dovish) is True


@pytest.mark.unit
def test_disagrees_on_large_magnitude_gap_same_sign() -> None:
    weakly_hawkish = LexiconScore(score=0.1, hawkish_hits=1, dovish_hits=0)
    assert disagrees(0.9, weakly_hawkish) is True  # gap 0.8 >= 0.5 threshold


@pytest.mark.unit
def test_does_not_disagree_when_scores_are_close_and_aligned() -> None:
    hawkish = LexiconScore(score=0.8, hawkish_hits=2, dovish_hits=0)
    assert disagrees(0.7, hawkish) is False

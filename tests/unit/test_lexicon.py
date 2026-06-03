"""Tests for the deterministic hawkish/dovish lexicon baseline (ADR 0008)."""

from __future__ import annotations

import pytest

from cbt_core.analysis.lexicon import HawkishDovishLexicon


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

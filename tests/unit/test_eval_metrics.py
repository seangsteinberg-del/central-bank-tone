"""Regression tests for the eval-script statistics (the headline research numbers; ADR 0012/0013).

The eval scripts live under ``scripts/``, which is excluded from the coverage source, so a bug in a
confusion index or an F1 denominator could silently corrupt the published accuracy / macro-F1 /
McNemar numbers. These pin the scoring helpers to hand-computed fixtures. Both eval scripts carry
their own copy of the label metrics, so the shared ones are checked against both: if one copy
drifts, this fails.
"""

from __future__ import annotations

import pytest
from scripts import eval_cross_dataset, eval_tone

# Chosen so every confusion cell and class metric is hand-checkable.
_GOLD = ["hawkish", "hawkish", "dovish", "dovish", "neutral", "neutral"]
_PRED = ["hawkish", "dovish", "dovish", "neutral", "neutral", "neutral"]

_MODULES = [eval_tone, eval_cross_dataset]
_IDS = ["eval_tone", "eval_cross_dataset"]


@pytest.mark.parametrize("module", _MODULES, ids=_IDS)
def test_accuracy_matches_hand_count(module: object) -> None:
    assert module._accuracy(_GOLD, _PRED) == pytest.approx(4 / 6)


@pytest.mark.parametrize("module", _MODULES, ids=_IDS)
def test_confusion_is_gold_rows_by_predicted_columns(module: object) -> None:
    # rows = gold in (hawkish, dovish, neutral) order; columns = predicted, same order.
    assert module._confusion(_GOLD, _PRED) == [[1, 1, 0], [0, 1, 1], [0, 0, 2]]


@pytest.mark.parametrize("module", _MODULES, ids=_IDS)
def test_per_class_precision_recall_f1_support(module: object) -> None:
    pc = module._per_class(_GOLD, _PRED)
    assert pc["hawkish"]["precision"] == pytest.approx(1.0)
    assert pc["hawkish"]["recall"] == pytest.approx(0.5)
    assert pc["hawkish"]["f1"] == pytest.approx(2 / 3)
    assert pc["hawkish"]["support"] == pytest.approx(2.0)
    assert pc["dovish"]["f1"] == pytest.approx(0.5)
    assert pc["neutral"]["precision"] == pytest.approx(2 / 3)
    assert pc["neutral"]["recall"] == pytest.approx(1.0)
    assert pc["neutral"]["f1"] == pytest.approx(0.8)


@pytest.mark.parametrize("module", _MODULES, ids=_IDS)
def test_macro_f1_is_the_unweighted_mean_of_class_f1(module: object) -> None:
    assert module._macro_f1(_GOLD, _PRED) == pytest.approx((2 / 3 + 0.5 + 0.8) / 3)


def test_significance_mcnemar_and_bootstrap_ci() -> None:
    # 10 items, all gold "a". A is right on 9, B on 5; the 6 discordant split 5 (A only) / 1 (B only).
    gold = ["a"] * 10
    pred_a = ["a", "a", "a", "a", "a", "b", "a", "a", "a", "a"]
    pred_b = ["b", "b", "b", "b", "b", "a", "a", "a", "a", "a"]
    sig = eval_tone._significance(gold, pred_a, pred_b)
    assert sig.a_only_correct == 5
    assert sig.b_only_correct == 1
    assert sig.mcnemar_chi2 == pytest.approx(1.5)  # (|5 - 1| - 1)**2 / (5 + 1)
    assert 0.0 < sig.mcnemar_p < 1.0
    assert sig.acc_diff == pytest.approx(0.4)  # 9/10 - 5/10
    assert -1.0 <= sig.ci_low <= sig.ci_high <= 1.0

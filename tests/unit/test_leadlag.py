"""Tests for the shared lead-lag statistics (ADR 0023).

Deterministic and offline: every RNG is seeded, so the bootstrap intervals are reproducible
(CLAUDE.md section 5). These cover the corrections the module exists to make: chronological pairing
against forward changes, a block length that spans the window overlap, and a family-wise interval
that is at least as wide as a per-test one.
"""

from __future__ import annotations

import numpy as np
import pytest

from cbt_core.analysis.leadlag import block_bootstrap_ci, block_length, lead_pairs, pearson


@pytest.mark.unit
def test_pearson_is_one_for_a_line_and_zero_for_a_constant_series() -> None:
    assert pearson(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 4.0, 6.0, 8.0])) == pytest.approx(
        1.0
    )
    assert pearson(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])) == 0.0  # constant -> 0


@pytest.mark.unit
def test_lead_pairs_are_chronological_and_use_forward_changes() -> None:
    tone = {3: 0.3, 1: 0.1, 2: 0.2}  # deliberately unsorted insertion order
    rate = {0: 5.0, 1: 5.1, 2: 5.3, 3: 5.6, 4: 5.6, 5: 6.1}

    tones0, changes0 = lead_pairs(tone, rate, horizon=0)
    assert list(tones0) == [0.1, 0.2, 0.3]  # ascending month order, not dict order
    # horizon 0 is the change over the prior month (m-1 -> m)
    assert list(changes0) == pytest.approx([5.1 - 5.0, 5.3 - 5.1, 5.6 - 5.3])

    tones3, changes3 = lead_pairs(tone, rate, horizon=3)
    # month m needs m and m+3 in the rate series: m=1 -> (1,4), m=2 -> (2,5); m=3 -> (3,6) is dropped
    assert list(tones3) == [0.1, 0.2]
    assert list(changes3) == pytest.approx([5.6 - 5.1, 6.1 - 5.3])


@pytest.mark.unit
def test_block_length_spans_the_horizon_and_is_bounded() -> None:
    assert block_length(30, 0) >= 2
    assert block_length(30, 6) >= 7  # at least horizon + 1, to span the window overlap
    assert block_length(30, 6) <= 30 // 2  # capped so there are always at least two blocks
    assert block_length(8, 6) == 4  # the n // 2 cap binds for a small sample


@pytest.mark.unit
def test_block_bootstrap_ci_widens_with_family_size_and_is_reproducible() -> None:
    rng = np.random.default_rng(0)
    n = 40
    xs = np.linspace(0.0, 1.0, n) + rng.normal(0.0, 0.1, n)
    ys = xs * 0.8 + rng.normal(0.0, 0.2, n)

    point1, lo1, hi1 = block_bootstrap_ci(xs, ys, samples=4000, seed=7, horizon=3, family_size=1)
    point12, lo12, hi12 = block_bootstrap_ci(
        xs, ys, samples=4000, seed=7, horizon=3, family_size=12
    )

    assert point1 == pytest.approx(point12)  # the point estimate does not depend on the correction
    assert (hi12 - lo12) >= (hi1 - lo1)  # the family-wise interval is at least as wide
    # deterministic: identical arguments give an identical interval
    assert block_bootstrap_ci(xs, ys, samples=4000, seed=7, horizon=3, family_size=12) == (
        point12,
        lo12,
        hi12,
    )


@pytest.mark.unit
def test_block_bootstrap_ci_is_degenerate_for_a_tiny_sample() -> None:
    xs = np.array([1.0, 2.0])
    ys = np.array([1.0, 2.0])
    point, lo, hi = block_bootstrap_ci(xs, ys, samples=100, seed=0, horizon=0, family_size=12)
    assert lo == point == hi  # fewer than three pairs: a degenerate interval, never a fake CI

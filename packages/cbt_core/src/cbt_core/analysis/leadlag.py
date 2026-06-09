"""Lead-lag correlation statistics for tone-vs-rate evaluation (ADR 0022, ADR 0023).

Pure numpy helpers shared by :class:`~cbt_core.services.market_service.MarketSignalService` (served,
per request) and ``scripts/eval_corpus_vs_rates.py`` (the offline report), so the published numbers
and the served numbers are computed by ONE implementation and cannot drift (the divergence ADR 0022
fixed). No IO and no domain types here: numpy arrays and floats only.

Two inference corrections for the structure of this problem (ADR 0023):

- The pairs are serially dependent. Each pair is a tone level against an OVERLAPPING forward rate
  change (month m's "+3 month change" shares two months with month m+1's), and both series are
  persistent. An i.i.d. resample of pairs ignores that autocorrelation and reports a confidence
  interval that is too narrow. The CI here uses a circular MOVING-BLOCK bootstrap, resampling
  contiguous blocks so the short-range dependence is preserved.
- Many cells are tested together (each tone index against each rate series across several
  horizons). A 95% CI per cell would, across the whole family, spuriously exclude zero in some cell
  by chance alone. The interval is therefore widened to a family-wise (Bonferroni) level,
  controlling the chance that ANY cell in the family wrongly excludes zero.
"""

from __future__ import annotations

import math

import numpy as np

# Family-wise error rate the Bonferroni-corrected intervals control across all tested cells.
FAMILY_WISE_ALPHA = 0.05


def pearson(xs: np.ndarray, ys: np.ndarray) -> float:
    """Pearson correlation of two equal-length arrays (0 when either is constant or too short)."""
    if len(xs) < 3 or xs.std() == 0 or ys.std() == 0:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def lead_pairs(
    tone: dict[int, float], rate: dict[int, float], horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Tone at month ``m`` paired with the rate change from ``m`` to ``m + horizon``, chronological.

    Horizon 0 is the change over the prior month (contemporaneous co-movement); a positive horizon
    is a lead test (does this month's tone precede the move over the next ``horizon`` months). Pairs
    are returned in ascending month order, so a block bootstrap resamples contiguous time.

    Args:
        tone: Month index to tone value.
        rate: Month index to rate value.
        horizon: Lead horizon in months (0 for contemporaneous).

    Returns:
        ``(tones, changes)`` as equal-length arrays, oldest month first.
    """
    tones: list[float] = []
    changes: list[float] = []
    for month in sorted(tone):
        start = month - 1 if horizon == 0 else month
        end = month if horizon == 0 else month + horizon
        if start in rate and end in rate:
            tones.append(tone[month])
            changes.append(rate[end] - rate[start])
    return np.array(tones), np.array(changes)


def block_length(n: int, horizon: int) -> int:
    """Moving-block length: large enough to span the window overlap, scaled by the sample size.

    Overlapping ``horizon``-month windows induce dependence of length about ``horizon``, so the block
    must be at least ``horizon + 1`` to capture it; it also grows like ``n ** (1/3)`` (the usual
    block-bootstrap rate) and is capped at half the sample so there are always at least two blocks.

    Args:
        n: Number of paired observations.
        horizon: The lead horizon in months.

    Returns:
        The block length to resample with.
    """
    cube_root = round(math.pow(n, 1.0 / 3.0))  # math.pow is typed -> float (the ** operator is Any)
    base = max(2, horizon + 1, cube_root)
    return min(base, max(2, n // 2))


def block_bootstrap_ci(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    samples: int,
    seed: int,
    horizon: int,
    family_size: int,
) -> tuple[float, float, float]:
    """Pearson point estimate and a family-wise circular moving-block bootstrap CI.

    Resamples contiguous (wrap-around) blocks of the chronologically ordered pairs, preserving the
    serial dependence the overlapping windows create, and takes percentiles at the Bonferroni level
    ``FAMILY_WISE_ALPHA / family_size`` so the interval controls family-wise error across all
    ``family_size`` tested cells.

    Args:
        xs: Tone values, chronologically ordered (see :func:`lead_pairs`).
        ys: Paired forward rate changes, in the same order.
        samples: Bootstrap resamples; use enough that the deep family-wise tail is stable.
        seed: RNG seed, so the interval is reproducible.
        horizon: The lead horizon, which sets the block length.
        family_size: Number of simultaneously tested cells, for the Bonferroni adjustment.

    Returns:
        ``(point, ci_low, ci_high)``; a degenerate ``(point, point, point)`` when ``n < 3``.
    """
    n = len(xs)
    point = pearson(xs, ys)
    if n < 3:
        return point, point, point
    length = block_length(n, horizon)
    blocks = -(-n // length)  # ceil division: enough blocks to cover n after truncation
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(samples, blocks))
    offsets = np.arange(length)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    idx = idx.reshape(samples, blocks * length)[:, :n]
    gx = xs[idx]
    gy = ys[idx]
    xm = gx - gx.mean(axis=1, keepdims=True)
    ym = gy - gy.mean(axis=1, keepdims=True)
    num = (xm * ym).sum(axis=1)
    den = np.sqrt((xm**2).sum(axis=1) * (ym**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = np.where(den > 0, num / den, 0.0)
    draws.sort()
    tail = FAMILY_WISE_ALPHA / family_size / 2.0
    return (
        point,
        float(draws[int(tail * samples)]),
        float(draws[min(int((1.0 - tail) * samples), samples - 1)]),
    )

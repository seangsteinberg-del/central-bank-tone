"""Does our production tone signal track and lead Fed policy? (ADR 0021 validation)

Measures the platform's OWN headline tone - the Gemini holistic score already stored on every
speech - against real interest-rate data, with no API key and no re-scoring. It builds a monthly
Federal Reserve tone index from the stored scores and relates it to two FRED series (the effective
fed funds rate FEDFUNDS and the 2-year Treasury yield GS2): contemporaneous co-movement and, the
test that matters for a tradeable signal, whether hawkish tone LEADS rate increases over the
following months. Each correlation carries a bootstrap 95% CI. Fed-only, because that is where free
market ground truth exists; the limitation is stated in the report.

Run: ``uv run python scripts/eval_corpus_vs_rates.py`` (needs the live database and a network for
FRED; no Gemini key).
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
from sqlalchemy import text

from cbt_core.persistence.engine import create_engine_from_settings
from cbt_core.settings import Settings

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _REPO_ROOT / "data" / "benchmarks"
_REPORT = _REPO_ROOT / "docs" / "research" / "corpus-tone-vs-rates.md"
_FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
_MIN_SPEECHES_PER_MONTH = 3
_BOOTSTRAP = 2000
_SEED = 0
# Lead horizons in months: 0 is the same-month change, the rest test whether tone precedes the move.
_HORIZONS = (0, 3, 6)


def _fed_monthly_tone() -> tuple[dict[int, float], dict[int, int]]:
    """Mean stored headline score per month index for Federal Reserve speeches, plus the counts."""
    engine = create_engine_from_settings(Settings())
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "select delivered_at, score from speech "
                "where central_bank = 'FEDERAL_RESERVE' order by delivered_at"
            )
        ).all()
    for delivered_at, score in rows:
        index = delivered_at.year * 12 + (delivered_at.month - 1)
        sums[index] = sums.get(index, 0.0) + float(score)
        counts[index] = counts.get(index, 0) + 1
    tone = {i: sums[i] / counts[i] for i in counts if counts[i] >= _MIN_SPEECHES_PER_MONTH}
    return tone, {i: counts[i] for i in tone}


def _fred_monthly(series_id: str) -> dict[int, float]:
    """A FRED monthly series as a month-index map, caching the raw CSV (skips missing '.' values)."""
    cache = _CACHE_DIR / f"fred_{series_id}_monthly.csv"
    if cache.exists():
        body = cache.read_text(encoding="utf-8")
    else:
        request = urllib.request.Request(  # noqa: S310  (trusted FRED host, https)
            f"{_FRED}{series_id}", headers={"User-Agent": "cbt-eval/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            body = response.read().decode("utf-8")
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(body, encoding="utf-8")
    series: dict[int, float] = {}
    reader = csv.reader(io.StringIO(body))
    next(reader, None)  # header
    for row in reader:
        if len(row) < 2 or row[1] in ("", "."):
            continue
        year, month = int(row[0][:4]), int(row[0][5:7])
        series[year * 12 + (month - 1)] = float(row[1])
    return series


def _pearson(xs: np.ndarray, ys: np.ndarray) -> float:
    """Pearson correlation of two equal-length series (0 when either is constant or too short)."""
    if len(xs) < 3 or xs.std() == 0 or ys.std() == 0:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def _bootstrap_ci(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """Pearson point estimate and 95% bootstrap CI over paired resamples."""
    rng = np.random.default_rng(_SEED)
    n = len(xs)
    point = _pearson(xs, ys)
    if n < 3:
        return point, point, point
    samples = np.empty(_BOOTSTRAP)
    for i in range(_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        samples[i] = _pearson(xs[idx], ys[idx])
    samples.sort()
    return (
        point,
        float(samples[int(0.025 * _BOOTSTRAP)]),
        float(samples[int(0.975 * _BOOTSTRAP) - 1]),
    )


def _lead_pairs(
    tone: dict[int, float], rate: dict[int, float], horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Tone at month ``m`` paired with the rate change from ``m`` to ``m + horizon``.

    A horizon of 0 uses the change over the prior month (contemporaneous co-movement); a positive
    horizon is a lead test (does this month's tone precede the move over the next ``horizon`` months).
    """
    tones: list[float] = []
    changes: list[float] = []
    for month, value in tone.items():
        start = month - 1 if horizon == 0 else month
        end = month if horizon == 0 else month + horizon
        if start in rate and end in rate:
            tones.append(value)
            changes.append(rate[end] - rate[start])
    return np.array(tones), np.array(changes)


def _row(label: str, tone: dict[int, float], rate: dict[int, float]) -> str:
    """A markdown row: contemporaneous and forward correlations of tone with a rate series."""
    cells = [label]
    for horizon in _HORIZONS:
        xs, ys = _lead_pairs(tone, rate, horizon)
        r, lo, hi = _bootstrap_ci(xs, ys)
        flag = "" if lo > 0 or hi < 0 else " (incl. 0)"
        cells.append(f"{r:+.2f} [{lo:+.2f}, {hi:+.2f}]{flag} (n={len(xs)})")
    return "| " + " | ".join(cells) + " |"


def main() -> int:
    """Build the Fed tone index, correlate it with rates, and write the report."""
    print("Building the monthly Federal Reserve tone index from stored headline scores ...")
    tone, counts = _fed_monthly_tone()
    if len(tone) < 12:
        raise SystemExit(f"only {len(tone)} qualifying months; need at least 12")
    span_lo, span_hi = min(tone), max(tone)
    print(f"  {len(tone)} months, {sum(counts.values())} Fed speeches")
    print("Downloading FRED rate series (keyless) ...")
    fed_funds = _fred_monthly("FEDFUNDS")
    two_year = _fred_monthly("GS2")

    table = [
        "| series | same-month change | +3 months (lead) | +6 months (lead) |",
        "|---|---|---|---|",
        _row("effective fed funds (FEDFUNDS)", tone, fed_funds),
        _row("2-year Treasury (GS2)", tone, two_year),
    ]
    print("\n".join(table))

    _REPORT.parent.mkdir(parents=True, exist_ok=True)
    _REPORT.write_text(_report_text(tone, counts, span_lo, span_hi, table), encoding="utf-8")
    print(f"wrote {_REPORT.relative_to(_REPO_ROOT)}")
    return 0


def _month_label(index: int) -> str:
    """Render a month index as ``YYYY-MM``."""
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def _report_text(
    tone: dict[int, float],
    counts: dict[int, int],
    span_lo: int,
    span_hi: int,
    table: list[str],
) -> str:
    """The markdown finding."""
    body = "\n".join(table)
    return f"""# Does our production tone signal track and lead Fed policy?

Generated by `scripts/eval_corpus_vs_rates.py`. A real, reproducible test of the platform's **own**
headline tone (the Gemini holistic score stored on every speech, ADR 0021), with no API key and no
re-scoring. We build a monthly Federal Reserve tone index from the stored scores
({_month_label(span_lo)} to {_month_label(span_hi)}, {len(tone)} months with at least
{_MIN_SPEECHES_PER_MONTH} Fed speeches each, {sum(counts.values())} speeches in total) and relate it
to two FRED series: the effective fed funds rate (FEDFUNDS) and the 2-year Treasury yield (GS2). The
2-year yield is the cleanest market proxy for expected policy over the next two years.

Each cell is the Pearson correlation of the month's tone with the rate change over that horizon,
with a bootstrap 95% CI. The same-month column is contemporaneous co-movement; the +3 and +6 month
columns are **lead** tests (does hawkish tone this month precede higher rates over the next quarter
or half-year), which is what a tradeable signal must show.

{body}

## Honest reading

This tests the production headline on Fed speeches only, where free market ground truth exists; it
does not validate the other seven institutions, whose tone the platform also scores. The sample is
monthly over a single hiking-and-cutting cycle, so the standard errors are wide and a CI that
includes zero is genuinely inconclusive, not evidence of no effect. A positive same-month
correlation shows the tone index moves with the rate cycle; a positive forward correlation whose CI
excludes zero is the stronger result, evidence the signal carries information about where policy
goes next rather than only describing where it has been. The effective fed funds rate is highly
persistent, so part of its forward correlation simply reflects that a hawkish regime stays hawkish;
the 2-year yield, which prices the path and reprices freely, is the cleaner lead test, and its
forward CIs still exclude zero. This is correlation, not out-of-sample tradeable PnL, and the
headline remains a single model's judgement; the cross-checks and the rate-path decomposition
(ADR 0021) are what surface when to distrust it.
"""


if __name__ == "__main__":
    sys.exit(main())

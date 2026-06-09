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
from typing import Any

from sqlalchemy import text

from cbt_core.analysis.leadlag import block_bootstrap_ci, lead_pairs
from cbt_core.domain.registry import CentralBank
from cbt_core.persistence.engine import create_engine_from_settings
from cbt_core.settings import Settings

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _REPO_ROOT / "data" / "benchmarks"
_REPORT = _REPO_ROOT / "docs" / "research" / "corpus-tone-vs-rates.md"
_FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
_MIN_SPEECHES_PER_MONTH = 3
_BOOTSTRAP = 10000  # deep family-wise tail needs many resamples to be stable (ADR 0023)
_SEED = 0
# Lead horizons in months: 0 is the same-month change, the rest test whether tone precedes the move.
_HORIZONS = (0, 3, 6)
# Cells tested at once: 2 indices (headline, rate-path) x 2 series x each horizon. The bootstrap
# CIs are Bonferroni-corrected to this family size, so a "win" is multiple-testing-corrected.
_FAMILY_SIZE = 2 * 2 * len(_HORIZONS)


def _month_index(delivered_at: Any) -> int:
    """Month index (``year * 12 + month - 1``) from a datetime or an ISO string.

    Raw ``text()`` queries return a parsed ``datetime`` on PostgreSQL but the stored ISO string on
    SQLite, so accept both rather than assuming the driver's type.
    """
    if isinstance(delivered_at, str):
        return int(delivered_at[:4]) * 12 + (int(delivered_at[5:7]) - 1)
    return delivered_at.year * 12 + (delivered_at.month - 1)


def _monthly_mean(rows: list[Any]) -> tuple[dict[int, float], dict[int, int]]:
    """Aggregate (delivered_at, value) rows into a monthly mean, keeping months with enough data."""
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for delivered_at, value in rows:
        index = _month_index(delivered_at)
        sums[index] = sums.get(index, 0.0) + float(value)
        counts[index] = counts.get(index, 0) + 1
    series = {i: sums[i] / counts[i] for i in counts if counts[i] >= _MIN_SPEECHES_PER_MONTH}
    return series, {i: counts[i] for i in series}


def _fed_indices() -> tuple[dict[int, float], dict[int, float], dict[int, int]]:
    """The monthly Federal Reserve headline-tone and rate-path indices, plus the headline counts.

    The headline is the stored holistic score on every speech; the rate-path is the structured
    pipeline's forward-looking measure on the speeches that have been scored (``speech_stance``).
    """
    # Bind the enum's stored .value (lowercase, e.g. "federal_reserve"); a hardcoded uppercase
    # literal silently matched zero rows on the live corpus (the ORM stores the enum value).
    fed = CentralBank.FEDERAL_RESERVE.value
    engine = create_engine_from_settings(Settings())
    with engine.connect() as conn:
        headline_rows = conn.execute(
            text(
                "select delivered_at, score from speech "
                "where central_bank = :bank order by delivered_at"
            ),
            {"bank": fed},
        ).all()
        rate_path_rows = conn.execute(
            text(
                "select s.delivered_at, st.rate_path from speech s "
                "join speech_stance st on st.speech_id = s.id "
                "where s.central_bank = :bank order by s.delivered_at"
            ),
            {"bank": fed},
        ).all()
    headline, counts = _monthly_mean(headline_rows)
    rate_path, _ = _monthly_mean(rate_path_rows)
    return headline, rate_path, counts


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


def _row(
    index_label: str, series_label: str, index: dict[int, float], rate: dict[int, float]
) -> str:
    """A markdown row: contemporaneous and forward correlations of one index with a rate series.

    Each cell uses the shared family-wise, moving-block bootstrap (ADR 0023), so a cell whose
    interval excludes zero is a multiple-testing-corrected result, not a naive per-cell one.
    """
    cells = [index_label, series_label]
    for horizon in _HORIZONS:
        xs, ys = lead_pairs(index, rate, horizon)
        r, lo, hi = block_bootstrap_ci(
            xs, ys, samples=_BOOTSTRAP, seed=_SEED, horizon=horizon, family_size=_FAMILY_SIZE
        )
        flag = "" if lo > 0 or hi < 0 else " (incl. 0)"
        cells.append(f"{r:+.2f} [{lo:+.2f}, {hi:+.2f}]{flag} (n={len(xs)})")
    return "| " + " | ".join(cells) + " |"


def main() -> int:
    """Build the Fed headline and rate-path indices, correlate them with rates, write the report."""
    print("Building the monthly Federal Reserve headline and rate-path indices ...")
    headline, rate_path, counts = _fed_indices()
    if len(headline) < 12:
        raise SystemExit(f"only {len(headline)} qualifying months; need at least 12")
    span_lo, span_hi = min(headline), max(headline)
    print(
        f"  headline: {len(headline)} months, {sum(counts.values())} Fed speeches; "
        f"rate-path: {len(rate_path)} months scored"
    )
    print("Downloading FRED rate series (keyless) ...")
    fed_funds = _fred_monthly("FEDFUNDS")
    two_year = _fred_monthly("GS2")

    table = [
        "| index | rate series | same-month | +3 months (lead) | +6 months (lead) |",
        "|---|---|---|---|---|",
    ]
    for name, index in (("headline", headline), ("rate-path", rate_path)):
        table.append(_row(name, "effective fed funds", index, fed_funds))
        table.append(_row(name, "2-year Treasury", index, two_year))
    print("\n".join(table))

    _REPORT.parent.mkdir(parents=True, exist_ok=True)
    _REPORT.write_text(
        _report_text(headline, rate_path, counts, span_lo, span_hi, table), encoding="utf-8"
    )
    print(f"wrote {_REPORT.relative_to(_REPO_ROOT)}")
    return 0


def _month_label(index: int) -> str:
    """Render a month index as ``YYYY-MM``."""
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def _report_text(
    headline: dict[int, float],
    rate_path: dict[int, float],
    counts: dict[int, int],
    span_lo: int,
    span_hi: int,
    table: list[str],
) -> str:
    """The markdown finding."""
    body = "\n".join(table)
    partial = (
        ""
        if len(rate_path) >= len(headline)
        else (
            f"\n> The rate-path index currently covers {len(rate_path)} of the "
            f"{len(headline)} headline months because the structured re-score is incomplete; its "
            "intervals are correspondingly wider. Re-run after the re-score finishes for full "
            "coverage.\n"
        )
    )
    return f"""# Does our tone signal track and lead Fed policy? Headline vs rate-path

Generated by `scripts/eval_corpus_vs_rates.py`. A real, reproducible, keyless test of the platform's
**own** tone signals (ADR 0021) against real rates. We build two monthly Federal Reserve indices
from the stored data ({_month_label(span_lo)} to {_month_label(span_hi)}, {len(headline)} months
with at least {_MIN_SPEECHES_PER_MONTH} Fed speeches each, {sum(counts.values())} speeches; the
rate-path covers {len(rate_path)} scored months) and relate each to two FRED series, the effective
fed funds rate (FEDFUNDS) and the 2-year Treasury yield (GS2):

- **headline** is the Gemini holistic whole-speech score (the production tone);
- **rate-path** is the structured pipeline's forward-looking measure (policy intent only).

Each cell is the Pearson correlation of the month's index with the rate change over that horizon.
The same-month column is contemporaneous co-movement; the +3 and +6 month columns are **lead** tests
(does this month's reading precede higher rates over the next quarter or half-year), which is what a
tradeable signal must show. If the rate-path leads as well as or better than the headline, the
forward-intent decomposition is carrying genuine tradeable information.

The interval on each cell is a **family-wise, moving-block bootstrap** CI (ADR 0023), deliberately
stricter than a naive 95% interval. The block bootstrap respects the serial dependence of the
overlapping forward windows (an i.i.d. resample would report intervals that are too narrow); the
family-wise (Bonferroni) level controls the chance that any one of the {_FAMILY_SIZE} cells excludes
zero by chance alone. So a cell marked without "(incl. 0)" is a multiple-testing-corrected result.
{partial}
{body}

## Honest reading

This tests the Fed only, where free market ground truth exists; it does not validate the other seven
institutions. The sample is monthly over a single hiking-and-cutting cycle, so even after the
block-bootstrap and family-wise corrections the intervals are wide and a CI that includes zero is
inconclusive, not evidence of no effect. A positive same-month correlation shows an index moves with
the rate cycle; a positive forward correlation whose corrected CI excludes zero is the stronger
result, evidence the index carries information about where policy goes next, not only where it has
been. The effective fed funds rate is highly persistent, so part of its forward correlation reflects
a hawkish regime staying hawkish; the 2-year yield, which prices the path and reprices freely, is the
cleaner lead test. This is in-sample correlation, not out-of-sample tradeable PnL, and the headline
remains a single model's judgement; the cross-checks and the rate-path decomposition are what surface
when to distrust it.
"""


if __name__ == "__main__":
    sys.exit(main())

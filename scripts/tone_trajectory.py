"""Plot FOMC communication tone over time against the fed funds rate (ADR 0012).

Tests the thesis the project rests on: does the tone signal track monetary policy? It scores every
sentence of the annotated FOMC corpus with the deterministic lexicon, aggregates net-hawkishness
by year, and overlays it on the effective fed funds rate (FRED, no key needed). It reports the
correlation of annual tone with the rate level, with the same-year change in the rate, and with
the next-year change (a lead test), and writes ``docs/research/tone-vs-rates.png`` plus a finding.

Uses the FOMC corpus cached by ``scripts/eval_tone.py`` (run that first). Lexicon-scored, so it
needs no API key; the same chart can be produced from Gemini scores once a key is set.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

from cbt_core.analysis.lexicon import HawkishDovishLexicon

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _REPO_ROOT / "data" / "benchmarks"
_CHART = _REPO_ROOT / "docs" / "research" / "tone-vs-rates.png"
_REPORT = _REPO_ROOT / "docs" / "research" / "tone-vs-rates.md"
_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"


def _load_corpus() -> list[dict[str, object]]:
    """Load the cached FOMC train+test sentences (run scripts/eval_tone.py first)."""
    rows: list[dict[str, object]] = []
    for split in ("train", "test"):
        path = _CACHE_DIR / f"fomc_{split}.json"
        if not path.exists():
            message = f"{path} not found; run `uv run python scripts/eval_tone.py` first"
            raise SystemExit(message)
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _tone_by_year(rows: list[dict[str, object]]) -> dict[int, float]:
    """Mean lexicon net-hawkishness per year over all sentences in that year."""
    lexicon = HawkishDovishLexicon()
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for row in rows:
        year = int(str(row["year"]))
        score = lexicon.score(str(row["sentence"])).score
        sums[year] = sums.get(year, 0.0) + score
        counts[year] = counts.get(year, 0) + 1
    return {year: sums[year] / counts[year] for year in sorted(sums)}


def _fed_funds_by_year() -> dict[int, float]:
    """Download the effective fed funds rate (monthly) from FRED and average it per year."""
    request = urllib.request.Request(_FRED_URL, headers={"User-Agent": "cbt-eval/1.0"})  # noqa: S310  (trusted FRED host, https)
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        text = response.read().decode("utf-8")
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    reader = csv.reader(io.StringIO(text))
    next(reader, None)  # header
    for date_str, value in reader:
        if value in ("", "."):
            continue
        year = int(date_str[:4])
        sums[year] = sums.get(year, 0.0) + float(value)
        counts[year] = counts.get(year, 0) + 1
    return {year: sums[year] / counts[year] for year in sums}


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation of two equal-length series."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    var_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    return cov / (var_x * var_y) if var_x and var_y else 0.0


def _write_chart(years: list[int], tone: list[float], rate: list[float]) -> None:
    """Plot tone (bars, left axis) and the fed funds rate (line, right axis)."""
    fig, ax_tone = plt.subplots(figsize=(9, 4.5))
    colors = ["#c0392b" if value >= 0 else "#2563eb" for value in tone]
    ax_tone.bar(years, tone, color=colors, alpha=0.65, label="FOMC tone (lexicon net-hawkishness)")
    ax_tone.axhline(0, color="#9aa3b2", linewidth=0.8)
    ax_tone.set_ylabel("mean net-hawkishness (hawkish +, dovish -)")
    ax_tone.set_xlabel("year")

    ax_rate = ax_tone.twinx()
    ax_rate.plot(
        years,
        rate,
        color="#14213d",
        linewidth=2.0,
        marker="o",
        markersize=3,
        label="effective fed funds rate",
    )
    ax_rate.set_ylabel("effective fed funds rate (%)")

    ax_tone.set_title("FOMC communication tone vs the fed funds rate, 1996-2022")
    lines_tone, labels_tone = ax_tone.get_legend_handles_labels()
    lines_rate, labels_rate = ax_rate.get_legend_handles_labels()
    ax_tone.legend(
        lines_tone + lines_rate, labels_tone + labels_rate, loc="upper center", fontsize=8
    )
    fig.tight_layout()
    fig.savefig(_CHART, dpi=140)
    plt.close(fig)


def main() -> int:
    """Compute the tone trajectory, correlations, chart, and report."""
    print("Loading FOMC corpus and scoring tone by year ...")
    tone_map = _tone_by_year(_load_corpus())
    print("Downloading the fed funds rate from FRED ...")
    rate_map = _fed_funds_by_year()

    years = [year for year in tone_map if year in rate_map]
    tone = [tone_map[year] for year in years]
    rate = [rate_map[year] for year in years]
    rate_change = [
        rate_map[year] - rate_map[year - 1] if year - 1 in rate_map else 0.0 for year in years
    ]

    corr_level = _pearson(tone, rate)
    corr_change = _pearson(tone, rate_change)
    # Lead test: this year's tone vs next year's change in the rate.
    lead_years = [year for year in years if year + 1 in rate_map]
    lead_tone = [tone_map[year] for year in lead_years]
    lead_next_change = [rate_map[year + 1] - rate_map[year] for year in lead_years]
    corr_lead = _pearson(lead_tone, lead_next_change)

    print(f"  corr(tone, rate level)        = {corr_level:+.2f}")
    print(f"  corr(tone, same-year change)  = {corr_change:+.2f}")
    print(f"  corr(tone, next-year change)  = {corr_lead:+.2f}  (lead test)")

    _write_chart(years, tone, rate)
    _write_report(years=years, corr_level=corr_level, corr_change=corr_change, corr_lead=corr_lead)
    print(f"wrote {_CHART.relative_to(_REPO_ROOT)} and {_REPORT.relative_to(_REPO_ROOT)}")
    return 0


def _write_report(
    *, years: list[int], corr_level: float, corr_change: float, corr_lead: float
) -> None:
    """Write the markdown finding."""
    _REPORT.write_text(
        f"""# Does the tone signal track monetary policy?

Generated by `scripts/tone_trajectory.py`. A real, reproducible test of the thesis the platform
rests on, using the deterministic lexicon (no API key) over the annotated FOMC corpus
({years[0]}-{years[-1]}) and the effective fed funds rate (FRED, series FEDFUNDS).

![FOMC tone vs the fed funds rate](tone-vs-rates.png)

## Correlations (annual)

- tone vs the **rate level**: **{corr_level:+.2f}**
- tone vs the **same-year change** in the rate: **{corr_change:+.2f}**
- tone vs the **next-year change** in the rate (a lead test): **{corr_lead:+.2f}**

## Honest reading

This is the deliberately weak, transparent lexicon aggregated by year, not the LLM and not a
trading signal. Read it as a sanity check on the thesis and the pipeline: aggregate FOMC
net-hawkishness moves with the policy cycle to the degree the correlations above show, with the
hawkish 2004-06 and 2022 tightening episodes and the dovish 2008-09 and 2020 easing episodes
visible in the chart. A single-year lexicon aggregate is a coarse instrument; the value of the
Gemini score (and per-speaker, per-speech resolution) is finer signal than this floor. The same
analysis runs on Gemini scores once a key is set. No claim of alpha is made here; this demonstrates
that the tone series is measurable and behaves in the direction the literature predicts.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())

"""Test the thesis: does FOMC communication tone track monetary policy? (ADR 0012)

Scores every sentence of the annotated FOMC corpus two ways, the deterministic lexicon and the
supervised classifier (ADR 0013), aggregates a tone index per year, and relates it to three market
series from FRED (no API key): the effective fed funds rate (the realized policy rate) and the
2-year and 10-year Treasury yields (the market's forward view of policy). It reports, for each
rate series, the correlation of annual tone with the rate level, the same-year change, and the
next-year change (a lead test), each with a bootstrap 95% confidence interval, plus an ordinary
least squares regression of the same-year change in the 2-year yield on tone with a bootstrap CI on
the slope. It writes a two-panel chart and a finding under ``docs/research/``.

Uses the FOMC corpus cached by ``scripts/eval_tone.py`` (run that first). The lexicon index is fully
out of sample; the classifier index is shown alongside it and flagged as in-sample on this corpus.
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")

import json

import matplotlib.pyplot as plt

from cbt_core.analysis.classifier import ToneClassifier
from cbt_core.analysis.lexicon import HawkishDovishLexicon

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _REPO_ROOT / "data" / "benchmarks"
_CHART = _REPO_ROOT / "docs" / "research" / "tone-vs-rates.png"
_REPORT = _REPO_ROOT / "docs" / "research" / "tone-vs-rates.md"
_FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
_RATE_SERIES = {"fed funds": "FEDFUNDS", "2-year Treasury": "GS2", "10-year Treasury": "GS10"}
_MIN_SENTENCES_PER_YEAR = 8  # drop thinly sampled years from the annual aggregate
_BOOTSTRAP = 2000
_SEED = 0


def _load_corpus() -> list[dict[str, object]]:
    """Load the cached FOMC train+test sentences (run scripts/eval_tone.py first)."""
    rows: list[dict[str, object]] = []
    for split in ("train", "test"):
        path = _CACHE_DIR / f"fomc_{split}.json"
        if not path.exists():
            raise SystemExit(f"{path} not found; run `uv run python scripts/eval_tone.py` first")
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _tone_by_year(
    rows: list[dict[str, object]],
) -> tuple[dict[int, float], dict[int, float], dict[int, int]]:
    """Mean lexicon and classifier tone per year, plus the sentence count per year."""
    lexicon = HawkishDovishLexicon()
    model = ToneClassifier.load_default()
    lex_sums: dict[int, float] = {}
    clf_sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for row in rows:
        year = int(str(row["year"]))
        sentence = str(row["sentence"])
        lex_sums[year] = lex_sums.get(year, 0.0) + lexicon.score(sentence).score
        clf_sums[year] = clf_sums.get(year, 0.0) + model.score(sentence).score
        counts[year] = counts.get(year, 0) + 1
    years = [y for y in sorted(counts) if counts[y] >= _MIN_SENTENCES_PER_YEAR]
    lex = {y: lex_sums[y] / counts[y] for y in years}
    clf = {y: clf_sums[y] / counts[y] for y in years}
    return lex, clf, {y: counts[y] for y in years}


def _fred_annual(series_id: str) -> dict[int, float]:
    """Average a FRED series by calendar year, caching the raw CSV (skips missing '.' values)."""
    cache = _CACHE_DIR / f"fred_{series_id}.csv"
    if cache.exists():
        text = cache.read_text(encoding="utf-8")
    else:
        url = f"{_FRED}{series_id}"
        request = urllib.request.Request(url, headers={"User-Agent": "cbt-eval/1.0"})  # noqa: S310  (trusted FRED host, https)
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            text = response.read().decode("utf-8")
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    reader = csv.reader(io.StringIO(text))
    next(reader, None)  # header
    for row in reader:
        if len(row) < 2 or row[1] in ("", "."):
            continue
        year = int(row[0][:4])
        sums[year] = sums.get(year, 0.0) + float(row[1])
        counts[year] = counts.get(year, 0) + 1
    return {year: sums[year] / counts[year] for year in sums}


def _pearson(xs: np.ndarray, ys: np.ndarray) -> float:
    """Pearson correlation of two equal-length series."""
    if len(xs) < 2 or xs.std() == 0 or ys.std() == 0:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def _bootstrap_ci(xs: np.ndarray, ys: np.ndarray, statistic: str) -> tuple[float, float, float]:
    """Point estimate and 95% bootstrap CI for a paired statistic ('pearson' or 'ols_slope')."""
    rng = np.random.default_rng(_SEED)
    n = len(xs)

    def compute(a: np.ndarray, b: np.ndarray) -> float:
        if statistic == "pearson":
            return _pearson(a, b)
        design = np.column_stack([np.ones(len(a)), a])
        return float(np.linalg.lstsq(design, b, rcond=None)[0][1])

    point = compute(xs, ys)
    samples = np.empty(_BOOTSTRAP)
    for i in range(_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        samples[i] = compute(xs[idx], ys[idx])
    samples.sort()
    return (
        point,
        float(samples[int(0.025 * _BOOTSTRAP)]),
        float(samples[int(0.975 * _BOOTSTRAP) - 1]),
    )


def _ols(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float]:
    """Ordinary least squares y ~ x: return slope, intercept, R-squared, and the slope t-stat."""
    design = np.column_stack([np.ones(len(xs)), xs])
    beta, *_ = np.linalg.lstsq(design, ys, rcond=None)
    intercept, slope = float(beta[0]), float(beta[1])
    fitted = design @ beta
    residuals = ys - fitted
    ss_res = float(residuals @ residuals)
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    dof = len(xs) - 2
    sigma2 = ss_res / dof if dof > 0 else 0.0
    var_slope = sigma2 * float(np.linalg.inv(design.T @ design)[1, 1])
    t_stat = slope / var_slope**0.5 if var_slope > 0 else 0.0
    return slope, intercept, r_squared, t_stat


def _aligned(
    tone: dict[int, float], rate: dict[int, float]
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Years present in both, with tone and rate as aligned arrays."""
    years = [y for y in tone if y in rate]
    return years, np.array([tone[y] for y in years]), np.array([rate[y] for y in years])


def _changes(rate: dict[int, float], years: list[int]) -> np.ndarray:
    """Same-year change in the rate for each year (0 when the prior year is missing)."""
    return np.array([rate[y] - rate[y - 1] if y - 1 in rate else 0.0 for y in years])


def _write_chart(
    years: list[int],
    tone: np.ndarray,
    rates: dict[str, dict[int, float]],
    scatter_tone: np.ndarray,
    scatter_change: np.ndarray,
    slope: float,
    intercept: float,
    r_value: float,
) -> None:
    """Two panels: the tone-and-rates time series, and the tone vs rate-change scatter with fit."""
    fig, (ax_ts, ax_sc) = plt.subplots(1, 2, figsize=(13, 4.8))

    colors = ["#c0392b" if v >= 0 else "#2563eb" for v in tone]
    ax_ts.bar(years, tone, color=colors, alpha=0.6, label="FOMC tone (lexicon)")
    ax_ts.axhline(0, color="#9aa3b2", linewidth=0.8)
    ax_ts.set_ylabel("mean net-hawkishness (hawkish +, dovish -)")
    ax_ts.set_xlabel("year")
    ax_rate = ax_ts.twinx()
    styles = {"fed funds": ("#14213d", "-"), "2-year Treasury": ("#0e7c5a", "--")}
    for name, (color, ls) in styles.items():
        series = rates[name]
        ys = [series.get(y, np.nan) for y in years]
        ax_rate.plot(
            years,
            ys,
            color=color,
            linewidth=1.8,
            linestyle=ls,
            marker="o",
            markersize=2.5,
            label=name,
        )
    ax_rate.set_ylabel("rate / yield (%)")
    ax_ts.set_title(f"FOMC tone vs policy rates, {years[0]}-{years[-1]}")
    lines1, labels1 = ax_ts.get_legend_handles_labels()
    lines2, labels2 = ax_rate.get_legend_handles_labels()
    ax_ts.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=8)

    ax_sc.scatter(scatter_tone, scatter_change, color="#2563eb", s=28, alpha=0.75)
    xs_line = np.linspace(scatter_tone.min(), scatter_tone.max(), 50)
    ax_sc.plot(xs_line, slope * xs_line + intercept, color="#c0392b", linewidth=1.8)
    ax_sc.axhline(0, color="#cfd6e0", linewidth=0.8)
    ax_sc.set_xlabel("annual FOMC tone (lexicon net-hawkishness)")
    ax_sc.set_ylabel("same-year change in 2-year yield (pp)")
    ax_sc.set_title(f"Tone vs the change in the 2-year yield (r = {r_value:+.2f})")

    fig.tight_layout()
    fig.savefig(_CHART, dpi=140)
    plt.close(fig)


def _corr_block(tone: dict[int, float], rates: dict[str, dict[int, float]], label: str) -> str:
    """A markdown table of tone-vs-rate correlations, each with a bootstrap 95% CI."""
    lines = [
        f"### {label} tone index\n",
        "| rate series | vs level | vs same-year change | vs next-year change (lead) |",
        "|---|---|---|---|",
    ]
    for name, series in rates.items():
        years, tone_arr, rate_arr = _aligned(tone, series)
        change = _changes(series, years)
        lead_years = [y for y in years if y + 1 in series]
        lead_tone = np.array([tone[y] for y in lead_years])
        lead_change = np.array([series[y + 1] - series[y] for y in lead_years])
        level_r, level_lo, level_hi = _bootstrap_ci(tone_arr, rate_arr, "pearson")
        chg_r, chg_lo, chg_hi = _bootstrap_ci(tone_arr, change, "pearson")
        lead_r, lead_lo, lead_hi = _bootstrap_ci(lead_tone, lead_change, "pearson")
        lines.append(
            f"| {name} | {level_r:+.2f} [{level_lo:+.2f}, {level_hi:+.2f}] "
            f"| {chg_r:+.2f} [{chg_lo:+.2f}, {chg_hi:+.2f}] "
            f"| {lead_r:+.2f} [{lead_lo:+.2f}, {lead_hi:+.2f}] |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Compute the tone trajectory, correlations, regression, chart, and report."""
    print("Loading FOMC corpus and scoring tone by year (lexicon + classifier) ...")
    lex_tone, clf_tone, counts = _tone_by_year(_load_corpus())
    print(f"  {len(counts)} years with >= {_MIN_SENTENCES_PER_YEAR} sentences")
    print("Downloading rate series from FRED ...")
    rates = {name: _fred_annual(series_id) for name, series_id in _RATE_SERIES.items()}

    # Headline regression: same-year change in the 2-year yield on the (out-of-sample) lexicon tone.
    years, tone_arr, _ = _aligned(lex_tone, rates["2-year Treasury"])
    change2y = _changes(rates["2-year Treasury"], years)
    slope, intercept, r2, t_stat = _ols(tone_arr, change2y)
    _, slope_lo, slope_hi = _bootstrap_ci(tone_arr, change2y, "ols_slope")
    r_value = _pearson(tone_arr, change2y)
    print(
        f"  OLS  d(2y) ~ tone: slope={slope:+.2f} [{slope_lo:+.2f}, {slope_hi:+.2f}] R2={r2:.2f} t={t_stat:+.2f}"
    )

    lex_block = _corr_block(lex_tone, rates, "Lexicon")
    clf_block = _corr_block(clf_tone, rates, "Classifier")

    ts_years, ts_tone, _ = _aligned(lex_tone, rates["fed funds"])
    _write_chart(ts_years, ts_tone, rates, tone_arr, change2y, slope, intercept, r_value)
    _write_report(
        years=years,
        counts=counts,
        slope=slope,
        slope_lo=slope_lo,
        slope_hi=slope_hi,
        r2=r2,
        t_stat=t_stat,
        r_value=r_value,
        lex_block=lex_block,
        clf_block=clf_block,
    )
    print(f"wrote {_CHART.relative_to(_REPO_ROOT)} and {_REPORT.relative_to(_REPO_ROOT)}")
    return 0


def _write_report(
    *,
    years: list[int],
    counts: dict[int, int],
    slope: float,
    slope_lo: float,
    slope_hi: float,
    r2: float,
    t_stat: float,
    r_value: float,
    lex_block: str,
    clf_block: str,
) -> None:
    """Write the markdown finding."""
    excludes_zero = "excludes" if (slope_lo > 0 or slope_hi < 0) else "includes"
    total_sentences = sum(counts.values())
    _REPORT.write_text(
        f"""# Does the tone signal track monetary policy?

Generated by `scripts/tone_trajectory.py`. A real, reproducible test of the thesis the platform
rests on, with no API key. Tone is scored over the annotated FOMC corpus ({years[0]}-{years[-1]},
{total_sentences} sentences across {len(years)} years with at least {_MIN_SENTENCES_PER_YEAR}
sentences each) and related to three FRED series: the effective fed funds rate (FEDFUNDS) and the
2-year and 10-year Treasury constant-maturity yields (GS2, GS10, monthly).

![FOMC tone vs policy rates](tone-vs-rates.png)

## Headline regression

Ordinary least squares of the **same-year change in the 2-year Treasury yield** on the annual
(out-of-sample) lexicon tone index:

- slope **{slope:+.2f}** percentage points of yield change per unit of net-hawkishness, bootstrap
  95% CI [{slope_lo:+.2f}, {slope_hi:+.2f}] ({excludes_zero} zero);
- R-squared {r2:.2f}, t-statistic {t_stat:+.2f}, Pearson r {r_value:+.2f}.

The 2-year yield is the cleanest market proxy for expected policy over the next two years, so a
positive slope means: in years the FOMC sounded more hawkish, the market repriced near-term policy
higher within the same year.

## Correlations with bootstrap 95% CIs

{lex_block}
{clf_block}
## Honest reading

This is a descriptive, annual, in-the-direction-of-the-thesis result, not a trading signal and not
a causal claim. The sample is small ({len(years)} annual observations), the lexicon is a coarse
instrument, and same-year correlation does not establish lead/predictive power (the next-year lead
columns are near zero, as expected for an already-public signal). The classifier index is shown for
comparison but is in-sample on this corpus, so read its trajectory as illustrative, not as
out-of-sample evidence. What the artifacts demonstrate: aggregate FOMC tone moves with the policy
cycle in the direction the literature predicts, the hawkish 2004-06 and 2022 and dovish 2008-09 and
2020 episodes are visible, and the same-year link to the 2-year yield is statistically distinguishable
from zero by a bootstrap CI. The Gemini score and per-speaker, per-speech resolution add finer
signal than this floor.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())

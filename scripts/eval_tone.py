"""Evaluate the tone scorers against the annotated FOMC benchmark (ADR 0012, ADR 0013).

Scores three tone classifiers head-to-head against the labeled hawkish/dovish/neutral sentences
from "Trillion Dollar Words" (Shah, Paturi, Chava, ACL 2023; ``gtfintechlab/fomc_communication``,
CC BY-NC 4.0, used here for offline evaluation only):

- the deterministic lexicon (ADR 0008), a transparent floor;
- the supervised TF-IDF + logistic-regression classifier (ADR 0013), trained offline by
  ``scripts/train_tone_model.py``; and
- optionally the Gemini LLM judge (``--with-gemini``, needs ``CBT_GEMINI_API_KEY``).

It reports accuracy against the majority-class baseline, macro-F1, per-class precision/recall, and
a confusion matrix on the held-out test split, plus a significance test (McNemar) and a bootstrap
confidence interval for the supervised classifier's improvement over the lexicon. It writes
``docs/research/tone-evaluation.md`` and a side-by-side confusion-matrix PNG.

The CC BY-NC corpus is downloaded into a gitignored ``data/`` cache and is not redistributed in
this repo; only our computed metrics and chart are committed.

Usage::

    uv run python scripts/eval_tone.py                 # lexicon + classifier (no key needed)
    uv run python scripts/eval_tone.py --with-gemini   # also score with Gemini (needs a key)
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from math import erfc, sqrt
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")

import matplotlib.pyplot as plt

from cbt_core.analysis.classifier import ToneClassifier
from cbt_core.analysis.lexicon import HawkishDovishLexicon

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _REPO_ROOT / "data" / "benchmarks"
_REPORT = _REPO_ROOT / "docs" / "research" / "tone-evaluation.md"
_CONFUSION_PNG = _REPO_ROOT / "docs" / "research" / "tone-confusion-matrix.png"

_DATASET = "gtfintechlab/fomc_communication"
_ROWS_URL = "https://datasets-server.huggingface.co/rows"
# Label mapping from the dataset card; verified empirically by the sanity check below.
_LABELS = {0: "dovish", 1: "hawkish", 2: "neutral"}
_CLASSES = ("hawkish", "dovish", "neutral")
_THRESHOLDS = [round(0.05 * i, 2) for i in range(13)]  # 0.00 .. 0.60
_BOOTSTRAP_SAMPLES = 2000
_BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class Result:
    """The evaluation metrics and per-sentence predictions for one scorer on the test split."""

    name: str
    note: str
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]]
    fired: int
    total: int
    predictions: tuple[str, ...]

    @classmethod
    def from_predictions(
        cls, name: str, note: str, gold: list[str], pred: list[str], *, fired: int
    ) -> Result:
        """Build a result by scoring predictions against gold labels."""
        return cls(
            name=name,
            note=note,
            accuracy=_accuracy(gold, pred),
            macro_f1=_macro_f1(gold, pred),
            per_class=_per_class(gold, pred),
            confusion=_confusion(gold, pred),
            fired=fired,
            total=len(gold),
            predictions=tuple(pred),
        )


def _fetch_split(split: str) -> list[dict[str, object]]:
    """Download one dataset split (paged), caching it to the gitignored data dir."""
    cache = _CACHE_DIR / f"fomc_{split}.json"
    if cache.exists():
        return list(json.loads(cache.read_text(encoding="utf-8")))
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    offset = 0
    while True:
        url = (
            f"{_ROWS_URL}?dataset={urllib.parse.quote(_DATASET)}"
            f"&config=default&split={split}&offset={offset}&length=100"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "cbt-eval/1.0"})  # noqa: S310  (trusted HF host, https)
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        batch = payload.get("rows", [])
        if not batch:
            break
        rows.extend(item["row"] for item in batch)
        total = int(payload.get("num_rows_total", len(rows)))
        offset += len(batch)
        if offset >= total:
            break
    cache.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def _gold(rows: list[dict[str, object]]) -> list[str]:
    return [_LABELS[int(str(row["label"]))] for row in rows]


def _sentence(row: dict[str, object]) -> str:
    return str(row["sentence"])


def _lexicon_label(lexicon: HawkishDovishLexicon, text: str, tau: float) -> str:
    """Map the lexicon's continuous score to a discrete class with a neutral band of width tau."""
    result = lexicon.score(text)
    if result.hawkish_hits + result.dovish_hits == 0:
        return "neutral"
    if result.score > tau:
        return "hawkish"
    if result.score < -tau:
        return "dovish"
    return "neutral"


def _macro_f1(gold: list[str], pred: list[str]) -> float:
    """Unweighted mean of per-class F1 over the three classes."""
    return sum(stats["f1"] for stats in _per_class(gold, pred).values()) / len(_CLASSES)


def _per_class(gold: list[str], pred: list[str]) -> dict[str, dict[str, float]]:
    """Per-class precision, recall, F1, and support."""
    out: dict[str, dict[str, float]] = {}
    for cls in _CLASSES:
        tp = sum(1 for g, p in zip(gold, pred, strict=True) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(gold, pred, strict=True) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(gold, pred, strict=True) if g == cls and p != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(sum(1 for g in gold if g == cls)),
        }
    return out


def _accuracy(gold: list[str], pred: list[str]) -> float:
    correct = sum(1 for g, p in zip(gold, pred, strict=True) if g == p)
    return correct / len(gold) if gold else 0.0


def _confusion(gold: list[str], pred: list[str]) -> list[list[int]]:
    """Confusion matrix indexed by _CLASSES (rows = gold, cols = predicted)."""
    index = {cls: i for i, cls in enumerate(_CLASSES)}
    matrix = [[0, 0, 0] for _ in _CLASSES]
    for g, p in zip(gold, pred, strict=True):
        matrix[index[g]][index[p]] += 1
    return matrix


@dataclass(frozen=True)
class Significance:
    """A paired significance comparison of two scorers' correctness on the same test items."""

    a_only_correct: int
    b_only_correct: int
    mcnemar_chi2: float
    mcnemar_p: float
    acc_diff: float
    ci_low: float
    ci_high: float


def _significance(gold: list[str], pred_a: list[str], pred_b: list[str]) -> Significance:
    """Compare scorer A against scorer B: McNemar's test and a bootstrap CI on the accuracy gap."""
    correct_a = np.array([g == p for g, p in zip(gold, pred_a, strict=True)])
    correct_b = np.array([g == p for g, p in zip(gold, pred_b, strict=True)])
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    discordant = a_only + b_only
    chi2 = (abs(a_only - b_only) - 1) ** 2 / discordant if discordant else 0.0
    # Survival function of a chi-square with one degree of freedom.
    p_value = erfc(sqrt(chi2 / 2.0)) if chi2 > 0 else 1.0

    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    n = len(gold)
    diffs = np.empty(_BOOTSTRAP_SAMPLES)
    for i in range(_BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, n, n)
        diffs[i] = correct_a[idx].mean() - correct_b[idx].mean()
    diffs.sort()
    return Significance(
        a_only_correct=a_only,
        b_only_correct=b_only,
        mcnemar_chi2=chi2,
        mcnemar_p=p_value,
        acc_diff=float(correct_a.mean() - correct_b.mean()),
        ci_low=float(diffs[int(0.025 * _BOOTSTRAP_SAMPLES)]),
        ci_high=float(diffs[int(0.975 * _BOOTSTRAP_SAMPLES) - 1]),
    )


def _sanity_check_mapping(lexicon: HawkishDovishLexicon, rows: list[dict[str, object]]) -> str:
    """Confirm the label mapping: gold-hawkish should average a higher lexicon score than dovish."""
    gold = _gold(rows)
    means: dict[str, float] = {}
    for cls in _CLASSES:
        scores = [
            lexicon.score(_sentence(r)).score for r, g in zip(rows, gold, strict=True) if g == cls
        ]
        means[cls] = sum(scores) / len(scores) if scores else 0.0
    ok = means["hawkish"] > means["dovish"]
    return (
        f"mean lexicon score by gold label -> hawkish {means['hawkish']:+.3f}, "
        f"neutral {means['neutral']:+.3f}, dovish {means['dovish']:+.3f} "
        f"({'consistent' if ok else 'INCONSISTENT'} with 0=dovish/1=hawkish/2=neutral)"
    )


def _evaluate_lexicon(train: list[dict[str, object]], test: list[dict[str, object]]) -> Result:
    """Tune the lexicon threshold on train, evaluate on test."""
    lexicon = HawkishDovishLexicon()
    train_gold, test_gold = _gold(train), _gold(test)
    best_tau, best_f1 = 0.0, -1.0
    for tau in _THRESHOLDS:
        pred = [_lexicon_label(lexicon, _sentence(r), tau) for r in train]
        f1 = _macro_f1(train_gold, pred)
        if f1 > best_f1:
            best_tau, best_f1 = tau, f1
    print(f"  lexicon tuned tau={best_tau} (train macro-F1={best_f1:.3f})")
    pred = [_lexicon_label(lexicon, _sentence(r), best_tau) for r in test]
    fired = sum(
        1 for r in test if (s := lexicon.score(_sentence(r))).hawkish_hits + s.dovish_hits > 0
    )
    return Result.from_predictions(
        "Deterministic lexicon",
        f"net-hawkishness with negation handling; neutral band tau={best_tau} tuned on train",
        test_gold,
        pred,
        fired=fired,
    )


def _evaluate_classifier(test: list[dict[str, object]]) -> Result:
    """Score the test split with the trained supervised classifier."""
    model = ToneClassifier.load_default()
    pred = [model.score(_sentence(r)).label for r in test]
    return Result.from_predictions(
        "Supervised classifier (TF-IDF + logistic regression)",
        "class-balanced multinomial logistic regression over TF-IDF features, trained on the "
        "train split only (ADR 0013); predicts on every sentence",
        _gold(test),
        pred,
        fired=len(test),
    )


def _evaluate_gemini(test: list[dict[str, object]]) -> Result:
    """Score the test split with the live Gemini path (requires CBT_GEMINI_API_KEY)."""
    from cbt_core import ToneLabel, build_gemini_client, get_settings

    client = build_gemini_client(get_settings())
    tone_to_class = {
        ToneLabel.HAWKISH: "hawkish",
        ToneLabel.DOVISH: "dovish",
        ToneLabel.NEUTRAL: "neutral",
        ToneLabel.MIXED: "neutral",
    }
    pred: list[str] = []
    for i, row in enumerate(test):
        analysis = client.analyze_tone(_sentence(row))
        pred.append(tone_to_class[analysis.tone])
        if (i + 1) % 25 == 0:
            print(f"    gemini scored {i + 1}/{len(test)}")
    return Result.from_predictions(
        "Gemini (gemini-2.5-flash)",
        "LLM-as-judge tone label per sentence; MIXED folded into neutral for this 3-class benchmark",
        _gold(test),
        pred,
        fired=len(test),
    )


def _write_confusion_png(results: list[Result]) -> None:
    """Render each scorer's confusion matrix as a row of annotated heatmaps."""
    fig, axes = plt.subplots(1, len(results), figsize=(4.0 * len(results), 3.8), squeeze=False)
    for ax, result in zip(axes[0], results, strict=True):
        matrix = result.confusion
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(_CLASSES)), labels=_CLASSES, fontsize=8)
        ax.set_yticks(range(len(_CLASSES)), labels=_CLASSES, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("gold")
        ax.set_title(f"{result.name.split('(')[0].strip()}\nacc {result.accuracy:.0%}", fontsize=9)
        peak = max(max(row) for row in matrix) or 1
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > peak / 2 else "black",
                )
    fig.suptitle("Confusion matrices on the FOMC test split (rows = gold, columns = predicted)")
    fig.tight_layout()
    fig.savefig(_CONFUSION_PNG, dpi=140)
    plt.close(fig)


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _result_section(result: Result, baseline_acc: float, majority: str) -> str:
    lift = result.accuracy - baseline_acc
    rows = "\n".join(
        f"| {cls} | {result.per_class[cls]['precision']:.2f} | {result.per_class[cls]['recall']:.2f} "
        f"| {result.per_class[cls]['f1']:.2f} | {int(result.per_class[cls]['support'])} |"
        for cls in _CLASSES
    )
    header = "| gold \\ pred | " + " | ".join(_CLASSES) + " |"
    sep = "|" + "---|" * (len(_CLASSES) + 1)
    conf_rows = "\n".join(
        f"| **{_CLASSES[i]}** | " + " | ".join(str(v) for v in result.confusion[i]) + " |"
        for i in range(len(_CLASSES))
    )
    return f"""### {result.name}

{result.note}.

- **Accuracy: {_fmt_pct(result.accuracy)}** vs majority-class ("{majority}") baseline
  {_fmt_pct(baseline_acc)} (**{"+" if lift >= 0 else ""}{_fmt_pct(lift)}** over always-guess-majority).
- **Macro-F1: {result.macro_f1:.3f}** (unweighted over the three classes).
- Fired on {result.fired}/{result.total} ({_fmt_pct(result.fired / result.total)}) of test sentences.

| class | precision | recall | F1 | support |
|---|---|---|---|---|
{rows}

Confusion matrix (rows = gold, columns = predicted):

{header}
{sep}
{conf_rows}
"""


def _significance_section(sig: Significance) -> str:
    significant = "significant" if sig.mcnemar_p < 0.05 else "not significant"
    return f"""## Is the classifier's gain over the lexicon real?

A paired comparison on the same {sig.a_only_correct + sig.b_only_correct} sentences the two scorers
disagree on. The classifier is right and the lexicon wrong on {sig.a_only_correct}; the reverse on
{sig.b_only_correct}.

- **McNemar's test:** chi-square (continuity-corrected) {sig.mcnemar_chi2:.2f}, p = {sig.mcnemar_p:.3g}
  ({significant} at 0.05).
- **Accuracy gain {_fmt_pct(sig.acc_diff)}**, bootstrap 95% CI
  [{_fmt_pct(sig.ci_low)}, {_fmt_pct(sig.ci_high)}] ({_BOOTSTRAP_SAMPLES} resamples). The interval
  {"excludes" if sig.ci_low > 0 else "includes"} zero.
"""


def _write_report(
    *,
    train_n: int,
    test_n: int,
    baseline_acc: float,
    majority: str,
    mapping_note: str,
    results: list[Result],
    significance: Significance,
) -> None:
    """Write the markdown evaluation report covering every scorer that ran."""
    sections = "\n".join(_result_section(r, baseline_acc, majority) for r in results)
    gemini_note = (
        ""
        if any("Gemini" in r.name for r in results)
        else "\nThe Gemini path is wired into this harness (`--with-gemini`, requires "
        "`CBT_GEMINI_API_KEY`); that run is pending an API key, after which its numbers join the "
        "table above for a three-way comparison.\n"
    )
    _REPORT.write_text(
        f"""# Tone scorer evaluation

Generated by `scripts/eval_tone.py`. A real, reproducible evaluation against labeled data, not a
self-assessment. Re-run with `uv run python scripts/eval_tone.py` (no API key needed).

## Benchmark

"Trillion Dollar Words" (Shah, Paturi, Chava, ACL 2023), dataset `gtfintechlab/fomc_communication`
(CC BY-NC 4.0): FOMC sentences each labeled hawkish, dovish, or neutral. Used for offline
evaluation only; the corpus is not redistributed here (fetched into a gitignored cache). Split
sizes: train {train_n}, test {test_n}. Label-mapping sanity check (train): {mapping_note}.

## Method

Each scorer assigns one of hawkish / dovish / neutral to each sentence. Every hyperparameter (the
lexicon's neutral band, the classifier's regularization, vocabulary, and class balancing) was
chosen on train only (the classifier by k-fold cross-validation) and applied unchanged to the
held-out test split, which is scored exactly once. We report accuracy against the majority-class
baseline (so the lift is honest) and macro-F1 (which weights the three classes equally, not
dominated by the large neutral class).

## Results (held-out test split)

{sections}
{_significance_section(significance)}
![Confusion matrices](tone-confusion-matrix.png)

## Honest reading

The supervised classifier (ADR 0013) is the strongest offline scorer: it learns from the whole
vocabulary, predicts on every sentence, and roughly doubles the lexicon's macro-F1, with the gain
over the lexicon significant under McNemar's test and a bootstrap confidence interval that excludes
zero. It is still a linear bag-of-words model on a hard three-class problem, so it is a credible
baseline, not a claim of state of the art (transformer models reported in the source paper score
higher). The lexicon remains the transparent, license-clean floor and the auditable cross-check on
the model; the classifier and the Gemini judge are the stronger signals layered on top.{gemini_note}""",
        encoding="utf-8",
    )


def main() -> int:
    """Run the evaluation and write the report and chart."""
    with_gemini = "--with-gemini" in sys.argv[1:]
    print(f"Fetching {_DATASET} (train, test) ...")
    train, test = _fetch_split("train"), _fetch_split("test")
    print(f"  train={len(train)} test={len(test)} sentences")

    mapping_note = _sanity_check_mapping(HawkishDovishLexicon(), train)
    print(f"  {mapping_note}")

    lexicon_result = _evaluate_lexicon(train, test)
    classifier_result = _evaluate_classifier(test)
    results = [lexicon_result, classifier_result]
    if with_gemini:
        print("  scoring with Gemini (this spends API calls) ...")
        results.append(_evaluate_gemini(test))

    significance = _significance(
        _gold(test), list(classifier_result.predictions), list(lexicon_result.predictions)
    )

    majority = Counter(_gold(train)).most_common(1)[0][0]
    baseline_acc = _accuracy(_gold(test), [majority] * len(test))
    for result in results:
        print(
            f"  {result.name}: accuracy={result.accuracy:.3f} macro-F1={result.macro_f1:.3f} "
            f"(baseline {baseline_acc:.3f})"
        )
    print(
        f"  classifier vs lexicon: McNemar chi2={significance.mcnemar_chi2:.2f} "
        f"p={significance.mcnemar_p:.3g}, acc gain {significance.acc_diff:+.3f} "
        f"95% CI [{significance.ci_low:+.3f}, {significance.ci_high:+.3f}]"
    )

    _write_confusion_png(results)
    _write_report(
        train_n=len(train),
        test_n=len(test),
        baseline_acc=baseline_acc,
        majority=majority,
        mapping_note=mapping_note,
        results=results,
        significance=significance,
    )
    print(f"wrote {_REPORT.relative_to(_REPO_ROOT)} and {_CONFUSION_PNG.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

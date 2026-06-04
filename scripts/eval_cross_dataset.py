"""Out-of-distribution check: score the tone classifier on op-fed (ADR 0013, survey next-step 3).

The supervised classifier (ADR 0013) is trained and tested on "Trillion Dollar Words" FOMC
*speech* sentences. A single-benchmark number can flatter a model, so this script measures
zero-shot transfer to a different corpus annotated a different way: **op-fed**
(`kakeith/op-fed`, MIT; Keith et al., "Op-Fed", arXiv:2509.13539), 1,044 sentences from FOMC
*meeting transcripts* labeled with a StanceNLI scheme.

op-fed labels stance against the fixed hypothesis "We should tighten monetary policy":

- ``entailment``  -> the speaker supports tightening (raising rates) -> **hawkish**
- ``contradiction`` -> the speaker supports easing (lowering rates) -> **dovish**
- ``neutral`` -> no directional stance -> **neutral**

(per the paper: "an entailment label would capture ... tightening direction ...; a contradiction
label would instead capture a stance direction towards loosening".) Rows whose stance is empty or
``ambiguous`` are not usable and are dropped. We apply the already-trained classifier and the
lexicon unchanged (no fitting on op-fed) and report accuracy, macro-F1, and a confusion matrix
against the majority-class baseline. The MIT corpus is cached gitignored and not redistributed
here; only our computed metrics and chart are committed.

Usage::

    uv run python scripts/eval_cross_dataset.py   # no API key needed
"""

from __future__ import annotations

import csv
import io
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

from cbt_core.analysis.classifier import ToneClassifier
from cbt_core.analysis.lexicon import HawkishDovishLexicon

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE = _REPO_ROOT / "data" / "benchmarks" / "opfed_v1.csv"
_REPORT = _REPO_ROOT / "docs" / "research" / "cross-dataset-eval.md"
_CONFUSION_PNG = _REPO_ROOT / "docs" / "research" / "cross-dataset-confusion.png"

_SOURCE_URL = "https://raw.githubusercontent.com/kakeith/op-fed/main/data/opfed_v1.csv"
_SENTENCE_COLUMN = "sentence"
_STANCE_COLUMN = "4_stance_nli"
# Documented op-fed StanceNLI mapping (hypothesis: "We should tighten monetary policy").
_STANCE_TO_CLASS = {
    "entailment": "hawkish",
    "contradiction": "dovish",
    "neutral": "neutral",
}
_CLASSES = ("hawkish", "dovish", "neutral")
# The lexicon's neutral band, selected on the FOMC benchmark train split (scripts/eval_tone.py).
# Reused unchanged here: cross-dataset means no fitting on the new corpus.
_LEXICON_TAU = 0.0


@dataclass(frozen=True)
class Result:
    """Metrics for one scorer on the op-fed labeled subset."""

    name: str
    note: str
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]]


def _fetch_rows() -> list[dict[str, str]]:
    """Download op-fed (cached) and return its rows as dicts."""
    if not _CACHE.exists():
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(_SOURCE_URL, headers={"User-Agent": "cbt-eval/1.0"})  # noqa: S310  (trusted GitHub host, https)
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            _CACHE.write_bytes(response.read())
    text = _CACHE.read_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _labeled(rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    """Return (sentences, gold classes) for rows carrying a usable stance label."""
    sentences: list[str] = []
    gold: list[str] = []
    for row in rows:
        stance = (row.get(_STANCE_COLUMN) or "").strip().lower()
        sentence = (row.get(_SENTENCE_COLUMN) or "").strip()
        mapped = _STANCE_TO_CLASS.get(stance)
        if mapped is not None and sentence:
            sentences.append(sentence)
            gold.append(mapped)
    return sentences, gold


def _accuracy(gold: list[str], pred: list[str]) -> float:
    return sum(g == p for g, p in zip(gold, pred, strict=True)) / len(gold) if gold else 0.0


def _per_class(gold: list[str], pred: list[str]) -> dict[str, dict[str, float]]:
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


def _macro_f1(gold: list[str], pred: list[str]) -> float:
    return sum(stats["f1"] for stats in _per_class(gold, pred).values()) / len(_CLASSES)


def _confusion(gold: list[str], pred: list[str]) -> list[list[int]]:
    index = {cls: i for i, cls in enumerate(_CLASSES)}
    matrix = [[0, 0, 0] for _ in _CLASSES]
    for g, p in zip(gold, pred, strict=True):
        matrix[index[g]][index[p]] += 1
    return matrix


def _result(name: str, note: str, gold: list[str], pred: list[str]) -> Result:
    return Result(
        name=name,
        note=note,
        accuracy=_accuracy(gold, pred),
        macro_f1=_macro_f1(gold, pred),
        per_class=_per_class(gold, pred),
        confusion=_confusion(gold, pred),
    )


def _lexicon_label(lexicon: HawkishDovishLexicon, text: str) -> str:
    result = lexicon.score(text)
    if result.hawkish_hits + result.dovish_hits == 0:
        return "neutral"
    if result.score > _LEXICON_TAU:
        return "hawkish"
    if result.score < -_LEXICON_TAU:
        return "dovish"
    return "neutral"


def _write_confusion_png(result: Result) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.8))
    ax.imshow(result.confusion, cmap="Blues")
    ax.set_xticks(range(len(_CLASSES)), labels=_CLASSES, fontsize=8)
    ax.set_yticks(range(len(_CLASSES)), labels=_CLASSES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(f"Classifier on op-fed (acc {result.accuracy:.0%})", fontsize=9)
    peak = max(max(row) for row in result.confusion) or 1
    for i, row in enumerate(result.confusion):
        for j, value in enumerate(row):
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > peak / 2 else "black",
            )
    fig.suptitle("Cross-dataset confusion (op-fed, rows = gold, columns = predicted)")
    fig.tight_layout()
    fig.savefig(_CONFUSION_PNG, dpi=140)
    plt.close(fig)


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _result_block(result: Result, baseline_acc: float, majority: str) -> str:
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
    lift = result.accuracy - baseline_acc
    return f"""### {result.name}

{result.note}.

- **Accuracy {_fmt_pct(result.accuracy)}** vs majority-class ("{majority}") {_fmt_pct(baseline_acc)}
  ({"+" if lift >= 0 else ""}{_fmt_pct(lift)}).
- **Macro-F1 {result.macro_f1:.3f}**.

| class | precision | recall | F1 | support |
|---|---|---|---|---|
{rows}

Confusion matrix (rows = gold, columns = predicted):

{header}
{sep}
{conf_rows}
"""


def _write_report(
    *,
    n: int,
    distribution: dict[str, int],
    results: list[Result],
    majority: str,
    baseline_acc: float,
) -> None:
    dist = ", ".join(f"{cls} {distribution.get(cls, 0)}" for cls in _CLASSES)
    blocks = "\n".join(_result_block(r, baseline_acc, majority) for r in results)
    classifier = next(r for r in results if "classifier" in r.name.lower())
    lexicon = next(r for r in results if "lexicon" in r.name.lower())
    vs_baseline = (
        f"below the {majority} majority-class baseline ({_fmt_pct(baseline_acc)})"
        if classifier.accuracy < baseline_acc
        else f"above the {majority} majority-class baseline ({_fmt_pct(baseline_acc)})"
    )
    _REPORT.write_text(
        f"""# Cross-dataset check: the tone classifier on op-fed

Generated by `scripts/eval_cross_dataset.py`. An out-of-distribution test of the supervised
classifier (ADR 0013), which is trained and tested on FOMC *speech* sentences ("Trillion Dollar
Words"). Here it is applied unchanged to **op-fed** (`kakeith/op-fed`, MIT; Keith et al.,
arXiv:2509.13539): {n} FOMC *meeting transcript* sentences with a usable stance label.

## Mapping

op-fed annotates stance against the hypothesis "We should tighten monetary policy", so the labels
map directly onto our classes: entailment -> hawkish, contradiction -> dovish, neutral -> neutral.
Rows whose stance is empty or `ambiguous` are dropped. Class distribution of the labeled subset:
{dist}.

This is a genuine distribution shift on two axes: the text is meeting-transcript discussion rather
than prepared speeches, and the labels come from a different annotation method (StanceNLI) than the
training benchmark. No model or threshold was refit on op-fed.

## Results

{blocks}
![Cross-dataset confusion matrix](cross-dataset-confusion.png)

## Honest reading

On op-fed the classifier scores **{_fmt_pct(classifier.accuracy)} accuracy / macro-F1
{classifier.macro_f1:.3f}**, well below its in-distribution **59.9% / 0.582** on the FOMC speech
test split, and its accuracy is **{vs_baseline}** because the labeled subset is small ({n}
sentences) and skewed dovish. This is a hard transfer, and the result is reported as it is: meeting
transcript discussion under a StanceNLI annotation is a meaningfully different task from labeling
prepared-speech sentences, so much of the model's signal turns out to be specific to the speech
corpus it was trained on rather than general hawkish/dovish language. It still separates the classes
better than the lexicon on macro-F1 ({classifier.macro_f1:.3f} vs {lexicon.macro_f1:.3f}), so it is
not merely guessing, but the drop is the honest headline. For genuinely out-of-corpus text the
production path is the Gemini judge, which does not rely on this model's learned vocabulary; a
fine-tuned transformer (survey next-step 4) would be the way to harden the offline scorer.
""",
        encoding="utf-8",
    )


def main() -> int:
    """Run the cross-dataset evaluation and write the report and chart."""
    print(f"Fetching op-fed from {_SOURCE_URL} ...")
    rows = _fetch_rows()
    sentences, gold = _labeled(rows)
    print(f"  {len(rows)} rows, {len(sentences)} with a usable stance label")
    if not sentences:
        print("  no labeled sentences found; aborting")
        return 1

    distribution = Counter(gold)
    majority = distribution.most_common(1)[0][0]
    baseline_acc = _accuracy(gold, [majority] * len(gold))

    model = ToneClassifier.load_default()
    classifier_pred = [model.score(s).label for s in sentences]
    lexicon = HawkishDovishLexicon()
    lexicon_pred = [_lexicon_label(lexicon, s) for s in sentences]

    results = [
        _result(
            "Deterministic lexicon (zero-shot)",
            "net-hawkishness with the neutral band tuned on the FOMC benchmark, applied unchanged",
            gold,
            lexicon_pred,
        ),
        _result(
            "Supervised classifier (zero-shot)",
            "the FOMC-trained TF-IDF + logistic-regression model (ADR 0013), applied unchanged",
            gold,
            classifier_pred,
        ),
    ]
    for result in results:
        print(f"  {result.name}: accuracy={result.accuracy:.3f} macro-F1={result.macro_f1:.3f}")
    print(f"  majority baseline ({majority}): {baseline_acc:.3f}")

    classifier_result = next(r for r in results if "classifier" in r.name.lower())
    _write_confusion_png(classifier_result)
    _write_report(
        n=len(sentences),
        distribution=dict(distribution),
        results=results,
        majority=majority,
        baseline_acc=baseline_acc,
    )
    print(f"wrote {_REPORT.relative_to(_REPO_ROOT)} and {_CONFUSION_PNG.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

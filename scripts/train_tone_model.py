"""Train the supervised TF-IDF tone classifier offline (ADR 0013).

Fits a class-balanced multinomial logistic regression over TF-IDF features on the annotated FOMC
benchmark ("Trillion Dollar Words", Shah, Paturi, Chava, ACL 2023;
``gtfintechlab/fomc_communication``, CC BY-NC 4.0) and writes the trained artifact to
``packages/cbt_core/src/cbt_core/analysis/tone_model.json`` for :class:`cbt_core.ToneClassifier`.

Everything here is pure numpy: the design matrix uses the same featurizers the classifier uses at
inference time (so features cannot drift), and the optimizer is a plain full-batch Adam on the
class-balanced softmax cross-entropy loss with L2 regularization. The regularization strength is
selected by k-fold cross-validation on train (so no test-set peeking); the model is then refit on
all of train and the artifact is written. The held-out test numbers are reported by
``scripts/eval_tone.py`` in a single, honest pass.

Class balancing weights each class inversely to its frequency, so the minority hawkish and dovish
classes are not drowned out by the dominant neutral class; this is what raises macro-F1 on this
imbalanced benchmark. The benchmark is fetched into a gitignored cache and not redistributed; the
trained weights inherit its non-commercial term (ADR 0013).

Usage::

    uv run python scripts/train_tone_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from cbt_core.analysis.classifier import (
    TONE_CLASSES,
    ToneClassifier,
    build_vocabulary,
    compute_idf,
    transform,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _REPO_ROOT / "data" / "benchmarks"
_ARTIFACT = (
    _REPO_ROOT / "packages" / "cbt_core" / "src" / "cbt_core" / "analysis" / "tone_model.json"
)

# Dataset label ids -> our class names (verified by the sanity check in eval_tone.py).
_LABELS = {0: "dovish", 1: "hawkish", 2: "neutral"}

# Featurization and optimization hyperparameters. ngram_max=2 captures phrases like "rate hike".
# min_df / max_features / class balancing were chosen by the cross-validated search in
# scripts (see docs/research/tone-evaluation.md); L2 is re-selected here by k-fold CV. Fixed seed
# -> a reproducible artifact.
_NGRAM_MAX = 2
_MIN_DF = 3
_MAX_FEATURES = 5000
_SUBLINEAR_TF = True
_BALANCE_CLASSES = True
_CV_FOLDS = 3
_EPOCHS_CV = 400
_EPOCHS_FINAL = 600
_LEARNING_RATE = 0.5
_L2_GRID = (3e-4, 1e-3, 3e-3)
_SEED = 0


def _load(split: str) -> tuple[list[str], npt.NDArray[np.int64]]:
    """Load a cached split as parallel sentence and class-index arrays."""
    cache = _CACHE_DIR / f"fomc_{split}.json"
    if not cache.exists():
        raise SystemExit(
            f"missing {cache}; run `uv run python scripts/eval_tone.py` once to populate the cache"
        )
    rows = json.loads(cache.read_text(encoding="utf-8"))
    sentences = [str(row["sentence"]) for row in rows]
    labels = np.array(
        [TONE_CLASSES.index(_LABELS[int(row["label"])]) for row in rows], dtype=np.int64
    )
    return sentences, labels


def _one_hot(y: npt.NDArray[np.int64], n_classes: int) -> npt.NDArray[np.float64]:
    """One-hot encode integer labels into a ``(n, n_classes)`` matrix."""
    encoded = np.zeros((y.shape[0], n_classes), dtype=np.float64)
    encoded[np.arange(y.shape[0]), y] = 1.0
    return encoded


def _sample_weights(y: npt.NDArray[np.int64], n_classes: int) -> npt.NDArray[np.float64]:
    """Per-sample weights that balance the classes (inverse frequency, mean 1)."""
    if not _BALANCE_CLASSES:
        return np.ones(y.shape[0], dtype=np.float64)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    class_weight = y.shape[0] / (n_classes * np.maximum(counts, 1.0))
    weights = class_weight[y]
    return weights / weights.mean()


def _fit_softmax(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    *,
    n_classes: int,
    l2: float,
    epochs: int,
    learning_rate: float = _LEARNING_RATE,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fit class-balanced multinomial logistic regression by full-batch Adam."""
    n_samples, n_features = x.shape
    weights = np.zeros((n_features, n_classes), dtype=np.float64)
    bias = np.zeros(n_classes, dtype=np.float64)
    targets = _one_hot(y, n_classes)
    sample_weight = _sample_weights(y, n_classes)[:, None]
    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = np.zeros_like(bias)
    v_b = np.zeros_like(bias)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for step in range(1, epochs + 1):
        logits = x @ weights + bias
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        grad = sample_weight * (probs - targets) / n_samples
        grad_w = x.T @ grad + l2 * weights
        grad_b = grad.sum(axis=0)
        m_w = beta1 * m_w + (1 - beta1) * grad_w
        v_w = beta2 * v_w + (1 - beta2) * grad_w**2
        m_b = beta1 * m_b + (1 - beta1) * grad_b
        v_b = beta2 * v_b + (1 - beta2) * grad_b**2
        weights -= (
            learning_rate * (m_w / (1 - beta1**step)) / (np.sqrt(v_w / (1 - beta2**step)) + eps)
        )
        bias -= learning_rate * (m_b / (1 - beta1**step)) / (np.sqrt(v_b / (1 - beta2**step)) + eps)
    return weights, bias


def _predict(
    x: npt.NDArray[np.float64], weights: npt.NDArray[np.float64], bias: npt.NDArray[np.float64]
) -> npt.NDArray[np.int64]:
    """Return the argmax class index for each row."""
    return np.asarray((x @ weights + bias).argmax(axis=1), dtype=np.int64)


def _macro_f1(
    y_true: npt.NDArray[np.int64], y_pred: npt.NDArray[np.int64], n_classes: int
) -> float:
    """Unweighted mean F1 across classes."""
    f1s: list[float] = []
    for cls in range(n_classes):
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1s))


def _select_l2(
    texts: list[str], y: npt.NDArray[np.int64], n_classes: int, rng: np.random.Generator
) -> tuple[float, dict[float, tuple[float, float]]]:
    """Select L2 by k-fold cross-validated macro-F1 (no test-set peeking)."""
    folds = np.array_split(rng.permutation(len(texts)), _CV_FOLDS)
    scores: dict[float, tuple[float, float]] = {}
    for l2 in _L2_GRID:
        fold_f1: list[float] = []
        for k in range(_CV_FOLDS):
            val_idx = folds[k]
            fit_idx = np.concatenate([folds[j] for j in range(_CV_FOLDS) if j != k])
            fit_text = [texts[i] for i in fit_idx]
            val_text = [texts[i] for i in val_idx]
            vocab = build_vocabulary(
                fit_text, ngram_max=_NGRAM_MAX, min_df=_MIN_DF, max_features=_MAX_FEATURES
            )
            idf = compute_idf(fit_text, vocab, ngram_max=_NGRAM_MAX)
            x_fit = transform(
                fit_text, vocab, idf, ngram_max=_NGRAM_MAX, sublinear_tf=_SUBLINEAR_TF
            )
            x_val = transform(
                val_text, vocab, idf, ngram_max=_NGRAM_MAX, sublinear_tf=_SUBLINEAR_TF
            )
            weights, bias = _fit_softmax(
                x_fit, y[fit_idx], n_classes=n_classes, l2=l2, epochs=_EPOCHS_CV
            )
            fold_f1.append(_macro_f1(y[val_idx], _predict(x_val, weights, bias), n_classes))
        scores[l2] = (float(np.mean(fold_f1)), float(np.std(fold_f1)))
        print(f"  l2={l2:<7} cv macro-F1={scores[l2][0]:.3f} +/- {scores[l2][1]:.3f}")
    best_l2 = max(scores, key=lambda key: scores[key][0])
    return best_l2, scores


def main() -> int:
    """Train the model, select L2 by CV, refit on all of train, and write the artifact."""
    train_text, train_y = _load("train")
    n_classes = len(TONE_CLASSES)
    rng = np.random.default_rng(_SEED)
    print(f"train={len(train_text)} sentences, {_CV_FOLDS}-fold CV for L2:")
    best_l2, cv_scores = _select_l2(train_text, train_y, n_classes, rng)
    print(f"selected l2={best_l2} (cv macro-F1={cv_scores[best_l2][0]:.3f})")

    vocab = build_vocabulary(
        train_text, ngram_max=_NGRAM_MAX, min_df=_MIN_DF, max_features=_MAX_FEATURES
    )
    idf = compute_idf(train_text, vocab, ngram_max=_NGRAM_MAX)
    x_train = transform(train_text, vocab, idf, ngram_max=_NGRAM_MAX, sublinear_tf=_SUBLINEAR_TF)
    weights, bias = _fit_softmax(
        x_train, train_y, n_classes=n_classes, l2=best_l2, epochs=_EPOCHS_FINAL
    )
    train_acc = float(np.mean(_predict(x_train, weights, bias) == train_y))
    print(f"refit on full train (vocab={len(vocab)}): train accuracy={train_acc:.3f}")

    artifact = {
        "version": 1,
        "classes": list(TONE_CLASSES),
        "ngram_max": _NGRAM_MAX,
        "sublinear_tf": _SUBLINEAR_TF,
        "vocabulary": vocab,
        "idf": [round(float(v), 6) for v in idf.tolist()],
        "coef": [[round(float(v), 6) for v in row] for row in weights.tolist()],
        "intercept": [round(float(v), 6) for v in bias.tolist()],
        "metadata": {
            "benchmark": "gtfintechlab/fomc_communication (CC BY-NC 4.0)",
            "train_n": len(train_text),
            "balanced_classes": _BALANCE_CLASSES,
            "ngram_max": _NGRAM_MAX,
            "min_df": _MIN_DF,
            "max_features": _MAX_FEATURES,
            "selected_l2": best_l2,
            "cv_folds": _CV_FOLDS,
            "cv_macro_f1": round(cv_scores[best_l2][0], 4),
            "cv_macro_f1_std": round(cv_scores[best_l2][1], 4),
        },
    }
    _ARTIFACT.write_text(json.dumps(artifact), encoding="utf-8")
    size_kb = _ARTIFACT.stat().st_size / 1024
    print(f"wrote {_ARTIFACT.relative_to(_REPO_ROOT)} ({size_kb:.0f} KB)")

    model = ToneClassifier.from_artifact(_ARTIFACT)
    for text in (
        "The committee remains prepared to raise rates to combat persistent inflation.",
        "Given downside risks and labor market slack, further accommodation may be warranted.",
        "Broad equity price indexes fell over the intermeeting period.",
    ):
        verdict = model.score(text)
        print(f"  smoke: {verdict.label:<8} score={verdict.score:+.3f}  | {text[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

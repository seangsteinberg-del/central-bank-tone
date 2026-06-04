"""Supervised TF-IDF + logistic-regression tone classifier (ADR 0013).

A real, license-clean-to-run supervised model that scores monetary-policy tone
(hawkish / dovish / neutral) with no network and no paid API. It complements two existing
signals: the deterministic lexicon (a transparent floor, ADR 0008) and the Gemini LLM judge (the
multilingual production path, ADR 0007). Where the lexicon only fires on sentences containing its
curated terms, this model learns from the whole vocabulary and so carries signal on far more
sentences; its accuracy is measured against the annotated FOMC benchmark, not asserted (see
``scripts/eval_tone.py`` and ``docs/research/tone-evaluation.md``).

This module is inference only and pure: it loads a trained artifact (a JSON file of the
vocabulary, IDF weights, and the fitted softmax weights) and applies it. The artifact is produced
offline by ``scripts/train_tone_model.py``. Training and inference share the vectorization helpers
here so the features can never drift apart.

The artifact is trained on the CC BY-NC 4.0 FOMC benchmark, so the trained weights inherit that
non-commercial term; ADR 0013 records this, exactly as ADR 0008 records it for FOMC-RoBERTa. The
license-clean production path remains the lexicon and Gemini.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from cbt_core.exceptions import CbtError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    import numpy.typing as npt

_ARTIFACT_PACKAGE = "cbt_core.analysis"
_ARTIFACT_NAME = "tone_model.json"
_TOKEN_RE = re.compile(r"[a-z]+")

# The three classes the benchmark labels and the model predicts. Order is fixed: it indexes the
# columns of the weight matrix in the artifact.
TONE_CLASSES: tuple[str, ...] = ("hawkish", "dovish", "neutral")


class ToneModelError(CbtError):
    """Raised when the trained tone-model artifact is missing or malformed.

    A deployment/programming error (the artifact should ship with the package), not user input,
    so adapters surface it as a 5xx rather than a 4xx.
    """


def tokenize(text: str) -> list[str]:
    """Split text into lowercase alphabetic word tokens.

    Numbers and punctuation are dropped; the monetary signal lives in words and short phrases
    ("rate hike", "downside risks"), and digits are high-cardinality noise.

    Args:
        text: The raw text to tokenize.

    Returns:
        The list of lowercase word tokens, in order.
    """
    return _TOKEN_RE.findall(text.lower())


def ngrams(tokens: Sequence[str], ngram_max: int) -> list[str]:
    """Build unigrams through ``ngram_max``-grams from a token sequence.

    Args:
        tokens: The ordered tokens.
        ngram_max: The largest n-gram length to emit (1 for unigrams only, 2 to add bigrams).

    Returns:
        Every n-gram for ``n`` in ``1..ngram_max`` as space-joined strings, in order.
    """
    grams: list[str] = list(tokens)
    for n in range(2, ngram_max + 1):
        grams.extend(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return grams


def _features(text: str, ngram_max: int) -> list[str]:
    """Tokenize and expand one text into its n-gram feature strings."""
    return ngrams(tokenize(text), ngram_max)


def _tfidf_row(
    feature_strings: Iterable[str],
    vocabulary: dict[str, int],
    idf: npt.NDArray[np.float64],
    *,
    sublinear_tf: bool,
) -> npt.NDArray[np.float64]:
    """Build one L2-normalized TF-IDF row over a fixed vocabulary (shared by train and inference)."""
    vec = np.zeros(len(vocabulary), dtype=np.float64)
    counts: dict[int, int] = {}
    for feature in feature_strings:
        index = vocabulary.get(feature)
        if index is not None:
            counts[index] = counts.get(index, 0) + 1
    for index, count in counts.items():
        tf = 1.0 + math.log(count) if sublinear_tf else float(count)
        vec[index] = tf * idf[index]
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


def build_vocabulary(
    texts: Iterable[str], *, ngram_max: int, min_df: int, max_features: int
) -> dict[str, int]:
    """Select the model vocabulary from a training corpus by document frequency.

    Features appearing in fewer than ``min_df`` documents are dropped (noise); if more than
    ``max_features`` remain, the most frequent are kept. The surviving features are indexed in
    sorted order so the mapping is deterministic.

    Args:
        texts: The training documents.
        ngram_max: The largest n-gram length to consider.
        min_df: The minimum document frequency for a feature to be kept.
        max_features: The maximum vocabulary size.

    Returns:
        A deterministic map from feature string to column index.
    """
    document_frequency: dict[str, int] = {}
    for text in texts:
        for feature in set(_features(text, ngram_max)):
            document_frequency[feature] = document_frequency.get(feature, 0) + 1
    eligible = [feature for feature, df in document_frequency.items() if df >= min_df]
    eligible.sort(key=lambda feature: (-document_frequency[feature], feature))
    kept = sorted(eligible[:max_features])
    return {feature: index for index, feature in enumerate(kept)}


def compute_idf(
    texts: Iterable[str], vocabulary: dict[str, int], *, ngram_max: int
) -> npt.NDArray[np.float64]:
    """Compute smoothed inverse document frequencies for a vocabulary over a corpus.

    Uses the smoothed form ``idf = log((1 + n) / (1 + df)) + 1`` so no feature divides by zero
    and very common features are damped but never negative.

    Args:
        texts: The corpus to estimate document frequencies from.
        vocabulary: The feature-to-index mapping.
        ngram_max: The largest n-gram length (must match :func:`build_vocabulary`).

    Returns:
        The IDF weight for each vocabulary entry, indexed by its column.
    """
    document_frequency = np.zeros(len(vocabulary), dtype=np.float64)
    n_documents = 0
    for text in texts:
        n_documents += 1
        seen = {vocabulary[f] for f in set(_features(text, ngram_max)) if f in vocabulary}
        for index in seen:
            document_frequency[index] += 1.0
    return np.log((1.0 + n_documents) / (1.0 + document_frequency)) + 1.0


def transform(
    texts: Sequence[str],
    vocabulary: dict[str, int],
    idf: npt.NDArray[np.float64],
    *,
    ngram_max: int,
    sublinear_tf: bool,
) -> npt.NDArray[np.float64]:
    """Vectorize a list of texts into a TF-IDF design matrix.

    Args:
        texts: The texts to vectorize.
        vocabulary: The feature-to-index mapping.
        idf: The IDF weights aligned to ``vocabulary``.
        ngram_max: The largest n-gram length.
        sublinear_tf: Whether to dampen term frequencies by ``1 + log(tf)``.

    Returns:
        A matrix of shape ``(len(texts), len(vocabulary))``.
    """
    if not texts:
        return np.zeros((0, len(vocabulary)), dtype=np.float64)
    return np.vstack(
        [
            _tfidf_row(_features(text, ngram_max), vocabulary, idf, sublinear_tf=sublinear_tf)
            for text in texts
        ]
    )


@dataclass(frozen=True)
class ClassifierScore:
    """The supervised model's verdict for one text.

    Attributes:
        label: The argmax class, one of :data:`TONE_CLASSES`.
        score: A continuous net-hawkishness in ``[-1.0, 1.0]``, ``P(hawkish) - P(dovish)``.
        probabilities: The full softmax distribution over :data:`TONE_CLASSES`.
        confidence: The probability of the predicted ``label`` (the max of ``probabilities``).
    """

    label: str
    score: float
    probabilities: dict[str, float]
    confidence: float


class ToneClassifier:
    """A trained TF-IDF + multinomial-logistic-regression tone classifier (inference only).

    Construct it from a trained artifact with :meth:`load_default` (the bundled model) or
    :meth:`from_artifact` (an arbitrary path). The vectorization here is the same code the trainer
    uses, so a text is featurized identically at train and inference time.
    """

    def __init__(
        self,
        *,
        vocabulary: dict[str, int],
        idf: npt.NDArray[np.float64],
        coef: npt.NDArray[np.float64],
        intercept: npt.NDArray[np.float64],
        ngram_max: int,
        sublinear_tf: bool,
        classes: Sequence[str] = TONE_CLASSES,
    ) -> None:
        """Build a classifier from fitted parameters.

        Args:
            vocabulary: Map from feature string to its column index.
            idf: Inverse-document-frequency weights, one per vocabulary entry.
            coef: The fitted weight matrix, shape ``(n_features, n_classes)``.
            intercept: The per-class bias, shape ``(n_classes,)``.
            ngram_max: The largest n-gram length used when the model was trained.
            sublinear_tf: Whether term frequencies were dampened by ``1 + log(tf)``.
            classes: The class labels indexing the columns of ``coef``.

        Raises:
            ToneModelError: If the parameter shapes are inconsistent.
        """
        n_features = len(vocabulary)
        if idf.shape != (n_features,):
            raise ToneModelError(f"idf shape {idf.shape} != ({n_features},)")
        if coef.shape != (n_features, len(classes)):
            raise ToneModelError(f"coef shape {coef.shape} != ({n_features}, {len(classes)})")
        if intercept.shape != (len(classes),):
            raise ToneModelError(f"intercept shape {intercept.shape} != ({len(classes)},)")
        self._vocabulary = vocabulary
        self._idf = idf
        self._coef = coef
        self._intercept = intercept
        self._ngram_max = ngram_max
        self._sublinear_tf = sublinear_tf
        self._classes = tuple(classes)

    @property
    def classes(self) -> tuple[str, ...]:
        """The class labels, in the column order of the weight matrix."""
        return self._classes

    def _vector(self, text: str) -> npt.NDArray[np.float64]:
        """Featurize one text into an L2-normalized TF-IDF vector over the fixed vocabulary."""
        return _tfidf_row(
            _features(text, self._ngram_max),
            self._vocabulary,
            self._idf,
            sublinear_tf=self._sublinear_tf,
        )

    def predict_proba(self, text: str) -> dict[str, float]:
        """Return the softmax probability of each tone class for a text.

        Args:
            text: The text to score.

        Returns:
            A mapping from each class in :attr:`classes` to its probability (summing to 1).
        """
        logits = self._vector(text) @ self._coef + self._intercept
        logits -= logits.max()
        exp = np.exp(logits)
        probs = exp / exp.sum()
        return {cls: float(probs[i]) for i, cls in enumerate(self._classes)}

    def score(self, text: str) -> ClassifierScore:
        """Score the monetary-policy tone of a text.

        Args:
            text: The text to score.

        Returns:
            A :class:`ClassifierScore` with the argmax label, a continuous
            ``P(hawkish) - P(dovish)`` score, the full distribution, and the top probability.
        """
        probs = self.predict_proba(text)
        label = max(probs, key=lambda cls: probs[cls])
        score = probs.get("hawkish", 0.0) - probs.get("dovish", 0.0)
        return ClassifierScore(
            label=label,
            score=score,
            probabilities=probs,
            confidence=probs[label],
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ToneClassifier:
        """Build a classifier from a parsed artifact mapping.

        Args:
            payload: The parsed JSON artifact (see :meth:`to_dict` for its shape).

        Returns:
            The reconstructed classifier.

        Raises:
            ToneModelError: If a required key is missing or malformed.
        """
        # Deserializing an external JSON artifact: values are untyped at this IO boundary, so we
        # treat them as Any and let the except clause turn any shape mismatch into a typed error.
        data = cast("dict[str, Any]", payload)
        try:
            return cls(
                vocabulary={str(k): int(v) for k, v in data["vocabulary"].items()},
                idf=np.asarray(data["idf"], dtype=np.float64),
                coef=np.asarray(data["coef"], dtype=np.float64),
                intercept=np.asarray(data["intercept"], dtype=np.float64),
                ngram_max=int(data["ngram_max"]),
                sublinear_tf=bool(data["sublinear_tf"]),
                classes=tuple(str(c) for c in data.get("classes", TONE_CLASSES)),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ToneModelError(f"malformed tone-model artifact: {exc}") from exc

    @classmethod
    def from_artifact(cls, path: Path) -> ToneClassifier:
        """Load a classifier from a JSON artifact file.

        Args:
            path: The artifact path.

        Returns:
            The loaded classifier.

        Raises:
            ToneModelError: If the file is missing or malformed.
        """
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ToneModelError(f"cannot read tone-model artifact at {path}: {exc}") from exc
        return cls.from_dict(payload)

    @classmethod
    def load_default(cls) -> ToneClassifier:
        """Load the bundled trained model shipped with the package.

        Returns:
            The default classifier.

        Raises:
            ToneModelError: If the bundled artifact is missing (the model has not been trained)
                or malformed.
        """
        try:
            resource = resources.files(_ARTIFACT_PACKAGE) / _ARTIFACT_NAME
            text = resource.read_text(encoding="utf-8")
        except (FileNotFoundError, ModuleNotFoundError) as exc:
            raise ToneModelError(
                f"bundled tone model {_ARTIFACT_NAME!r} not found; train it with "
                "`uv run python scripts/train_tone_model.py`"
            ) from exc
        return cls.from_dict(json.loads(text))

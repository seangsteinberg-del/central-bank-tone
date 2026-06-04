"""Tests for the supervised TF-IDF tone classifier (ADR 0013).

Covers the shared featurizers (tokenization, vocabulary, IDF, the design matrix), the inference
model on hand-built weights (so the maths is pinned exactly), artifact (de)serialization and its
error handling, and a behavioural check of the bundled trained model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cbt_core import ClassifierScore, ToneClassifier, ToneModelError
from cbt_core.analysis.classifier import (
    TONE_CLASSES,
    build_vocabulary,
    compute_idf,
    ngrams,
    tokenize,
    transform,
)


@pytest.mark.unit
def test_tokenize_lowercases_and_drops_non_letters() -> None:
    assert tokenize("Rate HIKE of 25bps, again!") == ["rate", "hike", "of", "bps", "again"]


@pytest.mark.unit
def test_ngrams_emits_unigrams_and_bigrams() -> None:
    assert ngrams(["rate", "hike", "soon"], 2) == ["rate", "hike", "soon", "rate hike", "hike soon"]


@pytest.mark.unit
def test_ngrams_unigrams_only_when_max_is_one() -> None:
    assert ngrams(["a", "b", "c"], 1) == ["a", "b", "c"]


@pytest.mark.unit
def test_build_vocabulary_drops_below_min_df_and_is_deterministic() -> None:
    texts = ["inflation risk", "inflation risk", "growth"]
    # "growth" appears in one document; min_df=2 drops it. "inflation"/"risk"/"inflation risk" stay.
    vocab = build_vocabulary(texts, ngram_max=2, min_df=2, max_features=100)
    assert "growth" not in vocab
    assert set(vocab) == {"inflation", "risk", "inflation risk"}
    # Indices are assigned in sorted order, so the mapping is reproducible.
    assert vocab == dict(zip(sorted(vocab), range(len(vocab)), strict=True))


@pytest.mark.unit
def test_build_vocabulary_caps_at_max_features_by_document_frequency() -> None:
    texts = ["a a b c", "a b", "a"]  # df: a=3, b=2, c=1
    vocab = build_vocabulary(texts, ngram_max=1, min_df=1, max_features=2)
    assert set(vocab) == {"a", "b"}  # the two most frequent survive the cap


@pytest.mark.unit
def test_compute_idf_is_smoothed_and_lower_for_common_terms() -> None:
    texts = ["alpha beta", "alpha"]
    vocab = build_vocabulary(texts, ngram_max=1, min_df=1, max_features=10)
    idf = compute_idf(texts, vocab, ngram_max=1)
    # "alpha" in both docs -> idf = log((1+2)/(1+2)) + 1 = 1.0; "beta" in one -> higher.
    assert idf[vocab["alpha"]] == pytest.approx(1.0)
    assert idf[vocab["beta"]] > idf[vocab["alpha"]]


@pytest.mark.unit
def test_transform_rows_are_l2_normalized() -> None:
    texts = ["alpha beta", "alpha"]
    vocab = build_vocabulary(texts, ngram_max=1, min_df=1, max_features=10)
    idf = compute_idf(texts, vocab, ngram_max=1)
    matrix = transform(texts, vocab, idf, ngram_max=1, sublinear_tf=True)
    assert matrix.shape == (2, len(vocab))
    assert np.linalg.norm(matrix[0]) == pytest.approx(1.0)


@pytest.mark.unit
def test_transform_empty_input_returns_zero_by_vocab_matrix() -> None:
    vocab = {"x": 0, "y": 1}
    idf = np.ones(2)
    matrix = transform([], vocab, idf, ngram_max=1, sublinear_tf=True)
    assert matrix.shape == (0, 2)


def _toy_model() -> ToneClassifier:
    """A two-feature model where 'hawk' pushes hawkish and 'dove' pushes dovish."""
    classes = TONE_CLASSES  # (hawkish, dovish, neutral)
    return ToneClassifier(
        vocabulary={"hawk": 0, "dove": 1},
        idf=np.array([1.0, 1.0]),
        coef=np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]),
        intercept=np.zeros(3),
        ngram_max=1,
        sublinear_tf=True,
        classes=classes,
    )


@pytest.mark.unit
def test_predict_proba_is_a_distribution() -> None:
    probs = _toy_model().predict_proba("hawk")
    assert set(probs) == set(TONE_CLASSES)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


@pytest.mark.unit
def test_score_picks_argmax_and_reports_net_hawkishness() -> None:
    result = _toy_model().score("hawk")
    assert isinstance(result, ClassifierScore)
    assert result.label == "hawkish"
    assert result.score == pytest.approx(
        result.probabilities["hawkish"] - result.probabilities["dovish"]
    )
    assert result.confidence == pytest.approx(max(result.probabilities.values()))


@pytest.mark.unit
def test_score_is_dovish_for_dovish_feature() -> None:
    result = _toy_model().score("dove")
    assert result.label == "dovish"
    assert result.score < 0


@pytest.mark.unit
def test_unknown_tokens_fall_back_to_intercept() -> None:
    # No known features: the vector is all zeros, so the distribution is the softmax of the bias.
    result = _toy_model().score("completely unrelated words")
    assert result.probabilities["hawkish"] == pytest.approx(1 / 3)


@pytest.mark.unit
def test_from_dict_round_trips_through_to_inference() -> None:
    payload = {
        "classes": list(TONE_CLASSES),
        "ngram_max": 1,
        "sublinear_tf": True,
        "vocabulary": {"hawk": 0, "dove": 1},
        "idf": [1.0, 1.0],
        "coef": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
        "intercept": [0.0, 0.0, 0.0],
    }
    model = ToneClassifier.from_dict(payload)
    assert model.score("hawk").label == "hawkish"


@pytest.mark.unit
def test_constructor_rejects_inconsistent_shapes() -> None:
    with pytest.raises(ToneModelError):
        ToneClassifier(
            vocabulary={"a": 0, "b": 1},
            idf=np.array([1.0]),  # wrong length
            coef=np.zeros((2, 3)),
            intercept=np.zeros(3),
            ngram_max=1,
            sublinear_tf=True,
        )


@pytest.mark.unit
def test_from_dict_raises_on_malformed_payload() -> None:
    with pytest.raises(ToneModelError):
        ToneClassifier.from_dict({"vocabulary": {"a": 0}})  # missing idf/coef/intercept


@pytest.mark.unit
def test_from_artifact_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ToneModelError):
        ToneClassifier.from_artifact(tmp_path / "nope.json")


@pytest.mark.unit
def test_from_artifact_loads_written_model(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps(
            {
                "classes": list(TONE_CLASSES),
                "ngram_max": 1,
                "sublinear_tf": True,
                "vocabulary": {"hawk": 0, "dove": 1},
                "idf": [1.0, 1.0],
                "coef": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
                "intercept": [0.0, 0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )
    assert ToneClassifier.from_artifact(path).score("dove").label == "dovish"


@pytest.mark.unit
def test_bundled_model_separates_clear_hawkish_from_dovish() -> None:
    # A behavioural check on the shipped trained model: it should lean the right way on
    # unambiguous sentences (not a precise-probability assertion, which would be brittle).
    model = ToneClassifier.load_default()
    hawkish = model.score(
        "The committee will raise interest rates and tighten policy to fight persistent inflation."
    )
    dovish = model.score(
        "With downside risks and economic slack, we will cut rates and ease policy to support growth."
    )
    assert hawkish.probabilities["hawkish"] > hawkish.probabilities["dovish"]
    assert dovish.probabilities["dovish"] > dovish.probabilities["hawkish"]
    assert hawkish.score > dovish.score

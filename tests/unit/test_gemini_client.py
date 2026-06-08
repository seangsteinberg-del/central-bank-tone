"""Tests for the Gemini client, mocking the google-genai SDK (no network, CLAUDE.md section 5)."""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from google.genai import errors as genai_errors
from pydantic import SecretStr

from cbt_core.analysis import Aspect, Horizon, StanceLabel
from cbt_core.domain.analysis import ToneAnalysis
from cbt_core.domain.qa import EMBEDDING_DIM, RetrievedChunk
from cbt_core.domain.tone import ToneLabel
from cbt_core.exceptions import LlmError
from cbt_core.llm.gemini import (
    _EMBED_BATCH_SIZE,
    GeminiClient,
    _with_backoff,
    build_gemini_client,
)
from cbt_core.settings import Settings

_ANALYSIS = ToneAnalysis(
    summary="Policy will stay restrictive.",
    tone=ToneLabel.HAWKISH,
    score=0.7,
    rationale="Emphasis on inflation persistence.",
)
_CHUNK = RetrievedChunk(
    speech_id=UUID(int=1),
    chunk_index=0,
    text="excerpt",
    title="Speech",
    url="https://example.org/s/1",
    distance=0.1,
)


def _client() -> MagicMock:
    return MagicMock()


def _gemini(client: MagicMock) -> GeminiClient:
    return GeminiClient(client, model="gemini-2.5-flash", embedding_model="gemini-embedding-001")


# --- analyze_tone ----------------------------------------------------------------------------


@pytest.mark.unit
def test_analyze_tone_returns_the_parsed_analysis() -> None:
    client = _client()
    # Gemini returns JSON text (an explicit schema, not the Pydantic model, is sent to avoid the
    # additionalProperties the model's extra="forbid" emits); the client validates it back.
    client.models.generate_content.return_value.text = _ANALYSIS.model_dump_json()
    result = _gemini(client).analyze_tone("a speech")
    assert result == _ANALYSIS
    assert client.models.generate_content.call_args.kwargs["model"] == "gemini-2.5-flash"


@pytest.mark.unit
@pytest.mark.parametrize("text", [None, "", "not json", '{"summary": "x"}'])
def test_analyze_tone_raises_llm_error_on_unparseable_response(text: object) -> None:
    client = _client()
    client.models.generate_content.return_value.text = text
    with pytest.raises(LlmError):
        _gemini(client).analyze_tone("a speech")


# --- classify_sentences ----------------------------------------------------------------------

_VERDICTS_JSON = json.dumps(
    [
        {"stance": "hawkish", "aspect": "inflation", "horizon": "forward"},
        {"stance": "dovish", "aspect": "growth", "horizon": "backward"},
    ]
)


@pytest.mark.unit
def test_classify_sentences_maps_each_verdict_to_a_classified_sentence() -> None:
    client = _client()
    client.models.generate_content.return_value.text = _VERDICTS_JSON
    result = _gemini(client).classify_sentences(["We will hike.", "Growth slowed."])
    assert [item.label for item in result] == [StanceLabel.HAWKISH, StanceLabel.DOVISH]
    assert result[0].text == "We will hike."
    assert result[0].aspect is Aspect.INFLATION
    assert result[0].horizon is Horizon.FORWARD
    assert result[1].aspect is Aspect.GROWTH
    assert result[1].horizon is Horizon.BACKWARD


@pytest.mark.unit
def test_classify_sentences_empty_input_makes_no_call() -> None:
    client = _client()
    assert _gemini(client).classify_sentences([]) == []
    client.models.generate_content.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("text", [None, ""])
def test_classify_sentences_raises_on_empty_response(text: object) -> None:
    client = _client()
    client.models.generate_content.return_value.text = text
    with pytest.raises(LlmError):
        _gemini(client).classify_sentences(["a policy sentence"])


@pytest.mark.unit
@pytest.mark.parametrize("text", ["not json", '[{"stance": "banana"}]'])
def test_classify_sentences_raises_on_unparseable_response(text: str) -> None:
    client = _client()
    client.models.generate_content.return_value.text = text
    with pytest.raises(LlmError):
        _gemini(client).classify_sentences(["a policy sentence"])


@pytest.mark.unit
def test_classify_sentences_tolerates_minor_count_drift() -> None:
    client = _client()
    # Nine verdicts for ten sentences (a small drift): align to the shorter list, do not fail.
    nine = json.dumps([{"stance": "hawkish", "aspect": "other", "horizon": "forward"}] * 9)
    client.models.generate_content.return_value.text = nine
    result = _gemini(client).classify_sentences([f"sentence {i}" for i in range(10)])
    assert len(result) == 9
    assert all(item.label is StanceLabel.HAWKISH for item in result)


@pytest.mark.unit
def test_classify_sentences_rejects_a_wildly_misaligned_response() -> None:
    client = _client()
    client.models.generate_content.return_value.text = _VERDICTS_JSON  # two verdicts
    with pytest.raises(LlmError, match="were sent"):
        _gemini(client).classify_sentences([f"sentence {i}" for i in range(10)])


# --- embed -----------------------------------------------------------------------------------


@pytest.mark.unit
def test_embed_returns_one_normalized_vector_per_text() -> None:
    client = _client()
    # gemini-embedding-001 is not unit-length at a reduced dimensionality; embed must normalize so
    # the cosine-distance retrievers (which take a dot product as a cosine) work correctly.
    vector = [0.1] * EMBEDDING_DIM
    client.models.embed_content.return_value.embeddings = [
        MagicMock(values=vector),
        MagicMock(values=vector),
    ]
    result = _gemini(client).embed(["a", "b"])
    assert len(result) == 2
    for vec in result:
        assert len(vec) == EMBEDDING_DIM
        assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, rel_tol=1e-9)
    assert client.models.embed_content.call_args.kwargs["model"] == "gemini-embedding-001"


@pytest.mark.unit
def test_embed_empty_input_makes_no_call() -> None:
    client = _client()
    assert _gemini(client).embed([]) == []
    client.models.embed_content.assert_not_called()


@pytest.mark.unit
def test_embed_wrong_count_raises_llm_error() -> None:
    client = _client()
    client.models.embed_content.return_value.embeddings = [MagicMock(values=[0.1])]
    with pytest.raises(LlmError):
        _gemini(client).embed(["a", "b"])


@pytest.mark.unit
def test_embed_missing_values_raises_llm_error() -> None:
    client = _client()
    client.models.embed_content.return_value.embeddings = [MagicMock(values=None)]
    with pytest.raises(LlmError):
        _gemini(client).embed(["a"])


@pytest.mark.unit
def test_embed_batches_requests_over_the_endpoint_limit() -> None:
    # A long speech chunks past the endpoint's per-request cap; embed must split the input into
    # batches and return one vector per text in order, rather than failing the whole speech.
    client = _client()
    vector = [0.1] * EMBEDDING_DIM

    def _embed_one_per_text(**kwargs: object) -> MagicMock:
        contents = kwargs["contents"]
        assert isinstance(contents, list)
        assert len(contents) <= _EMBED_BATCH_SIZE
        return MagicMock(embeddings=[MagicMock(values=vector) for _ in contents])

    client.models.embed_content.side_effect = _embed_one_per_text
    texts = [f"chunk {i}" for i in range(_EMBED_BATCH_SIZE + 16)]
    result = _gemini(client).embed(texts)

    assert len(result) == len(texts)
    assert client.models.embed_content.call_count == 2  # one full batch plus the remainder


# --- retry/backoff -------------------------------------------------------------------------


@pytest.mark.unit
def test_with_backoff_wraps_a_non_retryable_api_error_in_llm_error() -> None:
    # A model failure that is non-retryable (or outlasts the retries) must reach adapters as an
    # LlmError (a CbtError they translate), never a raw google APIError that bypasses their handlers
    # and surfaces as an unhandled 500 mid-demo (CLAUDE.md section 3).
    def _boom() -> None:
        raise genai_errors.APIError(400, {"error": {"message": "bad request", "code": 400}})

    with pytest.raises(LlmError):
        _with_backoff(_boom, operation="analyze_tone")


@pytest.mark.unit
def test_transcribe_image_returns_the_transcribed_text() -> None:
    client = _client()
    client.models.generate_content.return_value.text = "Transcribed speech text."
    assert _gemini(client).transcribe_image(b"\x89PNG fake-bytes") == "Transcribed speech text."


@pytest.mark.unit
def test_transcribe_image_returns_empty_string_when_blank() -> None:
    client = _client()
    client.models.generate_content.return_value.text = None
    assert _gemini(client).transcribe_image(b"\x89PNG fake-bytes") == ""


# --- answer ----------------------------------------------------------------------------------


@pytest.mark.unit
def test_answer_returns_the_generated_text() -> None:
    client = _client()
    client.models.generate_content.return_value.text = "a grounded answer"
    assert _gemini(client).answer("question?", [_CHUNK]) == "a grounded answer"


@pytest.mark.unit
def test_answer_empty_response_raises_llm_error() -> None:
    client = _client()
    client.models.generate_content.return_value.text = ""
    with pytest.raises(LlmError):
        _gemini(client).answer("question?", [_CHUNK])


# --- build -----------------------------------------------------------------------------------


@pytest.mark.unit
def test_build_gemini_client_uses_settings() -> None:
    # genai.Client construction is lazy (no network); this just wires the models and key.
    settings = Settings(
        _env_file=None,
        gemini_api_key=SecretStr("dummy-key-for-tests"),
        gemini_model="gemini-2.5-flash",
    )
    client = build_gemini_client(settings)
    assert isinstance(client, GeminiClient)
    assert client._model == "gemini-2.5-flash"
    assert client._embedding_model == "gemini-embedding-001"

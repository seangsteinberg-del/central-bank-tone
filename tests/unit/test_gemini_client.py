"""Tests for the Gemini client, mocking the google-genai SDK (no network, CLAUDE.md section 5)."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import UUID

import pytest
from pydantic import SecretStr

from cbt_core.domain.analysis import ToneAnalysis
from cbt_core.domain.qa import EMBEDDING_DIM, RetrievedChunk
from cbt_core.domain.tone import ToneLabel
from cbt_core.exceptions import LlmError
from cbt_core.llm.gemini import GeminiClient, build_gemini_client
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


# --- embed -----------------------------------------------------------------------------------


@pytest.mark.unit
def test_embed_returns_one_vector_per_text() -> None:
    client = _client()
    vector = [0.1] * EMBEDDING_DIM
    client.models.embed_content.return_value.embeddings = [
        MagicMock(values=vector),
        MagicMock(values=vector),
    ]
    result = _gemini(client).embed(["a", "b"])
    assert result == [vector, vector]
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

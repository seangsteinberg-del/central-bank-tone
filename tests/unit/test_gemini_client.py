"""Tests for the Gemini client, mocking the google-genai SDK (no network, CLAUDE.md section 5)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from cbt_core.domain.analysis import ToneAnalysis
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


def _client_returning(parsed: object) -> MagicMock:
    client = MagicMock()
    client.models.generate_content.return_value.parsed = parsed
    return client


@pytest.mark.unit
def test_analyze_tone_returns_the_parsed_analysis() -> None:
    sdk = _client_returning(_ANALYSIS)
    result = GeminiClient(sdk, model="gemini-2.5-flash").analyze_tone("a speech")
    assert result == _ANALYSIS
    sdk.models.generate_content.assert_called_once()
    assert sdk.models.generate_content.call_args.kwargs["model"] == "gemini-2.5-flash"


@pytest.mark.unit
@pytest.mark.parametrize("parsed", [None, {"summary": "x"}, "not a model"])
def test_analyze_tone_raises_llm_error_on_unparseable_response(parsed: object) -> None:
    sdk = _client_returning(parsed)
    with pytest.raises(LlmError):
        GeminiClient(sdk, model="gemini-2.5-flash").analyze_tone("a speech")


@pytest.mark.unit
def test_build_gemini_client_uses_settings() -> None:
    # genai.Client construction is lazy (no network); this just wires the model and key.
    settings = Settings(
        _env_file=None,
        gemini_api_key=SecretStr("dummy-key-for-tests"),
        gemini_model="gemini-2.5-flash",
    )
    client = build_gemini_client(settings)
    assert isinstance(client, GeminiClient)
    assert client._model == "gemini-2.5-flash"

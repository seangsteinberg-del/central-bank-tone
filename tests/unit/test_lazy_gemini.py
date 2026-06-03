"""Tests for the lazy Gemini client that lets the app boot without a key (CLAUDE.md section 5)."""

from __future__ import annotations

import pytest
from tests._stubs import StubLlmClient

from cbt_core import LazyGeminiClient, LlmClient, Settings
from cbt_core.exceptions import LlmError


@pytest.mark.unit
def test_lazy_client_builds_once_on_first_use_and_delegates(
    dummy_settings: Settings, stub_llm_client: StubLlmClient
) -> None:
    builds = {"count": 0}

    def builder(_settings: Settings) -> LlmClient:
        builds["count"] += 1
        return stub_llm_client

    lazy = LazyGeminiClient(dummy_settings, builder=builder)
    assert builds["count"] == 0  # constructing the lazy client does not build the real one

    lazy.analyze_tone("a speech")
    lazy.embed(["one", "two"])
    lazy.answer("a question?", [])

    assert builds["count"] == 1  # built once, then cached
    assert stub_llm_client.calls == ["a speech"]
    assert stub_llm_client.answers == 1


@pytest.mark.unit
def test_lazy_client_raises_llm_error_when_the_key_is_missing(dummy_settings: Settings) -> None:
    def failing_builder(_settings: Settings) -> LlmClient:
        raise ValueError("No API key was provided")

    lazy = LazyGeminiClient(dummy_settings, builder=failing_builder)
    with pytest.raises(LlmError, match="CBT_GEMINI_API_KEY"):
        lazy.analyze_tone("a speech")

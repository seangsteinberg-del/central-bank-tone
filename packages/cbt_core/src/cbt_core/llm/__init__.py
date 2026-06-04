"""The LLM boundary and its Google Gemini implementation (ADR 0006).

Services depend on :class:`LlmClient`; ``build_gemini_client`` provides the concrete Gemini
client. Adapters never call Gemini directly.
"""

from __future__ import annotations

from cbt_core.llm.client import LlmClient
from cbt_core.llm.gemini import GeminiClient, LazyGeminiClient, build_gemini_client
from cbt_core.llm.offline import OfflineLlmClient

__all__ = [
    "GeminiClient",
    "LazyGeminiClient",
    "LlmClient",
    "OfflineLlmClient",
    "build_gemini_client",
]

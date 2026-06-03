"""Google Gemini implementation of the LLM boundary (ADR 0006).

The only model provider. ``GeminiClient`` takes an injected ``google-genai`` client so it can be
unit tested without the network, and ``build_gemini_client`` wires one from settings. A missing
or malformed model response raises :class:`LlmError` rather than fabricating a result.
"""

from __future__ import annotations

from google import genai
from google.genai import types

from cbt_core.domain.analysis import ToneAnalysis
from cbt_core.exceptions import LlmError
from cbt_core.logging import get_logger
from cbt_core.settings import Settings

_logger = get_logger(__name__)

_SYSTEM_INSTRUCTION = (
    "You are a central bank communications analyst. Read the speech and return a concise "
    "summary, the monetary-policy tone as one of hawkish, dovish, neutral, or mixed, a "
    "continuous score from -1.0 (most dovish) to 1.0 (most hawkish), and a one-line rationale. "
    "Judge only what the text supports; do not speculate beyond it."
)


class GeminiClient:
    """An :class:`~cbt_core.llm.client.LlmClient` backed by Google Gemini."""

    def __init__(self, client: genai.Client, *, model: str) -> None:
        """Build the client.

        Args:
            client: A configured ``google-genai`` client.
            model: The Gemini model id, for example ``gemini-2.5-flash``.
        """
        self._client = client
        self._model = model

    def analyze_tone(self, speech_text: str) -> ToneAnalysis:
        """Summarize a speech and judge its tone via Gemini structured output.

        Args:
            speech_text: The full text of the speech.

        Returns:
            The model's :class:`ToneAnalysis`.

        Raises:
            LlmError: If Gemini returns no parsed result or one of the wrong shape.
        """
        response = self._client.models.generate_content(
            model=self._model,
            contents=speech_text,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=ToneAnalysis,
                temperature=0.2,
            ),
        )
        parsed = response.parsed
        if not isinstance(parsed, ToneAnalysis):
            _logger.error("gemini_unparseable_response", model=self._model)
            raise LlmError("Gemini returned no parseable ToneAnalysis")
        _logger.info(
            "gemini_tone_analyzed",
            model=self._model,
            tone=parsed.tone.value,
            score=parsed.score,
            summary_chars=len(parsed.summary),
        )
        return parsed


def build_gemini_client(settings: Settings) -> GeminiClient:
    """Build a :class:`GeminiClient` from settings.

    Args:
        settings: Application settings holding the Gemini API key and model.

    Returns:
        A configured :class:`GeminiClient`.
    """
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    return GeminiClient(client, model=settings.gemini_model)

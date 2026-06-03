"""Google Gemini implementation of the LLM boundary (ADR 0006).

The only model provider. ``GeminiClient`` takes an injected ``google-genai`` client so it can be
unit tested without the network, and ``build_gemini_client`` wires one from settings. A missing
or malformed model response raises :class:`LlmError` rather than fabricating a result.
"""

from __future__ import annotations

from collections.abc import Sequence

from google import genai
from google.genai import types

from cbt_core.domain.analysis import ToneAnalysis
from cbt_core.domain.qa import RetrievedChunk
from cbt_core.exceptions import LlmError
from cbt_core.llm.client import EMBEDDING_DIM, Embedding
from cbt_core.logging import get_logger
from cbt_core.settings import Settings

_logger = get_logger(__name__)

_SYSTEM_INSTRUCTION = (
    "You are a central bank communications analyst. Read the speech and return a concise "
    "summary, the monetary-policy tone as one of hawkish, dovish, neutral, or mixed, a "
    "continuous score from -1.0 (most dovish) to 1.0 (most hawkish), and a one-line rationale. "
    "Judge only what the text supports; do not speculate beyond it."
)

_ANSWER_INSTRUCTION = (
    "You answer questions about a central bank speaker using only the provided excerpts from "
    "their speeches. Ground every statement in the excerpts and do not use outside knowledge. "
    "If the excerpts do not contain the answer, say so plainly."
)


class GeminiClient:
    """An :class:`~cbt_core.llm.client.LlmClient` backed by Google Gemini."""

    def __init__(self, client: genai.Client, *, model: str, embedding_model: str) -> None:
        """Build the client.

        Args:
            client: A configured ``google-genai`` client.
            model: The Gemini generative model id, for example ``gemini-2.5-flash``.
            embedding_model: The Gemini embedding model id, for example
                ``gemini-embedding-001``.
        """
        self._client = client
        self._model = model
        self._embedding_model = embedding_model

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

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        """Embed texts with the Gemini embedding model.

        Args:
            texts: The texts to embed.

        Returns:
            One :data:`EMBEDDING_DIM`-dimensional vector per input text, in order.

        Raises:
            LlmError: If Gemini returns the wrong number of embeddings or an empty vector.
        """
        if not texts:
            return []
        response = self._client.models.embed_content(
            model=self._embedding_model,
            # google-genai types `contents` as an invariant list union, so a plain list[str]
            # is rejected by mypy though it is a valid runtime input.
            contents=list(texts),  # type: ignore[arg-type]
            config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
        )
        embeddings = response.embeddings
        if embeddings is None or len(embeddings) != len(texts):
            raise LlmError("Gemini returned an unexpected number of embeddings")
        vectors: list[Embedding] = []
        for item in embeddings:
            if item.values is None:
                raise LlmError("Gemini returned an embedding with no values")
            vectors.append(list(item.values))
        return vectors

    def answer(self, question: str, chunks: Sequence[RetrievedChunk]) -> str:
        """Answer a question grounded only in the retrieved chunks.

        Args:
            question: The user's question.
            chunks: The retrieved context to ground the answer in.

        Returns:
            The grounded answer text.

        Raises:
            LlmError: If Gemini returns an empty answer.
        """
        context = "\n\n".join(
            f"[Speech {chunk.speech_id} | {chunk.title}]\n{chunk.text}" for chunk in chunks
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=f"Question: {question}\n\nExcerpts:\n{context}",
            config=types.GenerateContentConfig(
                system_instruction=_ANSWER_INSTRUCTION, temperature=0.2
            ),
        )
        if not response.text:
            raise LlmError("Gemini returned an empty answer")
        return response.text


def build_gemini_client(settings: Settings) -> GeminiClient:
    """Build a :class:`GeminiClient` from settings.

    Args:
        settings: Application settings holding the Gemini API key and models.

    Returns:
        A configured :class:`GeminiClient`.
    """
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    return GeminiClient(
        client, model=settings.gemini_model, embedding_model=settings.gemini_embedding_model
    )

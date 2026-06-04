"""Google Gemini implementation of the LLM boundary (ADR 0006).

The only model provider. ``GeminiClient`` takes an injected ``google-genai`` client so it can be
unit tested without the network, and ``build_gemini_client`` wires one from settings. A missing
or malformed model response raises :class:`LlmError` rather than fabricating a result.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from google import genai
from google.genai import types
from pydantic import ValidationError

from cbt_core.domain.analysis import ToneAnalysis
from cbt_core.domain.qa import RetrievedChunk
from cbt_core.domain.tone import ToneLabel
from cbt_core.exceptions import LlmError
from cbt_core.llm.client import EMBEDDING_DIM, Embedding, LlmClient
from cbt_core.logging import get_logger
from cbt_core.settings import Settings

_logger = get_logger(__name__)

# An explicit response schema for structured tone output. We do NOT hand Gemini the Pydantic
# model directly: ToneAnalysis uses ``extra="forbid"``, which makes Pydantic emit
# ``additionalProperties`` in its JSON schema, and the Gemini ``response_schema`` field rejects
# that (HTTP 400). This schema is the Gemini-compatible subset; the response JSON is then
# validated back into ToneAnalysis, so the domain model's strict validation still applies.
_TONE_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "summary": types.Schema(type=types.Type.STRING),
        "tone": types.Schema(type=types.Type.STRING, enum=[label.value for label in ToneLabel]),
        "score": types.Schema(type=types.Type.NUMBER),
        "rationale": types.Schema(type=types.Type.STRING),
    },
    required=["summary", "tone", "score", "rationale"],
    property_ordering=["summary", "tone", "score", "rationale"],
)

_SYSTEM_INSTRUCTION = (
    "You are a central bank communications analyst. Read the speech and return a concise "
    "summary, the monetary-policy tone as one of hawkish, dovish, neutral, or mixed, a "
    "continuous score, and a one-line rationale. Anchor the score on this scale so it is "
    "comparable across speeches: +1.0 is unambiguously hawkish (urging tighter policy: rate "
    "hikes, withdrawing accommodation, fighting inflation); -1.0 is unambiguously dovish (urging "
    "easier policy: rate cuts, stimulus, supporting growth); 0.0 is balanced or procedural; "
    "intermediate values reflect the strength of the lean (for example +0.5 is moderately "
    "hawkish, -0.3 mildly dovish). Use 'mixed' only when the speech makes strong arguments in "
    "both directions. Judge only what the text supports; do not speculate beyond it."
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
                response_schema=_TONE_RESPONSE_SCHEMA,
                # Greedy decoding so the tone score is reproducible: the same speech scores the
                # same way across runs, which a non-zero temperature would not guarantee.
                temperature=0.0,
            ),
        )
        text = response.text
        if not text:
            _logger.error("gemini_empty_response", model=self._model)
            raise LlmError("Gemini returned an empty tone analysis")
        try:
            parsed = ToneAnalysis.model_validate_json(text)
        except ValidationError as exc:
            # Deliberately no exc_info: the ValidationError detail echoes the model's response,
            # and we do not log external payloads (CLAUDE.md section 7).
            _logger.error("gemini_unparseable_response", model=self._model)  # noqa: TRY400
            raise LlmError("Gemini returned an unparseable ToneAnalysis") from exc
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


class LazyGeminiClient:
    """An :class:`~cbt_core.llm.client.LlmClient` that builds the real client on first use.

    This lets the application start without a Gemini API key: browsing speakers and tone history
    needs no model, so only the operations that actually call Gemini (ingest, index, ask) fail,
    and only when invoked, with an explicit :class:`LlmError` naming the missing key (CLAUDE.md
    section 3, no silent fallback). In production the settings validator already requires a real
    key, so the first use succeeds.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        builder: Callable[[Settings], LlmClient] = build_gemini_client,
    ) -> None:
        """Build the lazy client.

        Args:
            settings: Application settings holding the Gemini API key and models.
            builder: Factory for the real client, called once on first use. Injectable for tests.
        """
        self._settings = settings
        self._builder = builder
        self._delegate: LlmClient | None = None

    def _client(self) -> LlmClient:
        """Return the underlying client, building it on first use."""
        if self._delegate is None:
            try:
                self._delegate = self._builder(self._settings)
            except ValueError as exc:
                raise LlmError(
                    "Gemini is not configured; set CBT_GEMINI_API_KEY to use model features"
                ) from exc
        return self._delegate

    def analyze_tone(self, speech_text: str) -> ToneAnalysis:
        """Summarize a speech and judge its tone (see :meth:`GeminiClient.analyze_tone`)."""
        return self._client().analyze_tone(speech_text)

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        """Embed texts (see :meth:`GeminiClient.embed`)."""
        return self._client().embed(texts)

    def answer(self, question: str, chunks: Sequence[RetrievedChunk]) -> str:
        """Answer a question grounded in chunks (see :meth:`GeminiClient.answer`)."""
        return self._client().answer(question, chunks)

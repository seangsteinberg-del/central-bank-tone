"""Google Gemini implementation of the LLM boundary (ADR 0006).

The only model provider. ``GeminiClient`` takes an injected ``google-genai`` client so it can be
unit tested without the network, and ``build_gemini_client`` wires one from settings. A missing
or malformed model response raises :class:`LlmError` rather than fabricating a result.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from cbt_core.analysis.stance import Aspect, ClassifiedSentence, Horizon, StanceLabel
from cbt_core.domain.analysis import ToneAnalysis
from cbt_core.domain.qa import RetrievedChunk
from cbt_core.domain.tone import ToneLabel
from cbt_core.exceptions import LlmError
from cbt_core.llm.client import EMBEDDING_DIM, Embedding, LlmClient
from cbt_core.logging import get_logger
from cbt_core.settings import Settings

_logger = get_logger(__name__)

# Transient Gemini errors worth retrying: rate limits (429) and server errors (5xx). A 4xx other
# than 429 is a request problem and is not retried.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 6
_BACKOFF_BASE_SECONDS = 2.0

# The batched sentence classifier aligns the model's verdicts to the input sentences by position.
# Gemini usually returns exactly one verdict per sentence but occasionally drifts by a few; an
# overlap below this fraction means the response is too misaligned to trust and is rejected.
_MIN_ALIGN_RATIO = 0.8


def _with_backoff[T](call: Callable[[], T], *, operation: str) -> T:
    """Call ``call``, retrying transient Gemini errors with exponential backoff.

    A large ingestion makes many Gemini calls and the free tier rate-limits (HTTP 429), so a single
    throttled call is retried rather than failing the speech. A non-retryable error (a 4xx other
    than 429) is raised immediately.

    Args:
        call: The zero-argument Gemini call to make.
        operation: A short label for logging (for example ``"analyze_tone"``).

    Returns:
        The call's result.

    Raises:
        google.genai.errors.APIError: If the call keeps failing after the last attempt, or fails
            with a non-retryable status.
    """
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return call()
        except genai_errors.APIError as exc:
            last_attempt = attempt == _MAX_ATTEMPTS - 1
            if exc.code not in _RETRYABLE_STATUS or last_attempt:
                raise
            delay = _BACKOFF_BASE_SECONDS * (2**attempt)
            _logger.warning(
                "gemini_retry",
                operation=operation,
                status=exc.code,
                attempt=attempt + 1,
                delay=delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable: the loop returns or raises on the final attempt")


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

# Schema for the batched sentence classifier (ADR 0021): a JSON array with one object per input
# sentence. Enums are the domain stance, aspect, and horizon vocabularies, so Gemini cannot return a
# value outside the schema spine.
_SENTENCE_STANCE_SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "stance": types.Schema(type=types.Type.STRING, enum=[s.value for s in StanceLabel]),
            "aspect": types.Schema(type=types.Type.STRING, enum=[a.value for a in Aspect]),
            "horizon": types.Schema(type=types.Type.STRING, enum=[h.value for h in Horizon]),
        },
        required=["stance", "aspect", "horizon"],
        property_ordering=["stance", "aspect", "horizon"],
    ),
)

_CLASSIFY_INSTRUCTION = (
    "You are a central bank communications analyst. You are given a numbered list of sentences from "
    "a central bank speech. For each sentence, classify three things. (1) Monetary-policy stance: "
    "'hawkish' if it argues for or signals tighter policy (higher rates, withdrawing accommodation, "
    "fighting inflation), 'dovish' for easier policy (lower rates, stimulus, supporting growth), or "
    "'neutral' if it takes no policy direction. (2) Aspect: the policy topic it concerns, one of "
    "inflation, growth, employment, balance_sheet, financial_stability, guidance, or other. "
    "(3) Horizon: 'forward' if it states intent or expectation about future policy or the economy, "
    "'backward' if it describes what already happened, or 'unspecified'. Return a JSON array with "
    "exactly one object per input sentence, in the same order. Judge only what each sentence "
    "supports."
)


class _SentenceVerdict(BaseModel):
    """One sentence's classification as Gemini returns it, before mapping to a domain type."""

    model_config = ConfigDict(extra="ignore")

    stance: StanceLabel
    aspect: Aspect = Aspect.OTHER
    horizon: Horizon = Horizon.UNSPECIFIED


_VERDICTS_ADAPTER = TypeAdapter(list[_SentenceVerdict])

_ANSWER_INSTRUCTION = (
    "You answer questions about a central bank speaker using only the provided excerpts from "
    "their speeches. Ground every statement in the excerpts and do not use outside knowledge. "
    "If the excerpts do not contain the answer, say so plainly."
)

_TRANSCRIBE_INSTRUCTION = (
    "Transcribe the speech text in this page image verbatim. Output only the transcribed text, "
    "in reading order, with no commentary, headers, footers, or page numbers."
)


def _l2_normalize(values: list[float]) -> list[float]:
    """Scale a vector to unit length.

    ``gemini-embedding-001`` does not return unit vectors at a reduced ``output_dimensionality``,
    but the cosine-distance retrievers treat embeddings as L2-normalized (so a dot product is a
    cosine similarity), as does the offline client. Normalizing here keeps that contract. A zero
    vector is returned unchanged.

    Args:
        values: The raw embedding vector.

    Returns:
        The vector scaled to unit L2 norm.
    """
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


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
        response = _with_backoff(
            lambda: self._client.models.generate_content(
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
            ),
            operation="analyze_tone",
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

    def classify_sentences(self, sentences: Sequence[str]) -> list[ClassifiedSentence]:
        """Classify each policy-relevant sentence's stance, aspect, and horizon in one call.

        Args:
            sentences: The policy-relevant sentences, already filtered, in order.

        Returns:
            One :class:`~cbt_core.analysis.ClassifiedSentence` per input sentence, in order. An empty
            input returns an empty list with no model call.

        Raises:
            LlmError: If Gemini returns an empty response, an unparseable one, or a number of
                verdicts that does not match the input sentences.
        """
        if not sentences:
            return []
        numbered = "\n".join(f"{index + 1}. {sentence}" for index, sentence in enumerate(sentences))
        response = _with_backoff(
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=numbered,
                config=types.GenerateContentConfig(
                    system_instruction=_CLASSIFY_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=_SENTENCE_STANCE_SCHEMA,
                    # Greedy decoding so the same speech classifies the same way across runs.
                    temperature=0.0,
                ),
            ),
            operation="classify_sentences",
        )
        text = response.text
        if not text:
            _logger.error("gemini_empty_response", model=self._model)
            raise LlmError("Gemini returned an empty sentence classification")
        try:
            verdicts = _VERDICTS_ADAPTER.validate_json(text)
        except ValidationError as exc:
            # No exc_info: the detail echoes the model's response, which we do not log (section 7).
            _logger.error("gemini_unparseable_response", model=self._model)  # noqa: TRY400
            raise LlmError("Gemini returned an unparseable sentence classification") from exc
        if not verdicts:
            raise LlmError("Gemini returned no sentence classifications")
        # Gemini occasionally returns a few more or fewer verdicts than sentences (it re-splits on an
        # abbreviation or a decimal). The aggregate is count-based and does not depend on exact text
        # alignment, so a minor drift is tolerated by aligning to the shorter list; a wildly
        # misaligned response (less than _MIN_ALIGN_RATIO overlap) is rejected rather than trusted.
        shorter, longer = sorted((len(verdicts), len(sentences)))
        if shorter < _MIN_ALIGN_RATIO * longer:
            raise LlmError(
                f"Gemini classified {len(verdicts)} sentences but {len(sentences)} were sent"
            )
        if len(verdicts) != len(sentences):
            _logger.warning(
                "gemini_sentence_count_drift", sent=len(sentences), classified=len(verdicts)
            )
        classified = [
            ClassifiedSentence(
                text=sentence, label=verdict.stance, aspect=verdict.aspect, horizon=verdict.horizon
            )
            for sentence, verdict in zip(sentences, verdicts, strict=False)
        ]
        _logger.info(
            "gemini_sentences_classified",
            model=self._model,
            sentences=len(classified),
            hawkish=sum(1 for item in classified if item.label is StanceLabel.HAWKISH),
            dovish=sum(1 for item in classified if item.label is StanceLabel.DOVISH),
        )
        return classified

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
        response = _with_backoff(
            lambda: self._client.models.embed_content(
                model=self._embedding_model,
                # google-genai types `contents` as an invariant list union, so a plain list[str]
                # is rejected by mypy though it is a valid runtime input.
                contents=list(texts),  # type: ignore[arg-type]
                config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
            ),
            operation="embed",
        )
        embeddings = response.embeddings
        if embeddings is None or len(embeddings) != len(texts):
            raise LlmError("Gemini returned an unexpected number of embeddings")
        vectors: list[Embedding] = []
        for item in embeddings:
            if item.values is None:
                raise LlmError("Gemini returned an embedding with no values")
            vectors.append(_l2_normalize(list(item.values)))
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
        response = _with_backoff(
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=f"Question: {question}\n\nExcerpts:\n{context}",
                config=types.GenerateContentConfig(
                    system_instruction=_ANSWER_INSTRUCTION, temperature=0.2
                ),
            ),
            operation="answer",
        )
        if not response.text:
            raise LlmError("Gemini returned an empty answer")
        return response.text

    def transcribe_image(self, image: bytes, *, mime_type: str = "image/png") -> str:
        """Transcribe the text in a page image with Gemini vision.

        Used to recover a speech's text from a PDF that no text extractor can decode (subset fonts
        with no ToUnicode map), by rendering its pages and reading the pixels (ADR 0019).

        Args:
            image: The page image bytes.
            mime_type: The image mime type (``image/png`` by default).

        Returns:
            The transcribed text, or an empty string if the page held none.
        """
        # google-genai types a multimodal `contents` list as an invariant union; a list of a Part
        # and a str is a valid runtime input but mypy infers list[object], so the ignore stands.
        contents = [types.Part.from_bytes(data=image, mime_type=mime_type), _TRANSCRIBE_INSTRUCTION]
        response = _with_backoff(
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=contents,  # type: ignore[arg-type]
                config=types.GenerateContentConfig(temperature=0.0),
            ),
            operation="transcribe_image",
        )
        return response.text or ""


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

    def classify_sentences(self, sentences: Sequence[str]) -> list[ClassifiedSentence]:
        """Classify sentences (see :meth:`GeminiClient.classify_sentences`)."""
        return self._client().classify_sentences(sentences)

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        """Embed texts (see :meth:`GeminiClient.embed`)."""
        return self._client().embed(texts)

    def answer(self, question: str, chunks: Sequence[RetrievedChunk]) -> str:
        """Answer a question grounded in chunks (see :meth:`GeminiClient.answer`)."""
        return self._client().answer(question, chunks)

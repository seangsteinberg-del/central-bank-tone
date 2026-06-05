"""The LLM boundary (CLAUDE.md section 2).

All generative model access goes through this protocol: tone analysis, embeddings, and grounded
question answering. Services depend on ``LlmClient``, not on a concrete provider, so the domain
stays testable without the network or a paid API and the provider can be swapped behind the
interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from cbt_core.analysis.stance import ClassifiedSentence
from cbt_core.domain.analysis import ToneAnalysis
from cbt_core.domain.qa import EMBEDDING_DIM, RetrievedChunk

# A single embedding vector of dimension EMBEDDING_DIM.
type Embedding = list[float]

__all__ = ["EMBEDDING_DIM", "Embedding", "LlmClient"]


@runtime_checkable
class LlmClient(Protocol):
    """Protocol for the generative model used to analyze speeches and answer questions.

    Runtime-checkable so a test can assert (via ``isinstance``) that every full test double and
    production client implements the whole surface; this guards against a double silently lagging
    behind a newly added method (ADR 0021 added :meth:`classify_sentences`).
    """

    def analyze_tone(self, speech_text: str) -> ToneAnalysis:
        """Summarize a speech and judge its monetary-policy tone.

        Args:
            speech_text: The full text of the speech.

        Returns:
            The model's :class:`ToneAnalysis`.

        Raises:
            LlmError: If the model call fails or returns an unusable response.
        """
        ...

    def classify_sentences(self, sentences: Sequence[str]) -> list[ClassifiedSentence]:
        """Classify each policy-relevant sentence's stance, aspect, and horizon (ADR 0021).

        The structured-pipeline counterpart to :meth:`analyze_tone`: instead of one tone for the
        whole speech, label each sentence so :func:`cbt_core.analysis.aggregate_stances` can build a
        normalized net-hawkishness measure, a forward-looking sub-measure, and a per-aspect
        breakdown. Implementations classify in one batched call, not one call per sentence.

        Args:
            sentences: The policy-relevant sentences, already filtered, in order.

        Returns:
            One :class:`~cbt_core.analysis.ClassifiedSentence` per input sentence, in the same
            order. An empty input returns an empty list and makes no model call.

        Raises:
            LlmError: If the model call fails or returns a result that does not align one-to-one
                with the input sentences.
        """
        ...

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        """Embed texts into vectors of dimension :data:`EMBEDDING_DIM`.

        Args:
            texts: The texts to embed.

        Returns:
            One embedding vector per input text, in order.

        Raises:
            LlmError: If the model call fails or returns the wrong number of vectors.
        """
        ...

    def answer(self, question: str, chunks: Sequence[RetrievedChunk]) -> str:
        """Answer a question grounded only in the retrieved chunks.

        Args:
            question: The user's question.
            chunks: The retrieved context the answer must be grounded in.

        Returns:
            The grounded answer text.

        Raises:
            LlmError: If the model call fails or returns an empty answer.
        """
        ...

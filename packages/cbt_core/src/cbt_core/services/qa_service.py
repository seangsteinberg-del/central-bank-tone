"""Question-answering service over a speaker's speeches (CLAUDE.md sections 2, 3, 7).

Embeds the question, retrieves the most relevant speech chunks for the speaker, and asks the LLM
to answer grounded in them with citations. When retrieval finds nothing, it abstains with a
reason rather than fabricating an answer (no silent fallback).
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from cbt_core.domain.qa import Answer, Citation, RetrievedChunk
from cbt_core.llm.client import Embedding, LlmClient
from cbt_core.logging import get_logger, resolve_correlation_id
from cbt_core.services.speaker_service import SpeakerService

_logger = get_logger(__name__)

_ABSTENTION = "I could not find anything in this speaker's speeches that addresses that question."
_CORPUS_ABSTENTION = (
    "I could not find anything in the speeches I have analyzed that addresses that question."
)


class ChunkRetriever(Protocol):
    """Retrieves the chunks most relevant to a query embedding, per speaker or corpus-wide."""

    def search(
        self, speaker_id: UUID, query_embedding: Embedding, top_k: int
    ) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks nearest to ``query_embedding`` for the speaker."""
        ...

    def search_all(self, query_embedding: Embedding, top_k: int) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks nearest to ``query_embedding`` across all speakers."""
        ...


class QaService:
    """Answer questions about a speaker, grounded in their speeches."""

    def __init__(
        self,
        llm_client: LlmClient,
        retriever: ChunkRetriever,
        speaker_service: SpeakerService,
        *,
        top_k: int = 5,
        max_distance: float = 0.6,
    ) -> None:
        """Build the service.

        Args:
            llm_client: The LLM boundary used to embed the question and answer.
            retriever: Retrieves the relevant chunks.
            speaker_service: Verifies the speaker exists.
            top_k: The maximum number of chunks to retrieve.
            max_distance: The maximum cosine distance for a chunk to count as relevant. Chunks
                beyond it are dropped before answering, so an off-topic question abstains rather
                than grounding in the nearest-but-irrelevant chunks (CLAUDE.md section 3).
        """
        self._llm = llm_client
        self._retriever = retriever
        self._speaker_service = speaker_service
        self._top_k = top_k
        self._max_distance = max_distance

    def _relevant(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Drop chunks beyond the relevance threshold so off-topic questions abstain."""
        return [chunk for chunk in chunks if chunk.distance <= self._max_distance]

    def answer(
        self,
        *,
        speaker_id: UUID,
        question: str,
        actor: str = "system",
        correlation_id: UUID | None = None,
    ) -> Answer:
        """Answer a question about a speaker from their speeches.

        Args:
            speaker_id: The speaker the question is about.
            question: The natural-language question.
            actor: Who is performing the action.
            correlation_id: Correlation id for this call; one is minted if not supplied.

        Returns:
            An :class:`Answer`. When no relevant chunk is found it abstains, with ``abstained``
            set and no citations.

        Raises:
            EntityNotFoundError: If the speaker does not exist.
            LlmError: If embedding or answering fails.
        """
        correlation = resolve_correlation_id(correlation_id)
        log = _logger.bind(correlation_id=str(correlation), actor=actor, speaker_id=str(speaker_id))
        self._speaker_service.get_speaker(speaker_id, actor=actor, correlation_id=correlation)

        query_embedding = self._llm.embed([question])[0]
        chunks = self._relevant(self._retriever.search(speaker_id, query_embedding, self._top_k))
        if not chunks:
            log.info("qa_abstained")
            return Answer(text=_ABSTENTION, citations=(), abstained=True)

        text = self._llm.answer(question, chunks)
        citations = tuple(Citation.from_chunk(chunk) for chunk in chunks)
        log.info("qa_answered", chunk_count=len(chunks))
        return Answer(text=text, citations=citations, abstained=False)

    def answer_corpus(
        self, *, question: str, actor: str = "system", correlation_id: UUID | None = None
    ) -> Answer:
        """Answer a question across every speaker's speeches (platform-wide).

        The corpus-wide counterpart to :meth:`answer`: it retrieves the most relevant chunks
        from the whole analyzed corpus rather than a single speaker, so the platform itself is
        natural-language queryable. It abstains the same way when retrieval finds nothing.

        Args:
            question: The natural-language question.
            actor: Who is performing the action.
            correlation_id: Correlation id for this call; one is minted if not supplied.

        Returns:
            An :class:`Answer`. When no relevant chunk is found it abstains, with ``abstained``
            set and no citations.

        Raises:
            LlmError: If embedding or answering fails.
        """
        correlation = resolve_correlation_id(correlation_id)
        log = _logger.bind(correlation_id=str(correlation), actor=actor)

        query_embedding = self._llm.embed([question])[0]
        chunks = self._relevant(self._retriever.search_all(query_embedding, self._top_k))
        if not chunks:
            log.info("qa_corpus_abstained")
            return Answer(text=_CORPUS_ABSTENTION, citations=(), abstained=True)

        text = self._llm.answer(question, chunks)
        citations = tuple(Citation.from_chunk(chunk) for chunk in chunks)
        log.info("qa_corpus_answered", chunk_count=len(chunks))
        return Answer(text=text, citations=citations, abstained=False)

"""Speech indexing service for retrieval (CLAUDE.md sections 2 and 7).

Chunks an ingested speech, embeds the chunks with the LLM boundary, and stores the vectors for
RAG. Idempotent: a speech that is already indexed is skipped, so re-running spends no model call.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from cbt_core.analysis.chunking import chunk_text
from cbt_core.exceptions import LlmError
from cbt_core.llm.client import LlmClient
from cbt_core.logging import get_logger, resolve_correlation_id
from cbt_core.persistence.repositories import SpeechChunkRepository, SpeechRepository
from cbt_core.services._support import IdFactory, default_id_factory

_logger = get_logger(__name__)


class IndexingService:
    """Chunk and embed speeches into the vector store for retrieval."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        llm_client: LlmClient,
        *,
        max_chars: int = 1000,
        overlap: int = 100,
        id_factory: IdFactory = default_id_factory,
    ) -> None:
        """Build the service.

        Args:
            session_factory: Factory for sessions. The service owns the transaction.
            llm_client: The LLM boundary used to embed chunks.
            max_chars: Maximum chunk length in characters.
            overlap: Overlap between consecutive chunks in characters.
            id_factory: Source of new chunk identifiers. Inject a deterministic one in tests.
        """
        self._session_factory = session_factory
        self._llm = llm_client
        self._max_chars = max_chars
        self._overlap = overlap
        self._id_factory = id_factory

    def index_speech(
        self, speech_id: UUID, *, actor: str = "system", correlation_id: UUID | None = None
    ) -> int:
        """Chunk, embed, and store a speech for retrieval.

        Args:
            speech_id: The speech to index.
            actor: Who is performing the action.
            correlation_id: Correlation id for this call; one is minted if not supplied.

        Returns:
            The number of chunks stored, or 0 if the speech was already indexed.

        Raises:
            EntityNotFoundError: If the speech does not exist.
            LlmError: If embedding fails or returns the wrong number of vectors.
        """
        correlation = resolve_correlation_id(correlation_id)
        log = _logger.bind(correlation_id=str(correlation), actor=actor, speech_id=str(speech_id))

        with self._session_factory() as session:
            speech = SpeechRepository(session).get(speech_id)
            if SpeechChunkRepository(session).has_chunks(speech_id):
                log.info("speech_already_indexed")
                return 0

        chunks = chunk_text(speech.text, max_chars=self._max_chars, overlap=self._overlap)
        embeddings = self._llm.embed(chunks)
        if len(embeddings) != len(chunks):
            # Honour the documented contract with a typed CbtError rather than a bare ValueError
            # from a strict zip (which would bypass the adapter error handlers, CLAUDE.md section 3).
            raise LlmError(f"embedding returned {len(embeddings)} vectors for {len(chunks)} chunks")

        with self._session_factory() as session:
            repository = SpeechChunkRepository(session)
            # Lengths are already validated equal above, so strict is unnecessary here.
            for index, (text, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
                repository.add_chunk(
                    chunk_id=self._id_factory(),
                    speech_id=speech_id,
                    chunk_index=index,
                    text=text,
                    embedding=embedding,
                )
            session.commit()

        log.info("speech_indexed", chunk_count=len(chunks))
        return len(chunks)

"""An in-process chunk retriever for the keyless, Docker-less demo (ADR 0014).

`SpeechRetriever` (in `repositories.py`) uses pgvector and therefore needs PostgreSQL. For a local
demo running on SQLite with no pgvector, `InMemoryChunkRetriever` implements the same
search/search_all contract by holding embeddings in memory and computing cosine distance in numpy.
It is also handy in tests that want a real retriever without a database. Nothing is persisted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cbt_core.domain.qa import RetrievedChunk

if TYPE_CHECKING:
    from uuid import UUID

    from cbt_core.llm.client import Embedding


class InMemoryChunkRetriever:
    """A ChunkRetriever backed by in-process vectors (cosine distance), no database required.

    Implements the same protocol as :class:`~cbt_core.persistence.repositories.SpeechRetriever`:
    ``search`` restricts to one speaker and ``search_all`` spans the corpus, both returning the
    nearest chunks first with their citation metadata. Distances are cosine distances in ``[0, 2]``
    computed against the stored embeddings.
    """

    def __init__(self) -> None:
        """Build an empty retriever."""
        self._speaker_ids: list[UUID] = []
        self._vectors: list[np.ndarray] = []
        self._chunks: list[RetrievedChunk] = []

    def add(
        self,
        *,
        speaker_id: UUID,
        speech_id: UUID,
        chunk_index: int,
        text: str,
        title: str,
        url: str,
        embedding: Embedding,
    ) -> None:
        """Index one chunk's embedding and citation metadata.

        Args:
            speaker_id: The speaker who gave the speech (for per-speaker retrieval).
            speech_id: The speech the chunk belongs to.
            chunk_index: The chunk's position within the speech.
            text: The chunk text.
            title: The speech title (for citation).
            url: The speech source URL (for citation).
            embedding: The chunk embedding (assumed L2-normalized).
        """
        self._speaker_ids.append(speaker_id)
        self._vectors.append(np.asarray(embedding, dtype=np.float64))
        # The stored distance is a placeholder; it is recomputed per query in _rank.
        self._chunks.append(
            RetrievedChunk(
                speech_id=speech_id,
                chunk_index=chunk_index,
                text=text,
                title=title,
                url=url,
                distance=0.0,
            )
        )

    def search(
        self, speaker_id: UUID, query_embedding: Embedding, top_k: int
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` nearest chunks for one speaker (nearest first)."""
        return self._rank(query_embedding, top_k, speaker_id=speaker_id)

    def search_all(self, query_embedding: Embedding, top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` nearest chunks across every speaker (nearest first)."""
        return self._rank(query_embedding, top_k, speaker_id=None)

    def _rank(
        self, query_embedding: Embedding, top_k: int, *, speaker_id: UUID | None
    ) -> list[RetrievedChunk]:
        """Score every (optionally speaker-filtered) chunk by cosine distance and take the top_k."""
        query = np.asarray(query_embedding, dtype=np.float64)
        scored: list[tuple[float, RetrievedChunk]] = []
        for owner, vector, chunk in zip(
            self._speaker_ids, self._vectors, self._chunks, strict=True
        ):
            if speaker_id is not None and owner != speaker_id:
                continue
            distance = 1.0 - float(vector @ query)
            scored.append((distance, chunk.model_copy(update={"distance": distance})))
        scored.sort(key=lambda item: item[0])
        return [chunk for _, chunk in scored[:top_k]]

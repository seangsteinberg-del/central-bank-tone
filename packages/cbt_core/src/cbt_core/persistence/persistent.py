"""A PostgreSQL-backed chunk retriever for the no-pgvector live setup (ADR 0020).

The keyless demo's :class:`~cbt_core.persistence.memory.InMemoryChunkRetriever` holds embeddings in
process and loses them on restart. The local live setup (ADR 0018) runs on a native PostgreSQL with
no pgvector extension, but retrieval still needs to be persistent so that a filled corpus and its
question-answering index survive restarts and the Gemini embedding is computed once, not on every
boot.

:class:`PersistentChunkRetriever` subclasses the in-memory retriever (reusing its brute-force cosine
search), stores each chunk and its embedding in an ordinary ``bytea`` column, loads every stored
chunk into memory on construction, and writes through on each add. Brute-force cosine over a
single-operator corpus (tens of thousands of chunks at most) is one numpy matmul per query, well
within budget; a high-cardinality deployment uses the pgvector retriever instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, override
from uuid import UUID

import numpy as np
from sqlalchemy import Integer, LargeBinary, String, Text, Uuid, select
from sqlalchemy.orm import Mapped, mapped_column

from cbt_core.persistence.base import Base
from cbt_core.persistence.memory import InMemoryChunkRetriever

if TYPE_CHECKING:
    from sqlalchemy import Engine, Table
    from sqlalchemy.orm import Session, sessionmaker

    from cbt_core.llm.client import Embedding

# Embeddings are stored as packed 32-bit floats: half the size of float64 with negligible loss for a
# cosine index, and unit vectors round-trip cleanly.
_EMBED_DTYPE = np.float32


class SpeechChunkBlobRow(Base):
    """ORM row for one persisted chunk embedding (the vector packed as float32 ``bytea``).

    Keyed by ``(speech_id, chunk_index)`` so re-indexing a speech cannot create duplicate rows; the
    retriever also skips speeches it already holds (idempotent indexing).
    """

    __tablename__ = "speech_chunk_blob"

    speech_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    chunk_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    speaker_id: Mapped[UUID] = mapped_column(Uuid, index=True, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    url: Mapped[str] = mapped_column(String(1000), nullable=False)
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


class PersistentChunkRetriever(InMemoryChunkRetriever):
    """An in-memory chunk retriever that persists chunks to PostgreSQL and reloads them on startup."""

    def __init__(self, engine: Engine, session_factory: sessionmaker[Session]) -> None:
        """Build the retriever, creating its table if needed and loading any stored chunks.

        Args:
            engine: The engine the chunk table is created on.
            session_factory: The session factory used to read and write chunk rows.
        """
        super().__init__()
        self._session_factory = session_factory
        Base.metadata.create_all(engine, tables=cast("list[Table]", [SpeechChunkBlobRow.__table__]))
        self._load()

    def _load(self) -> None:
        """Load every persisted chunk into the in-memory cosine index."""
        with self._session_factory() as session:
            for row in session.execute(select(SpeechChunkBlobRow)).scalars():
                super().add(
                    speaker_id=row.speaker_id,
                    speech_id=row.speech_id,
                    chunk_index=row.chunk_index,
                    text=row.text,
                    title=row.title,
                    url=row.url,
                    embedding=np.frombuffer(row.embedding, dtype=_EMBED_DTYPE).tolist(),
                )

    @override
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
        """Persist the chunk to PostgreSQL, then index it in memory (write-through).

        Args:
            speaker_id: The speaker who gave the speech.
            speech_id: The speech the chunk belongs to.
            chunk_index: The chunk's position within the speech.
            text: The chunk text.
            title: The speech title (for citation).
            url: The speech source URL (for citation).
            embedding: The chunk embedding (L2-normalized).
        """
        blob = np.asarray(embedding, dtype=_EMBED_DTYPE).tobytes()
        with self._session_factory() as session:
            session.add(
                SpeechChunkBlobRow(
                    speech_id=speech_id,
                    chunk_index=chunk_index,
                    speaker_id=speaker_id,
                    text=text,
                    title=title,
                    url=url,
                    embedding=blob,
                )
            )
            session.commit()
        super().add(
            speaker_id=speaker_id,
            speech_id=speech_id,
            chunk_index=chunk_index,
            text=text,
            title=title,
            url=url,
            embedding=embedding,
        )

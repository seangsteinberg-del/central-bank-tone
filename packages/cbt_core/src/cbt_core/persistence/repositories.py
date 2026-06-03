"""Repositories: one aggregate each (CLAUDE.md section 2).

Repositories take a SQLAlchemy ``Session`` and expose domain-typed methods. They translate at
the persistence edge so callers only ever see domain models. They do not own the transaction;
the calling service commits or rolls back.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from cbt_core.domain.models import Speaker, ToneObservation
from cbt_core.domain.qa import RetrievedChunk
from cbt_core.domain.registry import CentralBank
from cbt_core.domain.speech import Speech
from cbt_core.exceptions import EntityNotFoundError
from cbt_core.persistence.mappers import (
    observation_to_row,
    row_to_observation,
    row_to_speaker,
    row_to_speech,
    speaker_to_row,
    speech_to_row,
)
from cbt_core.persistence.rows import (
    SpeakerRow,
    SpeechChunkRow,
    SpeechRow,
    ToneObservationRow,
)


class SpeakerRepository:
    """Persists and retrieves :class:`Speaker` aggregates."""

    def __init__(self, session: Session) -> None:
        """Bind the repository to a session.

        Args:
            session: The active SQLAlchemy session. The caller owns its lifecycle.
        """
        self._session = session

    def add(self, speaker: Speaker) -> None:
        """Stage a new speaker for insertion (the caller commits)."""
        self._session.add(speaker_to_row(speaker))

    def get(self, speaker_id: UUID) -> Speaker:
        """Return the speaker with ``speaker_id``.

        Raises:
            EntityNotFoundError: If no speaker has that id.
        """
        row = self._session.get(SpeakerRow, speaker_id)
        if row is None:
            raise EntityNotFoundError("Speaker", speaker_id)
        return row_to_speaker(row)

    def exists(self, speaker_id: UUID) -> bool:
        """Return whether a speaker with ``speaker_id`` exists."""
        return self._session.get(SpeakerRow, speaker_id) is not None

    def find_by_name_and_central_bank(self, name: str, central_bank: CentralBank) -> Speaker | None:
        """Return the speaker with this name at this institution, or ``None``."""
        row = self._session.scalars(
            select(SpeakerRow).where(
                SpeakerRow.name == name, SpeakerRow.central_bank == central_bank
            )
        ).first()
        return row_to_speaker(row) if row is not None else None

    def list_all(self) -> list[Speaker]:
        """Return every speaker, ordered by name."""
        rows = self._session.scalars(select(SpeakerRow).order_by(SpeakerRow.name)).all()
        return [row_to_speaker(row) for row in rows]


class ToneObservationRepository:
    """Append-only repository for :class:`ToneObservation` records.

    There is deliberately no update or delete method: observations are immutable (CLAUDE.md
    section 13), and the database enforces that with a trigger.
    """

    def __init__(self, session: Session) -> None:
        """Bind the repository to a session.

        Args:
            session: The active SQLAlchemy session. The caller owns its lifecycle.
        """
        self._session = session

    def append(self, observation: ToneObservation) -> None:
        """Stage a new observation for insertion (the caller commits)."""
        self._session.add(observation_to_row(observation))

    def list_for_speaker(self, speaker_id: UUID) -> list[ToneObservation]:
        """Return every observation for a speaker, oldest first."""
        statement = (
            select(ToneObservationRow)
            .where(ToneObservationRow.speaker_id == speaker_id)
            .order_by(ToneObservationRow.observed_at)
        )
        rows = self._session.scalars(statement).all()
        return [row_to_observation(row) for row in rows]


class SpeechRepository:
    """Append-only repository for analyzed :class:`Speech` records.

    Speeches are immutable; there is no update or delete method, and the database enforces that
    with a trigger. ``source_sha256`` is unique, so re-ingesting the same text is a no-op.
    """

    def __init__(self, session: Session) -> None:
        """Bind the repository to a session.

        Args:
            session: The active SQLAlchemy session. The caller owns its lifecycle.
        """
        self._session = session

    def add(self, speech: Speech) -> None:
        """Stage a new speech for insertion (the caller commits)."""
        self._session.add(speech_to_row(speech))

    def get(self, speech_id: UUID) -> Speech:
        """Return the speech with ``speech_id``.

        Raises:
            EntityNotFoundError: If no speech has that id.
        """
        row = self._session.get(SpeechRow, speech_id)
        if row is None:
            raise EntityNotFoundError("Speech", speech_id)
        return row_to_speech(row)

    def find_by_source_sha256(self, source_sha256: str) -> Speech | None:
        """Return the speech with this source hash, or ``None`` if not yet ingested."""
        row = self._session.scalars(
            select(SpeechRow).where(SpeechRow.source_sha256 == source_sha256)
        ).one_or_none()
        return row_to_speech(row) if row is not None else None

    def list_for_speaker(self, speaker_id: UUID) -> list[Speech]:
        """Return every speech for a speaker, most recent first."""
        statement = (
            select(SpeechRow)
            .where(SpeechRow.speaker_id == speaker_id)
            .order_by(SpeechRow.delivered_at.desc())
        )
        rows = self._session.scalars(statement).all()
        return [row_to_speech(row) for row in rows]


class SpeechChunkRepository:
    """Stores embedded speech chunks and retrieves them by vector similarity.

    ``search`` uses pgvector's cosine distance and therefore runs only on PostgreSQL; the write
    methods work on any backend.
    """

    def __init__(self, session: Session) -> None:
        """Bind the repository to a session.

        Args:
            session: The active SQLAlchemy session. The caller owns its lifecycle.
        """
        self._session = session

    def has_chunks(self, speech_id: UUID) -> bool:
        """Return whether the speech has already been chunked and embedded."""
        return (
            self._session.scalars(
                select(SpeechChunkRow.id).where(SpeechChunkRow.speech_id == speech_id).limit(1)
            ).first()
            is not None
        )

    def add_chunk(
        self,
        *,
        chunk_id: UUID,
        speech_id: UUID,
        chunk_index: int,
        text: str,
        embedding: list[float],
    ) -> None:
        """Stage one embedded chunk for insertion (the caller commits)."""
        self._session.add(
            SpeechChunkRow(
                id=chunk_id,
                speech_id=speech_id,
                chunk_index=chunk_index,
                body=text,
                embedding=embedding,
            )
        )

    def search(
        self, speaker_id: UUID, query_embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` chunks closest to ``query_embedding`` for a speaker.

        Args:
            speaker_id: Restrict retrieval to this speaker's speeches.
            query_embedding: The query vector.
            top_k: The maximum number of chunks to return.

        Returns:
            The closest chunks, nearest first, each carrying its speech's citation metadata.
        """
        return self._search(query_embedding, top_k, speaker_id=speaker_id)

    def search_all(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` chunks closest to ``query_embedding`` across every speaker.

        The corpus-wide variant of :meth:`search`, used to answer platform-level questions that
        are not scoped to one speaker.

        Args:
            query_embedding: The query vector.
            top_k: The maximum number of chunks to return.

        Returns:
            The closest chunks, nearest first, each carrying its speech's citation metadata.
        """
        return self._search(query_embedding, top_k, speaker_id=None)

    def _search(
        self, query_embedding: list[float], top_k: int, *, speaker_id: UUID | None
    ) -> list[RetrievedChunk]:
        """Run the nearest-neighbour query, optionally restricted to one speaker."""
        distance = SpeechChunkRow.embedding.cosine_distance(query_embedding)
        statement = (
            select(
                SpeechChunkRow.speech_id,
                SpeechChunkRow.chunk_index,
                SpeechChunkRow.body,
                SpeechRow.title,
                SpeechRow.url,
                distance.label("distance"),
            )
            .join(SpeechRow, SpeechChunkRow.speech_id == SpeechRow.id)
            .order_by(distance)
            .limit(top_k)
        )
        if speaker_id is not None:
            statement = statement.where(SpeechRow.speaker_id == speaker_id)
        rows = self._session.execute(statement).all()
        return [
            RetrievedChunk(
                speech_id=row.speech_id,
                chunk_index=row.chunk_index,
                text=row.body,
                title=row.title,
                url=row.url,
                distance=row.distance,
            )
            for row in rows
        ]


class SpeechRetriever:
    """A session-owning chunk retriever for the question-answering service."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Bind the retriever to a session factory.

        Args:
            session_factory: Factory for sessions; one is opened per search.
        """
        self._session_factory = session_factory

    def search(
        self, speaker_id: UUID, query_embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        """Retrieve the closest chunks for a speaker (see :meth:`SpeechChunkRepository.search`)."""
        with self._session_factory() as session:
            return SpeechChunkRepository(session).search(speaker_id, query_embedding, top_k)

    def search_all(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Retrieve the closest chunks across all speakers (corpus-wide).

        See :meth:`SpeechChunkRepository.search_all`.
        """
        with self._session_factory() as session:
            return SpeechChunkRepository(session).search_all(query_embedding, top_k)

"""Repositories: one aggregate each (CLAUDE.md section 2).

Repositories take a SQLAlchemy ``Session`` and expose domain-typed methods. They translate at
the persistence edge so callers only ever see domain models. They do not own the transaction;
the calling service commits or rolls back.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from cbt_core.domain.models import Speaker, ToneObservation
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
from cbt_core.persistence.rows import SpeakerRow, SpeechRow, ToneObservationRow


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

"""Tone observation service (CLAUDE.md sections 2, 4 and 7).

Records append-only tone observations for a speaker and reads them back. The source speech
text is hashed (sha256) for provenance and never stored or logged in full (section 7).
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from uuid import UUID

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.orm import Session, sessionmaker

from cbt_core.domain.models import ToneObservation
from cbt_core.domain.tone import ToneLabel
from cbt_core.exceptions import EntityNotFoundError, InvalidInputError
from cbt_core.logging import get_logger, resolve_correlation_id
from cbt_core.persistence.repositories import SpeakerRepository, ToneObservationRepository
from cbt_core.services._support import Clock, IdFactory, default_id_factory, utc_now

_logger = get_logger(__name__)


class ToneService:
    """Record and read append-only tone observations."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        id_factory: IdFactory = default_id_factory,
        clock: Clock = utc_now,
    ) -> None:
        """Build the service.

        Args:
            session_factory: Factory for sessions. The service owns the transaction.
            id_factory: Source of new observation identifiers. Inject a deterministic one in
                tests.
            clock: Source of the default ``observed_at``. Inject a frozen clock in tests.
        """
        self._session_factory = session_factory
        self._id_factory = id_factory
        self._clock = clock

    def record_observation(
        self,
        *,
        speaker_id: UUID,
        tone: ToneLabel,
        score: float,
        source_text: str,
        observed_at: datetime | None = None,
        actor: str = "system",
        correlation_id: UUID | None = None,
    ) -> ToneObservation:
        """Record a tone observation for a speaker.

        Args:
            speaker_id: The speaker the observation is about.
            tone: The discrete tone label.
            score: Continuous tone in ``[-1.0, 1.0]``.
            source_text: The source speech text; hashed for provenance, never stored in full.
            observed_at: When the observation was made; defaults to the injected clock.
            actor: Who is performing the action.
            correlation_id: Correlation id for this call; one is minted if not supplied.

        Returns:
            The recorded :class:`ToneObservation`.

        Raises:
            InvalidInputError: If the observation fails domain validation.
            EntityNotFoundError: If the speaker does not exist.
        """
        correlation = resolve_correlation_id(correlation_id)
        log = _logger.bind(correlation_id=str(correlation), actor=actor, speaker_id=str(speaker_id))
        encoded = source_text.encode("utf-8")
        source_sha256 = hashlib.sha256(encoded).hexdigest()
        when = observed_at if observed_at is not None else self._clock()
        try:
            observation = ToneObservation(
                id=self._id_factory(),
                speaker_id=speaker_id,
                observed_at=when,
                tone=tone,
                score=score,
                source_sha256=source_sha256,
            )
        except PydanticValidationError as exc:
            log.warning("observation_validation_failed", error_count=len(exc.errors()))
            raise InvalidInputError(f"invalid observation: {exc.errors()}") from exc

        with self._session_factory() as session:
            if not SpeakerRepository(session).exists(speaker_id):
                raise EntityNotFoundError("Speaker", speaker_id)
            ToneObservationRepository(session).append(observation)
            session.commit()

        log.info(
            "tone_observed",
            observation_id=str(observation.id),
            tone=tone.value,
            score=score,
            source_sha256=source_sha256,
            source_bytes=len(encoded),
        )
        return observation

    def observations_for(
        self, speaker_id: UUID, *, actor: str = "system", correlation_id: UUID | None = None
    ) -> list[ToneObservation]:
        """Return every observation for a speaker, oldest first.

        Args:
            speaker_id: The speaker whose observations to read.
            actor: Who is performing the action.
            correlation_id: Correlation id for this call; one is minted if not supplied.

        Returns:
            The list of observations.

        Raises:
            EntityNotFoundError: If the speaker does not exist.
        """
        correlation = resolve_correlation_id(correlation_id)
        log = _logger.bind(correlation_id=str(correlation), actor=actor, speaker_id=str(speaker_id))
        with self._session_factory() as session:
            if not SpeakerRepository(session).exists(speaker_id):
                raise EntityNotFoundError("Speaker", speaker_id)
            observations = ToneObservationRepository(session).list_for_speaker(speaker_id)
        log.info("observations_listed", count=len(observations))
        return observations

"""Speaker application service (CLAUDE.md sections 2 and 7).

The only entry point adapters use to register and read speakers. It owns its transaction
boundary and opens a log context with a correlation id, an actor, and the speaker id.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.orm import Session, sessionmaker

from cbt_core.domain.models import Speaker
from cbt_core.domain.registry import CentralBank
from cbt_core.exceptions import InvalidInputError
from cbt_core.logging import get_logger, resolve_correlation_id
from cbt_core.persistence.repositories import SpeakerRepository
from cbt_core.services._support import IdFactory, default_id_factory

_logger = get_logger(__name__)


class SpeakerService:
    """Register and read speakers."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        id_factory: IdFactory = default_id_factory,
    ) -> None:
        """Build the service.

        Args:
            session_factory: Factory for sessions. The service owns the transaction.
            id_factory: Source of new speaker identifiers. Inject a deterministic one in tests.
        """
        self._session_factory = session_factory
        self._id_factory = id_factory

    def register_speaker(
        self,
        *,
        name: str,
        central_bank: CentralBank,
        role: str,
        actor: str = "system",
        correlation_id: UUID | None = None,
    ) -> Speaker:
        """Register a new speaker.

        Args:
            name: The speaker's full name.
            central_bank: The institution the speaker belongs to.
            role: The speaker's role.
            actor: Who is performing the action (for the audit log).
            correlation_id: Correlation id for this call; one is minted if not supplied.

        Returns:
            The registered :class:`Speaker`.

        Raises:
            InvalidInputError: If the supplied fields fail domain validation.
        """
        correlation = resolve_correlation_id(correlation_id)
        log = _logger.bind(correlation_id=str(correlation), actor=actor)
        try:
            speaker = Speaker(
                id=self._id_factory(), name=name, central_bank=central_bank, role=role
            )
        except PydanticValidationError as exc:
            log.warning("speaker_validation_failed", error_count=len(exc.errors()))
            raise InvalidInputError(f"invalid speaker: {exc.errors()}") from exc

        with self._session_factory() as session:
            SpeakerRepository(session).add(speaker)
            session.commit()

        log.info("speaker_registered", speaker_id=str(speaker.id), central_bank=central_bank.value)
        return speaker

    def ensure_speaker(
        self,
        *,
        name: str,
        central_bank: CentralBank,
        role: str,
        actor: str = "system",
        correlation_id: UUID | None = None,
    ) -> Speaker:
        """Return the existing speaker with this name and institution, or register a new one.

        Used by the ingestion worker so scraping the same speaker repeatedly does not create
        duplicates.

        Args:
            name: The speaker's full name.
            central_bank: The institution the speaker belongs to.
            role: The speaker's role (used only when creating).
            actor: Who is performing the action.
            correlation_id: Correlation id for this call; one is minted if not supplied.

        Returns:
            The existing or newly registered :class:`Speaker`.

        Raises:
            InvalidInputError: If a new speaker's fields fail domain validation.
        """
        correlation = resolve_correlation_id(correlation_id)
        with self._session_factory() as session:
            existing = SpeakerRepository(session).find_by_name_and_central_bank(name, central_bank)
        if existing is not None:
            return existing
        return self.register_speaker(
            name=name, central_bank=central_bank, role=role, actor=actor, correlation_id=correlation
        )

    def get_speaker(
        self, speaker_id: UUID, *, actor: str = "system", correlation_id: UUID | None = None
    ) -> Speaker:
        """Return the speaker with ``speaker_id``.

        Args:
            speaker_id: The speaker to fetch.
            actor: Who is performing the action.
            correlation_id: Correlation id for this call; one is minted if not supplied.

        Returns:
            The :class:`Speaker`.

        Raises:
            EntityNotFoundError: If no speaker has that id.
        """
        correlation = resolve_correlation_id(correlation_id)
        log = _logger.bind(correlation_id=str(correlation), actor=actor, speaker_id=str(speaker_id))
        with self._session_factory() as session:
            speaker = SpeakerRepository(session).get(speaker_id)
        log.info("speaker_fetched")
        return speaker

    def list_speakers(
        self, *, actor: str = "system", correlation_id: UUID | None = None
    ) -> list[Speaker]:
        """Return every speaker, ordered by name.

        Args:
            actor: Who is performing the action.
            correlation_id: Correlation id for this call; one is minted if not supplied.

        Returns:
            The list of speakers.
        """
        correlation = resolve_correlation_id(correlation_id)
        log = _logger.bind(correlation_id=str(correlation), actor=actor)
        with self._session_factory() as session:
            speakers = SpeakerRepository(session).list_all()
        log.info("speakers_listed", count=len(speakers))
        return speakers

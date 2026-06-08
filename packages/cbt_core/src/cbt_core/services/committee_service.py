"""Committee tone-movement service (CLAUDE.md sections 2 and 7).

Builds the read model that situates one speech against its committee: as of the speech's delivery
date, each member's standing tone and most recent shift, and how far the committee has moved
overall. It reads stored speakers and their append-only tone observations; it records nothing.

A member is "on the committee as of the speech" if they have at least one tone observation on or
before the speech's delivery date. This is deliberately point-in-time: a future chair who had not
yet spoken is not counted, and a past chair's last standing reading is shown with its own date, so
the view never implies a reading is more recent than it is.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from cbt_core.domain.committee import CommitteeMovement, MemberMovement
from cbt_core.domain.models import Speaker
from cbt_core.exceptions import EntityNotFoundError
from cbt_core.logging import get_logger, resolve_correlation_id
from cbt_core.persistence.repositories import (
    SpeakerRepository,
    SpeechRepository,
    ToneObservationRepository,
)

_logger = get_logger(__name__)


class CommitteeService:
    """Compute committee-wide tone movement around a speech (a point-in-time read model)."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Build the service.

        Args:
            session_factory: Factory for sessions. Each call opens its own read transaction.
        """
        self._session_factory = session_factory

    def movement_for_speech(
        self, speech_id: UUID, *, actor: str = "system", correlation_id: UUID | None = None
    ) -> CommitteeMovement:
        """Return how the speech's committee has moved in tone, as of the speech.

        Args:
            speech_id: The speech to situate against its committee.
            actor: Who is performing the action.
            correlation_id: Correlation id for this call; one is minted if not supplied.

        Returns:
            A :class:`CommitteeMovement`: each member's standing tone and most recent shift as of
            the speech's delivery date, and the committee's aggregate tone and overall move.

        Raises:
            EntityNotFoundError: If no speech has that id.
        """
        correlation = resolve_correlation_id(correlation_id)
        log = _logger.bind(correlation_id=str(correlation), actor=actor, speech_id=str(speech_id))
        with self._session_factory() as session:
            speech = SpeechRepository(session).get(speech_id)
            as_of = speech.delivered_at
            members: list[MemberMovement] = []
            for speaker in SpeakerRepository(session).list_all():
                if speaker.central_bank is not speech.central_bank:
                    continue
                movement = self._member_movement(
                    session, speaker, as_of=as_of, is_subject=speaker.id == speech.speaker_id
                )
                if movement is not None:
                    members.append(movement)

        subject = next((member for member in members if member.is_subject), None)
        if subject is None:
            # The subject always has a reading as of its own speech in normal ingestion (the tone
            # observation is written atomically with the speech). Guard the data-integrity edge so a
            # speech missing its observation fails as a typed CbtError, not a raw StopIteration that
            # bypasses the adapter error handlers (CLAUDE.md section 3).
            raise EntityNotFoundError("Committee reading for speech", speech_id)
        members.sort(key=_movement_sort_key)
        committee_tone = sum(member.current_score for member in members) / len(members)
        deltas = [member.delta for member in members if member.delta is not None]
        overall_delta = sum(deltas) / len(deltas) if deltas else None

        log.info(
            "committee_movement_computed",
            central_bank=speech.central_bank.value,
            members=len(members),
            movers=len(deltas),
            committee_tone=round(committee_tone, 4),
        )
        return CommitteeMovement(
            central_bank=speech.central_bank,
            as_of=as_of,
            subject=subject,
            members=members,
            committee_tone=committee_tone,
            overall_delta=overall_delta,
            movers=len(deltas),
        )

    def _member_movement(
        self,
        session: Session,
        speaker: Speaker,
        *,
        as_of: datetime,
        is_subject: bool,
    ) -> MemberMovement | None:
        """Build one member's movement as of ``as_of``, or ``None`` if they had no reading yet."""
        history = [
            observation
            for observation in ToneObservationRepository(session).list_for_speaker(speaker.id)
            if observation.observed_at <= as_of
        ]
        if not history:
            return None
        current = history[-1]
        previous = history[-2] if len(history) >= 2 else None
        return MemberMovement(
            speaker=speaker,
            current_score=current.score,
            current_tone=current.tone,
            current_at=current.observed_at,
            previous_score=previous.score if previous is not None else None,
            previous_at=previous.observed_at if previous is not None else None,
            observation_count=len(history),
            is_subject=is_subject,
        )


def _movement_sort_key(member: MemberMovement) -> tuple[int, float, str]:
    """Sort key: members with a measurable shift first, by descending absolute shift, then name."""
    if member.delta is None:
        return (1, 0.0, member.speaker.name.casefold())
    return (0, -abs(member.delta), member.speaker.name.casefold())

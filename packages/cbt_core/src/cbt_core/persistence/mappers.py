"""Mappers between domain models and ORM rows (CLAUDE.md section 2).

This is the only place that knows both representations. ORM rows are constructed here on the
way in and converted back to immutable domain models on the way out; a ``*Row`` never escapes
the persistence layer.
"""

from __future__ import annotations

from datetime import UTC

from cbt_core.domain.models import Speaker, ToneObservation
from cbt_core.persistence.rows import SpeakerRow, ToneObservationRow


def speaker_to_row(speaker: Speaker) -> SpeakerRow:
    """Build a :class:`SpeakerRow` from a :class:`Speaker`."""
    return SpeakerRow(
        id=speaker.id,
        name=speaker.name,
        central_bank=speaker.central_bank,
        role=speaker.role,
    )


def row_to_speaker(row: SpeakerRow) -> Speaker:
    """Build a :class:`Speaker` from a :class:`SpeakerRow`."""
    return Speaker(
        id=row.id,
        name=row.name,
        central_bank=row.central_bank,
        role=row.role,
    )


def observation_to_row(observation: ToneObservation) -> ToneObservationRow:
    """Build a :class:`ToneObservationRow` from a :class:`ToneObservation`."""
    return ToneObservationRow(
        id=observation.id,
        speaker_id=observation.speaker_id,
        observed_at=observation.observed_at,
        tone=observation.tone,
        score=observation.score,
        source_sha256=observation.source_sha256,
    )


def row_to_observation(row: ToneObservationRow) -> ToneObservation:
    """Build a :class:`ToneObservation` from a :class:`ToneObservationRow`.

    The ``observed_at`` column is timezone-aware. Postgres preserves the offset; the SQLite
    dialect used in unit tests cannot store one, so a naive value read back is interpreted as
    UTC (the timezone it was written in). This is an explicit normalization at the persistence
    edge, not a silent guess about an unknown timezone.
    """
    observed_at = row.observed_at
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=UTC)
    return ToneObservation(
        id=row.id,
        speaker_id=row.speaker_id,
        observed_at=observed_at,
        tone=row.tone,
        score=row.score,
        source_sha256=row.source_sha256,
    )

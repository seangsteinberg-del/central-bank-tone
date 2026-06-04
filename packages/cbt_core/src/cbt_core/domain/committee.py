"""Committee tone-movement read models (CLAUDE.md sections 2 and 3).

A speech is given by one member of a committee (a central bank), but it lands against the backdrop
of how the whole committee has been moving. These read models answer, as of a given speech: how
hawkish or dovish does each member now read, how much has each one shifted since their previous
speech, and how far has the committee moved overall.

They are computed, immutable views built by :class:`~cbt_core.services.committee_service.CommitteeService`
from stored speakers and their append-only tone observations. They hold domain types (a
:class:`~cbt_core.domain.models.Speaker`, a :class:`~cbt_core.domain.tone.ToneLabel`), never ORM
rows or primitives.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from cbt_core.domain.models import Speaker
from cbt_core.domain.registry import CentralBank
from cbt_core.domain.tone import ToneLabel


class MemberMovement(BaseModel):
    """One committee member's standing tone and most recent shift, as of a point in time.

    Attributes:
        speaker: The committee member.
        current_score: The member's most recent tone score on or before the as-of date, in
            ``[-1.0, 1.0]`` (negative dovish, positive hawkish).
        current_tone: The discrete tone label of that most recent reading.
        current_at: When that most recent reading was observed.
        previous_score: The member's prior tone score (the reading before ``current``), or
            ``None`` if this is their first reading as of the as-of date.
        previous_at: When the prior reading was observed, or ``None``.
        observation_count: How many readings the member has on or before the as-of date.
        is_subject: True for the member who gave the speech this movement is computed against.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    speaker: Speaker
    current_score: float = Field(ge=-1.0, le=1.0)
    current_tone: ToneLabel
    current_at: datetime
    previous_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    previous_at: datetime | None = None
    observation_count: int = Field(ge=1)
    is_subject: bool = False

    @property
    def delta(self) -> float | None:
        """Signed change from the previous reading to the current one, or ``None`` if first.

        Returns:
            ``current_score - previous_score`` (positive means the member moved more hawkish),
            or ``None`` when there is no prior reading to compare against.
        """
        if self.previous_score is None:
            return None
        return self.current_score - self.previous_score


class CommitteeMovement(BaseModel):
    """How a central bank's committee has moved in tone, viewed as of one speech.

    Attributes:
        central_bank: The committee (institution) this movement is about.
        as_of: The date the view is taken as of (the subject speech's delivery date).
        subject: The movement of the member who gave the subject speech.
        members: Every member with at least one reading on or before ``as_of``, ordered with the
            largest absolute shift first (members with no prior reading last).
        committee_tone: The committee's standing tone, the mean of members' current scores, in
            ``[-1.0, 1.0]``.
        overall_delta: How far the committee moved overall, the mean of members' individual
            shifts (averaged over members that have a prior reading), or ``None`` if no member
            does. Not bounded to ``[-1.0, 1.0]`` because it is a mean of differences.
        movers: How many members have a measurable shift (a prior reading), the count
            ``overall_delta`` is averaged over.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    central_bank: CentralBank
    as_of: datetime
    subject: MemberMovement
    members: list[MemberMovement]
    committee_tone: float = Field(ge=-1.0, le=1.0)
    overall_delta: float | None = None
    movers: int = Field(ge=0)

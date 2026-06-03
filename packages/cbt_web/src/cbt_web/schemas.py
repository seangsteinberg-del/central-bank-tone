"""Form-validation models for the web UI (CLAUDE.md section 3).

Every form submission is validated against one of these models before any service call runs, so
the views pass typed values inward and never an unvalidated string. They are the web boundary
contract, separate from the core domain models.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from cbt_core import CentralBank


class AskForm(BaseModel):
    """A natural-language question submitted from an ask box."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)


class SpeakerForm(BaseModel):
    """A speaker submitted from the registration form."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    central_bank: CentralBank
    role: str = Field(min_length=1, max_length=200)


class IngestForm(BaseModel):
    """A speech submitted from the ingestion form."""

    model_config = ConfigDict(extra="forbid")

    speaker_id: UUID
    title: str = Field(min_length=1, max_length=500)
    url: str = Field(min_length=1, max_length=2000)
    delivered_on: date
    text: str = Field(min_length=1)
    language: str = Field(min_length=2, max_length=12, default="en")

    @property
    def delivered_at(self) -> datetime:
        """Return ``delivered_on`` as a timezone-aware datetime at UTC midnight.

        The core requires timezone-aware timestamps; an HTML date input carries no time or
        zone, so the day is anchored to UTC midnight.
        """
        return datetime.combine(self.delivered_on, time(0, 0), tzinfo=UTC)

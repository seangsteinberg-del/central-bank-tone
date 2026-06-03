"""Shared service support: injectable clock and id factory.

Services take these as constructor dependencies so tests can pin time and identifiers without
patching globals (CLAUDE.md section 5). Production uses the defaults below.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID, uuid4

IdFactory = Callable[[], UUID]
Clock = Callable[[], datetime]


def default_id_factory() -> UUID:
    """Return a fresh random identifier (uuid4)."""
    return uuid4()


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)

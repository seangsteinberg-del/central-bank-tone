"""The SpeechSource boundary and the scraped-speech DTO (CLAUDE.md section 2).

A source yields :class:`ScrapedSpeech` values ready to ingest. The HTTP fetcher is injected, so
sources are tested against HTML fixtures with no network.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from cbt_core import CentralBank

# Fetches the text of a URL. Injected so sources can be tested without the network.
type Fetcher = Callable[[str], str]
# Provides the bytes of a binary resource (for example a downloaded ZIP). Injected for testing.
type BytesProvider = Callable[[], bytes]

# Map BIS institution names onto the schema spine. Shared by every source so the free-text ->
# CentralBank knowledge lives in exactly one place. Only tracked banks are kept.
INSTITUTION_TO_BANK: tuple[tuple[str, CentralBank], ...] = (
    ("federal reserve", CentralBank.FEDERAL_RESERVE),
    ("european central bank", CentralBank.ECB),
    ("bank of england", CentralBank.BANK_OF_ENGLAND),
    ("bank of japan", CentralBank.BANK_OF_JAPAN),
    ("bank of canada", CentralBank.BANK_OF_CANADA),
    ("reserve bank of australia", CentralBank.RESERVE_BANK_OF_AUSTRALIA),
    ("swiss national bank", CentralBank.SWISS_NATIONAL_BANK),
    ("people's bank of china", CentralBank.PEOPLES_BANK_OF_CHINA),
    ("peoples bank of china", CentralBank.PEOPLES_BANK_OF_CHINA),
)

# BIS descriptions read "<type> by <Title> <Name>, <Role> of (the) <Institution>, at <venue>...".
# Capture the role between the first comma and " of "/" at ", and the speaker's own institution
# from the affiliation clause (not from the venue, which could also name a tracked bank).
_ROLE_RE = re.compile(r",\s*([A-Z][A-Za-z' .-]{1,80}?)\s+(?:of|at)\b")
_AFFILIATION_RE = re.compile(r",\s*[A-Z][^,]*?\bof\b\s+(?:the\s+)?(.+?)\s*,")


def central_bank_from(text: str) -> CentralBank | None:
    """Map a free-text institution string onto the schema spine, or ``None`` if untracked.

    Args:
        text: The institution text (an affiliation clause or institution name).

    Returns:
        The matched :class:`~cbt_core.CentralBank`, or ``None`` if no tracked bank is named.
    """
    lowered = text.lower()
    for needle, bank in INSTITUTION_TO_BANK:
        if needle in lowered:
            return bank
    return None


def role_from(description: str) -> str:
    """Best-effort speaker role from a BIS-style description, or a generic fallback.

    Args:
        description: The BIS description clause.

    Returns:
        The extracted role, or ``"Central banker"`` if none can be parsed.
    """
    match = _ROLE_RE.search(description)
    return match.group(1).strip() if match is not None else "Central banker"


def affiliation_from(description: str) -> str:
    """The speaker's own institution from a BIS-style description's affiliation clause.

    Reading the affiliation rather than scanning the whole description avoids misattributing a
    speaker to a tracked bank merely because they spoke at that bank's venue.

    Args:
        description: The BIS description clause.

    Returns:
        The affiliation substring, or the full description if no affiliation clause is found.
    """
    match = _AFFILIATION_RE.search(description)
    return match.group(1).strip() if match is not None else description


@dataclass(frozen=True)
class ScrapedSpeech:
    """A speech scraped from a source, ready to ingest.

    Attributes:
        speaker_name: The speaker's full name.
        central_bank: The institution (mapped onto the schema spine).
        role: The speaker's role.
        title: The speech title.
        url: The source URL.
        delivered_at: When the speech was delivered (timezone-aware).
        text: The full speech text.
        language: The source language code.
    """

    speaker_name: str
    central_bank: CentralBank
    role: str
    title: str
    url: str
    delivered_at: datetime
    text: str
    language: str = "en"


class SpeechSource(Protocol):
    """A source of central bank speeches."""

    @property
    def name(self) -> str:
        """A short, stable identifier for the source (for logging)."""
        ...

    def fetch(self, *, limit: int) -> list[ScrapedSpeech]:
        """Fetch up to ``limit`` recent speeches from the source."""
        ...

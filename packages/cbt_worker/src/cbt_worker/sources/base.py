"""The SpeechSource boundary and the scraped-speech DTO (CLAUDE.md section 2).

A source yields :class:`ScrapedSpeech` values ready to ingest. The HTTP fetcher is injected, so
sources are tested against HTML fixtures with no network.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from cbt_core import CentralBank

# Fetches the text of a URL. Injected so sources can be tested without the network.
type Fetcher = Callable[[str], str]


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

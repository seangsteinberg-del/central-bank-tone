"""Speech sources: one per scraping strategy, behind the SpeechSource protocol."""

from __future__ import annotations

from cbt_worker.sources.base import BytesProvider, Fetcher, ScrapedSpeech, SpeechSource
from cbt_worker.sources.bis import BisSpeechSource
from cbt_worker.sources.bis_bulk import BisArchiveError, BisBulkSpeechSource

__all__ = [
    "BisArchiveError",
    "BisBulkSpeechSource",
    "BisSpeechSource",
    "BytesProvider",
    "Fetcher",
    "ScrapedSpeech",
    "SpeechSource",
]

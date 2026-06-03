"""Speech sources: one per scraping strategy, behind the SpeechSource protocol."""

from __future__ import annotations

from cbt_worker.sources.base import Fetcher, ScrapedSpeech, SpeechSource
from cbt_worker.sources.bis import BisSpeechSource

__all__ = ["BisSpeechSource", "Fetcher", "ScrapedSpeech", "SpeechSource"]

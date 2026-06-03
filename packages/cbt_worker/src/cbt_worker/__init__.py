"""Central Bank Tone ingestion worker.

Scrapes central bank speeches from pluggable sources and drives them through the core ingestion
and indexing services. Depends on ``cbt_core`` only.
"""

from __future__ import annotations

from cbt_worker.runner import run_ingestion
from cbt_worker.sources import BisSpeechSource, ScrapedSpeech, SpeechSource

__all__ = ["BisSpeechSource", "ScrapedSpeech", "SpeechSource", "run_ingestion"]

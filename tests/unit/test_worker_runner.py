"""Tests for the ingestion worker runner (CLAUDE.md sections 2 and 5)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session, sessionmaker

from cbt_core import IndexingService, IngestionService, SpeakerService
from cbt_core.domain.registry import CentralBank
from cbt_core.exceptions import LlmError
from cbt_core.services._support import IdFactory
from cbt_worker import run_ingestion
from cbt_worker.sources.base import ScrapedSpeech


class _StubSource:
    name = "stub"

    def __init__(self, speeches: list[ScrapedSpeech]) -> None:
        self._speeches = speeches

    def fetch(self, *, limit: int) -> list[ScrapedSpeech]:
        return self._speeches[:limit]


class _RaisingLlm:
    def analyze_tone(self, speech_text: str) -> object:
        raise LlmError("model unavailable")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def answer(self, question: str, chunks: object) -> str:
        return ""


def _scraped(*, url: str, text: str, speaker: str = "Jerome Powell") -> ScrapedSpeech:
    return ScrapedSpeech(
        speaker_name=speaker,
        central_bank=CentralBank.FEDERAL_RESERVE,
        role="Chair",
        title="A speech",
        url=url,
        delivered_at=datetime(2026, 1, 1, tzinfo=UTC),
        text=text,
    )


@pytest.mark.unit
def test_run_ingestion_ingests_indexes_and_deduplicates_speakers(
    speaker_service: SpeakerService,
    ingestion_service: IngestionService,
    indexing_service: IndexingService,
) -> None:
    source = _StubSource(
        [
            _scraped(url="https://x/1", text="inflation will tighten and hike rates"),
            _scraped(url="https://x/2", text="growth is slowing with downside risks"),
        ]
    )
    count = run_ingestion(
        [source],
        speaker_service=speaker_service,
        ingestion_service=ingestion_service,
        indexing_service=indexing_service,
    )
    assert count == 2
    # Both speeches share one speaker, created once via ensure_speaker.
    assert len(speaker_service.list_speakers()) == 1


@pytest.mark.unit
def test_run_ingestion_skips_a_speech_that_fails_the_model(
    session_factory: sessionmaker[Session],
    speaker_service: SpeakerService,
    indexing_service: IndexingService,
    id_factory: IdFactory,
) -> None:
    failing_ingestion = IngestionService(session_factory, _RaisingLlm(), id_factory=id_factory)
    source = _StubSource([_scraped(url="https://x/1", text="some text")])
    count = run_ingestion(
        [source],
        speaker_service=speaker_service,
        ingestion_service=failing_ingestion,
        indexing_service=indexing_service,
    )
    assert count == 0  # the run completed despite the model failure

"""Vector retrieval over speech chunks on real Postgres + pgvector (CLAUDE.md section 5).

Uses a fresh container and stub (deterministic) embeddings, so it exercises the pgvector
similarity SQL and the full ingest -> index -> retrieve path without a live model call.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tests._stubs import StubLlmClient

import cbt_core.persistence as _persistence
from cbt_core import (
    IndexingService,
    IngestionService,
    SpeakerService,
    SpeechRetriever,
    ToneAnalysis,
    ToneLabel,
)
from cbt_core.domain.registry import CentralBank

_MIGRATIONS_DIR = Path(_persistence.__file__).resolve().parent / "migrations"
_LONG_TEXT = " ".join(f"word{i}" for i in range(300))


def _migrate(url: str) -> None:
    config = Config()
    config.set_main_option("script_location", str(_MIGRATIONS_DIR))
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")


@pytest.mark.integration
def test_vector_search_returns_the_speakers_chunks(fresh_postgres_url: str) -> None:
    _migrate(fresh_postgres_url)
    engine = create_engine(fresh_postgres_url, future=True)
    try:
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        counter = itertools.count(1)
        id_factory = lambda: UUID(int=next(counter))  # noqa: E731  (compact test id factory)
        stub = StubLlmClient(
            ToneAnalysis(
                summary="A summary.", tone=ToneLabel.HAWKISH, score=0.5, rationale="reason"
            )
        )

        speaker = SpeakerService(factory, id_factory=id_factory).register_speaker(
            name="Powell", central_bank=CentralBank.FEDERAL_RESERVE, role="Chair"
        )
        speech = IngestionService(factory, stub, id_factory=id_factory).ingest_speech(
            speaker_id=speaker.id,
            title="Outlook",
            url="https://example.org/s/1",
            delivered_at=datetime(2026, 1, 1, tzinfo=UTC),
            text=_LONG_TEXT,
        )
        chunk_count = IndexingService(
            factory, stub, max_chars=200, overlap=20, id_factory=id_factory
        ).index_speech(speech.id)
        assert chunk_count > 1

        query_embedding = stub.embed(["what is the outlook?"])[0]
        retriever = SpeechRetriever(factory)
        chunks = retriever.search(speaker.id, query_embedding, top_k=3)
        corpus_chunks = retriever.search_all(query_embedding, top_k=3)
    finally:
        engine.dispose()

    assert 1 <= len(chunks) <= 3
    assert all(chunk.speech_id == speech.id for chunk in chunks)
    # Distances are ordered nearest-first.
    assert chunks == sorted(chunks, key=lambda chunk: chunk.distance)
    # The corpus-wide search finds the same chunks when there is only one speaker.
    assert {chunk.speech_id for chunk in corpus_chunks} == {speech.id}
    assert corpus_chunks == sorted(corpus_chunks, key=lambda chunk: chunk.distance)

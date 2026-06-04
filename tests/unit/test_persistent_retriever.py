"""PersistentChunkRetriever stores chunks and reloads them, so retrieval survives a restart (ADR 0020).

Exercised against an in-process SQLite database (the ``bytea`` column maps to a BLOB), with no
PostgreSQL and no network, the same way the repository tests run.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import StaticPool

from cbt_core import PersistentChunkRetriever, make_session_factory


def _sqlite_engine() -> Engine:
    return create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )


@pytest.mark.unit
def test_persistent_retriever_reloads_chunks_after_a_restart() -> None:
    engine = _sqlite_engine()
    session_factory = make_session_factory(engine)
    speaker_id, speech_id = uuid4(), uuid4()

    first = PersistentChunkRetriever(engine, session_factory)
    first.add(
        speaker_id=speaker_id,
        speech_id=speech_id,
        chunk_index=0,
        text="Inflation will stay restrictive.",
        title="On policy",
        url="https://example.org/a",
        embedding=[1.0, 0.0, 0.0],
    )
    first.add(
        speaker_id=speaker_id,
        speech_id=speech_id,
        chunk_index=1,
        text="Growth is slowing.",
        title="On policy",
        url="https://example.org/a",
        embedding=[0.0, 1.0, 0.0],
    )
    assert first.has_speech(speech_id)

    # A fresh retriever on the same database (a restart) reloads the persisted chunks.
    second = PersistentChunkRetriever(engine, session_factory)
    assert second.has_speech(speech_id)
    hits = second.search_all([1.0, 0.0, 0.0], top_k=2)
    assert [hit.text for hit in hits] == ["Inflation will stay restrictive.", "Growth is slowing."]
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_persistent_retriever_starts_empty() -> None:
    engine = _sqlite_engine()
    retriever = PersistentChunkRetriever(engine, make_session_factory(engine))
    assert not retriever.has_speech(uuid4())
    assert retriever.search_all([1.0, 0.0, 0.0], top_k=5) == []

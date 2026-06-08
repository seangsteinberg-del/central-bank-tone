"""Keyless, Docker-less demo wiring for the web UI (ADR 0014).

Builds a fully working application backed by SQLite and the offline LLM client (ADR 0013, 0014),
seeded with a provided corpus, so the entire UI - tone scoring, the tone-over-time charts, and
natural-language search - runs with no Gemini key and no PostgreSQL. The production path
(`app.py` / `dependencies.py`, Gemini + Postgres) is untouched; this is for local demos and tests.

The reusable wiring lives here (and is exercised by tests); `scripts/run_demo.py` sources a real
seed corpus and serves it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, override
from uuid import UUID

from fastapi import FastAPI

from cbt_core import (
    CentralBank,
    CommitteeService,
    IndexingService,
    IngestionService,
    InMemoryChunkRetriever,
    LlmClient,
    LlmError,
    MarketSignalService,
    OfflineLlmClient,
    PersistentChunkRetriever,
    QaService,
    SpeakerService,
    ToneService,
    chunk_text,
    create_demo_schema,
    get_settings,
    make_demo_engine,
    make_session_factory,
)
from cbt_web.app import create_app
from cbt_web.dependencies import Services

if TYPE_CHECKING:
    from sqlalchemy import Engine
    from sqlalchemy.orm import Session, sessionmaker

# Hashing embeddings (ADR 0014) carry less absolute cosine similarity than learned embeddings, so
# the demo widens the relevance threshold; retrieval still abstains when nothing is close.
_DEMO_MAX_DISTANCE = 0.95
_DEMO_MODEL_ID = "offline-classifier-v1"


@dataclass(frozen=True)
class SeedSpeech:
    """One speech to seed into the demo corpus."""

    speaker_name: str
    central_bank: CentralBank
    role: str
    title: str
    url: str
    delivered_at: datetime
    text: str


class _DemoIndexer(IndexingService):
    """IndexingService variant that indexes into an in-memory retriever instead of pgvector."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        llm: LlmClient,
        *,
        retriever: InMemoryChunkRetriever,
        ingestion: IngestionService,
    ) -> None:
        """Build the demo indexer.

        Args:
            session_factory: The session factory (passed through to the base service).
            llm: The client used to embed chunks (the offline client, or a real Gemini client
                for the live setup).
            retriever: The in-memory retriever to populate.
            ingestion: Used to load a speech's text and metadata by id.
        """
        super().__init__(session_factory, llm)
        self._retriever = retriever
        self._ingestion = ingestion

    @override
    def index_speech(
        self, speech_id: UUID, *, actor: str = "system", correlation_id: UUID | None = None
    ) -> int:
        """Chunk, embed, and add a speech's chunks to the retriever, skipping ones already indexed."""
        if self._retriever.has_speech(speech_id):
            # Idempotent: a persisted speech is already in the (loaded) index, so do not re-embed or
            # duplicate its chunks on a re-run or restart.
            return 0
        speech = self._ingestion.get_speech(speech_id, actor=actor, correlation_id=correlation_id)
        chunks = chunk_text(speech.text, max_chars=self._max_chars, overlap=self._overlap)
        embeddings = self._llm.embed(chunks)
        if len(embeddings) != len(chunks):
            raise LlmError(f"embedding returned {len(embeddings)} vectors for {len(chunks)} chunks")
        for index, (text, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
            self._retriever.add(
                speaker_id=speech.speaker_id,
                speech_id=speech.id,
                chunk_index=index,
                text=text,
                title=speech.title,
                url=speech.url,
                embedding=embedding,
            )
        return len(chunks)


def build_demo_services(
    speeches: list[SeedSpeech],
    *,
    db_path: str | None = None,
    engine: Engine | None = None,
    llm: LlmClient | None = None,
    model_id: str = _DEMO_MODEL_ID,
    max_distance: float = _DEMO_MAX_DISTANCE,
    persistent_retrieval: bool = False,
) -> Services:
    """Build a service container backed by an in-memory retriever and seed it with a corpus.

    The defaults give the keyless demo: a SQLite database and the offline client, so the tone
    signal and retrieval need no Gemini key. Injecting ``engine`` and ``llm`` builds the live
    setup instead: the real Gemini client over a PostgreSQL engine, with the same in-memory
    vector index, so no pgvector is required (ADR 0018).

    Args:
        speeches: The corpus to seed (ingested and indexed in order); may be empty.
        db_path: A SQLite file path, or ``None`` for a shared in-memory database. Ignored when
            ``engine`` is provided.
        engine: An explicit engine (for example a PostgreSQL one); built from ``db_path`` if omitted.
        llm: The LLM client; the keyless :class:`OfflineLlmClient` if omitted.
        model_id: The model id recorded on each ingested speech.
        max_distance: The retrieval relevance threshold (cosine distance) for question answering.
        persistent_retrieval: When True, use a :class:`PersistentChunkRetriever` that stores chunk
            embeddings in the database and reloads them on startup (ADR 0020), so retrieval survives
            restarts; otherwise an in-process :class:`InMemoryChunkRetriever`.

    Returns:
        A fully wired :class:`Services` container.
    """
    settings = get_settings()
    engine = engine if engine is not None else make_demo_engine(db_path)
    create_demo_schema(engine)
    session_factory = make_session_factory(engine)

    llm = llm if llm is not None else OfflineLlmClient()
    offline = isinstance(llm, OfflineLlmClient)
    retriever: InMemoryChunkRetriever = (
        PersistentChunkRetriever(engine, session_factory)
        if persistent_retrieval
        else InMemoryChunkRetriever()
    )
    speaker_service = SpeakerService(session_factory)
    ingestion = IngestionService(session_factory, llm, model_id=model_id)
    indexer = _DemoIndexer(session_factory, llm, retriever=retriever, ingestion=ingestion)

    for seed in speeches:
        speaker = speaker_service.ensure_speaker(
            name=seed.speaker_name, central_bank=seed.central_bank, role=seed.role
        )
        speech = ingestion.ingest_speech(
            speaker_id=speaker.id,
            title=seed.title,
            url=seed.url,
            delivered_at=seed.delivered_at,
            text=seed.text,
        )
        indexer.index_speech(speech.id)

    return Services(
        settings=settings,
        engine=engine,
        speaker_service=speaker_service,
        tone_service=ToneService(session_factory),
        ingestion_service=ingestion,
        indexing_service=indexer,
        qa_service=QaService(llm, retriever, speaker_service, max_distance=max_distance),
        committee_service=CommitteeService(session_factory),
        market_service=MarketSignalService(session_factory, benchmark_dir=settings.benchmark_dir),
        offline=offline,
    )


def build_demo_app(
    speeches: list[SeedSpeech],
    *,
    db_path: str | None = None,
    engine: Engine | None = None,
    llm: LlmClient | None = None,
    model_id: str = _DEMO_MODEL_ID,
    max_distance: float = _DEMO_MAX_DISTANCE,
    persistent_retrieval: bool = False,
) -> FastAPI:
    """Build the web application over an in-memory retriever, seeded with ``speeches``.

    Reuses the production application factory (middleware, exception handlers, static mount, and
    routes) and swaps in the in-memory-retriever service container, so it exercises exactly the
    same views as production. With the defaults this is the keyless demo; injecting ``engine`` and
    ``llm`` builds the live setup (real Gemini over PostgreSQL; see :func:`build_demo_services`).

    Args:
        speeches: The corpus to seed.
        db_path: A SQLite file path, or ``None`` for a shared in-memory database. Ignored when
            ``engine`` is provided.
        engine: An explicit engine; built from ``db_path`` if omitted.
        llm: The LLM client; the keyless :class:`OfflineLlmClient` if omitted.
        model_id: The model id recorded on each ingested speech.
        max_distance: The retrieval relevance threshold (cosine distance) for question answering.

    Returns:
        The configured application.
    """
    app = create_app()
    app.state.services = build_demo_services(
        speeches,
        db_path=db_path,
        engine=engine,
        llm=llm,
        model_id=model_id,
        max_distance=max_distance,
        persistent_retrieval=persistent_retrieval,
    )
    return app

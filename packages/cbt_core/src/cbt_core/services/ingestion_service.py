"""Speech ingestion service (CLAUDE.md sections 2 and 7).

Ingests and analyzes a central bank speech: verifies the speaker, deduplicates by source hash
(so re-running a scraper is a no-op and spends no model call), scores the deterministic lexicon
baseline, asks Gemini for a summary and tone, and persists the immutable speech plus an
append-only tone observation atomically. The source text is hashed for provenance and never
logged in full.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy.orm import Session, sessionmaker

from cbt_core.analysis.lexicon import HawkishDovishLexicon
from cbt_core.domain.models import ToneObservation
from cbt_core.domain.speech import Speech
from cbt_core.llm.client import LlmClient
from cbt_core.logging import get_logger
from cbt_core.persistence.repositories import (
    SpeakerRepository,
    SpeechRepository,
    ToneObservationRepository,
)
from cbt_core.services._support import IdFactory, default_id_factory

_logger = get_logger(__name__)


class IngestionService:
    """Ingest and analyze central bank speeches, idempotently by source hash."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        llm_client: LlmClient,
        *,
        lexicon: HawkishDovishLexicon | None = None,
        id_factory: IdFactory = default_id_factory,
    ) -> None:
        """Build the service.

        Args:
            session_factory: Factory for sessions. The service owns the transaction.
            llm_client: The LLM boundary used to summarize and score the speech.
            lexicon: The deterministic lexicon baseline; a default one is used if not supplied.
            id_factory: Source of new identifiers. Inject a deterministic one in tests.
        """
        self._session_factory = session_factory
        self._llm = llm_client
        self._lexicon = lexicon if lexicon is not None else HawkishDovishLexicon()
        self._id_factory = id_factory

    def ingest_speech(
        self,
        *,
        speaker_id: UUID,
        title: str,
        url: str,
        delivered_at: datetime,
        text: str,
        language: str = "en",
        actor: str = "system",
        correlation_id: UUID | None = None,
    ) -> Speech:
        """Ingest one speech, analyze it, and persist it.

        Args:
            speaker_id: The speaker who gave the speech.
            title: The speech title.
            url: The source URL.
            delivered_at: When the speech was delivered (timezone-aware).
            text: The full speech text.
            language: The source language code.
            actor: Who is performing the action.
            correlation_id: Correlation id for this call; one is minted if not supplied.

        Returns:
            The ingested :class:`Speech`. If a speech with the same source hash already exists,
            that existing speech is returned unchanged and no model call is made.

        Raises:
            EntityNotFoundError: If the speaker does not exist.
            LlmError: If the model call fails or returns an unusable response.
        """
        correlation = correlation_id if correlation_id is not None else uuid4()
        encoded = text.encode("utf-8")
        source_sha256 = hashlib.sha256(encoded).hexdigest()
        log = _logger.bind(
            correlation_id=str(correlation),
            actor=actor,
            speaker_id=str(speaker_id),
            source_sha256=source_sha256,
        )

        # Phase 1: verify the speaker and deduplicate before any model spend.
        with self._session_factory() as session:
            speaker = SpeakerRepository(session).get(speaker_id)
            existing = SpeechRepository(session).find_by_source_sha256(source_sha256)
        if existing is not None:
            log.info("speech_already_ingested", speech_id=str(existing.id))
            return existing

        # Phase 2: analyze (no transaction held across the model call).
        lexicon_score = self._lexicon.score(text).score
        analysis = self._llm.analyze_tone(text)
        speech = Speech(
            id=self._id_factory(),
            speaker_id=speaker_id,
            central_bank=speaker.central_bank,
            title=title,
            url=url,
            delivered_at=delivered_at,
            language=language,
            text=text,
            source_sha256=source_sha256,
            summary=analysis.summary,
            tone=analysis.tone,
            score=analysis.score,
            lexicon_score=lexicon_score,
            rationale=analysis.rationale,
        )
        observation = ToneObservation(
            id=self._id_factory(),
            speaker_id=speaker_id,
            observed_at=delivered_at,
            tone=analysis.tone,
            score=analysis.score,
            source_sha256=source_sha256,
        )

        # Phase 3: persist the speech and its tone signal atomically.
        with self._session_factory() as session:
            SpeechRepository(session).add(speech)
            ToneObservationRepository(session).append(observation)
            session.commit()

        log.info(
            "speech_ingested",
            speech_id=str(speech.id),
            tone=analysis.tone.value,
            score=analysis.score,
            lexicon_score=lexicon_score,
            summary_chars=len(analysis.summary),
            source_bytes=len(encoded),
        )
        return speech

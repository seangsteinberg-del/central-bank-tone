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
from cbt_core.domain.speech import Speech, SpeechStance
from cbt_core.llm.client import LlmClient
from cbt_core.logging import get_logger
from cbt_core.persistence.repositories import (
    SpeakerRepository,
    SpeechRepository,
    SpeechStanceRepository,
    ToneObservationRepository,
)
from cbt_core.services._support import IdFactory, default_id_factory
from cbt_core.services.stance_service import StanceService, build_stance

_logger = get_logger(__name__)


class IngestionService:
    """Ingest and analyze central bank speeches, idempotently by source hash."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        llm_client: LlmClient,
        *,
        lexicon: HawkishDovishLexicon | None = None,
        stance_service: StanceService | None = None,
        id_factory: IdFactory = default_id_factory,
        model_id: str = "unknown",
    ) -> None:
        """Build the service.

        Args:
            session_factory: Factory for sessions. The service owns the transaction.
            llm_client: The LLM boundary used to summarize and score the speech.
            lexicon: The deterministic lexicon baseline; a default one is used if not supplied.
            stance_service: The structured stance pipeline (ADR 0021); one wired to the same LLM
                and lexicon is built if not supplied.
            id_factory: Source of new identifiers. Inject a deterministic one in tests.
            model_id: The configured model identifier, recorded on each speech so the tone
                series stays comparable as the model changes (the adapter injects it from
                settings).
        """
        self._session_factory = session_factory
        self._llm = llm_client
        self._lexicon = lexicon if lexicon is not None else HawkishDovishLexicon()
        self._stance = (
            stance_service
            if stance_service is not None
            else StanceService(llm_client, lexicon=self._lexicon)
        )
        self._id_factory = id_factory
        self._model_id = model_id

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

        # Phase 2: analyze (no transaction held across the model call). The model's holistic
        # judgement is the headline; the structured pipeline (ADR 0021) adds the rate-path and
        # per-aspect decomposition and cross-checks the headline by direction against the structured
        # net, the classifier (where it applies), and the lexicon. A majority disagreement is
        # flagged, not averaged away (ADR 0008), and surfaced on the speech and its observation.
        analysis = self._llm.analyze_tone(text)
        assessment = self._stance.assess(
            text,
            headline_score=analysis.score,
            headline_tone=analysis.tone,
            central_bank=speaker.central_bank,
        )
        needs_review = assessment.needs_review
        if needs_review:
            log.warning(
                "tone_cross_check_disagreement",
                headline_score=analysis.score,
                structured_net=assessment.structured_net,
                classifier_net=assessment.classifier_net,
                lexicon_score=assessment.lexicon_score,
                uncertainty=assessment.uncertainty,
            )
        speech_id = self._id_factory()
        speech = Speech(
            id=speech_id,
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
            lexicon_score=assessment.lexicon_score,
            rationale=analysis.rationale,
            needs_review=needs_review,
            model_id=self._model_id,
        )
        stance = build_stance(speech_id, assessment, model_id=self._model_id)
        observation = ToneObservation(
            id=self._id_factory(),
            speaker_id=speaker_id,
            observed_at=delivered_at,
            tone=analysis.tone,
            score=analysis.score,
            source_sha256=source_sha256,
            lexicon_score=assessment.lexicon_score,
            needs_review=needs_review,
        )

        # Phase 3: persist the speech, its derived stance decomposition, and its tone signal.
        with self._session_factory() as session:
            SpeechRepository(session).add(speech)
            SpeechStanceRepository(session).upsert(stance)
            ToneObservationRepository(session).append(observation)
            session.commit()

        log.info(
            "speech_ingested",
            speech_id=str(speech.id),
            tone=analysis.tone.value,
            score=analysis.score,
            rate_path=assessment.rate_path,
            uncertainty=assessment.uncertainty,
            lexicon_score=assessment.lexicon_score,
            needs_review=needs_review,
            summary_chars=len(analysis.summary),
            source_bytes=len(encoded),
        )
        return speech

    def get_speech(
        self, speech_id: UUID, *, actor: str = "system", correlation_id: UUID | None = None
    ) -> Speech:
        """Return a single analyzed speech by id.

        Args:
            speech_id: The speech to fetch.
            actor: Who is performing the action.
            correlation_id: Correlation id for this call; one is minted if not supplied.

        Returns:
            The :class:`Speech`.

        Raises:
            EntityNotFoundError: If no speech has that id.
        """
        correlation = correlation_id if correlation_id is not None else uuid4()
        log = _logger.bind(correlation_id=str(correlation), actor=actor, speech_id=str(speech_id))
        with self._session_factory() as session:
            speech = SpeechRepository(session).get(speech_id)
        log.info("speech_fetched")
        return speech

    def get_stance(
        self, speech_id: UUID, *, actor: str = "system", correlation_id: UUID | None = None
    ) -> SpeechStance | None:
        """Return a speech's structured stance decomposition, or ``None`` if it has none yet.

        Args:
            speech_id: The speech whose decomposition to fetch.
            actor: Who is performing the action.
            correlation_id: Correlation id for this call; one is minted if not supplied.

        Returns:
            The :class:`~cbt_core.domain.speech.SpeechStance`, or ``None`` when the speech has not
            been scored by the structured pipeline.
        """
        correlation = correlation_id if correlation_id is not None else uuid4()
        log = _logger.bind(correlation_id=str(correlation), actor=actor, speech_id=str(speech_id))
        with self._session_factory() as session:
            stance = SpeechStanceRepository(session).get(speech_id)
        log.info("speech_stance_fetched", found=stance is not None)
        return stance

    def stances_by_speech(
        self, *, actor: str = "system", correlation_id: UUID | None = None
    ) -> dict[UUID, SpeechStance]:
        """Return every stored stance decomposition, keyed by speech id, for read models.

        Args:
            actor: Who is performing the action.
            correlation_id: Correlation id for this call; one is minted if not supplied.

        Returns:
            A map from speech id to its :class:`~cbt_core.domain.speech.SpeechStance`.
        """
        correlation = correlation_id if correlation_id is not None else uuid4()
        log = _logger.bind(correlation_id=str(correlation), actor=actor)
        with self._session_factory() as session:
            stances = SpeechStanceRepository(session).all_by_speech()
        log.info("speech_stances_listed", count=len(stances))
        return stances

    def list_speeches(
        self, speaker_id: UUID, *, actor: str = "system", correlation_id: UUID | None = None
    ) -> list[Speech]:
        """Return a speaker's analyzed speeches, most recent first.

        Args:
            speaker_id: The speaker whose speeches to read.
            actor: Who is performing the action.
            correlation_id: Correlation id for this call; one is minted if not supplied.

        Returns:
            The speaker's speeches.

        Raises:
            EntityNotFoundError: If the speaker does not exist.
        """
        correlation = correlation_id if correlation_id is not None else uuid4()
        log = _logger.bind(correlation_id=str(correlation), actor=actor, speaker_id=str(speaker_id))
        with self._session_factory() as session:
            SpeakerRepository(session).get(speaker_id)
            speeches = SpeechRepository(session).list_for_speaker(speaker_id)
        log.info("speeches_listed", count=len(speeches))
        return speeches

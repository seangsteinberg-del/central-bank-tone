"""Worker orchestration: scrape sources, ensure speakers, ingest, and index (CLAUDE.md section 2).

Drives scraped speeches through the core services. A single speech that fails the model is logged
and skipped so one bad item does not abort the whole run; everything else is surfaced.
"""

from __future__ import annotations

from collections.abc import Sequence

from cbt_core import IndexingService, IngestionService, LlmError, SpeakerService, get_logger
from cbt_worker.sources.base import SpeechSource

_logger = get_logger(__name__)


def run_ingestion(
    sources: Sequence[SpeechSource],
    *,
    speaker_service: SpeakerService,
    ingestion_service: IngestionService,
    indexing_service: IndexingService,
    limit_per_source: int = 20,
) -> int:
    """Scrape each source and ingest and index its speeches.

    Args:
        sources: The speech sources to scrape.
        speaker_service: Resolves or creates the speaker for each speech.
        ingestion_service: Ingests and analyzes each speech (idempotent by source hash).
        indexing_service: Chunks and embeds each speech for retrieval.
        limit_per_source: Maximum speeches to fetch from each source per run.

    Returns:
        The number of speeches successfully ingested across all sources.
    """
    ingested = 0
    for source in sources:
        speeches = source.fetch(limit=limit_per_source)
        log = _logger.bind(source=source.name)
        log.info("source_scraped", scraped=len(speeches))
        for scraped in speeches:
            speaker = speaker_service.ensure_speaker(
                name=scraped.speaker_name, central_bank=scraped.central_bank, role=scraped.role
            )
            try:
                speech = ingestion_service.ingest_speech(
                    speaker_id=speaker.id,
                    title=scraped.title,
                    url=scraped.url,
                    delivered_at=scraped.delivered_at,
                    text=scraped.text,
                    language=scraped.language,
                )
            except LlmError:
                log.exception("speech_ingest_failed", url=scraped.url)
                continue
            indexing_service.index_speech(speech.id)
            ingested += 1
        log.info("source_ingested", ingested=ingested)
    return ingested

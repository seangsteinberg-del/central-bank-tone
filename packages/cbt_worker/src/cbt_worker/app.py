"""Worker entry point: build the services and run ingestion against a BIS source.

Run the live RSS scraper with ``uv run python -m cbt_worker.app``, or backfill from a downloaded
BIS bulk archive with ``uv run python -m cbt_worker.app --bulk path/to/speeches.zip [--limit N]``.
This is the composition root; the testable logic lives in ``runner`` and the sources.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

from cbt_core import (
    IndexingService,
    IngestionService,
    LazyGeminiClient,
    SpeakerService,
    configure_logging,
    create_engine_from_settings,
    get_settings,
    make_session_factory,
)
from cbt_worker.runner import run_ingestion
from cbt_worker.sources.base import SpeechSource
from cbt_worker.sources.bis import BisSpeechSource
from cbt_worker.sources.bis_bulk import BisBulkSpeechSource

_USER_AGENT = "cbt-worker/0.1 (central-bank-tone research; contact: ops@example.org)"
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 1.0
_REQUEST_DELAY_SECONDS = 0.5  # polite rate limit between requests to one host
_DEFAULT_LIMIT = 50


def _get(url: str) -> str:  # pragma: no cover - network IO, exercised in production only
    response = httpx.get(
        url, timeout=30.0, headers={"User-Agent": _USER_AGENT}, follow_redirects=True
    )
    response.raise_for_status()
    return response.text


def _http_fetcher(url: str) -> str:  # pragma: no cover - network IO, exercised in production only
    """Fetch a URL politely: a small inter-request delay and retry-with-backoff on failure."""
    time.sleep(_REQUEST_DELAY_SECONDS)
    for attempt in range(_MAX_ATTEMPTS - 1):
        try:
            return _get(url)
        except httpx.HTTPError:
            time.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
    return _get(url)  # the final attempt lets the error propagate


def _select_source(
    argv: list[str],
) -> tuple[SpeechSource, int]:  # pragma: no cover - argv wiring, exercised in production only
    """Pick the source and limit from argv: the bulk archive if ``--bulk`` is given, else the RSS."""
    limit = _DEFAULT_LIMIT
    if "--limit" in argv:
        limit = int(argv[argv.index("--limit") + 1])
    if "--bulk" in argv:
        path = Path(argv[argv.index("--bulk") + 1])
        return BisBulkSpeechSource(path.read_bytes), limit
    return BisSpeechSource(_http_fetcher), limit


def main() -> int:  # pragma: no cover - composition root wiring, exercised in production only
    """Build the services and run one ingestion pass over the selected BIS source."""
    settings = get_settings()
    configure_logging(environment=settings.environment)
    engine = create_engine_from_settings(settings)
    session_factory = make_session_factory(engine)
    llm = LazyGeminiClient(settings)
    source, limit = _select_source(sys.argv[1:])

    run_ingestion(
        [source],
        speaker_service=SpeakerService(session_factory),
        ingestion_service=IngestionService(session_factory, llm, model_id=settings.gemini_model),
        indexing_service=IndexingService(session_factory, llm),
        limit_per_source=limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

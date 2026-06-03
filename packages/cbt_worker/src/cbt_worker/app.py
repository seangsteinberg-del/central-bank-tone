"""Worker entry point: build the services and run ingestion against the BIS source.

Run with: ``uv run python -m cbt_worker.app``. This is the composition root; the testable logic
lives in ``runner`` and the sources.
"""

from __future__ import annotations

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
from cbt_worker.sources.bis import BisSpeechSource

_USER_AGENT = "cbt-worker/0.1 (central-bank-tone research; contact: ops@example.org)"


def _http_fetcher(url: str) -> str:  # pragma: no cover - network IO, exercised in production only
    response = httpx.get(url, timeout=30.0, headers={"User-Agent": _USER_AGENT})
    response.raise_for_status()
    return response.text


def main() -> int:  # pragma: no cover - composition root wiring, exercised in production only
    """Build the services and run one ingestion pass over the BIS source."""
    settings = get_settings()
    configure_logging(environment=settings.environment)
    engine = create_engine_from_settings(settings)
    session_factory = make_session_factory(engine)
    llm = LazyGeminiClient(settings)

    run_ingestion(
        [BisSpeechSource(_http_fetcher)],
        speaker_service=SpeakerService(session_factory),
        ingestion_service=IngestionService(session_factory, llm, model_id=settings.gemini_model),
        indexing_service=IndexingService(session_factory, llm),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

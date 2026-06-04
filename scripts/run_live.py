"""Serve the live application: real Gemini over your native PostgreSQL, no Docker (ADR 0018).

Builds the real stack without a container. Tone scoring, concise summaries, embeddings, and
question answering all go through the Google Gemini API (``CBT_GEMINI_API_KEY``); speakers and
speeches persist in PostgreSQL (``CBT_DATABASE_URL``) with the append-only immutability triggers;
and retrieval uses an in-process cosine index, so no pgvector extension is required (ADR 0018).
One ingestion pass scrapes real central-bank speeches from the live BIS index, scores and indexes
them, then the same UI as production is served at http://127.0.0.1:8000.

Prerequisites: a reachable PostgreSQL holding the ``cbt`` database and a valid Gemini key in
``.env``. The free Gemini tier is rate limited, so a pass of a dozen speeches takes a few minutes.

Usage::

    uv run python scripts/run_live.py [--limit N]
"""

from __future__ import annotations

import sys
import time

import httpx
import uvicorn

from cbt_core import (
    LazyGeminiClient,
    create_engine_from_settings,
    create_immutability_triggers,
    get_settings,
)
from cbt_web.demo import build_demo_app
from cbt_worker.runner import run_ingestion
from cbt_worker.sources.bis import BisSpeechSource

_HOST = "127.0.0.1"
_PORT = 8000
_DEFAULT_LIMIT = 12
_USER_AGENT = "cbt-worker/0.1 (central-bank-tone research; contact: ops@example.org)"
_REQUEST_DELAY_SECONDS = 0.5  # polite inter-request delay against the live BIS host
_LIVE_MAX_DISTANCE = 0.6  # learned Gemini embeddings; the production QaService relevance default


def _get(url: str) -> httpx.Response:
    """Fetch a URL politely (a small inter-request delay), raising on any HTTP error.

    Args:
        url: The URL to fetch.

    Returns:
        The HTTP response.
    """
    time.sleep(_REQUEST_DELAY_SECONDS)
    response = httpx.get(
        url, timeout=40.0, headers={"User-Agent": _USER_AGENT}, follow_redirects=True
    )
    response.raise_for_status()
    return response


def _http_fetcher(url: str) -> str:
    """Fetch a URL's text (the RSS listing and speech detail pages)."""
    return _get(url).text


def _pdf_fetcher(url: str) -> bytes:
    """Fetch a URL's raw bytes (a speech's linked full-text PDF)."""
    return _get(url).content


def _limit_from_argv(argv: list[str]) -> int:
    """Read ``--limit N`` from argv, defaulting to :data:`_DEFAULT_LIMIT`.

    Args:
        argv: The argument list (without the program name).

    Returns:
        The number of speeches to fetch from the source this run.
    """
    if "--limit" in argv:
        return int(argv[argv.index("--limit") + 1])
    return _DEFAULT_LIMIT


def main() -> int:
    """Build the live app, ingest one pass of real BIS speeches with Gemini, and serve it."""
    settings = get_settings()
    limit = _limit_from_argv(sys.argv[1:])
    engine = create_engine_from_settings(settings)
    llm = LazyGeminiClient(settings)

    host_and_db = str(settings.database_url).rsplit("@", 1)[-1]
    print("Central Bank Tone - live (real Gemini + PostgreSQL, no Docker, no pgvector)")
    print(f"  model: {settings.gemini_model}; db: {host_and_db}")
    print(f"  ingesting up to {limit} real BIS speeches (Gemini scores and summarizes each) ...")

    app = build_demo_app(
        [],
        engine=engine,
        llm=llm,
        model_id=settings.gemini_model,
        max_distance=_LIVE_MAX_DISTANCE,
    )
    create_immutability_triggers(engine)

    services = app.state.services
    ingested = run_ingestion(
        [BisSpeechSource(_http_fetcher, pdf_fetcher=_pdf_fetcher)],
        speaker_service=services.speaker_service,
        ingestion_service=services.ingestion_service,
        indexing_service=services.indexing_service,
        limit_per_source=limit,
    )

    print(f"  ingested {ingested} speeches; serving on http://{_HOST}:{_PORT}  (Ctrl+C to stop)")
    uvicorn.run(app, host=_HOST, port=_PORT, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

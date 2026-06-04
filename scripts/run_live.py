"""Serve the live application: real Gemini over native PostgreSQL, persistent, no Docker.

Builds the real stack without a container (ADR 0018) and makes it persistent (ADR 0020): tone
scoring, summaries, embeddings, and question answering go through Gemini; speakers, speeches, and
the tone history persist in PostgreSQL behind the append-only triggers; and chunk embeddings persist
in a ``bytea`` column and are reloaded into the in-process cosine index on startup, so a filled
corpus and its question-answering index survive restarts and the embedding is computed once.

It fills the corpus from two sources, both idempotent (deduplicated by source hash, never
re-embedded once stored):

- the BIS bulk per-year archives (``speeches_<year>.zip``), whose CSV carries the full speech text,
  for historical depth across every tracked central bank; and
- the live BIS RSS feed, for the most recent speeches right up to today.

Prerequisites: a reachable PostgreSQL holding the ``cbt`` database and a valid Gemini key in
``.env``. The free Gemini tier is rate limited (the client retries with backoff), so a large fill
takes a while; re-running resumes where it left off.

Usage::

    uv run python scripts/run_live.py [--limit N] [--years ...] [--no-serve] [--no-ocr] [--serve-only]

``--serve-only`` skips the fill and just serves the existing corpus (use it to restart the UI on
new code while a separate ``--no-serve`` fill keeps writing to the same database).

``--no-serve`` runs the fill and exits without serving, so a long backfill can run alongside an
already-serving instance (it writes to the same database); restart the server afterward to reload
the question-answering index over the enlarged corpus.

``--no-ocr`` disables the Gemini-vision OCR fallback for un-extractable (CID-font) PDFs, so a bulk
fill never stalls on the vision endpoint: such a PDF degrades to its HTML intro (or is skipped if
that is too short) instead. Bulk-archive speeches carry their full text in the CSV and never use
OCR regardless. Leave OCR on for a small, recency-focused run when the vision endpoint is healthy.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import uvicorn

from cbt_core import (
    build_gemini_client,
    configure_logging,
    create_engine_from_settings,
    create_immutability_triggers,
    get_settings,
)
from cbt_web.demo import build_demo_app
from cbt_worker.runner import run_ingestion
from cbt_worker.sources.base import BytesProvider, SpeechSource
from cbt_worker.sources.bis import BisSpeechSource, make_pdf_extractor
from cbt_worker.sources.bis_bulk import BisBulkSpeechSource

_HOST = "127.0.0.1"
_PORT = 8000
_DEFAULT_LIMIT = 120
_DEFAULT_YEARS = (2026, 2025, 2024)
_BULK_URL = "https://www.bis.org/speeches/speeches_{year}.zip"
_USER_AGENT = "cbt-worker/0.1 (central-bank-tone research; contact: ops@example.org)"
_REQUEST_DELAY_SECONDS = 0.5
_LIVE_MAX_DISTANCE = 0.6  # learned Gemini embeddings; the production QaService relevance default
_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "bulk"


def _get(url: str) -> httpx.Response:
    """Fetch a URL politely (a small inter-request delay), raising on any HTTP error."""
    time.sleep(_REQUEST_DELAY_SECONDS)
    response = httpx.get(
        url, timeout=120.0, headers={"User-Agent": _USER_AGENT}, follow_redirects=True
    )
    response.raise_for_status()
    return response


def _http_fetcher(url: str) -> str:
    """Fetch a URL's text (the RSS listing and speech detail pages)."""
    return _get(url).text


def _pdf_fetcher(url: str) -> bytes:
    """Fetch a URL's raw bytes (a speech's linked full-text PDF)."""
    return _get(url).content


def _bytes_provider(data: bytes) -> BytesProvider:
    """Wrap already-downloaded archive bytes as a no-argument provider (binds ``data`` per call)."""
    return lambda: data


def _download_bulk(year: int) -> bytes | None:
    """Return a year's BIS bulk archive bytes, cached under ``data/bulk``, or None if unavailable.

    Args:
        year: The archive year.

    Returns:
        The ZIP bytes, or ``None`` if the year is not published yet (a 404).
    """
    cache = _CACHE_DIR / f"speeches_{year}.zip"
    if cache.exists():
        return cache.read_bytes()
    try:
        data = _get(_BULK_URL.format(year=year)).content
    except httpx.HTTPError:
        return None
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(data)
    return data


def _build_sources(
    years: tuple[int, ...], pdf_extractor: Callable[[bytes], str]
) -> list[SpeechSource]:
    """Build the live RSS feed (freshest, fetched first) then the historical bulk archives.

    The RSS source runs first so the most recent speeches, right up to today, land before the long
    rate-limited bulk backfill can exhaust the free Gemini tier's daily quota; the bulk archives then
    fill the history, newest year first. Every source is idempotent (deduplicated by source hash), so
    the ordering only affects which speeches land first under a quota cap, never correctness.
    """
    sources: list[SpeechSource] = [
        BisSpeechSource(_http_fetcher, pdf_fetcher=_pdf_fetcher, pdf_extractor=pdf_extractor)
    ]
    for year in years:
        data = _download_bulk(year)
        if data is None:
            print(f"  bulk {year}: not published yet (skipped)")
            continue
        print(f"  bulk {year}: {len(data) / 1e6:.1f} MB")
        sources.append(BisBulkSpeechSource(_bytes_provider(data)))
    return sources


def _int_arg(flag: str, default: int) -> int:
    """Read an integer ``--flag N`` from argv, or the default."""
    argv = sys.argv[1:]
    return int(argv[argv.index(flag) + 1]) if flag in argv else default


def _years_arg(default: tuple[int, ...]) -> tuple[int, ...]:
    """Read ``--years 2026,2025`` from argv, or the default."""
    argv = sys.argv[1:]
    if "--years" not in argv:
        return default
    return tuple(int(part) for part in argv[argv.index("--years") + 1].split(",") if part.strip())


def main() -> int:
    """Build the persistent live app, fill it from the archives and the live feed, and serve it."""
    settings = get_settings()
    configure_logging(environment=settings.environment)
    limit = _int_arg("--limit", _DEFAULT_LIMIT)
    years = _years_arg(_DEFAULT_YEARS)
    engine = create_engine_from_settings(settings)
    llm = build_gemini_client(settings)

    host_and_db = str(settings.database_url).rsplit("@", 1)[-1]
    print("Central Bank Tone - live (real Gemini + PostgreSQL, persistent, no Docker, no pgvector)")
    print(f"  model: {settings.gemini_model}; db: {host_and_db}")

    app = build_demo_app(
        [],
        engine=engine,
        llm=llm,
        model_id=settings.gemini_model,
        max_distance=_LIVE_MAX_DISTANCE,
        persistent_retrieval=True,
    )
    create_immutability_triggers(engine)

    if "--serve-only" not in sys.argv:
        services = app.state.services
        print(f"  filling up to {limit} speeches/source (Gemini scores and summarizes each) ...")
        transcribe = None if "--no-ocr" in sys.argv else llm.transcribe_image
        sources = _build_sources(years, make_pdf_extractor(transcribe=transcribe))
        ingested = run_ingestion(
            sources,
            speaker_service=services.speaker_service,
            ingestion_service=services.ingestion_service,
            indexing_service=services.indexing_service,
            limit_per_source=limit,
        )
        if "--no-serve" in sys.argv:
            print(f"  ingested {ingested} new speeches; --no-serve set, not serving")
            return 0
        print(f"  ingested {ingested} new speeches; serving on http://{_HOST}:{_PORT}")
    else:
        print(f"  --serve-only: serving the existing corpus on http://{_HOST}:{_PORT}")

    uvicorn.run(app, host=_HOST, port=_PORT, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

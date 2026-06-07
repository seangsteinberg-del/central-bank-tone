"""Backfill every tracked-bank speech from a start year to a cutoff date onto SQLite (no Docker).

Two sources feed one ingest:

- the BIS bulk per-year archives (``speeches_<year>.zip``) for complete past years, whose CSV
  carries the full speech text; and
- a sitemap-driven scrape of the current year (whose bulk archive is not published yet):
  ``sitemap_documents_<year>.xml`` enumerates every ``/review/r<YYMMDD>x.htm`` speech, and each
  detail page's react-props carry the speaker (``short_title`` "Name: Title"), the institution and
  role (``description``), the delivery date, and the body (HTML plus the linked PDF).

Real Gemini scores and summarizes each speech; speakers, speeches, the append-only tone history,
the stance decomposition, and the chunk embeddings persist in a SQLite file (ADR 0018, 0020). The
ingest is idempotent by source hash, so a re-run resumes where it left off. SQLite serializes
writes, so this runs sequentially.

Usage::

    CBT_GEMINI_API_KEY=... uv run python scripts/run_backfill.py \
        [--db data/cbt_live.db] [--cutoff 2026-06-05] [--years 2023,2024,2025] \
        [--current-year 2026] [--limit N] [--smoke]
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import httpx
from selectolax.parser import HTMLParser
from sqlalchemy import create_engine

from cbt_core import build_gemini_client, configure_logging, get_logger, get_settings
from cbt_web.demo import build_demo_services
from cbt_worker.runner import run_ingestion
from cbt_worker.sources.base import (
    ScrapedSpeech,
    SpeechSource,
    affiliation_from,
    central_bank_from,
    role_from,
)
from cbt_worker.sources.bis import _looks_like_text, make_pdf_extractor
from cbt_worker.sources.bis_bulk import BisBulkSpeechSource

_logger = get_logger("run_backfill")

_BASE = "https://www.bis.org"
_BULK_URL = _BASE + "/speeches/speeches_{year}.zip"
_SITEMAP_URL = _BASE + "/sitemap_documents_{year}.xml"
_UA = {"User-Agent": "cbt-worker/0.1 (central-bank-tone research; contact: ops@example.org)"}
_CACHE = Path(__file__).resolve().parents[1] / "data" / "bulk"
_REQUEST_DELAY = 0.4
_MIN_WORDS = 50
_REVIEW_RE = re.compile(r"https://www\.bis\.org/review/r(\d{2})(\d{2})(\d{2})[a-z]?\.htm")


def _get(url: str) -> httpx.Response:
    time.sleep(_REQUEST_DELAY)
    response = httpx.get(url, headers=_UA, timeout=120.0, follow_redirects=True)
    response.raise_for_status()
    return response


def _download_bulk(year: int) -> bytes | None:
    cache = _CACHE / f"speeches_{year}.zip"
    if cache.exists():
        return cache.read_bytes()
    try:
        data = _get(_BULK_URL.format(year=year)).content
    except httpx.HTTPError:
        return None
    _CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(data)
    return data


class BisSitemapSource:
    """Scrape a year's BIS speeches from its document sitemap and detail pages."""

    name = "bis-sitemap"

    def __init__(self, year: int, cutoff: dt.date) -> None:
        self._year = year
        self._cutoff = cutoff
        self._extractor = make_pdf_extractor(transcribe=None)  # no Gemini-vision OCR in a bulk fill

    def fetch(self, *, limit: int) -> list[ScrapedSpeech]:
        sitemap = _get(_SITEMAP_URL.format(year=self._year)).text
        seen: set[str] = set()
        dated: list[tuple[dt.date, str]] = []
        for match in _REVIEW_RE.finditer(sitemap):
            url = match.group(0)
            if url in seen:
                continue
            seen.add(url)
            day = dt.date(2000 + int(match.group(1)), int(match.group(2)), int(match.group(3)))
            if dt.date(self._year, 1, 1) <= day <= self._cutoff:
                dated.append((day, url))
        dated.sort(reverse=True)  # newest first, so a capped --limit keeps the freshest
        print(f"  sitemap {self._year}: {len(dated)} speeches up to {self._cutoff}")
        results: list[ScrapedSpeech] = []
        for day, url in dated:
            if len(results) >= limit:
                break
            try:
                speech = self._scrape(url, day)
            except Exception:
                _logger.exception("sitemap_scrape_failed", url=url)
                continue
            if speech is not None:
                results.append(speech)
        return results

    def _scrape(self, url: str, day: dt.date) -> ScrapedSpeech | None:
        node = HTMLParser(_get(url).text).css_first("[data-react-props]")
        if node is None:
            return None
        raw = node.attributes.get("data-react-props")
        document = json.loads(raw).get("document") if raw else None
        if not isinstance(document, dict):
            return None
        description = str(document.get("description") or "")
        bank = central_bank_from(affiliation_from(description))
        if bank is None:
            return None  # an untracked institution
        short_title = str(document.get("short_title") or "")
        speaker, _, title = short_title.partition(":")
        speaker = speaker.strip()
        title = title.strip() or short_title.strip()
        if not speaker or not title:
            return None
        text = self._body(document)
        if len(text.split()) < _MIN_WORDS:
            return None
        published = document.get("publication_start_date")
        delivered = (
            dt.datetime.fromisoformat(published)
            if isinstance(published, str) and published
            else dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC)
        )
        return ScrapedSpeech(
            speaker_name=speaker,
            central_bank=bank,
            role=role_from(description),
            title=title,
            url=url,
            delivered_at=delivered,
            text=text,
        )

    def _body(self, document: dict[str, object]) -> str:
        content = document.get("content")
        html_text = (
            HTMLParser(content).text(separator=" ", strip=True)
            if isinstance(content, str) and content
            else ""
        )
        path = document.get("path")
        pdf_path = path if isinstance(path, str) and path.lower().endswith(".pdf") else None
        if pdf_path is None:
            return html_text
        pdf_url = pdf_path if pdf_path.startswith("http") else _BASE + pdf_path
        try:
            pdf_text = self._extractor(_get(pdf_url).content)
        except Exception:  # noqa: BLE001 - a bad PDF degrades to the HTML intro, never drops the speech
            return html_text
        if not _looks_like_text(pdf_text):
            return html_text
        return pdf_text if len(pdf_text.split()) > len(html_text.split()) else html_text


def _arg(flag: str, default: str) -> str:
    argv = sys.argv[1:]
    return argv[argv.index(flag) + 1] if flag in argv else default


def main() -> int:
    settings = get_settings()
    configure_logging(environment=settings.environment)
    smoke = "--smoke" in sys.argv
    db_path = Path(_arg("--db", "data/cbt_live.db")).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cutoff = dt.date.fromisoformat(_arg("--cutoff", "2026-06-05"))
    current_year = int(_arg("--current-year", "2026"))
    years = [int(y) for y in _arg("--years", "2023,2024,2025").split(",") if y.strip()]
    limit = 3 if smoke else int(_arg("--limit", "100000"))

    print("Central Bank Tone - backfill (real Gemini + SQLite, no Docker)")
    print(f"  db: {db_path}")
    print(f"  model: {settings.gemini_model}; cutoff: {cutoff}; limit/source: {limit}")

    engine = create_engine(
        f"sqlite+pysqlite:///{db_path.as_posix()}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    services = build_demo_services(
        [],
        engine=engine,
        llm=build_gemini_client(settings),
        model_id=settings.gemini_model,
        max_distance=0.6,
        persistent_retrieval=True,
    )

    sources: list[SpeechSource] = [BisSitemapSource(current_year, cutoff)]
    if not smoke:
        for year in years:
            data = _download_bulk(year)
            if data is None:
                print(f"  bulk {year}: not available (skipped)")
                continue
            print(f"  bulk {year}: {len(data) / 1e6:.1f} MB")
            sources.append(BisBulkSpeechSource(lambda d=data: d))

    print("  ingesting (Gemini scores and summarizes each; idempotent by source hash) ...")
    ingested = run_ingestion(
        sources,
        speaker_service=services.speaker_service,
        ingestion_service=services.ingestion_service,
        indexing_service=services.indexing_service,
        limit_per_source=limit,
    )
    print(f"  done: {ingested} speeches ingested this pass (re-run to resume any that failed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Scrape the BIS central bankers' speeches index (ADR 0010).

The BIS aggregates speeches from every central bank in one place, so a single source covers all
tracked institutions. The contract was verified against the live site (bis.org is a React app):

- The listing comes from the RSS feed (``/doclist/cbspeeches.rss``), an RSS 1.0 / Dublin Core
  document that is far more stable than the page's HTML. Each item carries the speaker
  (``dc:creator``), the delivery time (``dc:date``), the title, the speech URL, and a description
  that names the institution and role.
- The speech body lives in a ``data-react-props`` JSON blob on the detail page. ``document.content``
  is the speech HTML, but for many speeches that is only a short intro and the full text is the
  linked PDF at ``document.path``. We always bring in the full text (ADR 0019): the PDF is fetched
  and extracted, and preferred whenever it is fuller than the HTML intro. A PDF that extracts to
  glyph soup (subset fonts with no ToUnicode map) is rejected and the HTML intro is used instead,
  so no unreadable text is ever ingested (CLAUDE.md section 3).

The fetcher is injected, so this is tested against committed fixtures with no network. Speeches
from institutions outside the schema spine are skipped.
"""

from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from pdfminer.high_level import extract_text as _pdfminer_extract_text
from selectolax.parser import HTMLParser

from cbt_core import get_logger
from cbt_worker.sources.base import (
    Fetcher,
    ScrapedSpeech,
    affiliation_from,
    central_bank_from,
    role_from,
)

_logger = get_logger(__name__)

# Fetches the raw bytes of a URL (a PDF). Injected like the text fetcher, so the source is tested
# with no network.
PdfFetcher = Callable[[str], bytes]

_BASE_URL = "https://www.bis.org"
_LISTING_URL = "https://www.bis.org/doclist/cbspeeches.rss"
_RDF_ABOUT = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"

# The shortest extracted body we treat as a real speech. A body below this floor (an empty page, or
# a slides-only stub whose PDF is unreadable) has too little prose to score a meaningful tone, so it
# is skipped, not ingested.
_MIN_BODY_WORDS = 50

# A PDF text extraction must be at least this fraction letters-and-spaces to count as real prose;
# BIS PDFs with subset fonts and no ToUnicode map extract as "(cid:N)" glyph soup that fails this.
_MIN_LEGIBLE_FRACTION = 0.65


def _parse_date(text: str) -> datetime:
    """Parse a BIS ISO-8601 ``dc:date`` such as ``2024-03-25T12:40:00Z`` as a tz-aware datetime."""
    return datetime.fromisoformat(text)


def _extract_pdf_text(data: bytes) -> str:
    """Extract text from a PDF's bytes with pdfminer.six, collapsing runs of whitespace.

    Args:
        data: The raw bytes of a PDF document.

    Returns:
        The document text as a single whitespace-normalized string.
    """
    return " ".join(_pdfminer_extract_text(io.BytesIO(data)).split())


def _looks_like_text(body: str) -> bool:
    """Return whether ``body`` reads as natural-language prose rather than PDF glyph soup.

    A PDF whose embedded subset fonts carry no ToUnicode map extracts as ``(cid:N)`` runs (or other
    glyph-index noise). Ingesting that would feed the model garbage (CLAUDE.md section 3), so such
    an extraction is rejected and the caller falls back to the HTML intro.

    Args:
        body: The extracted text to judge.

    Returns:
        True if the text is non-empty, free of ``(cid:`` markers, and mostly letters and spaces.
    """
    if not body or "(cid:" in body:
        return False
    legible = sum(character.isalpha() or character.isspace() for character in body)
    return legible / len(body) > _MIN_LEGIBLE_FRACTION


def _clean_title(title: str, speaker: str) -> str:
    """Strip a BIS author prefix from a listing title.

    BIS formats a listing title as ``"Author[, Co-author]: Real title"``. The prefix can carry
    co-authors absent from ``dc:creator`` (for example ``"Carolyn Rogers,Toni Gravelle: ..."``), so
    it is removed only when the segment before the first colon names the speaker (by surname); a
    colon inside the real title (``"Inflation: the road ahead"``) is left intact.

    Args:
        title: The raw listing title.
        speaker: The ``dc:creator`` speaker name.

    Returns:
        The title without the author prefix, or the original title when no author prefix is found.
    """
    head, separator, rest = title.partition(":")
    parts = speaker.split()
    if separator and rest.strip() and parts and parts[-1].lower() in head.lower():
        return rest.strip()
    return title


def _localname(tag: str) -> str:
    """Return an XML tag's local name, dropping any ``{namespace}`` prefix."""
    return tag.rsplit("}", 1)[-1]


@dataclass(frozen=True)
class _ListingEntry:
    speaker: str
    role: str
    institution: str
    title: str
    url: str
    delivered_at: datetime


class BisSpeechSource:
    """Scrapes speeches from the BIS central bankers' speeches RSS feed."""

    name = "bis"

    def __init__(
        self,
        fetcher: Fetcher,
        *,
        base_url: str = _BASE_URL,
        listing_url: str = _LISTING_URL,
        pdf_fetcher: PdfFetcher | None = None,
        pdf_extractor: Callable[[bytes], str] = _extract_pdf_text,
    ) -> None:
        """Build the source.

        Args:
            fetcher: Fetches the text of a URL.
            base_url: The BIS base URL, used to absolutize relative links.
            listing_url: The speeches RSS feed URL.
            pdf_fetcher: Fetches the raw bytes of a PDF URL, used to pull a speech's full text from
                its linked PDF. When ``None`` (the default) only the HTML body is used.
            pdf_extractor: Extracts text from PDF bytes. Injectable for tests; defaults to
                :func:`_extract_pdf_text` (pdfminer.six).
        """
        self._fetcher = fetcher
        self._base_url = base_url
        self._listing_url = listing_url
        self._pdf_fetcher = pdf_fetcher
        self._pdf_extractor = pdf_extractor

    def fetch(self, *, limit: int) -> list[ScrapedSpeech]:
        """Fetch up to ``limit`` recent speeches from tracked institutions.

        Args:
            limit: The maximum number of speeches to return.

        Returns:
            The scraped speeches, skipping untracked institutions and bodies too short to score.
        """
        entries = self._parse_listing(self._fetcher(self._listing_url))
        results: list[ScrapedSpeech] = []
        for entry in entries:
            if len(results) >= limit:
                break
            bank = central_bank_from(entry.institution)
            if bank is None:
                _logger.info("bis_speech_skipped_untracked", institution=entry.institution)
                continue
            html_text, pdf_path = self._parse_detail(self._fetcher(entry.url))
            text = self._full_text(html_text, pdf_path)
            word_count = len(text.split())
            if word_count < _MIN_BODY_WORDS:
                # An empty page or an unreadable stub: too little prose to score a real tone.
                _logger.warning("bis_speech_skipped_short", url=entry.url, words=word_count)
                continue
            results.append(
                ScrapedSpeech(
                    speaker_name=entry.speaker,
                    central_bank=bank,
                    role=entry.role,
                    title=entry.title,
                    url=entry.url,
                    delivered_at=entry.delivered_at,
                    text=text,
                )
            )
        return results

    def _parse_listing(self, rss: str) -> list[_ListingEntry]:
        try:
            # Trusted BIS feed over HTTPS; ElementTree resolves no external entities by default.
            root = ET.fromstring(rss)  # noqa: S314
        except ET.ParseError:
            _logger.warning("bis_listing_unparseable")
            return []
        entries: list[_ListingEntry] = []
        for item in root.iter():
            if _localname(item.tag) != "item":
                continue
            fields = {_localname(child.tag): (child.text or "").strip() for child in item}
            url = fields.get("link") or item.attrib.get(_RDF_ABOUT, "")
            speaker = fields.get("creator", "")
            description = fields.get("description", "")
            date_text = fields.get("date", "")
            if not (url and speaker and date_text):
                continue
            entries.append(
                _ListingEntry(
                    speaker=speaker,
                    role=role_from(description),
                    institution=affiliation_from(description),
                    title=_clean_title(fields.get("title", ""), speaker),
                    url=self._absolute(url),
                    delivered_at=_parse_date(date_text),
                )
            )
        return entries

    def _parse_detail(self, html: str) -> tuple[str, str | None]:
        """Return ``(html_body_text, pdf_path)`` from a detail page's react props.

        Args:
            html: The detail page HTML.

        Returns:
            The HTML body text (empty if absent) and the linked PDF path (``None`` if absent).
        """
        node = HTMLParser(html).css_first("[data-react-props]")
        if node is None:
            return "", None
        raw = node.attributes.get("data-react-props")
        if not raw:
            return "", None
        try:
            props = json.loads(raw)
        except json.JSONDecodeError:
            return "", None
        document = props.get("document") if isinstance(props, dict) else None
        if not isinstance(document, dict):
            return "", None
        content = document.get("content")
        html_text = (
            HTMLParser(content).text(separator=" ", strip=True)
            if isinstance(content, str) and content
            else ""
        )
        path = document.get("path")
        pdf_path = path if isinstance(path, str) and path.lower().endswith(".pdf") else None
        return html_text, pdf_path

    def _full_text(self, html_text: str, pdf_path: str | None) -> str:
        """Return the fullest legible body: the linked PDF when it beats the HTML intro.

        BIS often publishes only a short HTML intro and the full speech as a linked PDF, so the PDF
        is fetched and preferred whenever it yields more legible text. An un-extractable PDF (glyph
        soup from subset fonts) or a failed fetch falls back to the HTML intro, so a speech is never
        ingested as unreadable text (CLAUDE.md section 3).

        Args:
            html_text: The HTML body text from the detail page.
            pdf_path: The linked PDF path, or ``None``.

        Returns:
            The chosen body text.
        """
        if pdf_path is None or self._pdf_fetcher is None:
            return html_text
        pdf_url = self._absolute(pdf_path)
        try:
            pdf_text = self._pdf_extractor(self._pdf_fetcher(pdf_url))
        except Exception:
            # The PDF fetcher and extractor are injected, so any implementation's failure can
            # surface here; a network or parse error must degrade to the HTML body, never drop the
            # speech (the same batch-isolation pattern as the runner).
            _logger.exception("bis_pdf_fetch_failed", url=pdf_url)
            return html_text
        if not _looks_like_text(pdf_text):
            _logger.info("bis_pdf_not_legible", url=pdf_url)
            return html_text
        return pdf_text if len(pdf_text.split()) > len(html_text.split()) else html_text

    def _absolute(self, href: str) -> str:
        if href.startswith("http"):
            return href
        return f"{self._base_url}{href}"

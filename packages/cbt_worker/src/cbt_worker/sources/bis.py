"""Scrape the BIS central bankers' speeches index (ADR 0010).

The BIS aggregates speeches from every central bank in one place, so a single source covers all
tracked institutions. The contract was verified against the live site (bis.org is a React app):

- The listing comes from the RSS feed (``/doclist/cbspeeches.rss``), an RSS 1.0 / Dublin Core
  document that is far more stable than the page's HTML. Each item carries the speaker
  (``dc:creator``), the delivery time (``dc:date``), the title, the speech URL, and a description
  that names the institution and role.
- The speech body lives in a ``data-react-props`` JSON blob on the detail page (``document.content``
  is the speech HTML), not in a server-rendered content div.

The fetcher is injected, so this is tested against committed fixtures with no network. Speeches
from institutions outside the schema spine are skipped.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime

from selectolax.parser import HTMLParser

from cbt_core import CentralBank, get_logger
from cbt_worker.sources.base import Fetcher, ScrapedSpeech

_logger = get_logger(__name__)

_BASE_URL = "https://www.bis.org"
_LISTING_URL = "https://www.bis.org/doclist/cbspeeches.rss"
_RDF_ABOUT = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"

# Map BIS institution names onto the schema spine. Only tracked banks are kept.
_INSTITUTION_TO_BANK: tuple[tuple[str, CentralBank], ...] = (
    ("federal reserve", CentralBank.FEDERAL_RESERVE),
    ("european central bank", CentralBank.ECB),
    ("bank of england", CentralBank.BANK_OF_ENGLAND),
    ("bank of japan", CentralBank.BANK_OF_JAPAN),
    ("bank of canada", CentralBank.BANK_OF_CANADA),
    ("reserve bank of australia", CentralBank.RESERVE_BANK_OF_AUSTRALIA),
    ("swiss national bank", CentralBank.SWISS_NATIONAL_BANK),
    ("people's bank of china", CentralBank.PEOPLES_BANK_OF_CHINA),
    ("peoples bank of china", CentralBank.PEOPLES_BANK_OF_CHINA),
)

# BIS descriptions read "<type> by <Title> <Name>, <Role> of (the) <Institution>, at <venue>...".
# Capture the role between the first comma and " of "/" at ", and the speaker's own institution
# from the affiliation clause (not from the venue, which could also name a tracked bank).
_ROLE_RE = re.compile(r",\s*([A-Z][A-Za-z' .-]{1,80}?)\s+(?:of|at)\b")
_AFFILIATION_RE = re.compile(r",\s*[A-Z][^,]*?\bof\b\s+(?:the\s+)?(.+?)\s*,")


def _central_bank_from(text: str) -> CentralBank | None:
    """Map an institution string onto the schema spine, or ``None`` if untracked."""
    lowered = text.lower()
    for needle, bank in _INSTITUTION_TO_BANK:
        if needle in lowered:
            return bank
    return None


def _role_from(description: str) -> str:
    """Best-effort role extracted from the RSS description, or a generic fallback."""
    match = _ROLE_RE.search(description)
    return match.group(1).strip() if match is not None else "Central banker"


def _affiliation_from(description: str) -> str:
    """The speaker's own institution from the affiliation clause, or the full description.

    Reading the affiliation rather than scanning the whole description avoids misattributing a
    speaker to a tracked bank merely because they spoke at that bank's venue.
    """
    match = _AFFILIATION_RE.search(description)
    return match.group(1).strip() if match is not None else description


def _parse_date(text: str) -> datetime:
    """Parse a BIS ISO-8601 ``dc:date`` such as ``2024-03-25T12:40:00Z`` as a tz-aware datetime."""
    return datetime.fromisoformat(text)


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
        self, fetcher: Fetcher, *, base_url: str = _BASE_URL, listing_url: str = _LISTING_URL
    ) -> None:
        """Build the source.

        Args:
            fetcher: Fetches the text of a URL.
            base_url: The BIS base URL, used to absolutize relative links.
            listing_url: The speeches RSS feed URL.
        """
        self._fetcher = fetcher
        self._base_url = base_url
        self._listing_url = listing_url

    def fetch(self, *, limit: int) -> list[ScrapedSpeech]:
        """Fetch up to ``limit`` recent speeches from tracked institutions.

        Args:
            limit: The maximum number of speeches to return.

        Returns:
            The scraped speeches, skipping untracked institutions and empty bodies.
        """
        entries = self._parse_listing(self._fetcher(self._listing_url))
        results: list[ScrapedSpeech] = []
        for entry in entries:
            if len(results) >= limit:
                break
            bank = _central_bank_from(entry.institution)
            if bank is None:
                _logger.info("bis_speech_skipped_untracked", institution=entry.institution)
                continue
            text = self._parse_detail(self._fetcher(entry.url))
            if not text:
                _logger.warning("bis_speech_skipped_empty_body", url=entry.url)
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
            title = fields.get("title", "")
            prefix = f"{speaker}:"
            clean_title = title[len(prefix) :].strip() if title.startswith(prefix) else title
            entries.append(
                _ListingEntry(
                    speaker=speaker,
                    role=_role_from(description),
                    institution=_affiliation_from(description),
                    title=clean_title or title,
                    url=self._absolute(url),
                    delivered_at=_parse_date(date_text),
                )
            )
        return entries

    def _parse_detail(self, html: str) -> str:
        node = HTMLParser(html).css_first("[data-react-props]")
        if node is None:
            return ""
        raw = node.attributes.get("data-react-props")
        if not raw:
            return ""
        try:
            props = json.loads(raw)
        except json.JSONDecodeError:
            return ""
        document = props.get("document") if isinstance(props, dict) else None
        content = document.get("content") if isinstance(document, dict) else None
        if not isinstance(content, str) or not content:
            return ""
        return HTMLParser(content).text(separator=" ", strip=True)

    def _absolute(self, href: str) -> str:
        if href.startswith("http"):
            return href
        return f"{self._base_url}{href}"

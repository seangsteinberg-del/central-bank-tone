"""Scrape the BIS central bankers' speeches index (ADR 0010).

The BIS aggregates speeches from every central bank in one place, so a single source covers all
tracked institutions. This parser targets a documented listing/detail HTML contract (see the
fixtures in the tests); the live BIS selectors must be verified and adjusted against the real
site. The fetcher is injected, so this is tested against fixtures with no network. Speeches from
institutions outside the schema spine are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from selectolax.parser import HTMLParser

from cbt_core import CentralBank, get_logger
from cbt_worker.sources.base import Fetcher, ScrapedSpeech

_logger = get_logger(__name__)

_BASE_URL = "https://www.bis.org"
_LISTING_URL = "https://www.bis.org/doclist/cbspeeches.htm"

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


def _central_bank_from(text: str) -> CentralBank | None:
    """Map an institution string onto the schema spine, or ``None`` if untracked."""
    lowered = text.lower()
    for needle, bank in _INSTITUTION_TO_BANK:
        if needle in lowered:
            return bank
    return None


def _parse_date(text: str) -> datetime:
    """Parse a BIS listing date such as ``25 Mar 2024`` as a UTC datetime."""
    return datetime.strptime(text, "%d %b %Y").replace(tzinfo=UTC)


@dataclass(frozen=True)
class _ListingEntry:
    speaker: str
    role: str
    institution: str
    title: str
    url: str
    delivered_at: datetime


class BisSpeechSource:
    """Scrapes speeches from the BIS central bankers' speeches index."""

    name = "bis"

    def __init__(
        self, fetcher: Fetcher, *, base_url: str = _BASE_URL, listing_url: str = _LISTING_URL
    ) -> None:
        """Build the source.

        Args:
            fetcher: Fetches the HTML of a URL.
            base_url: The BIS base URL, used to absolutize relative links.
            listing_url: The speeches listing URL.
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

    def _parse_listing(self, html: str) -> list[_ListingEntry]:
        tree = HTMLParser(html)
        entries: list[_ListingEntry] = []
        for row in tree.css("tr.speech"):
            date_node = row.css_first("td.date")
            speaker_node = row.css_first("td.speaker")
            institution_node = row.css_first("td.institution")
            link_node = row.css_first("td.title a")
            if date_node is None or speaker_node is None or institution_node is None:
                continue
            if link_node is None:
                continue
            href = link_node.attributes.get("href") or ""
            role_node = row.css_first("td.role")
            entries.append(
                _ListingEntry(
                    speaker=speaker_node.text(strip=True),
                    role=role_node.text(strip=True) if role_node is not None else "",
                    institution=institution_node.text(strip=True),
                    title=link_node.text(strip=True),
                    url=self._absolute(href),
                    delivered_at=_parse_date(date_node.text(strip=True)),
                )
            )
        return entries

    def _parse_detail(self, html: str) -> str:
        body = HTMLParser(html).css_first("div.speech-body")
        return body.text(separator=" ", strip=True) if body is not None else ""

    def _absolute(self, href: str) -> str:
        if href.startswith("http"):
            return href
        return f"{self._base_url}{href}"

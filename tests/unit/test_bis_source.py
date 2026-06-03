"""Tests for the BIS speech source against real-structure fixtures (CLAUDE.md section 5, no network).

The fixtures mirror the real bis.org contract verified against the live site: an RSS 1.0 / Dublin
Core listing feed and a detail page whose body lives in a ``data-react-props`` JSON blob. Real
institution names are used (to exercise the schema-spine mapping) with placeholder speakers and
bodies, so no copyrighted speech text is redistributed.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from cbt_core import CentralBank
from cbt_worker.sources.bis import BisSpeechSource

_BASE = "https://test.example"
_LISTING_URL = "https://test.example/cbspeeches.rss"

# RSS 1.0 / RDF feed: one Fed item, one ECB item (relative link), and one untracked item whose
# speaker is from an untracked bank but who spoke *at* the ECB (tests venue disambiguation).
_LISTING = """<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns="http://purl.org/rss/1.0/"
         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel rdf:about="https://test.example/cbspeeches.rss">
    <title>Central bankers' speeches</title>
    <link>https://test.example/cbspeeches</link>
  </channel>
  <item rdf:about="https://test.example/review/r1.htm">
    <title>Jane Doe: On inflation</title>
    <link>https://test.example/review/r1.htm</link>
    <description>Speech by Ms Jane Doe, Chair of the Federal Reserve System, at a forum, 25 March 2024.</description>
    <dc:creator>Jane Doe</dc:creator>
    <dc:date>2024-03-25T10:00:00Z</dc:date>
    <dc:title>Jane Doe: On inflation</dc:title>
  </item>
  <item rdf:about="/review/r2.htm">
    <title>Max Mustermann: The outlook</title>
    <link>/review/r2.htm</link>
    <description>Speech by Mr Max Mustermann, President of the European Central Bank, at a conference, 10 April 2024.</description>
    <dc:creator>Max Mustermann</dc:creator>
    <dc:date>2024-04-10T09:00:00Z</dc:date>
    <dc:title>Max Mustermann: The outlook</dc:title>
  </item>
  <item rdf:about="https://test.example/review/r3.htm">
    <title>A N Other: Visitor remarks</title>
    <link>https://test.example/review/r3.htm</link>
    <description>Speech by Mr A N Other, Governor of the Reserve Bank of Narnia, at the European Central Bank visitors' day, 1 May 2024.</description>
    <dc:creator>A N Other</dc:creator>
    <dc:date>2024-05-01T09:00:00Z</dc:date>
    <dc:title>A N Other: Visitor remarks</dc:title>
  </item>
</rdf:RDF>
"""


def _detail(body_html: str) -> str:
    """A detail page carrying the speech body in a data-react-props JSON blob, as BIS does."""
    props = '{"document":{"content":"' + body_html + '"}}'
    return f"<html><body><div data-react-class=\"MainMenu\" data-react-props='{props}'></div></body></html>"


_DETAIL_1 = _detail("<p>Inflation remains elevated; policy will stay restrictive.</p>")
_DETAIL_2 = _detail("<p>The outlook is balanced with some downside risks.</p>")
_DETAIL_NO_PROPS = "<html><body><div>no react props here</div></body></html>"


def _source(pages: dict[str, str]) -> BisSpeechSource:
    def fetch(url: str) -> str:
        return pages[url]

    return BisSpeechSource(fetch, base_url=_BASE, listing_url=_LISTING_URL)


def _pages(detail_2: str = _DETAIL_2) -> dict[str, str]:
    return {
        _LISTING_URL: _LISTING,
        "https://test.example/review/r1.htm": _DETAIL_1,
        "https://test.example/review/r2.htm": detail_2,
    }


@pytest.mark.unit
def test_fetch_parses_tracked_speeches_and_skips_untracked() -> None:
    speeches = _source(_pages()).fetch(limit=10)
    assert [s.speaker_name for s in speeches] == ["Jane Doe", "Max Mustermann"]
    assert speeches[0].central_bank is CentralBank.FEDERAL_RESERVE
    assert speeches[0].role == "Chair"
    assert speeches[0].url == "https://test.example/review/r1.htm"
    assert speeches[0].delivered_at == datetime(2024, 3, 25, 10, 0, tzinfo=UTC)
    assert "restrictive" in speeches[0].text
    # The relative link is absolutized; the speaker is mapped to the ECB.
    assert speeches[1].central_bank is CentralBank.ECB
    assert speeches[1].url == "https://test.example/review/r2.htm"


@pytest.mark.unit
def test_untracked_speaker_at_a_tracked_venue_is_not_misattributed() -> None:
    # The Narnia governor spoke at the ECB, but their affiliation is Narnia, so they are skipped.
    speeches = _source(_pages()).fetch(limit=10)
    assert all(s.speaker_name != "A N Other" for s in speeches)


@pytest.mark.unit
def test_fetch_respects_the_limit() -> None:
    speeches = _source(_pages()).fetch(limit=1)
    assert len(speeches) == 1
    assert speeches[0].speaker_name == "Jane Doe"


@pytest.mark.unit
def test_fetch_skips_speeches_with_an_empty_body() -> None:
    speeches = _source(_pages(detail_2=_DETAIL_NO_PROPS)).fetch(limit=10)
    assert [s.speaker_name for s in speeches] == ["Jane Doe"]


@pytest.mark.unit
def test_fetch_returns_empty_on_unparseable_feed() -> None:
    fetch: Callable[[str], str] = lambda _url: "<rdf:RDF><item>unclosed"  # noqa: E731
    assert BisSpeechSource(fetch, listing_url=_LISTING_URL).fetch(limit=5) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "html",
    [
        "<html><body><div>no react props here</div></body></html>",
        "<div data-react-props=''></div>",
        "<div data-react-props='not-json'></div>",
        "<div data-react-props='{\"document\":{}}'></div>",
        '<div data-react-props=\'{"document":{"content":""}}\'></div>',
    ],
)
def test_parse_detail_returns_empty_on_unusable_pages(html: str) -> None:
    source = BisSpeechSource(lambda _url: "", listing_url=_LISTING_URL)
    assert source._parse_detail(html) == ""  # exercise the detail parser directly


@pytest.mark.unit
def test_item_missing_required_fields_is_skipped() -> None:
    feed = """<?xml version="1.0"?>
    <rdf:RDF xmlns="http://purl.org/rss/1.0/"
             xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
             xmlns:dc="http://purl.org/dc/elements/1.1/">
      <item rdf:about="https://test.example/review/r9.htm">
        <title>No creator or date here</title>
      </item>
    </rdf:RDF>
    """
    fetch: Callable[[str], str] = lambda _url: feed  # noqa: E731
    assert BisSpeechSource(fetch, listing_url=_LISTING_URL).fetch(limit=5) == []

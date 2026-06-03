"""Tests for the BIS speech source against HTML fixtures (CLAUDE.md section 5, no network)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from cbt_core import CentralBank
from cbt_worker.sources.bis import BisSpeechSource

_LISTING = """
<table><tbody>
<tr class="speech">
  <td class="date">25 Mar 2024</td>
  <td class="speaker">Jerome Powell</td>
  <td class="role">Chair</td>
  <td class="institution">Federal Reserve System</td>
  <td class="title"><a href="/review/r1.htm">On inflation</a></td>
</tr>
<tr class="speech">
  <td class="date">10 Apr 2024</td>
  <td class="speaker">Christine Lagarde</td>
  <td class="role">President</td>
  <td class="institution">European Central Bank</td>
  <td class="title"><a href="https://test.example/review/r2.htm">The outlook</a></td>
</tr>
<tr class="speech">
  <td class="date">01 May 2024</td>
  <td class="speaker">A Governor</td>
  <td class="role">Governor</td>
  <td class="institution">Central Bank of Narnia</td>
  <td class="title"><a href="/review/r3.htm">Untracked</a></td>
</tr>
</tbody></table>
"""
_DETAIL_1 = (
    '<div class="speech-body">Inflation remains elevated; policy will stay restrictive.</div>'
)
_DETAIL_2 = '<div class="speech-body">The outlook is balanced with some downside risks.</div>'
_DETAIL_EMPTY = '<div class="other">no body element here</div>'

_BASE = "https://test.example"
_LISTING_URL = "https://test.example/cbspeeches"


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
    assert [s.speaker_name for s in speeches] == ["Jerome Powell", "Christine Lagarde"]
    assert speeches[0].central_bank is CentralBank.FEDERAL_RESERVE
    assert speeches[0].url == "https://test.example/review/r1.htm"
    assert speeches[0].delivered_at == datetime(2024, 3, 25, tzinfo=UTC)
    assert "restrictive" in speeches[0].text
    assert speeches[1].central_bank is CentralBank.ECB


@pytest.mark.unit
def test_fetch_respects_the_limit() -> None:
    speeches = _source(_pages()).fetch(limit=1)
    assert len(speeches) == 1
    assert speeches[0].speaker_name == "Jerome Powell"


@pytest.mark.unit
def test_fetch_skips_speeches_with_an_empty_body() -> None:
    speeches = _source(_pages(detail_2=_DETAIL_EMPTY)).fetch(limit=10)
    assert [s.speaker_name for s in speeches] == ["Jerome Powell"]


_MALFORMED = (
    "<table>"
    '<tr class="speech"><td class="date">x</td></tr>'  # missing speaker/institution
    '<tr class="speech"><td class="date">25 Mar 2024</td><td class="speaker">X</td>'
    '<td class="institution">Federal Reserve</td></tr>'  # has metadata but no title link
    "</table>"
)


@pytest.mark.unit
def test_fetch_tolerates_malformed_rows() -> None:
    fetch: Callable[[str], str] = lambda _url: _MALFORMED  # noqa: E731
    assert BisSpeechSource(fetch, listing_url=_LISTING_URL).fetch(limit=5) == []

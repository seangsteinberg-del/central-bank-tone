"""Tests for the BIS bulk-ZIP speech source against in-memory archives (no network, no files).

The fixtures build a ZIP containing a CSV with the documented columns. Real institution names are
used to exercise the schema-spine mapping and venue disambiguation; speakers, titles, and bodies
are placeholders, so no copyrighted speech text is redistributed.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Callable, Sequence

import pytest

from cbt_core import CentralBank
from cbt_worker.sources.bis_bulk import BisArchiveError, BisBulkSpeechSource

_COLUMNS = ["date", "author", "title", "text", "url", "description"]

# date, author, title, text, url, description. Two tracked rows (Fed, ECB), one untracked speaker
# who merely spoke *at* the ECB (venue disambiguation), one row with an empty body, one bad date.
_ROWS = [
    [
        "2024-03-25",
        "Jane Doe",
        "On inflation",
        "We will keep policy restrictive for some time.",
        "https://www.bis.org/review/r1.htm",
        "Speech by Ms Jane Doe, Chair of the Federal Reserve System, at a forum, 25 March 2024.",
    ],
    [
        "2024-04-10T09:00:00Z",
        "Max Mustermann",
        "The outlook",
        "Inflation has eased and the outlook is more balanced.",
        "https://www.bis.org/review/r2.htm",
        "Speech by Mr Max Mustermann, President of the European Central Bank, at a conference.",
    ],
    [
        "2024-05-01",
        "A N Other",
        "Visitor remarks",
        "A guest perspective on global policy.",
        "https://www.bis.org/review/r3.htm",
        "Speech by Mr A N Other, Governor of the Reserve Bank of Narnia, at the European Central "
        "Bank visitors' day.",
    ],
    [
        "2024-06-01",
        "Empty Body",
        "No text here",
        "   ",
        "https://www.bis.org/review/r4.htm",
        "Speech by Ms Empty Body, Governor of the Bank of England, at a hall.",
    ],
    [
        "not-a-date",
        "Bad Date",
        "When was this",
        "The body is fine but the date is not.",
        "https://www.bis.org/review/r5.htm",
        "Speech by Mr Bad Date, Governor of the Bank of Canada, at a venue.",
    ],
]


def _zip(
    rows: Sequence[Sequence[str]], *, header: Sequence[str] = _COLUMNS, name: str = "speeches.csv"
) -> bytes:
    """Build a ZIP holding one CSV with the given header and rows."""
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(header)
    writer.writerows(rows)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, text.getvalue())
    return buffer.getvalue()


def _provider(data: bytes) -> Callable[[], bytes]:
    return lambda: data


def test_fetch_parses_tracked_rows_and_maps_institutions() -> None:
    source = BisBulkSpeechSource(_provider(_zip(_ROWS)))
    speeches = source.fetch(limit=10)

    assert [s.speaker_name for s in speeches] == ["Jane Doe", "Max Mustermann"]
    fed, ecb = speeches
    assert fed.central_bank is CentralBank.FEDERAL_RESERVE
    assert fed.role == "Chair"
    assert fed.delivered_at.tzinfo is not None  # naive ISO date is made tz-aware
    assert ecb.central_bank is CentralBank.ECB
    assert ecb.role == "President"


def test_speaker_at_a_tracked_venue_is_not_misattributed() -> None:
    # "A N Other" works at the (untracked) Reserve Bank of Narnia but spoke at the ECB; the source
    # must read the affiliation, not the venue, and skip the row.
    source = BisBulkSpeechSource(_provider(_zip(_ROWS)))
    names = {s.speaker_name for s in source.fetch(limit=10)}
    assert "A N Other" not in names


def test_rows_with_empty_body_or_bad_date_are_skipped() -> None:
    source = BisBulkSpeechSource(_provider(_zip(_ROWS)))
    names = {s.speaker_name for s in source.fetch(limit=10)}
    assert "Empty Body" not in names  # blank text is skipped, not ingested as an empty speech
    assert "Bad Date" not in names  # an unparseable date is skipped, not coerced


def test_limit_caps_the_number_of_speeches() -> None:
    source = BisBulkSpeechSource(_provider(_zip(_ROWS)))
    assert len(source.fetch(limit=1)) == 1


def test_missing_required_column_raises() -> None:
    # Drop the url column from both header and rows.
    header = [c for c in _COLUMNS if c != "url"]
    rows = [[c for i, c in enumerate(row) if _COLUMNS[i] != "url"] for row in _ROWS]
    source = BisBulkSpeechSource(_provider(_zip(rows, header=header)))
    with pytest.raises(BisArchiveError, match="url"):
        source.fetch(limit=10)


def test_archive_without_a_csv_raises() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", "no csv here")
    source = BisBulkSpeechSource(_provider(buffer.getvalue()))
    with pytest.raises(BisArchiveError, match=r"no \.csv"):
        source.fetch(limit=10)


def test_multiple_csvs_require_an_explicit_name() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a.csv", "date,author,title,text,url,description\n")
        archive.writestr("b.csv", "date,author,title,text,url,description\n")
    data = buffer.getvalue()

    with pytest.raises(BisArchiveError, match="multiple CSV"):
        BisBulkSpeechSource(_provider(data)).fetch(limit=10)
    # Naming one resolves the ambiguity (it has only a header, so yields nothing).
    assert BisBulkSpeechSource(_provider(data), csv_name="a.csv").fetch(limit=10) == []


def test_custom_column_names_are_honoured() -> None:
    header = ["when", "who", "headline", "body", "link", "about"]
    rows = [
        [
            "2024-02-02",
            "Custom Col",
            "Mapped by config",
            "Policy remains data dependent.",
            "https://www.bis.org/review/c1.htm",
            "Speech by Mr Custom Col, Governor of the Bank of England, at a hall.",
        ]
    ]
    source = BisBulkSpeechSource(
        _provider(_zip(rows, header=header)),
        date_column="when",
        speaker_column="who",
        title_column="headline",
        text_column="body",
        url_column="link",
        institution_column="about",
    )
    speeches = source.fetch(limit=10)
    assert len(speeches) == 1
    assert speeches[0].central_bank is CentralBank.BANK_OF_ENGLAND

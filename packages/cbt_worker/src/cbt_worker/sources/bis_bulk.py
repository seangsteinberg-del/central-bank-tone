"""Ingest central bank speeches from the BIS bulk-download archive (ADR 0016).

The BIS publishes its central bankers' speeches as a bulk ZIP with no auth and no key: the full
archive ``speeches.zip`` and per-year ``speeches_<year>.zip``
(https://www.bis.org/cbspeeches/download.htm, noncommercial). Each ZIP contains a single CSV.
Reading that archive is a far more robust backfill path than scraping the live React site
(``sources/bis.py``): there is one stable, documented tabular contract instead of fragile selectors
and per-speech detail fetches.

The operator downloads the ZIP once; this source reads it from an injected bytes provider (a local
file in production, an in-memory archive in tests). Each row's institution is mapped onto the schema
spine with the same shared helpers the HTML source uses; rows from untracked institutions, or
missing a required field, an unparseable date, or an empty body, are skipped rather than guessed
(CLAUDE.md section 3). The column names are configurable because the published header has varied;
the defaults match the BIS / DRomelli ``cbspeeches`` layout (``date``, ``author``, ``title``,
``text``, ``url``, ``description``). The CC-noncommercial corpus is not redistributed in this repo.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime

from cbt_core import get_logger
from cbt_worker.sources.base import (
    BytesProvider,
    ScrapedSpeech,
    affiliation_from,
    central_bank_from,
    role_from,
)

_logger = get_logger(__name__)

# CSV fields can hold a whole speech, which exceeds Python's default 128 KiB field cap. Raise it to
# a generous bound (a fixed value, not sys.maxsize, which overflows the C long on some platforms).
_MAX_FIELD_BYTES = 16 * 1024 * 1024


class BisArchiveError(RuntimeError):
    """Raised when the bulk archive cannot be read: no CSV member, or required columns missing.

    A configuration/data error (the wrong file, or a CSV whose header does not match the configured
    columns), surfaced loudly rather than yielding an empty result silently.
    """


def _to_aware(value: datetime) -> datetime:
    """Attach UTC to a naive datetime so every stored timestamp is timezone-aware."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _parse_date(text: str) -> datetime | None:
    """Parse a BIS bulk date (ISO date or datetime) as a tz-aware datetime, or ``None``."""
    try:
        return _to_aware(datetime.fromisoformat(text.strip()))
    except ValueError:
        return None


class BisBulkSpeechSource:
    """Reads central bank speeches from a BIS bulk-download ZIP (one CSV inside)."""

    name = "bis-bulk"

    def __init__(
        self,
        provider: BytesProvider,
        *,
        csv_name: str | None = None,
        date_column: str = "date",
        speaker_column: str = "author",
        title_column: str = "title",
        text_column: str = "text",
        url_column: str = "url",
        institution_column: str = "description",
    ) -> None:
        """Build the source.

        Args:
            provider: Returns the bytes of the bulk ZIP (a local file read in production).
            csv_name: The CSV member to read; if ``None``, the single ``.csv`` in the archive.
            date_column: Header of the delivery-date column (ISO date or datetime).
            speaker_column: Header of the speaker-name column.
            title_column: Header of the title column.
            text_column: Header of the full-text column.
            url_column: Header of the source-URL column (provenance; rows without one are skipped).
            institution_column: Header of the column naming the speaker's institution; its text is
                mapped onto the schema spine (a BIS-style description works via substring match).
        """
        self._provider = provider
        self._csv_name = csv_name
        self._date = date_column.lower()
        self._speaker = speaker_column.lower()
        self._title = title_column.lower()
        self._text = text_column.lower()
        self._url = url_column.lower()
        self._institution = institution_column.lower()

    @property
    def _required(self) -> tuple[str, ...]:
        return (self._date, self._speaker, self._title, self._text, self._url, self._institution)

    def fetch(self, *, limit: int) -> list[ScrapedSpeech]:
        """Read up to ``limit`` speeches from the archive, in file order.

        Unlike the HTML source's "recent" feed, this is a backfill over the archive's rows in the
        order they appear; cap it with ``limit``.

        Args:
            limit: The maximum number of speeches to return.

        Returns:
            The parsed speeches, skipping untracked institutions, malformed rows, and empty bodies.

        Raises:
            BisArchiveError: If the ZIP has no CSV member or the CSV lacks a required column.
        """
        csv.field_size_limit(_MAX_FIELD_BYTES)
        with zipfile.ZipFile(io.BytesIO(self._provider())) as archive:
            member = self._csv_name or _single_csv(archive)
            with archive.open(member) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
                self._check_header(reader.fieldnames, member)
                return self._collect(reader, limit)

    def _check_header(self, fieldnames: Sequence[str] | None, member: str) -> None:
        header = {name.strip().lower() for name in (fieldnames or [])}
        missing = [column for column in self._required if column not in header]
        if missing:
            raise BisArchiveError(
                f"bulk CSV {member!r} is missing required column(s) {missing}; "
                f"present columns: {sorted(header)}"
            )

    def _collect(self, reader: csv.DictReader[str], limit: int) -> list[ScrapedSpeech]:
        results: list[ScrapedSpeech] = []
        for row in reader:
            if len(results) >= limit:
                break
            speech = self._row_to_speech(
                {(k or "").strip().lower(): (v or "") for k, v in row.items()}
            )
            if speech is not None:
                results.append(speech)
        return results

    def _row_to_speech(self, row: dict[str, str]) -> ScrapedSpeech | None:
        institution_text = row[self._institution]
        # Map from the affiliation clause, not the whole description, so a speaker is not
        # misattributed to a tracked bank they merely spoke *at* (a plain institution name has no
        # affiliation clause, so affiliation_from returns it unchanged and the match still works).
        bank = central_bank_from(affiliation_from(institution_text))
        if bank is None:
            _logger.info("bis_bulk_skipped_untracked", institution=institution_text[:80])
            return None
        url = row[self._url].strip()
        text = row[self._text].strip()
        title = row[self._title].strip()
        speaker = row[self._speaker].strip()
        if not (url and text and title and speaker):
            _logger.warning("bis_bulk_skipped_incomplete_row", url=url[:120])
            return None
        delivered_at = _parse_date(row[self._date])
        if delivered_at is None:
            _logger.warning(
                "bis_bulk_skipped_bad_date", url=url[:120], raw_date=row[self._date][:40]
            )
            return None
        return ScrapedSpeech(
            speaker_name=speaker,
            central_bank=bank,
            role=role_from(institution_text),
            title=title,
            url=url,
            delivered_at=delivered_at,
            text=text,
        )


def _single_csv(archive: zipfile.ZipFile) -> str:
    """Return the single ``.csv`` member of the archive, or raise if there is not exactly one."""
    members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    if not members:
        raise BisArchiveError("bulk archive contains no .csv member")
    if len(members) > 1:
        raise BisArchiveError(
            f"bulk archive has multiple CSV members {members}; pass csv_name to choose one"
        )
    return members[0]

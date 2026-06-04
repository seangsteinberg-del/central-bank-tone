# ADR 0016: A BIS bulk-archive speech source for backfill

Date: 2026-06-04

Status: Accepted

## Context

The ingestion worker scrapes the live BIS central bankers' speeches site (ADR 0010): it reads the
RSS listing and then fetches each speech's body from a `data-react-props` JSON blob on the detail
page. That path is correct for incremental updates (the RSS feed is the freshest view), but it is a
poor way to build the historical corpus the rest of the platform needs to be impressive: it is one
HTTP request per speech, it depends on the live site's HTML/JSON shape staying put, and it is
rate-limited out of politeness. The state-of-the-art survey (`docs/research/state-of-the-art.md`)
found that the BIS also publishes the whole archive as a bulk download
([bis.org/cbspeeches/download.htm](https://www.bis.org/cbspeeches/download.htm), noncommercial): a
single ZIP containing one CSV, no auth and no key, available as the full archive `speeches.zip` and
as per-year `speeches_<year>.zip`. That is a far more robust backfill source.

## Decision

Add `BisBulkSpeechSource`, a second implementation of the existing `SpeechSource` protocol that
reads speeches from a BIS bulk ZIP instead of scraping HTML.

- **Input is an injected bytes provider**, not a URL fetcher. The operator downloads the ZIP once;
  in production the provider is `Path(...).read_bytes`, and in tests it is a lambda over an
  in-memory archive, so the source is fully tested with no network and no files on disk. The
  protocol's `fetch(*, limit)` signature is unchanged, so the runner drives it identically.
- **One stable tabular contract.** The source opens the single CSV in the ZIP, validates that the
  header carries the columns it needs, and maps each row to a `ScrapedSpeech`. The column names are
  constructor parameters (defaults `date`, `author`, `title`, `text`, `url`, `description`) because
  the published header has varied across BIS / mirror layouts; pointing the source at a different
  layout is configuration, not code.
- **Shared schema-spine mapping.** The free-text-institution → `CentralBank` mapping, the BIS-style
  role parser, and the affiliation extractor moved from `sources/bis.py` into `sources/base.py`, so
  both sources use one copy. The bulk source maps from the *affiliation clause* (not the whole
  description), so a speaker is not misattributed to a tracked bank they merely spoke at; a plain
  institution name has no affiliation clause and is matched whole.
- **Skip, never guess (CLAUDE.md section 3).** A row from an untracked institution, missing a
  required field (URL, body, title, speaker), or carrying an unparseable date is logged and
  skipped. A structurally broken archive (no CSV member, or a CSV missing a required column) raises
  `BisArchiveError` loudly rather than silently yielding nothing.

The worker entry point gains `--bulk <path> [--limit N]`, so a backfill is
`python -m cbt_worker.app --bulk speeches.zip --limit 500`; with no flag it runs the RSS scraper as
before.

## Consequences

- The historical corpus can be built from one downloaded file with no per-speech HTTP and no
  dependence on the live site's HTML, which is the right tool for backfill; the RSS scraper remains
  the right tool for incremental updates. The two are interchangeable behind `SpeechSource`.
- Ingestion stays idempotent: the bulk and RSS paths can both run because `IngestionService`
  deduplicates by source-text hash, so a speech present in both is ingested once.
- The licensing rule is unchanged: the BIS corpus is noncommercial and is not redistributed in this
  repo (the ZIP is downloaded by the operator, never committed); only computed tone and metadata are
  stored.
- `BisArchiveError` is the worker's own error type. It is a configuration/data error surfaced to the
  operator, distinct from the per-row skips that keep a run going.

## Alternatives rejected

- **Download the ZIP over HTTP inside the source.** Couples the source to a 120 MiB network fetch
  and binary HTTP handling. Injecting a bytes provider keeps the download an operator concern and
  the source trivially testable; the provider can still wrap an HTTP call if desired.
- **Stream the CSV without loading the ZIP into memory.** The archive is ~120 MiB; reading it into a
  `BytesIO` is simple and well within a backfill tool's budget. Streaming adds complexity for no
  real gain here.
- **Make the URL column optional and synthesize a link when absent.** That would fabricate
  provenance. A speech with no source URL is skipped instead (the URL column is required), keeping
  every stored speech traceable to a real source.

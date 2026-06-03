# ADR 0010: BIS ingestion worker and the SpeechSource boundary

Date: 2026-06-03

Status: Accepted

## Context

The platform must scrape speeches from eight institutions automatically (ADR 0006). The research
(`docs/research/reusable-components.md`) found that the BIS central bankers' speeches index
(`bis.org/cbspeeches`) aggregates speeches from 130+ central banks, including all eight we track,
in one place. Writing eight separate per-site scrapers would be eight times the fragile parsing;
one BIS source covers them all. Existing scrapers (`scrape_bis`, GPLv3; `cbspeeches` dataset,
non-commercial) cannot be vendored into this repo for license reasons, so we write our own parser.

## Decision

Add a `cbt_worker` adapter (a workspace member depending on `cbt_core` only). It defines a
`SpeechSource` protocol and a `BisSpeechSource` that scrapes the BIS index plus each speech's
detail page. The HTTP fetcher is injected, so sources are tested against HTML fixtures with no
network. A `run_ingestion` runner resolves the speaker (`SpeakerService.ensure_speaker`), ingests
each speech (idempotent by source hash), and indexes it; a single speech that fails the model is
logged and skipped so one bad item does not abort the run.

Dependencies (in `cbt_worker`): `httpx` (Apache-2.0) for fetching and `selectolax` (MIT) for
fast HTML parsing. The parser targets a documented listing/detail HTML contract (the test
fixtures); the live BIS selectors must be verified and adjusted against the real site, and that
change is confined to `sources/bis.py`. Speeches from institutions outside the schema spine are
skipped, not guessed. The worker respects the source's robots.txt and rate limits, and sends an
identifying User-Agent.

## Consequences

One source covers all eight institutions, and adding a per-institution source later (for example
a Fed-specific scraper to fill recent gaps) is a new class behind the same protocol with no core
change. Because the fetcher is injected, the scraping logic is fully unit tested; the only
untested code is the thin composition root in `app.py` (`# pragma: no cover`). The live selectors
are a known follow-up: the first real run against BIS will likely need selector adjustments.

## Alternatives rejected

- Eight per-institution scrapers now: eight times the fragile parsing and maintenance for the
  same speeches BIS already aggregates.
- Vendor `scrape_bis` (GPLv3): copyleft would force this repo's license; we reuse its design idea
  only.
- Use the `cbspeeches` dataset: academic/non-commercial license, and it ends in 2023 so it cannot
  supply recent speeches.

# ADR 0017: The demo ships no synthetic seed corpus

Date: 2026-06-04

Status: Accepted

Amends: ADR 0014 (keyless, Docker-less demo)

## Context

ADR 0014 introduced a keyless, Docker-less demo so the whole UI could be shown with no Gemini key
and no PostgreSQL. To make that demo look populated, `scripts/run_demo.py` seeded a corpus two ways:

- It grouped a whole calendar year of real FOMC sentences (from the offline evaluation benchmark)
  into a single "speech", attributed it to the sitting Fed Chair, and stamped it with an invented
  date (July 1) and a generic press-releases URL.
- When that benchmark cache was absent, it fell back to three hand-composed "illustrative" passages
  written for the demo, with `(illustrative)` speaker names.

Both are synthetic. The first is the more dangerous: the underlying sentences are real, but the
*provenance* is fabricated. "Jerome Powell, Chair, delivered this on 2019-07-01" is not true of a
year's worth of mixed FOMC communications, and this is a tool whose entire purpose is to attribute
tone to a named speaker at a point in time. Presenting a synthetic aggregation as a real per-speaker
observation is exactly the confidently-wrong output CLAUDE.md section 3 ("no silent fallbacks")
forbids. The locally-cached benchmark corpora under `data/benchmarks/` (FOMC, FRED, op-fed) are real
external datasets, but they are licensed for offline evaluation only and were being repurposed as
display content via the seed path.

## Decision

The demo ships with no seed corpus, and no synthetic or aggregated speech is fabricated anywhere.

- `scripts/run_demo.py` serves an empty application (`build_demo_app([])`). The seeding helpers
  (`_seed_from_fomc_cache`, the `_CHAIRS` table, and the `_ILLUSTRATIVE` passages) are deleted.
- The corpus is populated only by real ingestion: the in-app "Add data" page (which scores and
  indexes a genuine speech keylessly via the offline client) or the worker against a configured
  database. Both store a speech only with its real title, date, source URL, and speaker.
- The dashboard renders an explicit empty state ("No speeches yet") that says the platform does not
  ship fabricated speeches and points to the Add data page, rather than showing a bare page.
- The locally-cached evaluation data under `data/benchmarks/` is purged. It is gitignored and was
  never committed; the evaluation scripts (`eval_tone.py`, `eval_cross_dataset.py`,
  `tone_trajectory.py`) re-download it on demand, so the reported accuracy stays reproducible.

The `SeedSpeech` container and `build_demo_services` / `build_demo_app` wiring remain: they take an
arbitrary corpus (including the empty one) and are exercised by the demo tests with in-test
fixtures. They are infrastructure for ingesting a real corpus, not a source of shipped data.

## Consequences

- The demo is honest by construction: nothing on screen is fabricated, and an empty corpus reads as
  empty rather than as a populated product. This costs the out-of-the-box "populated dashboard"
  that ADR 0014 provided; a reviewer now adds a real speech (a few seconds, keyless) or runs the
  worker to see the dashboard, leaderboards, and committee-movement views with data.
- The keyless demo's store is in-memory and per-process, so a speech added on the Add data page
  lives for the life of that process. A persistent, worker-populated demo is the production stack
  (a real database plus the worker), which is unchanged.
- Reproducing the evaluation now always re-downloads the benchmark corpora on first run. This is the
  pre-existing cache-miss path, so it adds a one-time download, not a behaviour change.

## Alternatives rejected

- **Keep the per-year FOMC aggregation as the seed.** The text is real, but bundling a year of
  communications into one Chair "speech" with an invented date fabricates provenance, which is the
  specific failure mode this tool exists to avoid.
- **Auto-ingest real speeches at demo startup.** A genuinely populated keyless demo is possible by
  running the worker over a live source on boot, but that needs network access at demo time and was
  explicitly out of scope here; the empty-until-ingested demo is the honest default.
- **Keep the `data/benchmarks/` cache.** It is real data, but the request was to purge it; since it
  is gitignored and re-downloaded on demand, deleting the local copy costs nothing reproducible.

# ADR 0022: Signal vs Market divergence view

Date: 2026-06-08

Status: Accepted

## Context

The dashboard answers two of the three questions a macro portfolio manager asks of a tone tool: is
it current (the Policy Monitor and the recent flow) and can I trust it (the methodology page and the
per-speech cross-checks). It does not answer the third, the "so what": where does the platform's
reading of central-bank communication diverge from what the rates market has already priced.

That validation already existed, but only as a standalone script (`scripts/eval_corpus_vs_rates.py`)
and a hand-typed table on the methodology page. Two problems followed. First, the script reads the
PostgreSQL live database and its raw SQL filters on the enum member name; it does not run against the
no-Docker SQLite corpus (ADR 0018) the demo serves, so the live numbers and the published numbers
could (and did) diverge: the committed figures were generated against a larger 2020-2026 corpus, and
on the served 2023-2026 corpus the "headline leads the 2-year yield" claim does not hold. Second, the
finding was not a live, queryable surface a reader could act on.

## Decision

Add a read-only core service, `MarketSignalService` (`cbt_core/services/market_service.py`), and a
web view at `/signal-vs-market`. The service builds the monthly Federal Reserve headline-tone and
forward-looking rate-path indices from the stored scores through the repository layer (no raw SQL,
typed enums, so it is correct on both SQLite and PostgreSQL), relates each to the effective fed funds
rate and the 2-year Treasury yield, reports the lead correlations with a seeded bootstrap 95% CI, and
computes the current divergence between the three-month tone shift and the three-month 2-year
repricing. It returns Pydantic domain read models (`cbt_core/domain/market.py`), never ORM rows.

The rate series are cached FRED CSVs read from `Settings.benchmark_dir` (the `CBT_BENCHMARK_DIR`
boundary, never `os.environ` in the service). The service never fetches from the network inside a
request: a missing cache raises `BenchmarkUnavailableError` and fewer than twelve qualifying months
raises `InsufficientDataError`, both rendered by the view as an honest "unavailable" panel rather than
a 500 or an empty chart (CLAUDE.md section 3). The pure statistics are copied from the eval script,
not imported, because scripts are off the package path and importing them would break the one-way
package dependency (CLAUDE.md section 2). `numpy`, already a `cbt_core` runtime dependency, supplies
the correlation and bootstrap.

The view is Federal Reserve only and labels its limits in the UI: Fed-only validation, the 2-year as
the market-path proxy, correlation not PnL, and a cached-snapshot data date. The methodology page's
rate table is updated to the real 2023-2026 figures and points to this page as the live source.

## Consequences

- The macro "so what" is now a live, tested surface, and the credibility-damaging mismatch between
  the published rate numbers and the served corpus is removed: both come from one code path.
- `Settings` gains a `benchmark_dir` field; adapters pass it to the service. The default is relative
  to the working directory (the app runs from the repo root); it is overridable via
  `CBT_BENCHMARK_DIR` and always injected explicitly in tests.
- The FRED CSVs live under the gitignored `data/` (CC BY-NC, not redistributed), so the cache is
  built locally by `scripts/eval_corpus_vs_rates.py`; where it is absent (CI, a fresh checkout) the
  page renders the honest unavailable state. Tests are hermetic, writing their own fixture CSVs.
- The view recomputes from all Fed observations and stances on each request, like the dashboard
  overview; acceptable at this corpus size, and a candidate for a cached read model if the corpus
  grows large.
- `scripts/eval_corpus_vs_rates.py` and `docs/research/corpus-tone-vs-rates.md` remain as the
  PostgreSQL-oriented offline report; the live service supersedes them for the served product.

## Alternatives rejected

- Import the script's helpers directly: breaks the layered-package rule; the helpers are copied.
- Fetch FRED inside the request with an on-disk cache fallback: network IO behind a page render is a
  latency and failure surface; the cache is refreshed out of band instead.
- Re-point the existing script at SQLite and keep the numbers in a hand-typed table: leaves the
  finding non-live and the table free to drift from the corpus again.
- A full OIS / fed-funds-futures curve as the market proxy: more faithful but needs a paid or
  heavier data source; the 2-year yield is a clean, free, honestly-labelled proxy.

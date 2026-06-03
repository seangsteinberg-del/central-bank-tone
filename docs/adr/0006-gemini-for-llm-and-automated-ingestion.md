# ADR 0006: Google Gemini for all LLM work, automated speech ingestion, no paid third-party APIs

Date: 2026-06-03

Status: Accepted

## Context

Central Bank Tone summarizes speeches, scores their tone, and answers questions about each
speaker. All three need a large language model, and the corpus needs to be ingested. Cost is a
primary constraint: the model spend dominates this kind of workload, and we want to avoid paid
third-party API bills during development and early operation. We also want the corpus to fill
itself rather than depend on manual uploads.

## Decision

- All generative work (summarization, tone scoring, and the per-speaker question answering) goes
  through the Google Gemini API via the `google-genai` SDK. No OpenAI, Anthropic, or other paid
  third-party API is introduced. Gemini is the single model provider because it is cheaper for
  this workload.
- Speeches are ingested by an automatic scraper, delivered as a new `cbt_worker` adapter (a
  workspace member depending on `cbt_core` only, like `cbt_api`). It fetches and normalizes
  speeches per central bank source on a schedule.
- The Gemini client and the scraper's HTTP access are isolated behind a `cbt_core` service
  boundary (an `LlmClient` protocol and an ingestion service). No adapter calls Gemini or the
  network directly. Tests use a stub/mock client; any test that exercises a live Gemini call
  sits behind the existing `llm` pytest marker, which is excluded from CI (CLAUDE.md section 5:
  no test hits a live or paid API).

This ADR records the direction; the `google-genai` dependency, the `cbt_worker` package, and the
LLM/ingestion services are added in follow-up changes, each with its own tests, a dependency
license check (CLAUDE.md section 9), and a CHANGELOG entry.

## Consequences

One provider keeps the integration surface and the cost model simple, and the `LlmClient`
boundary means the provider can be swapped behind an interface if that ever changes, without
touching adapters or the domain. Automated ingestion removes manual data entry but adds an
obligation: the scraper must respect each source's robots.txt and terms of use, rate-limit
politely, and record provenance (a source URL and the `source_sha256` already on
`ToneObservation`). Live model calls cost money and are non-deterministic, so they never run in
CI and are always mockable behind the boundary.

## Alternatives rejected

- OpenAI or Anthropic as the model provider: rejected for cost; the user's directive is the
  cheaper Gemini option and no paid third-party API.
- A managed ingestion/data vendor: a paid third-party dependency, and overkill versus scraping
  public speech pages directly.
- Calling Gemini directly from the API or worker adapter: would leak an external client past the
  service boundary and make the domain untestable without the network (CLAUDE.md section 2).

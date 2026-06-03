# ADR 0011: Server-rendered web UI (cbt_web) and corpus-wide natural-language query

Date: 2026-06-03

Status: Accepted

## Context

The platform had two adapters (`cbt_api`, `cbt_worker`) but no human-facing UI: speakers, tone
over time, and the grounded Q&A were reachable only as JSON over HTTP. The product is meant to be
read and queried in natural language by a person, and that natural-language query should cover the
whole corpus, not just one speaker at a time. We need a front end that meets the same engineering
bar as the rest of the repo and does not pull a second, heavyweight toolchain into a pure-Python
workspace.

## Decision

Add a `cbt_web` adapter (a workspace member depending on `cbt_core` only, machine-checked). It is
server-rendered with Jinja2 templates and progressively enhanced with htmx: the pages are real
forms and links that work without JavaScript, and htmx swaps in HTML fragments (the `/ui/*`
routes) for live search, asking, and ingestion. Screens: a speaker directory with live search and
a corpus-wide ask box (landing page), a speaker detail page with a tone-over-time chart (a pure
CSS diverging-bar timeline, no chart library) and the analyzed speeches, a per-speaker ask box,
and an admin page to register a speaker and ingest a speech. The adapter reaches the domain only
through the `cbt_core` service layer, validates every form against a `schemas.py` model before any
service call, and renders core exceptions as HTML error pages (`EntityNotFoundError` -> 404, any
other `CbtError` -> 500); user-input errors are caught in the views and re-rendered as 4xx.

To make the platform itself natural-language queryable, add a corpus-wide retrieval path in the
core: `SpeechChunkRepository.search_all` / `SpeechRetriever.search_all` (the nearest-neighbour
query without the speaker filter) and `QaService.answer_corpus`, which retrieves across every
speaker's chunks and answers grounded in them, abstaining the same way when nothing is found.

Dependencies (in `cbt_web`): `fastapi` and `uvicorn` (as the API adapter already uses), `jinja2`
(BSD-3) for templates, and `python-multipart` (Apache-2.0) for form parsing. htmx 2.0.4 (BSD-2)
is vendored as a static asset (`static/vendor/htmx.min.js`) rather than loaded from a CDN, so the
UI is self-contained and offline, with no third-party runtime dependency or supply-chain surface
at page load.

## Consequences

A person can now register a speaker, ingest a speech, watch its tone scored and summarized, see a
speaker's tone over time, and ask questions of one speaker or the whole corpus, all from the
browser. Because the adapter is server-rendered and progressively enhanced, it is fully testable
with the in-process `TestClient` against SQLite-backed services (happy path, bad input, not found,
and the server-error page), with no browser or JS runtime in the test loop. `cbt_web` is
independent of `cbt_api`; the small amount of wiring duplicated between them (the `Services`
container and `build_services`) is the cost of keeping the adapters decoupled.

## Alternatives rejected

- A React/Vite single-page app: richer client interactivity, but it introduces a Node/npm
  toolchain, a separate build and lint/test lane, and a JS runtime dependency, for a UI whose
  interactions (search, ask, ingest) are well served by server-rendered fragments.
- htmx from a CDN: simpler to reference, but adds a third-party runtime dependency and a
  supply-chain/offline failure mode; vendoring one pinned, license-clean file avoids both.
- A charting library for the tone timeline: unnecessary weight; a diverging bar around a neutral
  midline is a few lines of CSS and conveys hawkish/dovish over time directly.

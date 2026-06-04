# ADR 0018: A local live setup on native PostgreSQL, no Docker or pgvector

Date: 2026-06-04

Amends: ADR 0014 (keyless demo)

Status: Accepted

## Context

The production stack is PostgreSQL plus the pgvector extension (ADR 0003 / 0009), brought up for
local development with Docker (`docker compose up db`). Bringing the platform fully online on a real
operator machine ran into three environment facts at once: Docker was not installed (and installing
Docker Desktop needs administrator rights, WSL2, and a reboot), the machine had no administrator
session, and pgvector was not present in the PostgreSQL that *was* installed. Installing pgvector
also needs administrator rights (its files live under `Program Files`). What the machine did have
was a running PostgreSQL 16 server with `trust` authentication on localhost.

So the only thing standing between the code and a real, running system was the vector index, and the
codebase already abstracts that: `QaService` retrieves through the `ChunkRetriever` protocol, with a
pgvector-backed `SpeechRetriever` for production and an in-process `InMemoryChunkRetriever` (cosine
over vectors held in memory) for the keyless demo (ADR 0014). The relational store (speakers,
speeches, the append-only tone observations) needs only ordinary tables, which PostgreSQL provides
without any extension.

## Decision

Add `scripts/run_live.py`: a single-process live application that uses the **real Gemini client**
over the **native PostgreSQL** database, with the **in-memory vector index** instead of pgvector.

- The demo service factory (`cbt_web.demo.build_demo_services` / `build_demo_app`) gains optional
  `engine`, `llm`, `model_id`, and `max_distance` parameters. With the defaults it is still the
  keyless SQLite demo; injecting a PostgreSQL engine and a `GeminiClient` makes it the live setup.
  The wiring, views, and the in-memory retriever are otherwise identical, so the live app exercises
  exactly the same code as the demo and as production.
- The relational schema is created with `create_demo_schema` (speaker, speech, tone_observation;
  the pgvector `speech_chunk` table is not used because retrieval is in-memory). A new
  `create_immutability_triggers` installs the append-only triggers from migrations 0001 and 0002, so
  the database-level immutability guarantee (CLAUDE.md section 4) holds here too, not just the app.
- One ingestion pass scrapes real BIS speeches (full text via ADR 0019), Gemini scores tone and
  writes summaries, and each speech is embedded and indexed into the in-memory retriever in the same
  process, so browsing, the tone charts, the committee-movement view, and grounded question
  answering all work against a genuinely ingested corpus with no Docker and no pgvector.

## Consequences

- The platform runs end to end on a stock PostgreSQL with a Gemini key and no container, no
  extension, and no administrator rights. This is a local operator setup, not a replacement for the
  production deployment, which stays on pgvector (a shared, persistent vector index across
  processes).
- Because the vector index is in-process, question answering covers the speeches ingested in the
  running process. Relational reads (speakers, speeches, tone history, committee movement) come from
  PostgreSQL and so reflect everything ever ingested. A cross-process, persistent vector index is
  what pgvector provides; that is the upgrade path, gated only on installing the extension.
- The append-only trigger DDL now lives in two places: the migrations (the production source of
  truth) and `_IMMUTABILITY_DDL` in `engine.py` (for the create_all path). They must stay in sync;
  the constant carries a comment saying so, mirroring how `create_demo_schema` already mirrors the
  migration table definitions.

## Alternatives rejected

- **Install Docker Desktop (or pgvector) to match production exactly.** Both need administrator
  rights, and Docker additionally needs WSL2 and a reboot. None of that can be done hands-off, and
  it is unnecessary: the native PostgreSQL plus the in-memory retriever already runs the whole
  system.
- **A pgvector-free retriever that persists vectors in a normal PostgreSQL column.** A real option
  (brute-force cosine over a `double precision[]` column), but it is new persistence code for a
  single-operator local setup; the existing in-memory retriever already covers it.
- **SQLite for the live store too.** Simpler, but it loses the real append-only triggers (PL/pgSQL)
  and diverges further from production. Using the native PostgreSQL keeps the live setup faithful
  where it can be.

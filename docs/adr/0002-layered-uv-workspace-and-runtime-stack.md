# ADR 0002: Layered uv workspace and runtime stack

Date: 2026-06-03

Status: Accepted

## Context

Central Bank Tone ingests speeches, scores tone per speaker, and answers questions about each
speaker. It will grow several entry points (an HTTP API, an ingestion worker, possibly a CLI)
over one domain model. We need a structure that keeps the domain logic independent of any
delivery mechanism and that a type checker and a CI step can police, plus a concrete runtime
stack to build on.

## Decision

A single `uv` workspace with two members under `packages/`:

- `cbt_core` (import name `cbt_core`, distribution `cbt-core`): the domain heart. Holds the
  schema spine, domain models, services, persistence, settings, logging, and the exception
  hierarchy. It imports no adapter.
- `cbt_api` (distribution `cbt-api`): a FastAPI adapter that depends on `cbt_core` only.

The one-way dependency (adapters depend on core, never the reverse) and the settings boundary
(only `cbt_core.settings` reads the environment) are machine-enforced by
`scripts/check_imports.py`, run in CI and covered by a unit test. The package slug is `cbt`
(short, scales to `cbt_cli` / `cbt_worker`).

Runtime stack: Python 3.12, Pydantic v2 and pydantic-settings for models and configuration,
SQLAlchemy 2.0 with psycopg 3 against PostgreSQL, alembic for migrations, structlog for
structured logging, FastAPI for the HTTP adapter.

## Consequences

New adapters are added as workspace members depending on `cbt_core`; the checker keeps the
dependency direction honest. The domain can be tested without a web server or a database. One
`uv.lock` covers the workspace. The cost is the indirection of a service layer between adapters
and persistence, which is the boundary we want.

## Alternatives rejected

- Single flat package: nothing stops an adapter import from leaking into the domain.
- A `src/` monolith with import-linter only: the workspace also gives independent
  distributions and clean per-package dependency lists.
- Django: heavier than needed and couples the domain to the framework's ORM and request cycle.

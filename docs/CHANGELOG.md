# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Repository scaffold: `uv` workspace with `cbt_core` (domain core) and `cbt_api` (FastAPI
  adapter), the binding standards in `CLAUDE.md`, the `.claude/` agent harness, and the
  `scripts/check_imports.py` architecture checker.
- Schema spine: the `CentralBank` registry and the `ToneLabel` vocabulary as the single source
  of truth for the core type system.
- Domain models `Speaker` and `ToneObservation` (immutable, validated), the persistence layer
  (ORM rows, mappers, repositories, engine helpers), and the services `SpeakerService` and
  `ToneService`.
- Initial alembic migration creating `speaker` and the append-only `tone_observation` table,
  with a database trigger enforcing the append-only guarantee and a `RESTRICT` foreign key.
- `cbt_api` endpoints: register/list/get speakers and record/list tone observations, with a
  correlation-id middleware and core-exception-to-HTTP mapping.
- Test suite: unit tests (SQLite-backed repositories, services, schema spine, API via
  TestClient) and Docker-gated Postgres integration tests (migration round trip, append-only
  trigger, foreign-key semantics). Coverage gate at 90 percent.

- ADR 0006: decision to use Google Gemini for all LLM work, scrape speeches automatically via a
  future `cbt_worker` adapter, and use no paid third-party APIs.

### Changed

### Fixed

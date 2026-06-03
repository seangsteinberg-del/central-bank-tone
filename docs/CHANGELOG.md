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
- LLM boundary: the `LlmClient` protocol and a Gemini implementation (`google-genai`, ADR 0007),
  the `ToneAnalysis` domain value, and Gemini settings (`CBT_GEMINI_API_KEY` as `SecretStr`,
  `CBT_GEMINI_MODEL` defaulting to `gemini-2.5-flash`, `CBT_GEMINI_EMBEDDING_MODEL`); production
  requires a real key.
- Tone methodology (ADR 0008): a deterministic `HawkishDovishLexicon` baseline (our own word
  lists, the Apel and Blix Grimaldi net-hawkishness method) as a license-clean cross-check on the
  Gemini score.
- Speech ingestion: the `Speech` domain model (source plus its Gemini summary and tone and the
  lexicon score), an append-only `speech` table (migration 0002, trigger and unique source
  hash), `SpeechRepository`, and `IngestionService`, which verifies the speaker, deduplicates by
  source hash (no repeat model spend), analyzes, and persists the speech plus a tone observation
  atomically.
- Research notes (`docs/research/reusable-components.md`) on reusable prior art and licenses.
- Retrieval-augmented Q&A (ADR 0009): `pgvector` for vector storage, deterministic
  `chunk_text` chunking, `LlmClient.embed`/`answer` (Gemini `gemini-embedding-001`, 768-dim), a
  `speech_chunk` table (migration 0003, HNSW cosine index), `IndexingService`, and `QaService`,
  which answers grounded in retrieved chunks with citations and abstains when nothing relevant is
  found.
- Ingestion worker (`cbt_worker`, ADR 0010): a `SpeechSource` protocol and a `BisSpeechSource`
  that scrapes the BIS speeches index (one source covering all eight institutions, `httpx` +
  `selectolax`), and a `run_ingestion` runner that resolves the speaker, ingests, and indexes
  each speech. `SpeakerService.ensure_speaker` finds or creates a speaker by name and institution.

### Changed
- Replaced the interim `AnalysisService` (raw-text analysis) with `IngestionService`, which
  ingests a full speech with metadata. Unreleased, so no external consumers are affected.

### Fixed

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
- API endpoints (`cbt_api`): `POST/GET /speakers/{id}/speeches` to ingest+index and list a
  speaker's analyzed speeches, and `POST /speakers/{id}/ask` to answer a question about a speaker
  with citations. `IngestionService.list_speeches` reads a speaker's speeches.
- Corpus-wide question answering: `QaService.answer_corpus`, backed by
  `SpeechChunkRepository.search_all` / `SpeechRetriever.search_all` (nearest-neighbour retrieval
  across every speaker), so the whole platform is natural-language queryable, not just one speaker.
- Web UI (`cbt_web`, ADR 0011): a server-rendered (Jinja + htmx) FastAPI adapter with a speaker
  directory and live search, a corpus-wide ask box, a speaker page with a tone-over-time chart and
  analyzed speeches, a per-speaker ask box, and an admin page to register a speaker and ingest a
  speech. htmx 2.0.4 (BSD-2) is vendored as a static asset; the pages degrade gracefully without
  JavaScript. `SpeakerForm`/`AskForm`/`IngestForm` validate every submission at the boundary.
- Tone evaluation (ADR 0012): `scripts/eval_tone.py` scores the lexicon (and, with a key, Gemini)
  against the annotated FOMC benchmark and reports accuracy, macro-F1, and a confusion matrix vs a
  majority-class baseline; `scripts/tone_trajectory.py` charts FOMC net-hawkishness vs the fed funds
  rate with correlations. Results committed under `docs/research/`.
- Supervised tone classifier (ADR 0013): `cbt_core.ToneClassifier`, a pure-numpy TF-IDF +
  class-balanced multinomial logistic regression trained offline (`scripts/train_tone_model.py`) on
  the FOMC benchmark and shipped as a small committed JSON artifact. It runs with no API key and no
  Docker. On the held-out test split it scores 59.9% accuracy / 0.582 macro-F1, versus the lexicon's
  51.8% / 0.339; `scripts/eval_tone.py` now runs a three-way head-to-head and adds a McNemar
  significance test (p = 0.012) and a bootstrap confidence interval for the gain over the lexicon.
  Adds `numpy` as a `cbt_core` runtime dependency.
- Keyless offline LLM boundary (ADR 0014): `cbt_core.OfflineLlmClient` implements the `LlmClient`
  protocol with no network and no API key, tone from the supervised classifier (a speech's tone is
  the net share of its hawkish vs dovish sentences), a deterministic extractive summary,
  signed-feature-hashing embeddings for retrieval, and an explicitly extractive (never fabricated)
  answer. It is the offline implementation the demo runner wires in when no key is set; the Gemini
  path remains the production signal.
- Keyless, Docker-less demo: `cbt_core.InMemoryChunkRetriever` (cosine retrieval with no pgvector),
  `make_demo_engine` / `create_demo_schema` (SQLite), `cbt_web.demo.build_demo_app`, and
  `scripts/run_demo.py` (`make demo-lite`). The runner seeds the real FOMC corpus grouped by year
  and attributed to the sitting Fed Chair, so the whole UI - per-Chair tone trajectories, the
  cross-check markers, and natural-language search - runs populated with no Gemini key and no
  Postgres. Adds `IngestionService.get_speech`.
- Classifier calibration in the evaluation: `scripts/eval_tone.py` now reports the supervised
  classifier's expected and maximum calibration error (ECE 0.142, MCE 0.276), a multiclass Brier
  score, and the direction of the miscalibration, with a reliability diagram and confidence
  histogram (`docs/research/tone-reliability.png`). The model is under-confident on this benchmark
  (its predicted-class probability is a conservative lower bound on its accuracy); the methodology
  page surfaces this.
- Stronger thesis test: `scripts/tone_trajectory.py` now builds both a lexicon and a classifier
  annual tone index and relates them to three FRED series (the fed funds rate and the 2-year and
  10-year Treasury yields), reporting every correlation with a bootstrap 95% CI plus an OLS
  regression of the same-year change in the 2-year yield on tone (slope +6.89, bootstrap CI
  [+2.31, +14.22] excluding zero). FRED responses are cached for reproducibility.
- Real model/lexicon cross-check: `cbt_core.analysis.disagrees`, persisted as `lexicon_score` and
  `needs_review` on the tone observation (migration 0004) and on the speech, logged as a WARNING on
  divergence and shown as a "model/lexicon disagree" marker in the UI and on the API responses.
- `LazyGeminiClient`: the app boots without a Gemini key; model operations fail (with a clear
  `LlmError`) only when invoked. The model id is recorded on each speech (migration 0005).
- CI (`.github/workflows/ci.yml`): ruff, `mypy --strict`, the architecture check, the test suite
  with the coverage gate, and `pip-audit`; the Postgres + pgvector integration tests run there.
- Demo infrastructure: `docker-compose.yml` (pgvector), `scripts/migrate.py`, and a `Makefile`
  (`make demo`, `make eval`, `make gate`).
- Speech detail page and committee tone-movement read model (ADR 0015): clicking any speech opens
  `/speeches/{id}` with a concise summary and how the speech's committee has moved as of it. New
  `cbt_core.CommitteeService.movement_for_speech` builds an immutable, point-in-time
  `CommitteeMovement`: each member's standing tone and most recent shift (a member counts only once
  they have spoken, and their "current" reading is the latest on or before the speech, never a later
  one), the committee's standing tone (mean of current scores), and the overall move (mean of
  members' individual shifts over the members that have a prior reading, with that count). The page
  renders a per-member diverging movement bar scaled to the largest mover. Adds the
  `CommitteeMovement` and `MemberMovement` domain models.

### Changed
- Web UI overhaul: the landing page is now a dashboard (thesis hero with the headline finding and a
  prominent corpus ask box, a corpus-stats strip, hawkish/dovish leaderboards by latest tone, and
  recently-analyzed speeches); a new `/methodology` page surfaces the measured accuracy
  (classifier 59.9% vs lexicon 51.8%, McNemar p, bootstrap CI) and embeds the confusion-matrix and
  tone-vs-rates charts; the speaker page renders a real inline-SVG tone-over-time chart (trend line,
  per-point stems, hover detail) in place of CSS bars; and the stylesheet was reworked into a
  cohesive research-terminal design.
- Replaced the interim `AnalysisService` (raw-text analysis) with `IngestionService`, which
  ingests a full speech with metadata. Unreleased, so no external consumers are affected.
- The deterministic lexicon now uses longest-match phrase counting and a negation window, with a
  larger curated term set; ADR 0008 reframes it as a simplified net-hawkishness ratio (not the full
  Apel & Blix Grimaldi method) and the cross-check is now implemented, not just documented.
- Gemini tone scoring uses temperature 0 and a scale-anchoring rubric so scores are reproducible
  and comparable across speeches.

### Fixed
- The BIS scraper was rewritten against the live site (a React app): the listing now comes from the
  RSS feed and the speech body from the `data-react-props` JSON, with institution read from the
  affiliation clause (not the venue), plus fetcher retry/backoff. The previous selectors could not
  work against bis.org.
- The lexicon no longer double-counts a phrase via its substring or cancels a hawkish phrase
  against a dovish substring (for example "withdraw accommodation").
- Retrieval (`QaService`) now applies a maximum-distance relevance threshold, so an off-topic
  question abstains instead of grounding in the nearest-but-irrelevant chunks.
- The ingestion worker isolates per-source and per-speech failures, so one bad item no longer
  aborts the run.

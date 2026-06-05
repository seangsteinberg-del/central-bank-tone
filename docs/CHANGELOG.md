# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Ingestion runs the structured stance pipeline on every speech and persists its decomposition
  (ADR 0021) into a new derived `speech_stance` table (migration 0006): the forward-looking
  `rate_path` (policy intent), the directional `uncertainty` (the share of cross-checks that disagree
  with the headline), the three cross-check nets, the directional `needs_review`, and the per-aspect
  net-hawkishness map. The decomposition is recomputable from the speech text, so like the retrieval
  chunks it lives in its own non-append-only table (`SpeechStance` domain model,
  `SpeechStanceRepository.upsert`/`get`/`all_by_speech`) rather than on the immutable speech row, and
  can be re-derived for any speech as the method improves without touching the append-only tone
  record. The holistic Gemini score stays the headline; the structured pipeline supplies the
  decomposition and the directional cross-check.
- Production-signal validation against real rates (`scripts/eval_corpus_vs_rates.py`, ADR 0021): a
  reproducible, keyless test of the platform's own headline tone. It builds a monthly Federal Reserve
  tone index from the stored Gemini holistic scores and correlates it with FRED's effective fed funds
  rate and 2-year Treasury yield, contemporaneously and at 3- and 6-month leads, each with a
  bootstrap 95% CI. Measured result (2020-2026, 67 months, 538 Fed speeches): the tone index moves
  with the rate cycle (same-month +0.52 fed funds, +0.42 2-year) and leads it (3-month forward +0.51
  fed funds, +0.29 2-year; all CIs exclude zero), so the headline carries information about where
  policy goes next, not only where it has been. Fed-only (the limitation is stated): the other seven
  institutions have no free market ground truth. Report under `docs/research/`.
- Sentence-level stance aggregation engine (`cbt_core.analysis.stance`, ADR 0021), the pure,
  model-agnostic heart of a structured tone pipeline that replaces the single greedy whole-speech
  call (the weakest documented method; see `docs/research/tone-sota-blueprint.md`). It splits a
  speech, keeps policy-relevant sentences with a Gorodnichenko-style `PolicyRelevanceFilter`, and
  aggregates classifier-assigned sentence labels with the Trillion Dollar Words measure
  `(#Hawkish - #Dovish) / #relevant` (`aggregate_stances`), alongside a forward-looking sub-measure
  (rate-path intent, separated from backward-looking description) and a per-aspect breakdown
  (inflation, growth, employment, balance sheet, financial stability, guidance). The continuous
  measure maps to a `ToneLabel` with an honest `MIXED` only when both sides are materially present
  and an honest abstention (`NEUTRAL`) when no policy-relevant sentence is found. Who classifies each
  sentence is injected (Gemini in production, the supervised classifier offline, a stub in tests), so
  the reproducible accounting is tested with no network or GPU.
- `LlmClient.classify_sentences` (ADR 0021), a batched per-sentence stance/aspect/horizon classifier
  on the model boundary that feeds `aggregate_stances`. The Gemini client implements it as one
  structured call returning a JSON array constrained to the schema spine (stance, aspect, and
  horizon enums), not one call per sentence; the keyless offline client uses the supervised
  classifier for stance and deterministic cue heuristics (`infer_aspect`, `infer_horizon`) for the
  aspect and horizon axes.
- `StanceService` and `StanceAssessment` (ADR 0021): the ensemble brain. The headline score and tone
  stay the model's holistic whole-speech judgement (its dynamic range is what surfaces moves and
  divergence for a macro reader); the structured sentence-level pipeline adds the decision-relevant
  decomposition, a `rate_path` (forward-looking policy intent, separated from backward-looking
  description) and a per-aspect breakdown. Three independent signals (the structured net, the
  supervised classifier, the lexicon) are not averaged into the headline (`combine_signals`,
  ADR 0008) but compared against it. The comparison is by direction, not magnitude: each signal is
  reduced to hawkish/dovish/neutral and the uncertainty is the share of cross-checks that point the
  opposite way to the headline, so the compressed structured-net scale is never mistaken for
  disagreement and a single weak dissenter does not trip a review (a majority must). The FOMC-trained
  classifier is a cross-check only for the Federal Reserve, where it is measured to be valid; it is
  excluded for the other institutions, which it does not transfer to (ADR 0013). The cross-checks
  quantify when to distrust the headline, never replace it.
- Dashboard Policy Monitor, a macro-desk redesign of the landing page. The hero is now a sortable
  bank-by-bank matrix (current committee stance with a diverging move-track, 1-month and 3-month
  change, a 6-month inline sparkline, and the hawk/dove committee split), with server-rendered
  sorting (`GET /ui/monitor?sort=...`, a 422 on an unknown key). Above it sits a market-style KPI
  strip; below it a "who's turning" movers panel (the largest stance shifts over the trailing
  band) and a relative-value tone-spread chart between any two banks
  (`GET /ui/spread?a=...&b=...`, a 422 on an unknown bank). A policy-divergence-over-time chart and
  the per-bank committee view read one canonical per-bank committee-stance series, so a bank's
  "now" on the monitor equals its latest point on the divergence chart. The pooled corpus-aggregate
  band chart is demoted beneath these. Committee sparklines link through to the speaker page.
- Web UI: a dark research-desk theme (left rail with grouped nav and an active-item indicator, a
  sticky top bar with a breadcrumb and live clock, monospace metadata, metric cards).
- Dashboard: a per-bank committee view. A "Policy stance by bank" overview ranks each bank's
  committee by mean tone, and a bank toggle (`GET /ui/leaderboard?bank=...`) shows one committee's
  members ranked hawkish-to-dovish on a diverging tone map. Speakers are never pooled across banks
  (tone only compares within a committee).
- Per-speaker tone-over-time chart rebuilt with hawkish/dovish zones, value gridlines and axis
  ticks, and a highlighted most-recent reading.
- Inline tone-history sparklines on each committee member, a corpus-wide "Tone drift across central
  banks" band chart (monthly mean tone with a +/- 1 std envelope), and the speech-page committee
  movement chart restyled to match (a shared dovish/neutral/hawkish axis).
- `scripts/run_live.py` flags: `--no-serve` (fill alongside a running server) and `--no-ocr`
  (skip the Gemini-vision OCR fallback so a bulk fill never stalls on the vision endpoint). The
  live runner now fetches the RSS feed first so the newest speeches land before the bulk backfill.
- `scripts/run_live.py --concurrency N`: the historical backfill ingests speeches through a thread
  pool over the I/O-bound Gemini calls (speakers are pre-resolved serially so two workers never
  race to create the same one), about 9x faster than the sequential fill on a paid key. Default 8,
  capped at 12. Per-speech failures stay isolated and the fill remains idempotent and resumable.

### Fixed
- BIS RSS source: a transient HTTP error on one speech's detail fetch no longer aborts the whole
  feed; each entry is scraped in isolation (the regression that left the corpus short of today).

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
- BIS bulk-archive backfill source (`BisBulkSpeechSource`, ADR 0016): reads speeches from a
  downloaded BIS bulk ZIP (one CSV, no key) via an injected bytes provider, mapping each row onto
  the schema spine with the shared institution/role parsers (now in `sources/base.py`). Configurable
  column names; rows from untracked institutions or missing a required field are skipped, a
  structurally broken archive raises `BisArchiveError`. The worker entry point gains
  `--bulk <path> [--limit N]` for backfill; the RSS scraper remains the incremental path.
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
  `scripts/run_demo.py` (`make demo-lite`), so the whole UI - tone scoring, the tone charts, and
  natural-language search - runs with no Gemini key and no Postgres. The demo starts empty and is
  populated by real ingestion only (see Changed/Removed below; ADR 0017). Adds
  `IngestionService.get_speech`.
- Cross-dataset (out-of-distribution) evaluation: `scripts/eval_cross_dataset.py` applies the
  FOMC-trained classifier and the lexicon unchanged to op-fed (`kakeith/op-fed`, MIT; FOMC meeting
  transcripts labeled with a StanceNLI scheme), mapping entailment/contradiction/neutral onto
  hawkish/dovish/neutral via the documented "We should tighten monetary policy" hypothesis. An
  honest negative result: the classifier drops to 32.1% accuracy / 0.318 macro-F1 (below that
  corpus's dovish-skewed majority baseline), so much of its signal is speech-corpus-specific. Report
  and confusion matrix under `docs/research/`; surfaced in the methodology page's limitations.
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
- Local live setup with no Docker and no pgvector (ADR 0018): `scripts/run_live.py` runs the real
  Gemini client over a native PostgreSQL database with the in-process vector index, ingests a pass
  of real BIS speeches, and serves the production UI. `cbt_web.demo.build_demo_services` /
  `build_demo_app` gained optional `engine`, `llm`, `model_id`, and `max_distance` parameters so the
  same wiring serves the keyless SQLite demo or the live PostgreSQL + Gemini setup. New
  `cbt_core.create_immutability_triggers` installs the append-only triggers (mirroring migrations
  0001 and 0002) on a database built with `create_demo_schema`.
- Full speech text from the linked BIS PDF, always (ADR 0019): the BIS source fetches each speech's
  `document.path` PDF and extracts it with `pdfminer.six`, preferring the PDF whenever it is fuller
  than the short HTML intro. When a PDF is un-extractable (subset fonts with no ToUnicode map, which
  extract as `(cid:...)` glyph soup), `make_pdf_extractor` renders its pages with `pypdfium2` and
  has Gemini transcribe them (`GeminiClient.transcribe_image`), recovering the full text by reading
  the pixels; only if OCR also fails does the body fall back to the HTML intro, so unreadable text
  is never ingested. Adds an injectable `pdf_fetcher`/`pdf_extractor` on `BisSpeechSource` and the
  `pdfminer.six`, `pypdfium2`, and `Pillow` dependencies to `cbt_worker`.
- Persistent retrieval without pgvector (ADR 0020): `cbt_core.PersistentChunkRetriever` stores each
  chunk and its embedding (packed float32 in a `bytea` column) and reloads them into the in-process
  cosine index on startup, so a filled corpus and its question-answering index survive restarts and
  the Gemini embedding is computed once. Indexing is idempotent (a speech already stored is not
  re-embedded). `build_demo_services` / `build_demo_app` gained a `persistent_retrieval` flag.
- Historical backfill up to today in the live runner: `scripts/run_live.py` fills from the BIS bulk
  per-year archives (`speeches_<year>.zip`, full text in the CSV) for depth across the tracked
  central banks and from the live RSS feed for the newest speeches, both idempotent, so the corpus
  spans years and runs right up to the current day. `--limit N` and `--years` control the volume.
- Resilient Gemini calls: `GeminiClient` retries transient errors (HTTP 429 rate limits and 5xx)
  with exponential backoff, so a large fill is not derailed by the free tier's throttling.

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
- The keyless demo (`make demo-lite`) no longer ships a seed corpus. It starts empty and is
  populated only by real ingestion (the in-app Add data page, which scores and indexes keylessly,
  or the worker); the dashboard renders an explicit empty state. Previously it seeded a fabricated
  corpus: a whole year of FOMC sentences re-framed as one Chair "speech" with an invented date and
  URL, or three hand-composed "illustrative" passages. The platform no longer fabricates speeches to
  look populated (ADR 0017).

### Removed
- Purged all synthetic and locally-cached evaluation data: the fabricated demo seed corpus
  (above) and the on-demand benchmark cache under `data/benchmarks/` (the FOMC, FRED, and op-fed
  downloads). The benchmark data is gitignored and was never committed; the evaluation scripts
  re-download it on demand, so the reported accuracy stays reproducible.

### Fixed
- The live Gemini tone path returned HTTP 400. `analyze_tone` passed the `ToneAnalysis` Pydantic
  model directly as the Gemini `response_schema`; because the model uses `extra="forbid"`, Pydantic
  emits `additionalProperties`, which Gemini's `response_schema` rejects. The client now sends an
  explicit Gemini-compatible schema and validates the JSON response back into `ToneAnalysis`, so the
  domain model's strict validation is unchanged. Verified end to end against the live API (the bug
  was invisible to CI, where live calls are gated out). `embed` and `answer` were confirmed working
  against the live API in the same pass.
- The BIS scraper was rewritten against the live site (a React app): the listing now comes from the
  RSS feed and the speech body from the `data-react-props` JSON, with institution read from the
  affiliation clause (not the venue), plus fetcher retry/backoff. The previous selectors could not
  work against bis.org.
- The BIS listing title now strips a multi-author byline prefix (for example
  `"Carolyn Rogers,Toni Gravelle: Release of the FSR"`), not only a prefix matching the single
  `dc:creator`, while leaving a colon inside the real title intact. A body below a minimum word
  count (an empty page or an unreadable stub) is skipped rather than ingested as a non-scoreable
  speech.
- `GeminiClient.embed` now L2-normalizes each embedding. `gemini-embedding-001` is not unit-length
  at the reduced `output_dimensionality` the platform requests, but the cosine-distance retrievers
  treat a dot product as a cosine similarity, so un-normalized vectors made every chunk read as far
  away and question answering wrongly abstained. Normalizing at the source (as the offline client
  already did) restores grounded retrieval; pgvector's cosine operator was unaffected.
- `configure_logging` forces UTF-8 on the console streams. The Windows console defaults to cp1252,
  which cannot encode the non-ASCII characters in many central bankers' names, so structured logging
  crashed mid-ingest and the batch-isolation guard dropped the whole source. Forcing UTF-8 output
  keeps a large multilingual ingest from aborting.
- The lexicon no longer double-counts a phrase via its substring or cancels a hawkish phrase
  against a dovish substring (for example "withdraw accommodation").
- Retrieval (`QaService`) now applies a maximum-distance relevance threshold, so an off-topic
  question abstains instead of grounding in the nearest-but-irrelevant chunks.
- The ingestion worker isolates per-source and per-speech failures, so one bad item no longer
  aborts the run.

# Central Bank Tone

Central bank communication moves markets. The tone of a speech - how hawkish or dovish a
policymaker sounds - carries information about future policy beyond the words' literal content;
the "Voice of Monetary Policy" (Gorodnichenko, Pham, Talavera, *AER* 2023) shows the audio and
language of FOMC communication independently move asset prices. This project reads central bank
speeches, scores their tone per speaker over time, and makes the whole corpus queryable in natural
language, so that signal is measurable and inspectable.

## Does it work? (measured, not asserted)

The tone scorers are evaluated against the annotated FOMC benchmark ("Trillion Dollar Words", ACL
2023) and the resulting series is checked against actual policy. Everything here is reproducible
with no API key (`make eval`).

**Accuracy against human labels.** On the held-out FOMC test split, head to head:

| scorer | accuracy | macro-F1 |
|---|---|---|
| majority-class baseline | 49.8% | 0.222 |
| deterministic lexicon | 51.8% | 0.339 |
| **supervised classifier** (TF-IDF + logistic regression) | **59.9%** | **0.582** |

The supervised classifier (pure numpy, trained offline, no key) nearly doubles the lexicon's
macro-F1, and the gain is statistically significant: McNemar **p = 0.012**, bootstrap 95% CI on the
accuracy gap **[+2.2%, +14.1%]** (excludes zero). For context, fine-tuned RoBERTa-large reaches ~0.72
weighted F1 on this benchmark and zero-shot LLMs ~0.59 F1, so a transparent linear model at 0.58
macro-F1 is a credible floor in the zero-shot-LLM range, not a state-of-the-art claim (see
[docs/research/state-of-the-art.md](docs/research/state-of-the-art.md)). The evaluation also reports
**calibration** (ECE 0.142, MCE 0.276, with a reliability diagram): the classifier is under-confident
on this benchmark, so its predicted-class probability is a conservative lower bound on its accuracy.
Full numbers, confusion matrices, and the reliability diagram:
[docs/research/tone-evaluation.md](docs/research/tone-evaluation.md).

**Tone tracks the policy cycle.** Annual FOMC tone vs the fed funds rate and the 2- and 10-year
Treasury yields (FRED), 1996-2022. Regressing the same-year change in the 2-year yield on annual
tone gives a slope of **+6.89** (bootstrap 95% CI **[+2.31, +14.22]**, excludes zero; Pearson
r +0.35): in years the FOMC sounded more hawkish, the market repriced near-term policy higher within
the same year. This is a descriptive, same-year link, not a trading signal (the next-year lead terms
are ~0). Details: [docs/research/tone-vs-rates.md](docs/research/tone-vs-rates.md).

![FOMC tone vs policy rates](docs/research/tone-vs-rates.png)

The Gemini path scores head-to-head on the same benchmark once a key is set (`make eval` then
`uv run python scripts/eval_tone.py --with-gemini`).

## See it running with no key and no Docker

```bash
make demo-lite        # serves the keyless UI at :8000 (no key, no Docker); starts empty
```

This builds a real application on SQLite with the supervised classifier and an offline retriever,
plus a methodology page with the measured accuracy, all with no Gemini key and no database. It
ships with no seed corpus: the platform never fabricates speeches to look populated (ADR 0017), so
the demo starts empty and is populated only by real ingestion. Add a genuine speech (its real
title, date, URL, and speaker) on the in-app Add data page, where it is scored and indexed
keylessly, or run the worker to ingest at scale. Each speech then has a detail page: a concise
summary and, as of that speech, how far each member of the committee has shifted in tone since
their previous speech and how the committee moved overall (ADR 0015).

## Run it for real with Gemini, no Docker

If you have a Gemini key and a local PostgreSQL (no pgvector or Docker needed), one command brings
the real system online (ADR 0018):

```bash
make live             # real Gemini + native PostgreSQL, fills from the archives + live feed, serves :8000
```

It fills the corpus from two sources, both idempotent: the BIS bulk per-year archives
(`speeches_<year>.zip`, full text in the CSV) for historical depth across the tracked central banks,
and the live BIS RSS feed for the newest speeches right up to today. Each speech's **full text** is
always recovered (ADR 0019): the linked PDF when it is fuller than the HTML intro, and for a PDF no
text extractor can decode (subset fonts with no Unicode map), the pages are rendered and **Gemini
transcribes them**. Gemini scores tone and writes summaries; speakers, speeches, and tone history
persist in PostgreSQL behind the append-only triggers; and chunk embeddings persist in a `bytea`
column and reload on startup (ADR 0020), so the corpus and its question-answering index **survive
restarts** and the embedding is computed once. Set `CBT_DATABASE_URL` (default
`postgresql+psycopg://cbt:cbt@localhost:5432/cbt`) and `CBT_GEMINI_API_KEY` in `.env`; pass
`--limit N` and `--years 2026,2025,2024` to control the fill (re-running resumes where it left off).
The canonical production deployment uses pgvector for a shared, indexed vector store; this local
setup uses the persistent in-process index instead.

## How it works

Scrape a speech (the BIS speeches feed aggregates every tracked central bank) -> validate it
against the schema spine (the registry of banks and tone labels) -> score tone with three scorers: a
**Gemini** LLM-as-judge (summary, tone, a calibrated `[-1, 1]` score at temperature 0), a
**supervised classifier** (TF-IDF + logistic regression, trained offline), and a **deterministic
lexicon**; the lexicon is a transparent cross-check and a large disagreement flags the speech for
review -> persist an immutable speech and an append-only tone observation -> chunk and embed into
pgvector -> answer natural-language questions about one speaker or the whole corpus, grounded in
retrieved excerpts with citations, abstaining when nothing relevant is found.

There are two interchangeable implementations behind the LLM boundary: the **Gemini** client
(production) and a keyless **offline** client (classifier tone, hashing embeddings, extractive
answers) that powers `make demo-lite` with no key and no database.

## Layout

```
packages/
  cbt_core/   domain heart: schema spine, services, persistence, the LLM boundary, the lexicon,
              the supervised classifier, and the offline client. Imports no adapter (machine-checked).
  cbt_api/    FastAPI JSON adapter. Depends on cbt_core only.
  cbt_worker/ BIS speech sources behind one protocol: the RSS scraper (incremental) and a
              bulk-archive reader (backfill from the BIS bulk ZIP). Depends on cbt_core only.
  cbt_web/    server-rendered (Jinja + htmx) web UI: dashboard, per-speech committee-movement view,
              methodology page, SVG tone charts, and a keyless demo builder. Depends on cbt_core only.
scripts/      check_imports.py (architecture invariants), train_tone_model.py + eval_tone.py +
              tone_trajectory.py + eval_cross_dataset.py (the evaluation above), run_demo.py
              (the keyless demo), migrate.py.
docs/         CHANGELOG.md, adr/ (20 decision records), research/ (the evaluation with calibration,
              the tone-vs-rates study, the out-of-distribution check, and a state-of-the-art survey).
.github/      CI: ruff, mypy --strict, the import check, the test suite + coverage gate, pip-audit.
```

The binding engineering standards live in [CLAUDE.md](CLAUDE.md). Every change must comply.

## Run it

```bash
uv sync                                  # create .venv and install the workspace
cp .env.example .env                     # set CBT_GEMINI_API_KEY for model features (optional to boot)

make eval                                # reproduce the accuracy + tone-vs-rates results (no key)

# The full stack (needs Docker for Postgres + pgvector):
make demo                                # start the DB, migrate, and serve the web UI at :8000
```

The web UI boots without a Gemini key (you can browse speakers and tone history); ingesting and
asking need a key, and fail with a clear error until one is set. On Windows/PowerShell, run the
`Makefile` targets' commands directly (see the `Makefile`): `docker compose up -d db`,
`uv run python scripts/migrate.py`, `uv run uvicorn --factory cbt_web.app:create_app`.

## The quality gate

```bash
uv run ruff check . && uv run ruff format --check .   # lint + format
uv run mypy                                           # strict type check
uv run python scripts/check_imports.py                # architecture invariants
uv run pytest -m "not llm"                            # tests + coverage gate (fail_under 90)
```

Runs in CI on every push (`.github/workflows/ci.yml`), where Docker is available so the Postgres +
pgvector integration tests (append-only triggers, migration round trips, vector retrieval) execute
in addition to the unit suite. Locally those integration tests skip with an explicit reason when
Docker is unavailable; unit tests cover repository and mapper logic against in-process SQLite, and
live Gemini calls are gated behind the `llm` marker and excluded from CI. Coverage of the stubbed
code is not the same as signal correctness - for that, see the evaluation above.

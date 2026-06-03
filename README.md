# Central Bank Tone

Central bank communication moves markets. The tone of a speech - how hawkish or dovish a
policymaker sounds - carries information about future policy beyond the words' literal content;
the "Voice of Monetary Policy" (Gorodnichenko, Pham, Talavera, *AER* 2023) shows the audio and
language of FOMC communication independently move asset prices. This project reads central bank
speeches, scores their tone per speaker over time, and makes the whole corpus queryable in natural
language, so that signal is measurable and inspectable.

## Does it work? (measured, not asserted)

The tone scorers are evaluated against the annotated FOMC benchmark ("Trillion Dollar Words", ACL
2023) and the resulting series is checked against actual policy. Both runs are reproducible and
need no API key (`make eval`):

**Tone tracks the policy cycle.** Aggregate FOMC net-hawkishness by year vs the effective fed
funds rate, 1996-2022 (deterministic lexicon): correlation **+0.39** with the same-year change in
the rate, **+0.30** with the level, ~0 as a next-year lead. The hawkish 2004-06 and 2022 and dovish
2008-09 and 2020 episodes are visible.

![FOMC tone vs the fed funds rate](docs/research/tone-vs-rates.png)

**Accuracy against human labels.** On the held-out FOMC test split the deterministic lexicon scores
**51.8%** accuracy vs a 49.8% majority-class baseline (macro-F1 0.339), firing on ~12% of sentences
- a transparent, high-precision floor, not the production signal. Full numbers and the confusion
matrix: [docs/research/tone-evaluation.md](docs/research/tone-evaluation.md). The Gemini path scores
head-to-head on the same benchmark once a key is set (`make eval` then
`uv run python scripts/eval_tone.py --with-gemini`).

These are honest baselines, not claims of alpha. The point is that the signal is measured and
behaves in the direction the literature predicts; the LLM and per-speaker resolution add finer
signal than this floor.

## How it works

Scrape a speech (the BIS speeches feed aggregates every tracked central bank) -> validate it
against the schema spine (the registry of banks and tone labels) -> score tone two ways: a **Gemini**
LLM-as-judge (summary, tone, a calibrated `[-1, 1]` score at temperature 0) and a **deterministic
lexicon** cross-check; a large disagreement flags the speech for review -> persist an immutable
speech and an append-only tone observation -> chunk and embed into pgvector -> answer
natural-language questions about one speaker or the whole corpus, grounded in retrieved excerpts
with citations, abstaining when nothing relevant is found.

## Layout

```
packages/
  cbt_core/   domain heart: schema spine, services, persistence, the LLM boundary, the lexicon.
              Imports no adapter; the one-way dependency is machine-checked.
  cbt_api/    FastAPI JSON adapter. Depends on cbt_core only.
  cbt_worker/ BIS speeches scraper (RSS feed + speech bodies). Depends on cbt_core only.
  cbt_web/    server-rendered (Jinja + htmx) web UI. Depends on cbt_core only.
scripts/      check_imports.py (architecture invariants), eval_tone.py + tone_trajectory.py
              (the evaluation above), migrate.py.
docs/         CHANGELOG.md, adr/ (12 decision records), research/ (the evaluation + the prior-art
              survey with licensing).
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

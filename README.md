# Central Bank Tone

Ingests central bank speeches, summarizes them and scores their tone (hawkish, dovish,
neutral, mixed) per speaker, and answers questions about each speaker's communication. The
domain core (`cbt_core`) holds the model, services, and persistence; adapters (`cbt_api`,
and later a CLI or worker) depend on the core and never the reverse.

## Layout

```
packages/
  cbt_core/   domain heart: settings, schema spine, services, persistence. Imports no adapter.
  cbt_api/    FastAPI adapter. Depends on cbt_core only.
scripts/      check_imports.py: machine-enforces the layering and the settings boundary.
tests/        unit/ (no IO) and integration/ (Postgres via testcontainers).
docs/         CHANGELOG.md (keep-a-changelog) and adr/ (architecture decision records).
.claude/      agent harness: format-on-edit, guarded bash, protected paths.
```

The binding engineering standards live in [CLAUDE.md](CLAUDE.md). Every change must comply.

## Getting started

```bash
uv sync                                  # create .venv and install the workspace
cp .env.example .env                     # then set real values locally (.env is gitignored)
uv run pre-commit install                # install the local commit gate
```

## The quality gate

```bash
uv run ruff check . && uv run ruff format --check .   # lint + format
uv run mypy                                           # strict type check
uv run python scripts/check_imports.py                # architecture invariants
uv run pytest -m "not llm"                            # CI-equivalent test run
```

Integration tests need Docker (Postgres via testcontainers). When Docker is unavailable they
skip with an explicit reason; unit tests cover repository and mapper logic against an
in-process SQLite engine so the coverage gate holds without Docker.

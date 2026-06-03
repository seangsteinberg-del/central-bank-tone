# tests: local map and harnesses

The root `CLAUDE.md` section 5 holds the binding testing rules; this is the local map.
Category is implied by directory; markers are declared in `pyproject.toml` and enforced with
`--strict-markers`.

## Layout

- `tests/unit/` pure Python, no IO or Docker. Most business logic lives here, including
  repository and mapper coverage against an in-process SQLite engine. Runs by default.
- `tests/integration/` DB-level invariants that need real Postgres: the append-only trigger,
  foreign-key `ondelete` behaviour, and the migration up/down round trip. Uses a
  session-scoped testcontainer. Skips with an explicit reason when Docker is unavailable.
- `tests/conftest.py` shared fixtures (settings, SQLite engine, deterministic clock/id
  factories, the optional Postgres container).

## Conventions

- Name tests for behaviour, not implementation.
- Each test is hermetic: no shared mutable state, no order dependency. Seed any RNG; freeze or
  inject the clock and id factory.
- No test calls a live external/paid API. Use stubs, mocks, or cassettes. CI runs
  `pytest -m "not llm"`.
- Use only declared markers; a typo fails under `--strict-markers`.
- Coverage floor is 90 percent (branch on). Security-sensitive code (the append-only
  guarantee, input validation, the production-secret check) is tested exhaustively.

## Commands

```
.venv/Scripts/pytest.exe -m "not llm"     # CI-equivalent
.venv/Scripts/pytest.exe tests/unit -q    # fast loop, no Docker
```

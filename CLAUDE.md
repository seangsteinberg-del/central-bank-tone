# CLAUDE.md: Engineering Standards (binding)

This file is the source of truth for *how* this codebase is built. Every change, whether
human or agent, must comply. Treat the rules here as non-negotiable unless an ADR in
`docs/adr/` explicitly amends one of them.

## 1. What this project is

Central Bank Tone ingests central bank speeches, summarizes them, and scores their tone
(hawkish, dovish, neutral, mixed) per speaker, then answers natural-language questions about
each speaker's communication over time. End to end: fetch and normalize a speech, validate it
against the schema spine (the registry of central banks and tone labels the system knows),
persist the speaker and an append-only tone observation, expose a service API, and answer
questions against the stored corpus. The moving parts: `cbt_core` (domain models, the schema
spine, services, and persistence) and adapters such as `cbt_api` (a FastAPI surface) that
depend on the core and never the reverse. A later worker ingests speeches and an LLM
question-answering path (gated behind the `llm` test marker) answers queries.

## 2. Architecture invariants

- **Layered packages, strict one-way dependency:** every adapter (`cbt_api`, `cbt_cli`, …)
  depends on `cbt_core`; `cbt_core` imports no adapter. Enforced by
  `scripts/check_imports.py` (run in CI and covered by a unit test) plus code review.
- **The service layer is the only entry point** to `cbt_core` for adapters. Adapters never
  reach into repositories, the logger, or external clients directly. `cbt_core/services/*`
  owns transactional boundaries.
- **Domain models and persistence rows are separate types.** Pydantic domain models live in
  `domain/`; ORM rows live in `persistence/`. Mappers in `persistence/mappers.py` translate.
  Never leak an ORM object past the persistence layer.
- **One schema spine.** The core domain's type system (its enums, its registry of "things the
  system knows about": the central banks in `domain/registry.py` and the tone labels in
  `domain/tone.py`) has a single source of truth. Adding one entry there makes validation,
  persistence, and every adapter pick it up for free. Do not hardcode that knowledge anywhere
  else.

## 3. Code quality bar

### Type safety
- `mypy --strict` passes on every push. Zero `# type: ignore` without a one-line reason
  comment naming the upstream bug or library limitation.
- Every public function and class has full annotations. `Any` is permitted only at IO
  boundaries with a comment explaining why.
- Domain types over primitives: pass `Speaker` not `dict`, a `CentralBank` not `str`.

### Lint and format
- `ruff check` and `ruff format --check` pass on every push.
- No unused imports or variables. No commented-out code. No `print()` in library code (use
  structured logging, see section 7).

### Public API discipline
- Every package's public surface is re-exported from its `__init__.py` and listed in
  `__all__`. Imports from deep paths inside another package signal a missing public API.
- A breaking API change is allowed but always documented in `docs/CHANGELOG.md`, maintained
  from the first commit.

### Error handling
- Never swallow exceptions silently. Catch narrowly, re-raise wrapped with context, or log
  and re-raise. Bare `except:` and bare `except Exception:` without re-raise are bugs.
- Define explicit exception types per module. Adapters translate these to HTTP status codes
  or CLI exit codes; they do not invent new ones.
- User-input failures map to 4xx (or the CLI equivalent), not 5xx.

### No silent fallbacks (binding principle)
- Never ship a silent inaccurate or wrong fallback. When the system cannot do the right
  thing, surface a visible, explicit error or honest abstention that says WHY, rather than
  guessing, fabricating, or quietly degrading. A wrong answer presented confidently is worse
  than an honest "could not, here is the reason."

### Inputs and boundaries
- Validate at the boundary, trust inside. Every public service method takes typed values or
  validated models; everything that crosses a boundary (HTTP, CLI, file, external API
  response) is validated against a model before any business logic runs.
- File uploads: enforce mime type, a magic-byte check, a max size from settings, and any
  other format limit. Reject silently-oversize inputs at the framework layer.

## 4. Security

- Secrets via `SecretStr` (`pydantic-settings`). They never appear in logs, repr, exception
  messages, or test fixtures. Tests use dummy values.
- No secret in source, ever. `.env` is gitignored. `.env.example` documents shape only.
- The app connects to the database as a least-privilege role. Append-only tables
  (`tone_observation`) are enforced at the DB level (a trigger that rejects UPDATE/DELETE),
  not by the app alone.
- If you have auth: hash passwords with `bcrypt` directly (cost >= 12), never store or
  transmit plaintext. Sessions use signed cookies (`itsdangerous`), `HttpOnly`, `Secure`
  outside development, `SameSite=Lax`. State-changing routes carry a signed CSRF token.
- SQL: an ORM or `text()` with bound parameters. No string-formatted SQL, ever.
- Dependency vulnerabilities: `pip-audit` runs in CI and breaks the build on a known CVE in a
  pinned version.
- If you record external API interactions (vcrpy cassettes): the recording path strips API
  keys and sensitive headers via a `vcr_config` fixture, and committed cassettes are reviewed.

## 5. Testing: non-negotiable rules

- **Every public function in `cbt_core` has at least one test.** Adapters cover happy path +
  one auth/permission failure + one bad-input failure.
- **Security-sensitive code requires exhaustive tests:** anything touching auth, hashing,
  signing, access control, an audit/immutability guarantee, or input validation.
- **No test calls a live external/paid API.** Use stubs, `unittest.mock`, or `vcrpy`
  cassettes. A `@pytest.mark.llm` marker exists for opt-in manual runs only and is excluded
  from CI.
- **No test depends on wall-clock time or `random` without seeding.** Use `freezegun` /
  `monkeypatch` for clocks, or inject the clock and id factory; seed every RNG.
- **Integration tests use `testcontainers`** (a session-scoped Postgres container,
  function-scoped transaction-rollback isolation), with an in-process SQLite equivalent for
  repository and mapper coverage. CI runs the same migrations.
- **Coverage gate:** overall `fail_under = 90` (branch coverage on). Security-critical
  modules aim well above that.
- **Test names describe behaviour, not implementation:**
  `test_tone_observation_rejects_mutation_of_historical_row`, not `test_trigger_2`.
- **Each test is hermetic.** No global state, no shared mutable fixtures across modules, no
  test-order dependency.

## 6. Database and migrations

- Every schema change ships as a single migration with a working `downgrade()`. The round
  trip (`upgrade head` → `downgrade base` → `upgrade head`) is tested in CI.
- Migrations are forward-only in production; downgrade exists for development and CI.
- No business logic in migrations. Data backfills are a separate, idempotent script.
- Foreign keys use explicit `ondelete` semantics. `CASCADE` only where the parent owns the
  child's lifetime; otherwise `RESTRICT` (e.g. `tone_observation.speaker_id`).
- Constraints (CHECK, UNIQUE, NOT NULL) are declared on the table, not enforced by the app
  alone.

## 7. Observability

- Structured logging via `structlog`. JSON in production, key=value in development.
- Every service method opens a log context with a `correlation_id` (UUID4 per request / call),
  an `actor`, and the relevant entity IDs. Adapters inject the correlation id.
- Log levels: DEBUG (verbose flow), INFO (lifecycle events), WARNING (degraded but
  continuing), ERROR (operation failed). Never WARN as INFO.
- Never log secrets or full external payloads. Log the sha256 of inputs and sizes/counts
  (e.g. `source_sha256` of a speech, never its text).

## 8. Documentation

- **Docstrings:** every public module, class, and function. Google-style sections (`Args:`,
  `Returns:`, `Raises:`, `Example:`). Presence is enforced in CI: ruff `D101`/`D102`/`D103`/
  `D106` fail the build on an undocumented public class, method, function, or nested class.
- **ADRs** in `docs/adr/`, sequentially numbered, each ~one page: context, decision,
  consequences. Required for: dependency additions of meaningful surface, design decisions a
  future reader would otherwise question, and security trade-offs.
- **`docs/CHANGELOG.md`** updated for every user-visible or API-visible change (keep-a-
  changelog format).
- **Writing rules** (UI copy, docs, commit messages): direct plain prose. No em dashes. No
  emojis in chrome or formal docs. Avoid filler superlatives ("seamless", "leverage",
  "unlock", "supercharge", and similar marketing words).

## 9. Dependency management

- `uv` workspace. **One** `uv.lock`, committed. No drift.
- Pinning is conservative on majors (`lib>=1.2,<2`), permissive on minors. Renovate /
  Dependabot opens PRs; CI gates them.
- Adding a dependency requires: (a) an ADR if it is non-obvious, (b) listing it in the *right*
  package (don't put a test-only dep in runtime), (c) a license check (no GPL in runtime).

## 10. Commit and PR discipline

- Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `test:`).
- One concern per PR. PRs over ~400 LOC of non-generated change should be split.
- PRs must: pass CI, include tests for new behaviour, update docs if user-facing, link the
  ADR if architectural.
- `main` is protected: linear history, required reviews, CI green, branch up to date.

## 11. Configuration and environments

- All config goes through `cbt_core.settings.Settings` (`pydantic-settings`, the `CBT_`
  prefix). Never read `os.environ` directly outside that module (machine-enforced by
  `scripts/check_imports.py`).
- No "magic" defaults that differ between environments. Defaults are development-safe and
  production loudly rejects insecure values at startup (a placeholder `CBT_SECRET_KEY` raises
  when `CBT_ENVIRONMENT=production`).

## 12. Useful commands

```bash
uv sync                                  # install workspace
uv run pytest                            # full suite
uv run pytest -m unit                    # fast loop, no Docker
uv run pytest -m "not llm"               # CI-equivalent
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run python scripts/check_imports.py   # architecture invariants
```

## 13. Things you must not do

- Add Python files outside `packages/*/src/<pkg>/`, `tests/`, or `scripts/`.
- Bypass pre-commit hooks (`--no-verify`) or skip CI.
- Print, log, or include secrets in errors. If you find a leak, treat it as a security
  incident and rotate the key.
- Hand-edit `uv.lock`. Use `uv add` / `uv lock`.
- Mutate any append-only/immutable record (`tone_observation`). If state is wrong, append a
  correcting observation.
- Force-push to `main`.

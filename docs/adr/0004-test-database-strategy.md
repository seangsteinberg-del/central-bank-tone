# ADR 0004: Test database strategy (testcontainers Postgres with SQLite unit coverage)

Date: 2026-06-03

Status: Accepted

## Context

Production runs on PostgreSQL. Some invariants are Postgres-specific and can only be verified
against the real engine: the append-only trigger, native foreign-key `ondelete` behaviour, the
enum types, and the alembic migration up/down round trip. At the same time, the test suite must
run fast in the inner loop and must reach the 90 percent coverage gate on machines and CI jobs
that do not have Docker available.

## Decision

Two tiers:

- Unit tests run the repository and mapper logic against an in-process SQLite engine (a shared
  in-memory database with the schema created from the ORM metadata). This is the same
  SQLAlchemy code path the production engine uses, and it covers the Python logic.
- Integration tests (marked `integration`) run against a real PostgreSQL `testcontainers`
  container: the migration round trip, the append-only trigger, and foreign-key semantics. A
  session-scoped container is migrated once and per-test isolation comes from a rolled-back
  connection transaction. When Docker is unavailable the integration fixtures call
  `pytest.skip` with an explicit reason; they are never silently dropped.

Alembic migration scripts are excluded from coverage and from mypy: they are generated,
operational DDL, exercised by the round-trip integration test rather than unit tested.

## Consequences

The inner loop (`pytest -m unit`) needs no Docker and is fast. CI runs `pytest -m "not llm"`
with Docker present, exercising the Postgres-only guarantees. Coverage holds at 90 percent
without Docker because the repository logic is covered by SQLite. The trade-off is that
SQLite and Postgres are not identical; anything that depends on Postgres behaviour (the trigger,
native enums, FK enforcement) must be tested in the integration tier, not assumed from the unit
tier.

## Alternatives rejected

- Postgres for every test: too slow for the inner loop and requires Docker everywhere.
- SQLite only: cannot verify the trigger, native enums, or FK `ondelete`; the security-critical
  append-only guarantee would be untested.
- Mocking the database: would not test the mappers, the SQL, or the constraints at all.

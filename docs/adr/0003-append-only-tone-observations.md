# ADR 0003: Append-only tone observations enforced by the database

Date: 2026-06-03

Status: Accepted

## Context

A tone observation is a judgement about a speaker's communication at a point in time, tied to a
specific source speech (by its sha256). Re-scoring a speech, or correcting a mistake, must not
rewrite history: the prior judgement is part of the record. CLAUDE.md sections 4 and 13 require
append-only guarantees to be enforced at the database level, not by application code alone.

## Decision

The `tone_observation` table is append-only. A PostgreSQL `BEFORE UPDATE OR DELETE` trigger
(`tone_observation_append_only`) calls a `cbt_block_mutation()` function that raises an
exception, both installed by the initial migration, so any UPDATE or DELETE of a row is
rejected. The `ToneObservationRepository` exposes only `append` and `list_for_speaker`; there
is no update or delete path in code. A correction is recorded as a new observation, never an edit. The foreign
key `tone_observation.speaker_id -> speaker.id` uses `ondelete=RESTRICT` so a speaker with
history cannot be deleted out from under its observations.

## Consequences

The immutability guarantee holds even against a bug or a direct SQL statement, and is covered by
integration tests that attempt an UPDATE and a DELETE and assert both are rejected. Corrections
cost a new row, and consumers that want "the current tone" read the latest observation rather
than a mutable field. Deleting a speaker requires first deciding what to do with their history
(an explicit operation), which is the intended friction.

## Alternatives rejected

- App-level "don't update" convention only: a single careless write breaks the guarantee.
- Soft-delete / versioned rows with an `is_current` flag: still mutates historical rows and
  invites partial-update bugs.
- An ORM event hook instead of a trigger: bypassed by any non-ORM access to the database.

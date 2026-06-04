# ADR 0020: Persistent chunk embeddings without pgvector

Date: 2026-06-04

Amends: ADR 0018 (local live setup), ADR 0014 (keyless demo)

Status: Accepted

## Context

The local live setup (ADR 0018) retrieves with the in-process `InMemoryChunkRetriever` because the
native PostgreSQL has no pgvector extension. That retriever holds embeddings in memory and loses
them when the process exits, so every restart had to re-embed the whole corpus to answer questions
again. That is fine for a handful of seeded speeches but wrong for a real, filled-up corpus:
re-embedding hundreds or thousands of speeches through Gemini on every boot is slow and burns the
rate-limited API for no new information. The system needs to be persistent end to end, not just for
the relational rows (speakers, speeches, tone) that already live in PostgreSQL.

pgvector would solve this (a persistent, indexed vector column), but installing it needs
administrator rights the local setup does not assume (ADR 0018). The embeddings themselves are just
vectors; they can be stored in an ordinary column.

## Decision

Add `PersistentChunkRetriever`, an `InMemoryChunkRetriever` subclass that persists each chunk and
its embedding to PostgreSQL and reloads them on startup.

- **Storage is an ordinary `bytea` column.** Each chunk row (`speech_chunk_blob`) holds the chunk
  text and citation metadata plus the embedding packed as float32 bytes, keyed by
  `(speech_id, chunk_index)`. No extension is required; float32 halves the storage versus float64
  with negligible loss for a cosine index.
- **Load on construction, write through on add.** The retriever loads every stored chunk into the
  in-memory cosine index when it is built (so question answering works immediately after a restart),
  and each `add` writes the row to PostgreSQL and then to memory. Retrieval is the base class's
  brute-force cosine, one numpy matmul per query, which is well within budget for a single-operator
  corpus of up to tens of thousands of chunks.
- **Idempotent indexing.** The retriever tracks which speeches it holds (`has_speech`), and the
  indexer skips a speech already present. So a restart or a re-run neither duplicates chunks nor
  re-embeds, and the expensive Gemini embedding is a one-time cost per speech.

`build_demo_services` / `build_demo_app` take a `persistent_retrieval` flag; the live runner sets
it, the keyless demo does not.

## Consequences

- The whole system is persistent: speakers, speeches, and tone in their tables, and now the chunk
  embeddings and the question-answering index. A filled corpus survives restarts and a large fill
  is paid for once.
- Retrieval stays in-process (loaded from the database), so within a process question answering
  spans the full persisted corpus. The remaining reason to run pgvector is a corpus too large to
  hold in memory or a need to share the index across processes; that is the documented upgrade.
- `speech_chunk_blob` is created by the retriever (create-all), not by the alembic migrations, the
  same way `create_demo_schema` builds the relational tables for the no-pgvector setup. It is
  additive and does not touch the production pgvector schema.

## Alternatives rejected

- **Install pgvector.** The right tool for a large or shared deployment, but it needs administrator
  rights the local setup does not have (ADR 0018). The `bytea` store is the no-admin equivalent for
  a single-operator corpus.
- **A sidecar cache file (pickle / npz of the vectors).** Persists without a schema change, but
  introduces cache-invalidation against the database as the source of truth; storing the chunks in
  the database keeps one source of truth.
- **Compute cosine in SQL per query** (no in-memory index). Without pgvector this is a full scan
  with array math in SQL on every question, far slower than one in-memory matmul; loading once into
  memory is both simpler and faster at this scale.

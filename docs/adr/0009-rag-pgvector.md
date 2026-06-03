# ADR 0009: Retrieval-augmented Q&A with pgvector and Gemini embeddings

Date: 2026-06-03

Status: Accepted

## Context

The platform answers natural-language questions about each speaker. A speaker accumulates many
speeches, so the answer path must retrieve the relevant passages rather than stuff every speech
into the prompt. The research (`docs/research/reusable-components.md`) confirms pgvector +
SQLAlchemy + Gemini embeddings as a validated stack; we already run Postgres and use Gemini.

## Decision

Add `pgvector` (the Postgres vector extension and its Python/SQLAlchemy binding) as a runtime
dependency of `cbt_core` (`>=0.3,<2`; license: PostgreSQL/MIT-style, permissive). Each speech is
chunked deterministically (`cbt_core.analysis.chunk_text`), each chunk is embedded with Gemini
(`gemini-embedding-001`, 768 dimensions) through the `LlmClient.embed` boundary, and the vectors
are stored in a `speech_chunk` table (migration 0003) with an HNSW cosine index.

`QaService.answer` embeds the question, retrieves the nearest chunks for the speaker via a
`ChunkRetriever` (the pgvector `SpeechChunkRepository.search`), and asks Gemini to answer grounded
only in those chunks, returning citations. When retrieval finds nothing it abstains with a reason
(`abstained=True`, no citations) rather than fabricating an answer (CLAUDE.md section 3). The
retriever is a protocol so the service is unit tested without a database; the pgvector SQL is
covered by Postgres integration tests.

## Consequences

Q&A scales as the corpus grows and every answer is grounded and citable. `speech_chunk` is the
one table SQLite cannot fully exercise (the `<=>` operators are Postgres-only), so similarity
search is integration-tested; the chunk write path and the Q&A orchestration are unit tested.
Embeddings cost model calls at ingest/index time; indexing is idempotent so re-runs are free.
The vector dimension (768) is fixed in the schema; changing it is a migration plus a re-embed.

## Alternatives rejected

- Stuff all of a speaker's text into the prompt: breaks on context limits and cost as speeches
  accumulate, and gives no citations.
- A separate vector database (Pinecone, Weaviate): another service and likely a paid dependency
  when we already run Postgres; pgvector keeps one datastore.
- Keyword search only: misses paraphrases and semantically related passages.

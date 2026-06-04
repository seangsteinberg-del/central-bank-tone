# ADR 0014: A keyless offline implementation of the LLM boundary

Date: 2026-06-04

Status: Accepted

## Context

The platform's generative work, the speech summary, the tone score, the question-answering, goes
through the `LlmClient` boundary, implemented by Gemini (ADR 0007). Gemini needs an API key and is
gated out of CI. The app already boots without a key (ADR: `LazyGeminiClient`), but ingestion,
indexing, and question answering all fail until a key is set. For a demo, and for anyone evaluating
this project, that means there is nothing to look at without first provisioning a paid key and a
Postgres database.

With the supervised classifier (ADR 0013) there is now a real tone signal that needs no key. That
makes a fully offline path realistic: tone from the classifier, plus a license-clean way to embed
and to answer, so the whole stack runs locally with no key and no Docker.

## Decision

Add `cbt_core.OfflineLlmClient`, a second implementation of the same `LlmClient` boundary, with no
network and no key:

- **`analyze_tone`** scores tone with the classifier and produces a deterministic **extractive**
  summary (the lead sentence plus the most tonally salient remaining sentences) and a rationale
  that names the offline model and its class probabilities. It never emits the MIXED label, which
  the three-class classifier does not produce.
- **`embed`** uses **signed feature hashing**: each unigram/bigram is hashed to a bucket and a
  sign and accumulated into an `EMBEDDING_DIM`-vector, then L2-normalized. It is deterministic
  across processes (a stable BLAKE2b hash, not Python's per-run-salted `hash`) and good enough for
  nearest-neighbour retrieval over a demo corpus.
- **`answer`** returns an explicitly **extractive** answer: the retrieved sentences closest to the
  question, each with its source title, under a clear "no language model is configured (offline
  mode)" preamble. It assembles passages; it never writes prose as if a model generated it.

The honesty point is load-bearing (CLAUDE.md section 3, no silent fallbacks): with no LLM, the
correct behaviour is a visibly-extractive answer that says so, not a fabricated generative-looking
one. The Gemini path remains the production signal; `OfflineLlmClient` is what the demo runner
injects when no key is present.

## Consequences

- The platform is key-optional end to end. A local demo can ingest, score, index, and answer with
  no Gemini key and (with a SQLite engine and an in-process retriever) no Docker.
- The offline answer is extractive, not generative, and is labelled as such. That is a deliberate,
  honest limitation, not a hidden degradation.
- Hashing embeddings are weaker than learned embeddings (no semantics beyond surface n-grams), but
  they are deterministic, free, and adequate for retrieving relevant passages in a demo. The
  Gemini embeddings remain the production retrieval signal.
- It composes with the rest of the architecture for free: because services depend on the
  `LlmClient` protocol and `QaService` depends on a retriever protocol, nothing in the service
  layer changes.

## Alternatives rejected

- **Leave the app non-functional without a key.** A demo nobody can run without provisioning a paid
  key and a database is a poor demo.
- **Fabricate a generative-looking answer offline.** Violates the no-silent-fallback principle; an
  extractive answer that admits it is extractive is the honest behaviour.
- **Bundle a local generative LLM** (e.g. a small instruct model). A large dependency and runtime
  cost for a demo path, when an extractive answer is honest and sufficient.

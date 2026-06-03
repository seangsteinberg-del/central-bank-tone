# ADR 0007: Add the google-genai dependency behind an LlmClient boundary

Date: 2026-06-03

Status: Accepted

## Context

ADR 0006 chose Google Gemini as the single model provider. Calling it needs a client library.
The library is meaningful surface (it reaches the network and handles auth), so per CLAUDE.md
section 9 its addition is recorded here with a license check, and per section 2 it must not leak
past a `cbt_core` boundary.

## Decision

Add `google-genai` (the official `from google import genai` SDK) as a runtime dependency of
`cbt_core`, pinned conservatively (`>=1.0,<2`). License: Apache-2.0, which is permitted in
runtime (no GPL, CLAUDE.md section 9).

It sits behind the `LlmClient` protocol (`cbt_core.llm.client`). The concrete `GeminiClient`
(`cbt_core.llm.gemini`) takes an injected `genai.Client`, so it is unit tested by mocking the
SDK with no network; `build_gemini_client(settings)` wires a real one from
`CBT_GEMINI_API_KEY` and `CBT_GEMINI_MODEL`. Gemini's structured-output mode returns the
`ToneAnalysis` pydantic model directly; a missing or wrong-shaped response raises `LlmError`
rather than fabricating a result (CLAUDE.md section 3). A live smoke test exists behind the
`llm` marker and is excluded from CI.

## Consequences

The domain and services depend only on `LlmClient`, so they stay testable without the network or
a key, and the provider can be replaced behind the interface. `google-genai` pulls a transitive
tree (google-auth, websockets, and others) into the runtime; that is the cost of a maintained
official SDK. The API key is a `SecretStr` and is required in production (validated at startup).

## Alternatives rejected

- Call the Gemini REST endpoint with `httpx` directly: reinvents auth, retries, and streaming
  that the SDK already handles, with no boundary benefit.
- A third-party multi-provider wrapper (for example litellm): extra surface and indirection for
  a project that deliberately uses one provider (ADR 0006).

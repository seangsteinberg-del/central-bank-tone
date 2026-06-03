# ADR 0005: Defer authentication to a later milestone

Date: 2026-06-03

Status: Accepted

## Context

CLAUDE.md section 5 expects adapters to cover an auth/permission failure case, and section 4
describes the auth posture (bcrypt password hashing, signed session cookies, CSRF tokens). The
current scaffold is a green vertical slice over the domain (speakers and tone observations); it
has no users, sessions, or protected routes yet. Building auth now would add surface that is not
exercised by any feature.

## Decision

Authentication and authorization are deferred. The initial API is unauthenticated. Services
already thread an `actor` field through their log context so an authenticated principal can be
attached later without reshaping the service signatures. When auth lands it will follow
CLAUDE.md section 4 (bcrypt cost >= 12, `itsdangerous`-signed `HttpOnly` `SameSite=Lax`
cookies, `Secure` outside development, a signed CSRF token on state-changing routes), ship with
its own ADR, and add the auth-failure adapter test that section 5 requires.

## Consequences

The scaffold stays small and every line is exercised. The API must not be exposed to untrusted
networks until auth exists; this is a known, documented gap rather than a silent one. The
`actor` plumbing means adding auth is additive, not a refactor.

## Update (2026-06-04): the web UI ships with this gap

The `cbt_web` adapter (ADR 0011) now exposes state-changing routes (`/ui/speakers`, `/ui/ingest`,
the ask routes) with no authentication and no CSRF token, which the deferred-auth decision above
covers but which CLAUDE.md section 4 otherwise requires. This is a deliberate, documented
trust-boundary decision for a single-operator demo/research tool: it is intended to run on
localhost or a trusted network, and the write routes trigger model spend, so it must not be
exposed publicly until auth lands. When auth ships, the CSRF/signed-cookie scheme in CLAUDE.md
section 4 applies to these routes and the adapter gains its auth-failure test.

## Alternatives rejected

- Build full auth now: speculative surface with no feature using it, and untested branches.
- Pretend the gap does not exist: violates the no-silent-fallback principle; the limitation is
  recorded here instead.

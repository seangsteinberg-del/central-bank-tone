# ADR 0001: Record architecture decisions

Date: 2026-06-03

Status: Accepted

## Context

We need a durable, reviewable record of the decisions that shape this codebase, so a future
reader (human or agent) understands why the code is the way it is without re-litigating settled
questions.

## Decision

We use Architecture Decision Records, one Markdown file per decision in `docs/adr/`, numbered
sequentially, following the template in `0000-adr-template.md`. An ADR is required for
dependency additions of meaningful surface, non-obvious design decisions, and security
trade-offs (CLAUDE.md section 8). An ADR amends a binding rule in `CLAUDE.md` only when it says
so explicitly.

## Consequences

Decisions are discoverable and stable. Reviews can point at an ADR instead of repeating the
rationale. The cost is a short writeup per significant decision, which is the point.

## Alternatives rejected

- Decisions in commit messages only: not discoverable, not amendable.
- A wiki: drifts from the code; ADRs live in the repo and version with it.

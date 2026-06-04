# ADR 0015: Point-in-time committee tone-movement read model

Date: 2026-06-04

Status: Accepted

## Context

The platform stored tone per speaker and charted one speaker's trajectory, but a speech was read in
isolation. A speech is given by one member of a committee (a central bank), and the question a
reader actually has when they open it is comparative: how much did this speaker shift, how does each
of their colleagues now read, and which way has the committee moved overall. We had all the data to
answer that (speakers and their append-only tone observations) but no view that assembled it, and no
way to click from a speech into that context.

## Decision

Add a `CommitteeService.movement_for_speech(speech_id)` read model and a speech detail page that
renders it. The semantics are deliberately **point-in-time as of the speech's delivery date**, so
the view never implies a reading is fresher than it is:

- A speaker is "on the committee as of the speech" only if they have at least one tone observation
  on or before that date. A colleague who had not yet spoken is not counted; a former member's last
  standing reading is shown with its own date.
- A member's `current` reading is their latest observation on or before the date (never a later
  one), and their `previous` is the one before that. The shift (`delta`) is `current - previous`, or
  absent when they have only one reading.
- The committee's **standing tone** is the mean of members' current scores. The **overall move** is
  the mean of members' individual shifts, averaged only over members that have a prior reading
  (reported with that count), so a member with a single reading does not silently count as "no
  change".

The read models (`MemberMovement`, `CommitteeMovement`) are immutable domain values in
`cbt_core.domain.committee`; the service reads through the existing repositories and records
nothing. The page itself derives presentation only (a diverging movement bar scaled to the largest
mover, direction words); the numbers are the service's.

## Consequences

- Clicking any speech (dashboard, speaker page) opens a detail page with a concise summary and the
  committee context, which is the comparative read a user wants and a stronger demo surface.
- The point-in-time rule means the same committee renders differently as of different speeches,
  which is correct: it reflects who had spoken and what their latest reading was at that moment. It
  also handles the demo's single-active-chair corpus gracefully (a 2006 speech shows the chair plus
  the prior chair's last standing reading, not chairs who post-date it).
- The aggregate is honest about coverage: `overall_delta` is averaged only over members with a
  measurable shift and the page states how many that is, rather than diluting the mean with
  zero-shift placeholders or hiding members with no prior reading.
- This is a read model, not a new write path: it adds no migration and cannot mutate the append-only
  tone history (CLAUDE.md sections 3 and 13). It iterates speakers in process, which is fine for a
  single-operator tool; a high-cardinality deployment would push it into a dedicated read store.

## Alternatives rejected

- **Compare every member's most recent reading regardless of date.** Simpler, but it would let a
  chair who left a decade earlier appear to "move" on a speech they had nothing to do with. The
  point-in-time filter is what keeps the view defensible.
- **Define the overall move as the change in the committee mean across all members.** Mixing members
  who have a prior reading with those who do not makes the aggregate depend on roster size in a way
  that is hard to read; the mean of individual shifts over the members that actually have one is the
  honest quantity, and the count is shown alongside it.
- **A fixed trailing window (for example, the last 12 months) instead of "previous reading".** More
  configuration and a tunable that biases the result; "their previous analyzed speech" is the
  unambiguous, data-defined comparison the corpus supports.

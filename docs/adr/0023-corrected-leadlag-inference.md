# ADR 0023: Corrected lead-lag inference (block bootstrap, family-wise CI, one shared implementation)

Date: 2026-06-09

Status: Accepted

Supersedes part of ADR 0022 (the "copy the statistics, do not import them" decision).

## Context

The Signal vs Market lead-lag test (ADR 0021, ADR 0022) reported a per-cell 95% bootstrap CI that
was, on review, too optimistic for the structure of the data, and the offline eval that is supposed
to regenerate the published numbers was silently broken.

Three concrete defects:

1. **The bootstrap ignored serial dependence.** Each pair is a tone level against a forward rate
   *change*; the 3- and 6-month windows overlap (month m's "+3" shares two months with m+1's), and
   both series are persistent. The CI resampled pairs i.i.d., which treats overlapping, autocorrelated
   observations as independent and reports intervals that are too narrow. So "CI excludes zero" was
   not trustworthy as published.
2. **No multiple-testing correction.** The page tests twelve cells at once (two tone indices x two
   rate series x three horizons) and highlights the ones that exclude zero. With no family-wise
   adjustment, one or two cells excluding zero is expected under the null.
3. **The eval did not run on the served corpus.** `scripts/eval_corpus_vs_rates.py` filtered on the
   uppercase enum *name* (`'FEDERAL_RESERVE'`), but the ORM stores the enum *value*
   (`'federal_reserve'`), so against the live SQLite corpus it matched zero rows and raised. The
   published methodology figures therefore came from an earlier run, free to drift from the served
   data, which is precisely the failure ADR 0022 set out to close.

The binding no-silent-overstatement principle (CLAUDE.md section 3) makes a displayed significance
that is stronger than the statistics justify a defect, not a nuance.

## Decision

1. **One shared pure-stats module**, `cbt_core/analysis/leadlag.py`, imported by *both*
   `MarketSignalService` (served, per request) and `scripts/eval_corpus_vs_rates.py` (the offline
   report). The script imports from `cbt_core` (allowed; scripts depend on the core) and the service
   imports within `cbt_core` (allowed), so there is no layering violation. This supersedes ADR 0022's
   decision to copy the statistics into the service: the copy was what let the two surfaces drift, and
   a single implementation makes published-equals-served true by construction.
2. **Circular moving-block bootstrap** instead of i.i.d. pair resampling. Pairs are returned in
   chronological order and resampled in contiguous (wrap-around) blocks whose length is at least
   `horizon + 1` (to span the window overlap), grows like `n**(1/3)`, and is capped at half the
   sample so there are always at least two blocks. This preserves the short-range dependence the
   overlapping windows create.
3. **Family-wise (Bonferroni) corrected CI.** Each interval is taken at the percentile
   `FAMILY_WISE_ALPHA / family_size`, controlling the chance that *any* of the twelve cells excludes
   zero by chance. `LeadCorrelation` gains a `family_size` field recording the correction, and its
   `excludes_zero` (which drives the "leads" / "win" highlight on both pages) is now a
   multiple-testing-corrected significance, deliberately stricter than a naive per-cell interval.
4. **Fix the eval and regenerate.** Bind `CentralBank.FEDERAL_RESERVE.value` in the eval's SQL so it
   runs keyless on the SQLite corpus, parse the timestamp from either a datetime (PostgreSQL) or an
   ISO string (SQLite), bump the resample count to 10000 (the family-wise tail is deep), and
   regenerate `docs/research/corpus-tone-vs-rates.md`. The methodology page is updated to the
   corrected framing and the current 366-speech count.

## Consequences

- The intervals widen, as they should. On the served 2023-2026 corpus the headline tone's lead on the
  (administered, telegraphed) effective fed funds rate *survives* the correction at every horizon
  (same-month +0.67 [+0.31, +0.84], +3mo +0.72 [+0.42, +0.85], +6mo +0.59 [+0.06, +0.81]); against the
  freely-repricing 2-year yield every interval includes zero, so no lead is claimed; the rate-path
  index co-moves with fed funds only contemporaneously, its leads no longer excluding zero.
- The published and served numbers are now one implementation; the drift ADR 0022 fought is closed
  structurally, not by discipline. The methodology table's point estimates and win/loss pattern are
  unchanged (the correction widened intervals without flipping the headline result), and the prose now
  names the moving-block, family-wise method and the fed-funds-is-administered caveat.
- This remains in-sample correlation over a single hiking-and-cutting cycle. The correction makes the
  stated uncertainty honest; it does not turn the descriptive finding into out-of-sample tradeable
  alpha (a PnL backtest against a market-surprise series stays out of scope, per ADR 0012).
- The per-request bootstrap is vectorized over numpy, so 10000 resamples across twelve cells is still
  fast enough for a page render.

## Alternatives rejected

- **Keep the i.i.d. pair bootstrap.** It understates the variance on overlapping windows; the whole
  point is to stop overstating significance.
- **Per-cell 95% with no family-wise correction.** Across twelve simultaneous tests the family will
  surface false positives; the highlight must reflect that.
- **Analytic Newey-West / HAC standard errors.** A defensible alternative, but the block bootstrap is
  already in place, is distribution-free, and reuses the existing percentile machinery.
- **Duplicate the corrected statistics in both files.** Reintroduces exactly the copy-and-drift that
  caused the published-vs-served mismatch; the shared module is the fix.

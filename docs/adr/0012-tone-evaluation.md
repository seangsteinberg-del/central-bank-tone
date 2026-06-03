# ADR 0012: Evaluate the tone scorers against labeled data, and test the thesis

Date: 2026-06-04

Status: Accepted

## Context

The platform's value is a tone signal, and a signal nobody has measured is worthless to the
audience that would use it. The repo previously asserted a methodology (ADR 0008) but never
measured whether either scorer agrees with human labels, and never checked whether the tone series
tracks anything in the world. "99% coverage" measures the stubbed code, not the correctness of the
score. We need a real, reproducible evaluation, and we need it to run offline (no API key) so it is
always available.

## Decision

Two committed, reproducible evaluation scripts under `scripts/`, both lexicon-scored so they need
no API key (the Gemini path plugs into the same harness once a key is set):

- `scripts/eval_tone.py` scores the tone classifiers against the annotated FOMC benchmark from
  "Trillion Dollar Words" (Shah, Paturi, Chava, ACL 2023; `gtfintechlab/fomc_communication`,
  CC BY-NC 4.0, used for offline evaluation only). It tunes the lexicon's decision threshold on the
  train split and reports accuracy, macro-F1, per-class precision/recall, and a confusion matrix on
  the held-out test split, against the majority-class baseline so the lift is honest. It includes a
  label-mapping sanity check. Output: `docs/research/tone-evaluation.md` plus a confusion-matrix PNG.

- `scripts/tone_trajectory.py` tests the thesis: it aggregates lexicon net-hawkishness by year over
  the FOMC corpus and overlays it on the effective fed funds rate (FRED, no key), reporting the
  correlation with the rate level, the same-year change, and the next-year change (a lead test).
  Output: `docs/research/tone-vs-rates.md` plus a chart.

The CC BY-NC corpus is fetched into a gitignored cache and is **not** redistributed in the repo;
only our computed metrics and charts are committed. The Gemini path (`--with-gemini`) runs the same
metrics head-to-head when `CBT_GEMINI_API_KEY` is set.

## Consequences

The repo can answer the two questions the target audience asks - "how accurate is it?" and "does it
track policy?" - with committed numbers rather than claims. The current honest readings: the
lexicon beats the majority-class baseline modestly and fires on a minority of sentences (it is a
coarse, high-precision floor), and aggregate FOMC net-hawkishness correlates positively with the
same-year change in the fed funds rate, ~0 as a next-year lead. No claim of alpha is made; the
artifacts demonstrate the signal is measurable and behaves in the literature-predicted direction,
and they set up the Gemini-vs-lexicon comparison. Reproducing them needs network (HuggingFace,
FRED); the scripts cache the corpus locally after the first run.

## Alternatives rejected

- Keep asserting the methodology without measuring it: the gap a reviewer finds fastest, and the
  thing the role actually screens for.
- Commit the labeled corpus for turnkey reproduction: simpler, but redistributes a CC BY-NC dataset;
  fetching on demand respects the license.
- A full market backtest with returns: out of scope for a demo and easy to overclaim; a transparent
  correlation against the policy rate is the honest, defensible floor.

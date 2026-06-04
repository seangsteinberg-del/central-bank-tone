# ADR 0021: Structured sentence-level stance pipeline, ensembled and calibrated

Date: 2026-06-05

Status: Accepted (amends ADR 0008)

## Context

ADR 0008 set the production tone signal as a single Gemini call over the whole speech, scored on an
anchored scale, with the deterministic lexicon as a cross-check. A research pass over the state of
the art (27 primary sources, adversarially verified; see `docs/research/tone-sota-blueprint.md`)
found that this single-greedy-call shape is the weakest documented method: zero-shot frontier LLMs
score about 0.57 to 0.61 on the FOMC hawkish/dovish benchmark, barely above a majority-class
baseline, while the field's standard pipeline reaches about 0.71. The conclusion held across every
benchmark: a structured, sentence-level pipeline beats one greedy call, and the largest part of the
gain is in the pipeline shape, not the model, so it is capturable on Gemini today without new heavy
dependencies.

The same literature is consistent on three further points: classify at the sentence level but
filter to policy-relevant sentences before aggregating (so neutral filler does not dilute the
signal); separate orthogonal axes (rate-path intent vs description, and aspect) rather than emitting
one blended tone; and validate the text signal against market repricing.

## Decision

Move the production tone score from one whole-speech call to a structured pipeline, built around a
new pure, model-agnostic core module `cbt_core.analysis.stance`:

- **Split and filter.** Split the speech into sentences and keep only policy-relevant ones with a
  Gorodnichenko-style keyword gate (`PolicyRelevanceFilter`). Relevance is about topic, not
  polarity; the filter is broad and includes neutral policy nouns.
- **Classify each sentence** as hawkish / dovish / neutral, and tag each with an `Aspect` (inflation,
  growth, employment, balance sheet, financial stability, guidance, other) and a `Horizon` (forward,
  backward, unspecified). Who classifies is injected: the Gemini classifier in production (one
  batched structured call, not one call per sentence), the supervised classifier offline, or a stub
  in tests. The irreproducible model judgement stays out of this module.
- **Aggregate** with the Trillion Dollar Words measure `(#Hawkish - #Dovish) / #relevant`
  (`aggregate_stances`). The same measure is computed over forward-looking sentences only
  (`forward_net`, the rate-path intent signal) and within each aspect (`by_aspect`). The continuous
  measure maps to a discrete `ToneLabel` with an honest `MIXED` only when both sides are materially
  present, and an honest abstention (`NEUTRAL`, `relevant == 0`) rather than a silent neutral.
- **Ensemble and calibrate** (subsequent steps building on this module): combine the Gemini stance
  measure with the supervised classifier and the lexicon, surface ensemble disagreement as an
  explicit uncertainty band (extending the existing `needs_review`), and calibrate the production
  measure against free FRBSF / Bauer-Swanson monetary-policy surprises so the signal carries a
  measured number, not just an anchored prompt.

The accounting (split, filter, measure, label mapping) is pure and deterministic, so it is tested
without a network or a GPU and is exactly reproducible between production, the offline path, and
tests.

## Consequences

- The production score stops being a single greedy call and gains a rate-path (forward-looking)
  sub-score and a per-aspect breakdown, which are the decision-relevant axes for a macro reader.
- The pipeline is model-agnostic: it runs on Gemini now, and a stronger sentence classifier (a local
  fine-tuned model, a future phase) can be dropped in behind the same boundary without touching the
  aggregation.
- The honest limits stand and are documented: the field ceiling is about 0.71 to 0.73 F1, so the
  signal is noisy and must carry uncertainty; every labeled benchmark is Fed-only, so transfer to
  the other tracked banks is measured, not assumed.

## License provenance

The production path stays license-clean: Gemini plus our own self-authored lexicon and supervised
classifier. The CC BY-NC artifacts the research recommends as the strongest off-the-shelf baseline
(FOMC-RoBERTa) and the FOMC benchmark are used only offline, as an evaluation oracle and a
calibration aid, exactly as ADR 0008 and ADR 0013 already treat the benchmark. They are never shipped
in the always-on scorer.

## Alternatives rejected

- **Keep the single whole-speech call.** It is the weakest documented method and gives no
  sentence-level, aspect, or rate-path signal.
- **One Gemini call per sentence.** Faithful but slow and costly; a single batched call that returns
  a per-sentence label array gives the same granularity at one call per speech.
- **Average the model, classifier, and lexicon into one number.** Hides disagreement, which ADR 0008
  deliberately surfaces; the ensemble reports disagreement as uncertainty instead.
- **Ship a fine-tuned transformer (FOMC-RoBERTa) as the production model now.** Fed-only and
  out-of-distribution for our eight-bank corpus, CC BY-NC, and a heavy runtime dependency; it earns
  its place as an offline oracle, not the live scorer.

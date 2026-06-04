# ADR 0013: A supervised TF-IDF tone classifier, trained and measured offline

Date: 2026-06-04

Status: Accepted

## Context

The platform scored tone two ways (ADR 0008): a deterministic lexicon and the Gemini LLM judge.
The evaluation against the annotated FOMC benchmark (ADR 0012) showed the lexicon is a weak floor
on its own: 51.8% accuracy and 0.339 macro-F1, firing on only ~12% of sentences (it abstains to
neutral whenever none of its curated terms appear). The Gemini judge is the intended production
signal, but it needs an API key and is gated out of CI, so with no key the platform had no
*measured*, learned tone signal at all. A demo whose central claim is "we score tone" should be
able to show a real supervised model that beats the transparent baseline, offline, with numbers.

## Decision

Add a third scorer: a supervised **TF-IDF + class-balanced multinomial logistic regression**,
implemented in pure numpy and trained offline on the benchmark's train split.

- **Features.** Unigrams and bigrams, document-frequency pruned (`min_df=3`, top 5000 by
  frequency), smoothed IDF, sublinear term frequency, L2-normalized rows. The featurizers live in
  `cbt_core.analysis.classifier` and are shared by training and inference, so a sentence is
  vectorized identically in both, by construction.
- **Model.** Softmax regression fit by full-batch Adam on the class-balanced cross-entropy loss
  with L2 regularization. Class balancing (inverse-frequency sample weights) stops the dominant
  neutral class from drowning out hawkish and dovish; it is what lifts macro-F1. The L2 strength
  is chosen by k-fold cross-validation **on train only**; the held-out test split is scored once.
- **Shape.** Training (`scripts/train_tone_model.py`) writes a small JSON artifact (vocabulary,
  IDF, weights, ~300 KB) committed under `cbt_core/analysis/tone_model.json`. At runtime
  `cbt_core.ToneClassifier` loads it and does inference only (no training code, no network). It
  returns a label, the full softmax distribution, and a continuous `P(hawkish) - P(dovish)` score.

Measured result on the held-out test split (`scripts/eval_tone.py`,
`docs/research/tone-evaluation.md`): **59.9% accuracy, 0.582 macro-F1**, versus the lexicon's
51.8% / 0.339 and a 49.8% majority-class baseline. The gain over the lexicon is significant
(McNemar chi-square 6.3, p = 0.012) with a bootstrap 95% CI on the accuracy gap of [+2.2%, +14.1%]
that excludes zero. Per-class F1 is balanced (hawkish 0.50, dovish 0.60, neutral 0.65) rather than
collapsing onto neutral.

**Dependency.** This adds `numpy` (BSD-3, ubiquitous) to `cbt_core` at runtime. It is the only new
runtime dependency; the model deliberately avoids scikit-learn / PyTorch so the runtime stays
light and the maths stays transparent and auditable.

**License provenance.** The artifact is trained on the CC BY-NC 4.0 FOMC benchmark, so the trained
weights inherit the non-commercial term, exactly as FOMC-RoBERTa does (ADR 0008). The classifier
is therefore a research/evaluation artifact and an additional offline scorer, not a relicensing of
the benchmark. The license-clean production path is unchanged: the self-authored lexicon plus the
Gemini judge. The labeled corpus itself is never redistributed (gitignored cache; only weights and
computed metrics are committed).

## Consequences

- There is now a real, learned tone signal that runs with no API key and no Docker, and the
  evaluation is a genuine head-to-head (lexicon vs classifier vs, with a key, Gemini) with a
  significance test, not a single self-reported number.
- The model is a linear bag-of-words classifier on a hard three-class problem: a credible,
  reproducible baseline, not a claim of state of the art (the source paper's fine-tuned
  transformer scores higher). The evaluation says so plainly.
- Maintenance cost is the training script and the committed artifact; retraining is one command
  and fully deterministic (fixed seed).
- The classifier slots in as a peer of the lexicon as a deterministic, offline scorer, which makes
  a keyless end-to-end demo path possible (see the offline scoring work that builds on it).

## Alternatives rejected

- **Ship FOMC-RoBERTa as the model.** Stronger, but CC BY-NC (non-commercial), FOMC-only,
  English-only, and a heavy PyTorch/Transformers runtime dependency for a demo.
- **Use scikit-learn.** Convenient, but a large runtime dependency for what is a few lines of
  numpy; implementing the optimizer ourselves keeps the runtime light and the method transparent.
- **Keep only the lexicon offline.** Too weak (macro-F1 0.339) to credibly stand behind.
- **Average the model and lexicon into one score.** Hides disagreement, which ADR 0008 deliberately
  surfaces for review rather than smoothing away.

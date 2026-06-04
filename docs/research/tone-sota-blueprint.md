# How the best measure central-bank tone: a copyable blueprint

A synthesis of the state of the art for turning central-bank text into a tradeable hawkish/dovish
signal, scoped for a macro reader. Produced by a fan-out research pass (27 primary sources, 25
claims adversarially verified, 21 confirmed, 4 refuted). This is the evidence base for ADR 0021;
the headline conclusions are cited inline so a future reader can check them.

## The one finding that reframes everything

**A single greedy LLM call is the weakest documented method.** Across independent benchmarks,
zero-shot frontier models score around 0.57 to 0.61 on FOMC hawkish/dovish, barely above a
majority-class baseline and far below the fine-tuned ceiling:

- Zero-shot ChatGPT 0.5872 vs a fine-tuned RoBERTa-large 0.7171 on the same test split
  ([Trillion Dollar Words, ACL 2023](https://aclanthology.org/2023.acl-long.368/)).
- On the harder sentence-level stance task, the best zero-shot model (Claude Opus 4.1) reaches
  0.61 accuracy vs a 0.59 majority baseline and a 0.89 human baseline; DeepSeek-V3.1 (0.29) loses
  to a bag-of-words logistic regression ([Op-Fed, 2025](https://arxiv.org/pdf/2509.13539)).
- Zero-shot GPT-4 reaches F1 0.57 (0.58 few-shot) on a five-point hawk-dove scale
  ([Peskoff/Blinder et al., 2024](https://arxiv.org/abs/2407.19110)).

Our current production scorer is exactly this weak shape: one greedy whole-speech Gemini call. The
structural conclusion held across every benchmark to date: fine-tuned and/or structured beats a
single greedy call.

## The canonical recipe (the spine to copy)

The "Trillion Dollar Words" pipeline (Shah, Paturi and Chava, ACL 2023) is the public, copyable
baseline and it is four steps:

1. **Ingest the full public feed**, not just polished statements. FOMC public statements show
   dissent 47% of the time vs 82% in transcripts; a signal built on statements alone misses the
   dispersion markets react to ([Peskoff et al., 2024](https://arxiv.org/abs/2407.19110)). Use
   speeches, statements, minutes, press-conference transcripts, and decisions.
2. **Sentence-split, then filter to policy-relevant sentences** with a Gorodnichenko-style keyword
   gate (inflation, rates, employment, growth). This drops the roughly 62% neutral filler that
   otherwise dilutes the average.
3. **Classify each surviving sentence** hawkish / dovish / neutral, with a model trained on that
   specific three-class scheme. "Increase" can be hawkish or dovish by context, so generic
   positive/negative sentiment models are wrong for the task
   ([Fatemi et al., 2024](https://arxiv.org/pdf/2411.02476)). Off-the-shelf
   [FOMC-RoBERTa](https://huggingface.co/gtfintechlab/FOMC-RoBERTa) (label map LABEL_0=Dovish,
   LABEL_1=Hawkish, LABEL_2=Neutral) is the free ~0.71-F1 baseline.
4. **Aggregate per release**: `Measure = (#Hawkish - #Dovish) / #Total`, where `#Total` is the
   count of policy-relevant sentences (neutrals in the denominator). One normalized
   net-hawkishness number per release.

The reconciliation worth copying on granularity: classify at the **sentence** level (for the aspect
and forward-guidance axes below), but **filter** before aggregating, so you keep granularity
without the neutral-dilution that motivated some whole-document scoring.

## Separate orthogonal axes, do not emit one blended tone

The strongest 2025 systems decompose tone instead of collapsing it:

- **Rate-PATH intent vs opinion vs fact.** Op-Fed frames stance as natural-language inference
  against the fixed hypothesis "We should tighten monetary policy" (entailment = hawkish,
  contradiction = dovish), via a five-stage schema that separates subjective opinion from
  MP-relevance from rate direction ([Op-Fed, 2025](https://arxiv.org/pdf/2509.13539)). This is the
  axis a macro reader trades: intent to move, not mood.
- **Forward- vs backward-looking.** The IMF fine-tunes a multilingual sentence encoder (BGE-m3,
  deliberately not a generative model) over 21M sentences across 169 central banks, keeping the
  directional hawkish/dovish label on a separate axis from the temporal one; only the
  forward-looking component predicts market-based rates
  ([IMF WP/2025/109](https://www.imf.org/-/media/files/publications/wp/2025/english/wpiea2025109-print-pdf.pdf)).
- **Which agent.** CentralBankRoBERTa classifies the economic agent a sentence addresses
  (households, firms, financial sector, government, central bank) on a separate axis from sentiment
  ([Pfeifer and Marohl, 2023](https://www.sciencedirect.com/science/article/pii/S2405918823000302)).
  Note: it does not output a rate-path tone directly.
- **Composite indices.** The IMF collapses sentence labels into a small set of named numbers (Net
  Policy Sentiment, decomposed into forward/backward, plus Straightforwardness, Explanation, and
  Net Confidence) rather than one opaque score. A copyable template for interpretable aggregation.

## Market-implied ground truth (and the honest free-data limit)

Text tone is validated against market repricing around the release: the Trillion Dollar Words
measure is studied against the treasury and equity markets, and the IMF's forward-looking sentiment
predicts changes in market-based interest rates. Market repricing is the calibration target.

The honest constraint: central-bank **text** is fully public, but the **intraday tick** market data
to build a clean surprise around a release is generally not free. The best free proxies, several
published by the Fed itself:

- [FRBSF Monetary Policy Surprises](https://www.frbsf.org/research-and-insights/data-and-indicators/monetary-policy-surprises/)
  and the
  [US Monetary Policy Event-Study Database](https://www.frbsf.org/research-and-insights/data-and-indicators/us-monetary-policy-event-study-database/)
  (Bauer-Swanson) publish already-computed target/path factors.
- [Atlanta Fed Market Probability Tracker](https://www.atlantafed.org/cenfis/market-probability-tracker)
  publishes rate-move probabilities implied by fed-funds futures.

These are Fed-only and daily-or-coarser. The research could not independently verify the
surprise-construction specifics (Kuttner, Gurkaynak-Sack-Swanson, Nakamura-Steinsson, Swanson), so
the market layer is lower-confidence than the text-modeling findings and is treated as a
calibration check, not a label source.

## The frontier (Phase 2 territory)

The current best is a **fine-tuned** open model with reasoning grounded in the monetary-policy
transmission mechanism: a fine-tuned Qwen3-14B reaches 0.7327 Macro-F1, a 6.6% gain over the best
zero-shot baseline (GPT-4.1 at 0.6662) and above RoBERTa-large's ~0.71
([Yao et al., AAAI 2026](https://arxiv.org/html/2508.08001v2)). The path past ~0.71 is
fine-tuning plus transmission reasoning, not bigger zero-shot prompts. This needs local GPU compute
to train and serve.

## Honest ceiling and limits

- The realistic ceiling for sentence-level hawkish/dovish classification today is ~0.71 to 0.73
  F1. The signal is irreducibly noisy; any tradeable number must carry explicit uncertainty.
- Every labeled benchmark above is **Fed-only and mostly English**. Our corpus is eight banks.
  FOMC-trained accuracy on the ECB, BoE, or BoJ is unverified; transfer must be measured, not
  assumed. The IMF multilingual approach is the known answer at scale but ships no off-the-shelf
  labeled classifier.
- Several frontier numbers (Qwen3-14B 0.73; the IMF market-prediction results) are self-reported,
  in-sample, or working-paper results, not independent out-of-sample replications. None of the
  sources demonstrate realized out-of-sample tradeable PnL, only predictive correlation.

## What this implies for us (mapped to ADR 0021)

1. Replace the single greedy call with the **filter to classify to normalized-measure** pipeline.
   Most of the gain is in the pipeline shape and is model-agnostic, so it runs on Gemini today.
2. Add the **forward-looking (rate-path)** and **aspect** axes as structured per-sentence output.
3. **Ensemble** the Gemini stance with the supervised classifier (100% sentence coverage) and the
   lexicon, and surface ensemble disagreement as an explicit **uncertainty band**.
4. **Calibrate** against the free FRBSF/Bauer-Swanson surprises to finally put a measured number on
   the production signal (Fed-only; honest about it).
5. Keep CC BY-NC artifacts (FOMC-RoBERTa, the FOMC benchmark) **offline** as an oracle and
   calibration aid; the always-on production path stays license-clean (Gemini plus our own models).
6. A local fine-tuned model (Qwen3-14B style) is a later, GPU-gated phase, justified only if the
   measured numbers from steps 1 to 4 warrant the last few F1 points and the operational cost.

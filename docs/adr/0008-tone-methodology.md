# ADR 0008: Tone methodology - Gemini LLM-as-judge plus a deterministic lexicon baseline

Date: 2026-06-03

Status: Accepted

## Context

The platform's core output is a hawkish/dovish tone for each speech. Prior art (research notes in
`docs/research/reusable-components.md`) offers three families: finance/central-bank lexicons
(Apel & Blix Grimaldi 2012, Loughran-McDonald), fine-tuned classifiers (FOMC-RoBERTa from the ACL
2023 "Trillion Dollar Words" benchmark, CentralBankRoBERTa), and market/structural measures
(structural shadow rate). The strongest single classifier, FOMC-RoBERTa, is CC BY-NC 4.0
(non-commercial) and FOMC-only; Loughran-McDonald is commercial-by-permission. We need a method
that is license-clean, multi-institution, and works on non-English originals, with a transparent
cross-check.

## Decision

Two complementary signals per speech, both license-clean:

- Primary: a **Gemini LLM-as-judge** score (`ToneAnalysis`: summary, tone label, score in
  `[-1, 1]`, rationale), already built behind the `LlmClient` boundary (ADR 0007). Gemini is
  multilingual and multi-institution, and a malformed response raises rather than guesses.
- Baseline: a **deterministic lexicon scorer** (`cbt_core.analysis.lexicon`) computing a
  *simplified net-hawkishness ratio* inspired by Apel & Blix Grimaldi (2012) - `(hawkish -
  dovish) / total` over our own curated hawkish/dovish term lists, with longest-match phrase
  counting (so a phrase is not double-counted by a substring) and a short negation window. It is
  not the full sentence-level categorization of the original method; it is a transparent,
  license-clean floor. We author the word lists ourselves rather than copying the licensed
  Loughran-McDonald or non-commercial FOMC dictionaries.

Both are stored on each analyzed speech (the model `score` and the `lexicon_score`). The
deterministic baseline is a visible cross-check on the model score, and the disagreement is acted
on, not just displayed: `analysis.disagrees()` flags an opposite-sign disagreement or a large
magnitude gap (when the lexicon fired), `IngestionService` sets `needs_review` on the speech and
its tone observation and logs a WARNING, and the UI shows a "model/lexicon disagree" marker. A
large disagreement is surfaced for review, not silently averaged away.

The lexicon's accuracy is measured, not asserted: `scripts/eval_tone.py` scores it against the
annotated FOMC benchmark and reports accuracy, macro-F1, and a confusion matrix (see ADR 0012 and
`docs/research/tone-evaluation.md`). The same harness scores the Gemini path head-to-head once a
key is set.

## Consequences

The product owns its tone signal end to end with no non-commercial dependency, and the lexicon
gives an auditable, reproducible baseline that works even when the model is unavailable. The cost
is maintaining the word lists, which are deliberately small and documented, and the baseline is a
deliberately coarse floor (it fires on a minority of sentences; see the evaluation). FOMC-RoBERTa
and the annotated FOMC benchmark are used for offline evaluation of both scorers under their
non-commercial terms, without shipping in the product.

## Alternatives rejected

- Ship FOMC-RoBERTa as the tone model: CC BY-NC 4.0 blocks commercial use, FOMC-only, English-only.
- Lexicon only: cheap and transparent but weak on nuance, sarcasm, and non-English text.
- Average the model and lexicon into one number: hides disagreement, which is exactly the signal
  worth surfacing (CLAUDE.md section 3, no silent degradation).

# State of the art: central bank tone analysis (survey and positioning)

A survey of the methods, datasets, and open-source code for scoring hawkish/dovish tone in central
bank communication, with each source fetched and verified, used to position this project's signal
honestly and to set the next steps. Compiled from a multi-source research pass; every claim below
links its primary source.

## Where this project's classifier sits

The supervised classifier here scores **59.9% accuracy / 0.582 macro-F1** on the FOMC benchmark
(`docs/research/tone-evaluation.md`). Against the literature on the same family of task:

- **Fine-tuned transformers are the ceiling.** On the exact benchmark we evaluate against, RoBERTa-large
  reaches weighted F1 ~0.717 ([gtfintechlab/fomc-hawkish-dovish](https://github.com/gtfintechlab/fomc-hawkish-dovish),
  Table 5), and on the larger 25-bank WorldCentralBanks corpus RoBERTa-large reaches stance weighted F1
  0.740 ([gtfintechlab/WorldCentralBanks](https://github.com/gtfintechlab/WorldCentralBanks)).
- **Zero-shot LLMs land lower.** Zero-shot ChatGPT (gpt-3.5-turbo) scores F1 ~0.587 on the combined FOMC
  data, ~0.13 below fine-tuned RoBERTa-large; the best zero-shot LLMs on WorldCentralBanks stance score
  ~0.60-0.62 (DeepSeek-V3 ~0.601, Llama-3-70B ~0.620).

So a transparent TF-IDF linear model at ~0.58 macro-F1 is a credible baseline that lands in the
zero-shot-LLM range and clearly below fine-tuned transformers. This is exactly how the evaluation and
methodology page frame it: a measured, honest floor, not a state-of-the-art claim. The Gemini path is the
production analogue of the zero-shot-LLM tier; a fine-tuned PLM would be the way to close the gap to ~0.72.

## Open-source landscape (fetched and license-checked)

| Resource | License | What it is | How we use it |
|---|---|---|---|
| [gtfintechlab/fomc-hawkish-dovish](https://github.com/gtfintechlab/fomc-hawkish-dovish) | CC BY-NC 4.0 | "Trillion Dollar Words" (ACL 2023): 2,480 labeled FOMC sentences + FOMC-RoBERTa | Our evaluation benchmark (offline only; not shipped). |
| [gtfintechlab/WorldCentralBanks](https://github.com/gtfintechlab/WorldCentralBanks) | CC BY-NC-SA 4.0 | 380k sentences across 25 banks, 25k annotated for stance/temporal/uncertainty | The 25-bank list and stance scheme are facts we can align our registry/tone spine to; the corpus is NC. |
| [ProsusAI/finBERT](https://github.com/ProsusAI/finBERT) | Apache-2.0 | Financial sentiment transformer (PhraseBank acc 0.86 / F1 0.84) | License-clean transformer baseline option (heavy torch dep); a candidate stronger offline scorer. |
| [hugodevere/FOMCAnalysis-Model](https://github.com/hugodevere/FOMCAnalysis-Model) | MIT | Hand-curated hawk/dove word lists | MIT-clean terms to enrich our lexicon (curate, do not copy noise). |
| [kakeith/op-fed](https://github.com/kakeith/op-fed) | MIT | 1,044 annotated FOMC sentences, opinion -> policy -> stance hierarchy | A second labeled set for cross-dataset evaluation. |
| [yukit-k/centralbank_analysis](https://github.com/yukit-k/centralbank_analysis) | MIT | A Fed speech scraper (requests + BeautifulSoup) | MIT-clean prior art for the worker's Fed scraping. |
| [HanssonMagnus/scrape_bis](https://github.com/HanssonMagnus/scrape_bis) | GPL-3.0 | A BIS speeches scraper | Do NOT vendor (GPL forbidden in runtime, CLAUDE.md section 9); reimplement the method cleanly (we did). |
| [hollance/reliability-diagrams](https://github.com/hollance/reliability-diagrams) | MIT | ECE/MCE calibration + reliability diagrams | Vendor the pure-math `compute_calibration` to add a calibration metric to the eval. |
| [Loughran-McDonald](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) | Non-commercial / paid | The canonical finance sentiment dictionary | Reference only; we author our own lexicon to stay license-clean (ADR 0008). |

## Datasets for ingestion (free, no scraping)

- **BIS bulk download** ([bis.org/cbspeeches/download.htm](https://www.bis.org/cbspeeches/download.htm),
  noncommercial): two direct ZIPs, no auth or key. The full archive `speeches.zip` (~121 MiB, one CSV) and a
  per-year `speeches_2025.zip` (4.2 MB; 741 speeches). This is a far more robust ingestion path than HTML
  scraping and is the recommended bootstrap/backfill corpus for `cbt_worker`.
- **DRomelli/cbspeeches**: a normalized flat table of 35,487 speeches across 131 banks, 1986-2023
  (English-translated). A broad raw corpus; tone/summary are exactly what our service boundary adds.

## Prioritized next steps (impact to effort)

1. **Switch `cbt_worker` to the BIS bulk ZIP** as the backfill source (no fragile HTML scraping; one CSV,
   noncommercial-clean for research). Keep the RSS path for incremental updates. (Highest impact.)
2. ~~**Add a calibration metric** (ECE/MCE) to `scripts/eval_tone.py`.~~ **Done.** The eval reports
   ECE 0.142, MCE 0.276, a multiclass Brier score, and a reliability diagram; the classifier is
   under-confident on this benchmark (`docs/research/tone-evaluation.md`).
3. **Cross-dataset evaluation**: score the classifier on op-fed (MIT) as an out-of-distribution check, so the
   number is not benchmark-specific.
4. **A fine-tuned PLM tier** (e.g. finBERT, Apache-2.0, behind the existing `LlmClient` boundary) to close the
   gap from ~0.58 toward the ~0.72 transformer ceiling, if a heavier runtime dependency is acceptable.
5. **Enrich the lexicon** from the MIT FOMCAnalysis word lists (curated, not copied wholesale).

All of these preserve the architecture (the scorer/boundary seams already exist) and the licensing rules: the
CC-BY-NC benchmarks stay offline-evaluation-only; production stays on the self-authored lexicon plus the LLM.

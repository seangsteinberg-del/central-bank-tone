# Reusable components and prior art (research notes)

A survey of existing open-source projects, datasets, and methods for central bank speech
ingestion and monetary-policy tone analysis, with the build decisions taken for this project.
Cited from a multi-source, fact-checked research pass (2026-06-03). Licensing is the dominant
constraint: several high-value assets are non-commercial, so this project reuses *methods* and
permissively licensed code, and generates its own tone signal with Gemini.

## Data ingestion

| Asset | What | License | Decision |
| --- | --- | --- | --- |
| BIS central bankers' speeches ([bis.org/cbspeeches](https://www.bis.org/cbspeeches)) | One index aggregating speeches from 130+ central banks, including all 8 we target | Public website (respect ToS/robots) | **Scrape the BIS index directly** as the single cross-institution source. |
| [HanssonMagnus/scrape_bis](https://github.com/HanssonMagnus/scrape_bis) | Maintained Python BIS scraper, date/institution filters, PDF-to-text | GPLv3 (copyleft) | Reference design only; we write our own parser (do not vendor copyleft into this repo). |
| [DRomelli/cbspeeches](https://github.com/DRomelli/cbspeeches) | Ready-made corpus, 35,487 speeches 1986-2023 | Academic / non-commercial | Avoid for redistribution; useful as an offline backfill for personal research only. |
| [FedTools](https://github.com/David-Woroniuk/FedTools) (MIT), fomc_speech_scraper | Fed statements/minutes/Beige Books; FOMC member speeches | MIT / unclear | Optional Fed-specific gap-fillers behind the same `SpeechSource` interface. |

Design: one robust `BisSpeechSource` behind a pluggable `SpeechSource` protocol gives all 8
institutions; per-institution sources can be added later without touching the core.

## Tone scoring (hawkish / dovish)

| Asset | What | License | Decision |
| --- | --- | --- | --- |
| [gtfintechlab/FOMC-RoBERTa](https://huggingface.co/gtfintechlab/FOMC-RoBERTa) (ACL 2023 "Trillion Dollar Words") | Best-in-class 3-class dovish/hawkish/neutral classifier + annotated FOMC benchmark | CC BY-NC 4.0 | Non-commercial; use only as an evaluation reference, not in the product. |
| [CentralBankRoBERTa](https://github.com/Moritz-Pfeifer/CentralBankRoBERTa) | Multi-institution (Fed/ECB/BIS) sentiment + agent classifier | MIT | Commercially usable; a candidate future cross-check, but scores agent sentiment, not hawkish/dovish directly. |
| Apel & Blix Grimaldi (2012, Riksbank WP 261); ECB WP 2085; Loughran-McDonald | Lexicon / semantic-orientation net-hawkishness methods; finance sentiment dictionary | Methods are public; L-M dict is commercial-by-permission | Reuse the **method** (context-specific hawk/dove word lists, net-hawkishness) with our own word lists; do not copy the licensed dictionaries. |

Decision: **Gemini LLM-as-judge is the primary tone signal** (license-clean, multi-institution,
multilingual), complemented by a **self-authored deterministic lexicon baseline** for a
transparent, no-network cross-check (ADR 0008). The "Voice of Monetary Policy" (AER 2023) result
that tone independently moves markets motivates measuring it carefully.

## RAG / Q&A

pgvector + SQLAlchemy + Gemini embeddings is a validated stack for document RAG. Plan: chunk
each speech, embed chunks with `gemini-embedding-001`, store vectors in Postgres `pgvector`,
retrieve top-k per question, and have Gemini answer grounded in the retrieved chunks with
citations (speech id + url). Answers abstain when retrieval is empty rather than hallucinate
(CLAUDE.md section 3).

## Open questions (from the research, deferred)

- Reference RAG architectures specific to central-bank corpora and retrieval evaluation were not
  strongly corroborated; we follow the general pgvector + embeddings pattern and add our own
  retrieval tests.
- Cross-institution / non-English tone calibration (BoJ, PBoC originals) is unverified; Gemini's
  multilingual capability is the bet, with the lexicon baseline as an English cross-check.
- A clean commercially-licensable stack is: scrape BIS/institutions directly + Gemini tone +
  self-authored lexicon + CentralBankRoBERTa (MIT) if a model cross-check is later wanted.

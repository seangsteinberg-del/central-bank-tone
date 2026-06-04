# ADR 0019: Bring in the full speech text from the linked BIS PDF

Date: 2026-06-04

Amends: ADR 0010 (BIS speech source)

Status: Accepted

## Context

The BIS speech source (ADR 0010) reads each speech body from the `document.content` HTML in the
detail page's `data-react-props` blob. Validating the extractor against the live feed showed that
for many speeches `document.content` is only a short intro (often 150 to 250 words), and the full
speech is the linked PDF named in `document.path`; other speeches carry their full text in the HTML.
Scoring tone on a 150-word intro when the real speech is several thousand words is a weak,
potentially misleading signal, and the goal is to always bring in the full text.

Testing PDF extraction libraries against real BIS PDFs gave a clear, mixed result:

- Many BIS PDFs are ordinary text PDFs and extract cleanly (for example a Fed governor's speech
  came back as 4,600 well-formed words).
- Some BIS PDFs (notably graphics-heavy ECB talks) embed subset fonts with no ToUnicode map. Every
  pure-Python extractor returns glyph-index soup for these (`(cid:0)(cid:2)...` from pdfminer,
  `/0/1/2...` from pypdf). The character-to-text mapping is simply not in the file, so no extractor
  short of OCR can recover it.

Licensing also constrains the choice: `pymupdf`, the strongest extractor, is AGPL and so is
disallowed in this codebase's runtime (CLAUDE.md section 9, no GPL-family in runtime).

## Decision

Fetch and extract the linked PDF, prefer it when it is fuller than the HTML, and never ingest
unreadable extraction output.

- **Library: `pdfminer.six` (MIT).** It extracts the good PDFs cleanly and is license-clean. Added
  to `cbt_worker` only (it is a scraping concern). `pypdf` extracted the same broken PDFs no better
  and the clean ones no better; `pymupdf` is AGPL.
- **Injected, like the text fetcher.** `BisSpeechSource` gains a `pdf_fetcher` (`Callable[[str],
  bytes]`) and a `pdf_extractor` (`Callable[[bytes], str]`, defaulting to the pdfminer wrapper), so
  the PDF path is fully tested with no network and no real PDFs, and the source stays decoupled from
  any specific HTTP client. The worker and `run_live.py` inject an httpx bytes fetcher.
- **Prefer the fuller legible text.** The PDF (from `document.path`) is fetched and extracted; the
  speech body is the PDF text when it has more words than the HTML intro, otherwise the HTML.
- **OCR an un-extractable PDF with Gemini vision, so the full text is always recovered.** A
  `_looks_like_text` guard rejects extraction output that contains `(cid:` markers or is not mostly
  letters and spaces (a CID-encoded PDF that no text extractor, pdfminer or pypdfium2 included, can
  decode). When that happens, `make_pdf_extractor` renders each page to an image with `pypdfium2`
  and has Gemini transcribe it (`GeminiClient.transcribe_image`), reading the pixels rather than the
  broken font tables. Because the OCR goes through Gemini, it stays inside the model boundary
  (ADR 0006) and needs no OCR engine or system binary. If OCR also yields nothing legible, the body
  degrades to the HTML intro and is logged, never dropping the speech or ingesting glyph soup
  (CLAUDE.md section 3).
- **A minimum-words floor.** A body below 50 words (an empty page, or a stub even OCR cannot read)
  is skipped with a logged reason rather than ingested as a non-scoreable "speech".

## Consequences

- A speech's full text is recovered in every reachable case: a clean PDF via pdfminer, a
  CID-encoded PDF via Gemini-vision OCR, and a born-HTML speech from its HTML. Glyph soup is never
  ingested.
- OCR costs Gemini calls (one per page) and runs only when text extraction fails, so it is bounded
  to the few graphics-heavy PDFs that need it. `cbt_worker` gains MIT/BSD runtime dependencies:
  `pdfminer.six` (text) and `pypdfium2` plus `Pillow` (page rendering).

## Alternatives rejected

- **`pymupdf` / `fitz`.** Best extraction quality, but AGPL; disallowed in runtime by CLAUDE.md
  section 9.
- **`pypdf`.** Permissive and lightweight, but extracted the broken PDFs no better (its own glyph
  soup) and offered no advantage on the clean ones; `pdfminer.six` gives better text on the PDFs
  that do extract, and `pypdfium2` confirmed the CID PDFs are unrecoverable by any text extractor.
- **OCR with a local engine (Tesseract, or an ONNX OCR like RapidOCR).** Recovers the CID PDFs too,
  but Tesseract needs a system binary and the ONNX route adds a heavy runtime; Gemini vision reuses
  the model dependency already in the system and reads document images well.
- **Scrape the originating central bank's own site for the full text.** Each institution's site has
  a different structure; that is a per-source scraping project, not a change to the BIS source.

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
- **Reject glyph soup (CLAUDE.md section 3).** A `_looks_like_text` guard rejects extraction output
  that contains `(cid:` markers or is not mostly letters and spaces; a rejected PDF falls back to
  the HTML intro, so a speech is never ingested as unreadable text. A fetch or parse failure
  likewise degrades to the HTML body and is logged, never dropping the speech.
- **A minimum-words floor.** A body below 50 words (an empty page, or a stub whose only PDF is
  unreadable) is skipped with a logged reason rather than ingested as a non-scoreable "speech".

## Consequences

- Speeches whose full text is a clean PDF are now ingested in full, which is the right input for
  tone scoring; speeches with full HTML are unaffected.
- Speeches whose PDF is CID-encoded glyph soup fall back to the HTML intro. That is the honest best
  available text; the alternative (ingesting `(cid:...)` runs, or rendering and OCR-ing the PDF) is
  either garbage or far out of scope. The fallback is logged (`bis_pdf_not_legible`) so the limit is
  visible, not silent.
- `cbt_worker` gains a runtime dependency (`pdfminer.six`, MIT). Extraction adds a PDF fetch per
  speech and a second of CPU on large PDFs, which is acceptable for an ingestion/backfill tool.

## Alternatives rejected

- **`pymupdf` / `fitz`.** Best extraction quality, but AGPL; disallowed in runtime by CLAUDE.md
  section 9.
- **`pypdf`.** Permissive and lightweight, but extracted the broken PDFs no better (its own glyph
  soup) and offered no advantage on the clean ones; `pdfminer.six` gives better text on the PDFs
  that do extract.
- **OCR (render the PDF to images, run Tesseract).** Could in principle recover the CID-encoded
  PDFs, but it needs a system binary (Tesseract) and a rasterizer, is slow, and adds error of its
  own. Out of scope for the gain.
- **Scrape the originating central bank's own site for the full text.** Each institution's site has
  a different structure; that is a per-source scraping project, not a change to the BIS source.

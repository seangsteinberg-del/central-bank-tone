# cbt_worker: local map

The ingestion worker. The root `CLAUDE.md` holds the binding rules; this is the local map.
`cbt_worker` depends on `cbt_core` only (machine-checked by `scripts/check_imports.py`).

## What lives here (src/cbt_worker)

- `sources/` one module per scraping strategy behind the `SpeechSource` protocol
  (`sources/base.py`). `sources/bis.py` scrapes the BIS central bankers' speeches index, which
  aggregates every tracked institution, so it is the single primary source. The HTTP fetcher is
  injected, so sources are tested against HTML fixtures with no network.
- `runner.py` orchestration: for each scraped speech it resolves the speaker
  (`SpeakerService.ensure_speaker`), ingests it (`IngestionService`, idempotent by source hash),
  and indexes it (`IndexingService`). A single speech that fails the model is logged and skipped.
- `app.py` the composition root and ASGI-free entry point (`python -m cbt_worker.app`). It is
  thin glue (network and Gemini wiring) and is `# pragma: no cover`; the logic is in `runner` and
  the sources.

## Local conventions

- Sources never touch `cbt_core` internals; the runner calls services only.
- The BIS selectors target a documented HTML contract (see the test fixtures). Verify and adjust
  them against the live site (ADR 0010); changes are confined to `sources/bis.py`.
- Speeches from institutions outside the schema spine are skipped, not guessed.

## Commands

```
.venv/Scripts/pytest.exe -m unit -q
.venv/Scripts/python.exe -m cbt_worker.app   # one ingestion pass (needs DB + Gemini key)
```

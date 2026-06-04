"""Serve the web UI in keyless demo mode: no Gemini key, no Docker (ADR 0014, ADR 0017).

Builds a real application backed by an in-memory SQLite database and the offline LLM client
(ADR 0014): tone is scored by the supervised classifier (ADR 0013); retrieval and question
answering run on the offline client over an in-memory index.

The demo ships with no seed corpus. It starts empty and is populated only by real ingestion:
add a genuine speech (its real title, date, URL, and speaker) on the in-app "Add data" page,
which scores and indexes it keylessly, or run the worker (``cbt_worker``) against a configured
database. The platform never fabricates speeches to look populated (ADR 0017).

Usage::

    uv run python scripts/run_demo.py
"""

from __future__ import annotations

from cbt_web.demo import build_demo_app


def main() -> int:
    """Build the empty keyless demo app and serve it."""
    import uvicorn

    print("Central Bank Tone - keyless demo (no Gemini key, no Docker)")
    print("  corpus: empty - add a real speech on the Add data page, or run the worker to ingest")
    print("  tone: supervised classifier (ADR 0013); search: offline extractive (ADR 0014)")
    print("  serving on http://127.0.0.1:8000  (Ctrl+C to stop)")
    app = build_demo_app([])
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

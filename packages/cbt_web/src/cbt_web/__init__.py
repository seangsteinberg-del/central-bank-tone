"""Central Bank Tone web UI.

A server-rendered (Jinja + HTMX) FastAPI adapter over ``cbt_core``: it lists speakers, shows a
speaker's tone over time and analyzed speeches, answers natural-language questions about a
speaker or across the whole corpus, and triggers ingestion. It depends on ``cbt_core`` only
(machine-checked) and reaches the domain through the service layer, never a repository.

Run with: ``uv run uvicorn --factory cbt_web.app:create_app``.
"""

from __future__ import annotations

from cbt_web.app import create_app

__all__ = ["create_app"]

"""Opt-in live Gemini smoke test (CLAUDE.md section 5).

Marked ``llm`` so it is excluded from CI (`pytest -m "not llm"`), and skipped unless a real
``CBT_GEMINI_API_KEY`` is present. Run it manually to confirm the real provider path:
``uv run pytest -m llm``.
"""

from __future__ import annotations

import os

import pytest

from cbt_core.llm.gemini import build_gemini_client
from cbt_core.settings import Settings

_SPEECH = (
    "The Committee judges that inflation remains too high and that further policy firming may "
    "be appropriate to return inflation to target over time."
)


@pytest.mark.llm
@pytest.mark.skipif(
    not os.environ.get("CBT_GEMINI_API_KEY"), reason="CBT_GEMINI_API_KEY is not set"
)
def test_gemini_analyze_tone_live() -> None:
    client = build_gemini_client(Settings())
    result = client.analyze_tone(_SPEECH)
    assert -1.0 <= result.score <= 1.0
    assert result.summary
    assert result.rationale

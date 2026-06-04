"""Serve the web UI in keyless demo mode: no Gemini key, no Docker (ADR 0014).

Builds a real, populated demo and serves it on http://127.0.0.1:8000. Tone is scored by the
supervised classifier (ADR 0013); retrieval and question answering run on the offline client
(ADR 0014) over an in-memory index; everything is backed by an in-memory SQLite database.

Seed corpus: if the FOMC benchmark cache is present (run ``scripts/eval_tone.py`` once, or
``make eval``), this groups the real FOMC communications by year and attributes each year to the
sitting Fed Chair, producing genuine per-Chair tone trajectories from real central-bank text.
Otherwise it falls back to a small set of clearly-labelled illustrative passages, so the demo
still runs with no prerequisites.

Usage::

    uv run python scripts/run_demo.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from cbt_core import CentralBank
from cbt_web.demo import SeedSpeech, build_demo_app

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _REPO_ROOT / "data" / "benchmarks"
_FED_URL = "https://www.federalreserve.gov/newsevents/pressreleases.htm"

# Fed Chair by calendar year over the benchmark's span (the chair who held most of the year).
_CHAIRS: tuple[tuple[int, int, str], ...] = (
    (1987, 2005, "Alan Greenspan"),
    (2006, 2013, "Ben Bernanke"),
    (2014, 2017, "Janet Yellen"),
    (2018, 2026, "Jerome Powell"),
)


def _chair_for(year: int) -> str | None:
    """The Fed Chair attributed to a calendar year, or None if outside the known span."""
    for start, end, name in _CHAIRS:
        if start <= year <= end:
            return name
    return None


def _seed_from_fomc_cache() -> list[SeedSpeech]:
    """Build real per-Chair seed speeches from the cached FOMC corpus, grouped by year."""
    by_year: dict[int, list[str]] = {}
    for split in ("train", "test"):
        path = _CACHE_DIR / f"fomc_{split}.json"
        if not path.exists():
            return []
        for row in json.loads(path.read_text(encoding="utf-8")):
            year = int(str(row["year"]))
            by_year.setdefault(year, []).append(str(row["sentence"]))
    speeches: list[SeedSpeech] = []
    for year in sorted(by_year):
        chair = _chair_for(year)
        if chair is None or len(by_year[year]) < 8:
            continue
        speeches.append(
            SeedSpeech(
                speaker_name=chair,
                central_bank=CentralBank.FEDERAL_RESERVE,
                role="Chair",
                title=f"FOMC communications, {year}",
                url=_FED_URL,
                delivered_at=datetime(year, 7, 1, tzinfo=UTC),
                text=" ".join(by_year[year]),
            )
        )
    return speeches


# A tiny, clearly-illustrative fallback so the demo runs with no prerequisites. These passages are
# composed for the demo (not verbatim transcripts); the tone scores applied to them are real.
_ILLUSTRATIVE: tuple[SeedSpeech, ...] = (
    SeedSpeech(
        speaker_name="Jerome Powell (illustrative)",
        central_bank=CentralBank.FEDERAL_RESERVE,
        role="Chair",
        title="Illustrative: restoring price stability (2022)",
        url=_FED_URL,
        delivered_at=datetime(2022, 7, 1, tzinfo=UTC),
        text=(
            "Inflation is much too high and the committee is strongly committed to bringing it "
            "back to two percent. We have raised the policy rate substantially and anticipate "
            "that ongoing increases will be appropriate. We are prepared to tighten policy further "
            "and to keep at it until the job is done, even if that brings some pain to households."
        ),
    ),
    SeedSpeech(
        speaker_name="Jerome Powell (illustrative)",
        central_bank=CentralBank.FEDERAL_RESERVE,
        role="Chair",
        title="Illustrative: supporting the recovery (2020)",
        url=_FED_URL,
        delivered_at=datetime(2020, 7, 1, tzinfo=UTC),
        text=(
            "The pandemic has caused tremendous economic hardship and downside risks remain "
            "elevated. We have cut the policy rate to near zero and will use our tools to provide "
            "accommodation for as long as needed. Policy will remain highly supportive until the "
            "recovery is firmly on track and the labour market has substantially healed."
        ),
    ),
    SeedSpeech(
        speaker_name="Christine Lagarde (illustrative)",
        central_bank=CentralBank.ECB,
        role="President",
        title="Illustrative: the inflation outlook (2023)",
        url="https://www.ecb.europa.eu/press/html/index.en.html",
        delivered_at=datetime(2023, 3, 1, tzinfo=UTC),
        text=(
            "Underlying price pressures remain strong and inflation is projected to stay above our "
            "target for too long. We are determined to raise interest rates to a sufficiently "
            "restrictive level and to hold them there until inflation returns durably to two "
            "percent. Future decisions will remain data dependent."
        ),
    ),
)


def _load_seed() -> tuple[list[SeedSpeech], str]:
    """Return the seed corpus and a one-line description of its source."""
    real = _seed_from_fomc_cache()
    if real:
        chairs = sorted({s.speaker_name for s in real})
        return real, f"real FOMC corpus: {len(real)} yearly speeches across {len(chairs)} Chairs"
    return list(_ILLUSTRATIVE), "illustrative fallback (run `make eval` for the real FOMC corpus)"


def main() -> int:
    """Build the seeded demo app and serve it."""
    import uvicorn

    speeches, source = _load_seed()
    print("Central Bank Tone - keyless demo (no Gemini key, no Docker)")
    print(f"  seed: {source}")
    print("  tone: supervised classifier (ADR 0013); search: offline extractive (ADR 0014)")
    print("  serving on http://127.0.0.1:8000  (Ctrl+C to stop)")
    app = build_demo_app(speeches)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

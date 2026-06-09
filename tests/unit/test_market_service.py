"""Tests for the Signal vs Market service (ADR 0022).

Hermetic: the FRED rate series are written as small fixture CSVs into a tmp directory (the real
cached data lives under the gitignored ``data/`` and is never required by a test), and the Fed
corpus is seeded through the services with a scripted LLM so the indices are reproducible with no
network and no live model.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.orm import Session, sessionmaker
from tests._stubs import ScriptedLlmClient

from cbt_core import (
    CentralBank,
    IngestionService,
    MarketSignalService,
    SpeakerService,
    ToneLabel,
)
from cbt_core.analysis.leadlag import pearson
from cbt_core.exceptions import BenchmarkUnavailableError, InsufficientDataError
from cbt_core.services._support import IdFactory
from cbt_core.services.market_service import (
    _aligned,
    _change_over,
    load_fred_monthly,
)

_FED = CentralBank.FEDERAL_RESERVE


def _write_rate_csvs(directory: Path) -> None:
    """Write small FEDFUNDS and GS2 fixture CSVs covering 2024-01 .. 2025-06."""
    months = [(2024 + (i // 12), (i % 12) + 1) for i in range(18)]
    for code, base in (("FEDFUNDS", 5.0), ("GS2", 4.5)):
        lines = [f"observation_date,{code}"]
        for n, (year, month) in enumerate(months):
            lines.append(f"{year:04d}-{month:02d}-01,{base - n * 0.05:.2f}")
        (directory / f"fred_{code}_monthly.csv").write_text("\n".join(lines), encoding="utf-8")


def _seed_fed(
    session_factory: sessionmaker[Session],
    id_factory: IdFactory,
    *,
    months: int,
    per_month: int,
) -> tuple[float, int]:
    """Seed ``months`` Fed months (plus one excluded ECB speech). Return (jan-2024 mean, fed count)."""
    by_text: dict[str, tuple[float, ToneLabel]] = {}
    plan: list[tuple[int, int, str]] = []
    for mi in range(months):
        score = max(-1.0, min(1.0, round(-0.5 + mi * 0.07, 2)))
        tone = ToneLabel.HAWKISH if score >= 0 else ToneLabel.DOVISH
        for k in range(per_month):
            text = f"fed-{mi}-{k}"
            by_text[text] = (score, tone)
            plan.append((2024 + (mi // 12), (mi % 12) + 1, text))
    by_text["ecb-speech"] = (0.1, ToneLabel.HAWKISH)

    speakers = SpeakerService(session_factory, id_factory=id_factory)
    ingestion = IngestionService(session_factory, ScriptedLlmClient(by_text), id_factory=id_factory)
    fed = speakers.ensure_speaker(name="Jerome Powell", central_bank=_FED, role="Chair")
    ecb = speakers.ensure_speaker(
        name="Christine Lagarde", central_bank=CentralBank.ECB, role="President"
    )
    for year, month, text in plan:
        ingestion.ingest_speech(
            speaker_id=fed.id,
            title=text,
            url=f"https://example.org/{text}",
            delivered_at=datetime(year, month, 1, tzinfo=UTC),
            text=text,
        )
    ingestion.ingest_speech(
        speaker_id=ecb.id,
        title="ecb",
        url="https://example.org/ecb",
        delivered_at=datetime(2024, 1, 1, tzinfo=UTC),
        text="ecb-speech",
    )
    jan_2024_mean = max(-1.0, min(1.0, round(-0.5, 2)))
    return jan_2024_mean, len(plan)


@pytest.mark.unit
def test_load_fred_monthly_parses_and_skips_missing_values(tmp_path: Path) -> None:
    path = tmp_path / "fred_GS2_monthly.csv"
    path.write_text(
        "observation_date,GS2\n2024-01-01,4.00\n2024-02-01,.\n2024-03-01,\n2024-04-01,4.20\n",
        encoding="utf-8",
    )
    series = load_fred_monthly(path)
    assert series == {2024 * 12 + 0: 4.00, 2024 * 12 + 3: 4.20}


@pytest.mark.unit
def test_load_fred_monthly_raises_when_cache_missing(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkUnavailableError):
        load_fred_monthly(tmp_path / "fred_GS2_monthly.csv")


@pytest.mark.unit
def test_signal_vs_market_builds_indices_and_excludes_other_banks(
    session_factory: sessionmaker[Session], id_factory: IdFactory, tmp_path: Path
) -> None:
    jan_mean, fed_count = _seed_fed(session_factory, id_factory, months=14, per_month=3)
    _write_rate_csvs(tmp_path)
    result = MarketSignalService(session_factory, benchmark_dir=tmp_path).signal_vs_market()

    assert result.central_bank is _FED
    assert result.months == 14
    assert result.speeches == fed_count  # the ECB speech is excluded from the Fed count
    assert result.headline_index.points[2024 * 12 + 0] == pytest.approx(jan_mean)
    # Four index-vs-rate blocks (two indices x two rate series), each with three horizons.
    assert len(result.correlations) == 4
    assert all(len(ivr.correlations) == 3 for ivr in result.correlations)
    assert {s.code for s in result.rate_series} == {"FEDFUNDS", "GS2"}


@pytest.mark.unit
def test_signal_vs_market_raises_insufficient_data_below_twelve_months(
    session_factory: sessionmaker[Session], id_factory: IdFactory, tmp_path: Path
) -> None:
    _seed_fed(session_factory, id_factory, months=6, per_month=3)
    _write_rate_csvs(tmp_path)
    with pytest.raises(InsufficientDataError):
        MarketSignalService(session_factory, benchmark_dir=tmp_path).signal_vs_market()


@pytest.mark.unit
def test_change_over_returns_none_when_no_earlier_reading() -> None:
    assert _change_over({100: 1.0}, 3) is None
    assert _change_over({}, 3) is None


@pytest.mark.unit
def test_change_over_computes_the_delta() -> None:
    assert _change_over({100: 1.0, 103: 1.5}, 3) == pytest.approx(0.5)


@pytest.mark.unit
def test_aligned_is_none_when_a_side_is_missing() -> None:
    # No fabricated alignment when one side has no reading (CLAUDE.md section 3, no silent fallback).
    assert _aligned(None, 0.05, 10.0, 5.0) is None
    assert _aligned(0.2, 0.05, None, 5.0) is None


@pytest.mark.unit
def test_aligned_true_for_same_direction_false_for_opposite() -> None:
    assert _aligned(0.2, 0.05, 12.0, 5.0) is True  # tone up, yield up
    assert _aligned(0.2, 0.05, -12.0, 5.0) is False  # tone up, yield down
    assert _aligned(0.01, 0.05, 1.0, 5.0) is True  # both within their flat bands


@pytest.mark.unit
def test_pearson_is_one_for_a_perfect_positive_relationship() -> None:
    xs = np.array([1.0, 2.0, 3.0, 4.0])
    ys = np.array([2.0, 4.0, 6.0, 8.0])
    assert pearson(xs, ys) == pytest.approx(1.0)

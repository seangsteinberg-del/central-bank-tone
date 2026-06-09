"""Signal vs Market service (ADR 0022).

Builds the macro desk's "so what" view: it relates the platform's own monthly Federal Reserve tone
signals (the holistic headline and the forward-looking rate-path, both from stored scores) to two
cached FRED rate series (the effective fed funds rate and the 2-year Treasury yield), reports the
lead correlations with a bootstrap confidence interval, and states the current divergence between
the tone trend and market repricing.

It is a read-only service (it records nothing) and owns its read transaction, mirroring
:class:`~cbt_core.services.committee_service.CommitteeService`. The rate data is a cached snapshot
read from disk; the service never fetches from the network inside a request (refreshing the cache is
the job of ``scripts/eval_corpus_vs_rates.py``). The lead-lag statistics live in
:mod:`cbt_core.analysis.leadlag` and are imported by BOTH this service and that script, so the
served numbers and the published report are one implementation and cannot drift (ADR 0023; the
divergence ADR 0022 fixed).

Federal Reserve only: that is where free market ground truth exists. The 2-year yield is the
market-path proxy (it reprices freely), in contrast to the highly persistent fed funds rate.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from cbt_core.analysis.leadlag import block_bootstrap_ci, lead_pairs
from cbt_core.domain.market import (
    Divergence,
    IndexVsRate,
    LeadCorrelation,
    MonthlySeries,
    SignalVsMarket,
)
from cbt_core.domain.registry import CentralBank
from cbt_core.exceptions import BenchmarkUnavailableError, InsufficientDataError
from cbt_core.logging import get_logger, resolve_correlation_id
from cbt_core.persistence.repositories import (
    SpeakerRepository,
    SpeechRepository,
    SpeechStanceRepository,
    ToneObservationRepository,
)

_logger = get_logger(__name__)

# Lead horizons in months: 0 is contemporaneous co-movement, the rest test whether tone leads.
_HORIZONS = (0, 3, 6)
_MIN_MONTHS = 12  # below this the correlation is not meaningful; abstain rather than mislead.
# The two FRED series, with the human label and whether each is the freely-repricing market proxy.
_RATE_SERIES: tuple[tuple[str, str, bool], ...] = (
    ("FEDFUNDS", "effective fed funds", False),
    ("GS2", "2-year Treasury", True),
)
# The two tone indices (headline, rate-path) times each rate series times each horizon: every cell
# the page tests at once. The bootstrap CIs are Bonferroni-corrected to this family size (ADR 0023).
_FAMILY_SIZE = 2 * len(_RATE_SERIES) * len(_HORIZONS)
_TONE_FLAT_BAND = 0.05  # a 3-month tone change within this is "little changed"
_RATE_FLAT_BP = 5.0  # a 3-month 2-year change within this many basis points is "little changed"


def load_fred_monthly(path: Path) -> dict[int, float]:
    """Load a cached FRED monthly series as a month-index map (cache-only, no network).

    The file is the raw FRED CSV (``observation_date,VALUE``); rows with a missing value (``""`` or
    ``"."``) are skipped. The key is ``year * 12 + (month - 1)``.

    Args:
        path: Path to the cached CSV (for example ``data/benchmarks/fred_GS2_monthly.csv``).

    Returns:
        A map from month index to the series value.

    Raises:
        BenchmarkUnavailableError: If the cached file does not exist. The service never fetches from
            the network inside a request, so a missing cache is an explicit, honest failure.
    """
    if not path.exists():
        raise BenchmarkUnavailableError(
            f"cached rate series not found at {path}; refresh it with "
            "scripts/eval_corpus_vs_rates.py"
        )
    series: dict[int, float] = {}
    reader = csv.reader(io.StringIO(path.read_text(encoding="utf-8")))
    next(reader, None)  # header
    for row in reader:
        if len(row) < 2 or row[1] in ("", "."):
            continue
        year, month = int(row[0][:4]), int(row[0][5:7])
        series[year * 12 + (month - 1)] = float(row[1])
    return series


def _monthly_mean(
    rows: list[tuple[datetime, float]], *, min_per_month: int
) -> tuple[dict[int, float], dict[int, int]]:
    """Aggregate (timestamp, value) rows into a monthly mean, keeping months with enough data."""
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for when, value in rows:
        index = when.year * 12 + (when.month - 1)
        sums[index] = sums.get(index, 0.0) + float(value)
        counts[index] = counts.get(index, 0) + 1
    series = {i: sums[i] / counts[i] for i in counts if counts[i] >= min_per_month}
    return series, {i: counts[i] for i in series}


def _month_label(index: int) -> str:
    """Render a month index as ``YYYY-MM``."""
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def _clamp(value: float) -> float:
    """Clamp a correlation to ``[-1, 1]`` so a degenerate bootstrap bound never fails validation."""
    return max(-1.0, min(1.0, value))


class MarketSignalService:
    """Relate the platform's Federal Reserve tone signals to market rates (read-only, ADR 0022)."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        benchmark_dir: Path,
        min_speeches_per_month: int = 3,
        bootstrap_samples: int = 10000,
        seed: int = 0,
    ) -> None:
        """Build the service.

        Args:
            session_factory: Factory for read sessions; the service owns the transaction.
            benchmark_dir: Directory holding the cached FRED CSVs (from settings, not the environment).
            min_speeches_per_month: Minimum speeches a month needs to enter an index.
            bootstrap_samples: Resamples for the confidence interval. The default is large because the
                family-wise (Bonferroni) tail is deep, so it needs many resamples to be stable.
            seed: RNG seed, so the intervals are reproducible.
        """
        self._session_factory = session_factory
        self._benchmark_dir = benchmark_dir
        self._min_per_month = min_speeches_per_month
        self._bootstrap = bootstrap_samples
        self._seed = seed

    def signal_vs_market(
        self,
        central_bank: CentralBank = CentralBank.FEDERAL_RESERVE,
        *,
        actor: str = "system",
        correlation_id: UUID | None = None,
    ) -> SignalVsMarket:
        """Build the Signal vs Market view from stored tone signals and the cached rate series.

        Args:
            central_bank: The institution to analyze. Only the Federal Reserve is rate-validated.
            actor: Who is performing the action (for the log context).
            correlation_id: Correlation id for this call; resolved from context or minted if absent.

        Returns:
            The :class:`~cbt_core.domain.market.SignalVsMarket` read model.

        Raises:
            BenchmarkUnavailableError: If a required cached FRED CSV is missing.
            InsufficientDataError: If fewer than twelve qualifying months of tone exist.
        """
        correlation = resolve_correlation_id(correlation_id)
        log = _logger.bind(
            correlation_id=str(correlation), actor=actor, central_bank=central_bank.value
        )

        headline_rows, rate_path_rows, speeches = self._collect(central_bank)
        headline, _ = _monthly_mean(headline_rows, min_per_month=self._min_per_month)
        rate_path, _ = _monthly_mean(rate_path_rows, min_per_month=self._min_per_month)
        if len(headline) < _MIN_MONTHS:
            raise InsufficientDataError(
                f"only {len(headline)} qualifying months for {central_bank.value}; "
                f"need at least {_MIN_MONTHS} to relate tone to rates"
            )

        rates = {
            code: load_fred_monthly(self._benchmark_dir / f"fred_{code}_monthly.csv")
            for code, _, _ in _RATE_SERIES
        }
        correlations = tuple(
            index_vs_rate
            for name, index in (("headline", headline), ("rate-path", rate_path))
            for index_vs_rate in self._index_vs_rate(name, index, rates)
        )
        two_year = rates["GS2"]
        divergence = self._divergence(headline, two_year)

        span_months = sorted(set(headline) | set(rate_path) | self._rate_span(rates, headline))
        log.info(
            "signal_vs_market_built",
            months=len(headline),
            speeches=speeches,
            aligned=divergence.aligned,
        )
        return SignalVsMarket(
            central_bank=central_bank,
            headline_index=MonthlySeries(
                label="Headline tone",
                code="headline",
                is_rate=False,
                is_market_proxy=False,
                points=headline,
            ),
            rate_path_index=MonthlySeries(
                label="Rate-path tone",
                code="rate-path",
                is_rate=False,
                is_market_proxy=False,
                points=rate_path,
            ),
            rate_series=tuple(
                MonthlySeries(
                    label=label,
                    code=code,
                    is_rate=True,
                    is_market_proxy=is_proxy,
                    points=rates[code],
                )
                for code, label, is_proxy in _RATE_SERIES
            ),
            correlations=correlations,
            divergence=divergence,
            span_start=span_months[0],
            span_end=span_months[-1],
            months=len(headline),
            speeches=speeches,
            corpus_through=_month_label(max(headline)),
            market_through=_month_label(max(two_year)) if two_year else "n/a",
            min_speeches_per_month=self._min_per_month,
        )

    def _collect(
        self, central_bank: CentralBank
    ) -> tuple[list[tuple[datetime, float]], list[tuple[datetime, float]], int]:
        """Gather the bank's (date, headline) and (date, rate-path) rows and the speech count."""
        headline_rows: list[tuple[datetime, float]] = []
        rate_path_rows: list[tuple[datetime, float]] = []
        with self._session_factory() as session:
            speakers = [
                s for s in SpeakerRepository(session).list_all() if s.central_bank is central_bank
            ]
            observations = ToneObservationRepository(session)
            speeches_repo = SpeechRepository(session)
            stances = SpeechStanceRepository(session).all_by_speech()
            for speaker in speakers:
                headline_rows.extend(
                    (obs.observed_at, obs.score)
                    for obs in observations.list_for_speaker(speaker.id)
                )
                for speech in speeches_repo.list_for_speaker(speaker.id):
                    stance = stances.get(speech.id)
                    if stance is not None:
                        rate_path_rows.append((speech.delivered_at, stance.rate_path))
        return headline_rows, rate_path_rows, len(headline_rows)

    def _index_vs_rate(
        self, index_name: str, index: dict[int, float], rates: dict[str, dict[int, float]]
    ) -> tuple[IndexVsRate, ...]:
        """Correlate one tone index against every rate series across the lead horizons."""
        return tuple(
            IndexVsRate(
                index_name=index_name,
                rate_label=label,
                rate_code=code,
                correlations=tuple(
                    self._correlation(index, rates[code], horizon) for horizon in _HORIZONS
                ),
            )
            for code, label, _ in _RATE_SERIES
        )

    def _correlation(
        self, index: dict[int, float], rate: dict[int, float], horizon: int
    ) -> LeadCorrelation:
        """The lead correlation of a tone index with a rate series at one horizon (ADR 0023).

        The CI is a circular moving-block bootstrap, corrected to a family-wise level across all
        ``_FAMILY_SIZE`` tested cells, so ``excludes_zero`` is a multiple-testing-corrected
        significance, not a naive per-cell interval.
        """
        xs, ys = lead_pairs(index, rate, horizon)
        r, lo, hi = block_bootstrap_ci(
            xs,
            ys,
            samples=self._bootstrap,
            seed=self._seed,
            horizon=horizon,
            family_size=_FAMILY_SIZE,
        )
        return LeadCorrelation(
            horizon_months=horizon,
            r=_clamp(r),
            ci_low=_clamp(lo),
            ci_high=_clamp(hi),
            n=len(xs),
            family_size=_FAMILY_SIZE,
        )

    def _divergence(self, headline: dict[int, float], two_year: dict[int, float]) -> Divergence:
        """Compare the last three months of headline tone with 2-year-yield repricing."""
        tone_change = _change_over(headline, 3)
        market_change_pp = _change_over(two_year, 3)
        market_change_bp = market_change_pp * 100.0 if market_change_pp is not None else None
        tone_dir = _direction(tone_change, _TONE_FLAT_BAND, "more hawkish", "more dovish")
        market_dir = _direction(
            market_change_bp, _RATE_FLAT_BP, "repricing higher", "repricing lower"
        )
        aligned = _aligned(tone_change, _TONE_FLAT_BAND, market_change_bp, _RATE_FLAT_BP)
        return Divergence(
            tone_change_3m=tone_change,
            market_change_bp_3m=market_change_bp,
            tone_direction=tone_dir,
            market_direction=market_dir,
            aligned=aligned,
            headline=_divergence_headline(tone_change, market_change_bp, tone_dir, aligned),
        )

    def _rate_span(
        self, rates: dict[str, dict[int, float]], headline: dict[int, float]
    ) -> set[int]:
        """Rate months that fall within the tone span, so the chart axis covers the overlap only."""
        if not headline:
            return set()
        lo, hi = min(headline), max(headline)
        return {m for series in rates.values() for m in series if lo <= m <= hi}


def _change_over(series: dict[int, float], months: int) -> float | None:
    """The change in a monthly series over the last ``months``, or ``None`` if a side is missing."""
    if not series:
        return None
    latest = max(series)
    earlier = series.get(latest - months)
    if earlier is None:
        return None
    return series[latest] - earlier


def _direction(change: float | None, band: float, up: str, down: str) -> str:
    """A direction phrase for a signed change, with a flat band, or a missing-data phrase."""
    if change is None:
        return "not enough history"
    if change > band:
        return up
    if change < -band:
        return down
    return "little changed"


def _aligned(
    tone_change: float | None, tone_band: float, market_change: float | None, market_band: float
) -> bool | None:
    """Whether the tone shift and market repricing point the same way (None if a side is missing)."""
    if tone_change is None or market_change is None:
        return None
    tone_sign = 0 if abs(tone_change) <= tone_band else (1 if tone_change > 0 else -1)
    market_sign = 0 if abs(market_change) <= market_band else (1 if market_change > 0 else -1)
    return tone_sign == market_sign


def _divergence_headline(
    tone_change: float | None,
    market_change_bp: float | None,
    tone_dir: str,
    aligned: bool | None,
) -> str:
    """A plain-prose one-line read of the tone-vs-market comparison."""
    if aligned is None:
        return (
            "Not enough recent Federal Reserve history to compare the tone trend with market "
            "repricing over the last quarter."
        )
    tone_part = f"Fed communication is {tone_dir} ({tone_change:+.2f} over the quarter)"
    move = "risen" if (market_change_bp or 0.0) > 0 else "fallen"
    market_part = f"the 2-year yield has {move} {abs(market_change_bp or 0.0):.0f}bp"
    if aligned:
        return (
            f"{tone_part} and {market_part}: the platform's read and market pricing are moving "
            "together."
        )
    return (
        f"{tone_part} while {market_part}: the platform reads the committee differently from how "
        "the market is pricing the path."
    )

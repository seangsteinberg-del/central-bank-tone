"""Signal vs Market divergence read models (ADR 0022; CLAUDE.md sections 2 and 3).

These answer the macro reader's "so what" question: does the platform's reading of central-bank
communication agree with what the market has already priced, and where does it diverge. They are
computed, immutable views built by
:class:`~cbt_core.services.market_service.MarketSignalService` from the stored Federal Reserve tone
signals and cached FRED rate series. They hold domain types and plain numbers, never ORM rows.

The validation is the Federal Reserve only, because that is where free market ground truth exists.
The "what the market is pricing" proxy is the 2-year Treasury yield: a freely repricing read of the
expected policy path, in contrast to the highly persistent effective fed funds rate.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from cbt_core.domain.registry import CentralBank


class LeadCorrelation(BaseModel):
    """The correlation of a tone index with a rate series at one lead horizon.

    Attributes:
        horizon_months: 0 for contemporaneous co-movement, a positive value for a lead test (does
            this month's tone precede the rate move over the next ``horizon_months`` months).
        r: Pearson correlation point estimate, in ``[-1.0, 1.0]``.
        ci_low: Lower bound of the 95% bootstrap confidence interval.
        ci_high: Upper bound of the 95% bootstrap confidence interval.
        n: Number of paired observations the correlation is computed over.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon_months: int = Field(ge=0)
    r: float = Field(ge=-1.0, le=1.0)
    ci_low: float = Field(ge=-1.0, le=1.0)
    ci_high: float = Field(ge=-1.0, le=1.0)
    n: int = Field(ge=0)

    @property
    def excludes_zero(self) -> bool:
        """True when the confidence interval excludes zero (a non-inconclusive result)."""
        return self.ci_low > 0.0 or self.ci_high < 0.0


class IndexVsRate(BaseModel):
    """One tone index correlated against one rate series across the lead horizons.

    Attributes:
        index_name: The tone index, ``"headline"`` or ``"rate-path"``.
        rate_label: Human label for the rate series, for example ``"2-year Treasury"``.
        rate_code: The FRED series code, for example ``"GS2"``.
        correlations: One :class:`LeadCorrelation` per horizon, in ascending horizon order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    index_name: str
    rate_label: str
    rate_code: str
    correlations: tuple[LeadCorrelation, ...]


class MonthlySeries(BaseModel):
    """A monthly series keyed by month index (``year * 12 + (month - 1)``).

    Used for both the platform's tone indices and the FRED rate series so the view can plot them on
    one shared month axis.

    Attributes:
        label: Human label, for example ``"Headline tone"`` or ``"2-year Treasury"``.
        code: A short machine code, for example ``"headline"`` or ``"GS2"``.
        is_rate: True for a FRED rate series (a percentage), False for a tone index (``[-1, 1]``).
        is_market_proxy: True for the 2-year yield, the freely repricing market-path proxy.
        points: Month index to value.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    code: str
    is_rate: bool
    is_market_proxy: bool
    points: dict[int, float]

    @property
    def latest_month(self) -> int | None:
        """The most recent month index present, or ``None`` when the series is empty."""
        return max(self.points) if self.points else None


class Divergence(BaseModel):
    """The headline read: how the platform's recent tone shift compares to market repricing.

    Both sides are three-month changes so they are directly comparable. ``aligned`` is ``None`` when
    either side is missing, never a fabricated zero or a forced alignment (CLAUDE.md section 3).

    Attributes:
        tone_change_3m: Change in the headline Fed tone index over the last three months, or
            ``None`` when there is no reading three months back.
        market_change_bp_3m: Change in the 2-year Treasury yield over the last three months, in
            basis points, or ``None`` when there is no reading three months back.
        tone_direction: ``"more hawkish"`` / ``"more dovish"`` / ``"little changed"``.
        market_direction: ``"repricing higher"`` / ``"repricing lower"`` / ``"little changed"``.
        aligned: True when tone and market moved the same way, False when they diverged, ``None``
            when a side is missing.
        headline: A plain-prose one-line read for the macro desk.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tone_change_3m: float | None
    market_change_bp_3m: float | None
    tone_direction: str
    market_direction: str
    aligned: bool | None
    headline: str


class SignalVsMarket(BaseModel):
    """The full Signal vs Market view for one institution (the Federal Reserve).

    Attributes:
        central_bank: The institution; the Federal Reserve, the only rate-validated one.
        headline_index: The monthly holistic-tone index.
        rate_path_index: The monthly forward-looking rate-path index (ADR 0021).
        rate_series: The FRED rate series shown, effective fed funds and the 2-year Treasury.
        correlations: Each index correlated against each rate series across the lead horizons.
        divergence: The headline tone-vs-market read.
        span_start: Earliest month index spanned by the chart.
        span_end: Latest month index spanned by the chart.
        months: Number of qualifying headline months.
        speeches: Number of Federal Reserve speeches counted.
        corpus_through: The latest tone month, as ``YYYY-MM``.
        market_through: The latest rate month, as ``YYYY-MM`` (the cached-snapshot date).
        min_speeches_per_month: The minimum speeches a month needs to enter an index.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    central_bank: CentralBank
    headline_index: MonthlySeries
    rate_path_index: MonthlySeries
    rate_series: tuple[MonthlySeries, ...]
    correlations: tuple[IndexVsRate, ...]
    divergence: Divergence
    span_start: int
    span_end: int
    months: int = Field(ge=0)
    speeches: int = Field(ge=0)
    corpus_through: str
    market_through: str
    min_speeches_per_month: int = Field(ge=1)

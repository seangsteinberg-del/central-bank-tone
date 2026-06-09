"""Server-rendered views for the web UI (CLAUDE.md sections 2 and 3).

Each view validates its inputs, calls a service with typed values, and renders a template. Views
never touch a repository, the engine, or the logger directly; core exceptions propagate to the
handlers in ``errors.py``. Routes under ``/ui`` return HTML fragments for htmx to swap in; the
rest return full pages. The fragments degrade gracefully: the forms also submit as normal
requests when htmx (or JavaScript) is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Form, Request, Response
from pydantic import ValidationError

from cbt_core import (
    Aspect,
    BenchmarkUnavailableError,
    CentralBank,
    CommitteeMovement,
    InsufficientDataError,
    MemberMovement,
    SignalVsMarket,
    Speaker,
    Speech,
    SpeechStance,
    ToneLabel,
    ToneObservation,
)
from cbt_web.dependencies import (
    CommitteeServiceDep,
    IndexingServiceDep,
    IngestionServiceDep,
    MarketSignalServiceDep,
    QaServiceDep,
    SpeakerServiceDep,
    ToneServiceDep,
)
from cbt_web.schemas import AskForm, IngestForm, SpeakerForm
from cbt_web.templating import templates

router = APIRouter()

# SVG tone-chart geometry (a 720x232 viewBox); the score axis runs +1 (hawkish) .. -1 (dovish),
# with a left gutter for the value-axis ticks and a bottom strip for the date labels.
_CHART_W = 720.0
_CHART_H = 232.0
_PAD_L = 34.0
_PAD_R = 16.0
_PLOT_TOP = 16.0
_PLOT_BOTTOM = 192.0
_MID_Y = (_PLOT_TOP + _PLOT_BOTTOM) / 2.0
_HALF_H = (_PLOT_BOTTOM - _PLOT_TOP) / 2.0
_MAX_LABELS = 14
_TICK_VALUES = (1.0, 0.5, 0.0, -0.5, -1.0)  # value-axis gridlines and labels

# Inline committee-row sparkline geometry (a wide, short trend line on a fixed +1 .. -1 scale).
_SPARK_W = 240.0
_SPARK_H = 30.0
_SPARK_PAD_X = 2.0
_SPARK_MID = 15.0
_SPARK_AMP = 12.0

# Corpus tone-drift band chart geometry (monthly mean tone with a +/- 1 std band over time).
_BAND_W = 720.0
_BAND_H = 196.0
_BAND_PAD_L = 34.0
_BAND_PAD_R = 16.0
_BAND_TOP = 14.0
_BAND_BOTTOM = 162.0
_BAND_MID = (_BAND_TOP + _BAND_BOTTOM) / 2.0
_BAND_HALF = (_BAND_BOTTOM - _BAND_TOP) / 2.0
_BAND_MAX_LABELS = 12

# Number of categorical line colours (CSS ``--cat-0`` .. ``--cat-7``). Each tracked bank gets a
# stable, distinct slot by its registry order (``_BANK_PALETTE``); a test asserts the registry
# never outgrows the palette, so colours are never silently reused (CLAUDE.md section 3).
_PALETTE_SIZE = 8
_BANK_PALETTE: dict[CentralBank, int] = {bank: index for index, bank in enumerate(CentralBank)}

# Short, desk-standard codes for the per-bank line end labels. Keyed off the registry enum (it
# stays the single source of truth for which banks exist); an unlisted bank falls back to a
# derived code rather than failing, so adding a registry member never breaks the chart.
_BANK_CODE: dict[CentralBank, str] = {
    CentralBank.FEDERAL_RESERVE: "FED",
    CentralBank.ECB: "ECB",
    CentralBank.BANK_OF_ENGLAND: "BoE",
    CentralBank.BANK_OF_JAPAN: "BoJ",
    CentralBank.BANK_OF_CANADA: "BoC",
    CentralBank.RESERVE_BANK_OF_AUSTRALIA: "RBA",
    CentralBank.SWISS_NATIONAL_BANK: "SNB",
    CentralBank.PEOPLES_BANK_OF_CHINA: "PBoC",
}

# How many model/lexicon disagreements to surface on the dashboard, largest split first.
_MAX_FLAGGED = 8

# How many recent speeches to surface in the "Latest speeches" feed (newest first, deduped).
_RECENT_LIMIT = 10

# Per-bank divergence chart geometry: its own wider box with a right gutter for the end-of-line
# labels (one "CODE +0.xx" per bank, vertically de-collided), so a reader names a line on the
# chart without tracing back to the legend or relying on colour alone.
_DIV_W = 760.0
_DIV_H = 244.0
_DIV_PAD_L = 34.0
_DIV_PAD_R = 96.0
_DIV_TOP = 16.0
_DIV_BOTTOM = 196.0
_DIV_MID = (_DIV_TOP + _DIV_BOTTOM) / 2.0
_DIV_HALF = (_DIV_BOTTOM - _DIV_TOP) / 2.0
_DIV_MAX_LABELS = 12
_DIV_LABEL_GAP = 11.0  # minimum vertical spacing between adjacent end-of-line labels
_DIV_LABEL_X = _DIV_W - _DIV_PAD_R + 10.0  # x where the end-of-line labels begin (right gutter)

# A committee-stance move smaller than this (between the compared periods) reads as flat, not a
# turn; it is the rounding-noise band shared with the movement rows below.
_TURN_BAND = 0.02
_MAX_MOVERS = 6  # how many "who's turning" rows to surface, largest one-month move first

# Allowed Policy Monitor sort keys (validated at the boundary; an unknown key is a 422, not a
# silent default), mapped to a (key function, descending?) pair.
_MONITOR_SORTS: frozenset[str] = frozenset({"stance", "delta_1m", "delta_3m", "committee", "bank"})


def _bank_code(bank: CentralBank) -> str:
    """Return a short desk code for a bank (e.g. ``FED``), deriving one if it is not listed."""
    return _BANK_CODE.get(bank, bank.value[:4].upper())


def _solo_speakers(speakers: list[Speaker]) -> list[Speaker]:
    """Hide joint-statement phantom speakers (names with ';' or ',') from the UI listings.

    The bulk source stored a joint speech's combined author field verbatim, creating one-speech
    'speakers' like ``"Thomas Jordan; Martin Schlegel"`` that pollute the leaderboard, search, and
    committee boards. This is a presentation filter only; the underlying rows are untouched (a data
    backfill is the durable fix).
    """
    return [speaker for speaker in speakers if ";" not in speaker.name and "," not in speaker.name]


def _month_index(key: tuple[int, int]) -> int:
    """Map a ``(year, month)`` to a comparable month ordinal for calendar arithmetic."""
    return key[0] * 12 + (key[1] - 1)


def _month_minus(key: tuple[int, int], months: int) -> tuple[int, int]:
    """Return the ``(year, month)`` exactly ``months`` calendar months before ``key``."""
    ordinal = _month_index(key) - months
    return (ordinal // 12, ordinal % 12 + 1)


@dataclass(frozen=True)
class ChartPoint:
    """One plotted tone observation in the SVG chart's coordinate space."""

    x: float
    y: float
    zero_y: float
    tone: ToneLabel
    score: float
    label: str
    show_label: bool


@dataclass(frozen=True)
class ChartTick:
    """A value-axis gridline at a tone level, with its left-axis label."""

    y: float
    label: str
    zero: bool


@dataclass(frozen=True)
class ToneChart:
    """Precomputed geometry for the speaker's tone-over-time SVG line chart."""

    width: float
    height: float
    pad_l: float
    pad_r: float
    mid_y: float
    plot_top: float
    plot_bottom: float
    points: list[ChartPoint]
    polyline: str
    ticks: list[ChartTick]


@dataclass(frozen=True)
class Spark:
    """A tiny inline trend line (a speaker's tone history) on a fixed +1 .. -1 scale."""

    width: float
    height: float
    zero_y: float
    polyline: str
    last_x: float
    last_y: float


@dataclass(frozen=True)
class LeaderRow:
    """A speaker's latest tone reading and a sparkline of their tone history."""

    speaker: Speaker
    score: float
    tone: ToneLabel
    count: int
    spark: Spark


@dataclass(frozen=True)
class BandPoint:
    """One monthly bucket of the corpus tone-drift band: the mean and its +/- 1 std envelope."""

    x: float
    mean_y: float
    lo_y: float
    hi_y: float
    label: str
    show_label: bool


@dataclass(frozen=True)
class CorpusToneSeries:
    """Precomputed geometry for the corpus-wide monthly tone-drift band chart."""

    width: float
    height: float
    pad_l: float
    pad_r: float
    mid_y: float
    plot_top: float
    plot_bottom: float
    points: list[BandPoint]
    mean_polyline: str
    band_path: str
    ticks: list[ChartTick]
    months: int


@dataclass(frozen=True)
class BankLinePoint:
    """One month of a bank's tone-index line: its committee mean tone at a point in time."""

    x: float
    y: float
    label: str  # the month, e.g. "02/26"
    mean: float


@dataclass(frozen=True)
class MonthLabel:
    """A bottom-axis month tick (its x position and short label) on the divergence chart."""

    x: float
    label: str


@dataclass(frozen=True)
class BankToneLine:
    """One central bank's monthly tone index, drawn as a single coloured line over time."""

    bank: CentralBank
    label: str
    code: str  # short desk code shown at the line end (e.g. "FED")
    palette: int  # index into the CSS categorical palette (``--cat-0`` .. ``--cat-7``)
    points: list[BankLinePoint]
    polyline: str
    last_x: float
    last_y: float
    label_y: float  # de-collided y for the end-of-line label (kept clear of its neighbours)
    latest: float  # the most recent month's committee mean tone


@dataclass(frozen=True)
class BankToneHistory:
    """Per-bank tone indices over a shared month axis, for the policy-divergence chart.

    Every line is plotted against the same global month axis (so the banks are directly
    comparable, and so the lines align with the corpus band chart drawn above it). A bank
    contributes a point only for the months it actually spoke, leaving honest gaps.
    """

    width: float
    height: float
    pad_l: float
    pad_r: float
    mid_y: float
    plot_top: float
    plot_bottom: float
    label_x: float  # x where the end-of-line labels begin (in the right gutter)
    lines: list[BankToneLine]
    ticks: list[ChartTick]
    month_labels: list[MonthLabel]
    months: int


@dataclass(frozen=True)
class BankBoard:
    """One central bank's committee, ranked by each member's most recent tone.

    Tone is only comparable within a committee, so the dashboard ranks each bank's speakers among
    themselves (most hawkish first, most dovish last) and summarizes the committee, never pooling
    speakers across banks.
    """

    bank: CentralBank
    label: str
    ranked: list[LeaderRow]  # every member, most hawkish first
    speaker_count: int
    mean_score: float  # the committee's mean latest tone (a per-bank tone index)
    hawk_count: int  # members leaning hawkish (score > 0)
    dove_count: int  # members leaning dovish (score < 0)


@dataclass(frozen=True)
class DeskRead:
    """A one-line, data-derived read of the corpus tone for the desk masthead."""

    now: float  # the most recent month's mean tone across the corpus
    headline: str  # e.g. "The central-bank chorus is broadly neutral and drifting hawkish."
    hawk_bank: str  # label of the bank whose committee mean is most hawkish
    dove_bank: str  # label of the bank whose committee mean is most dovish


@dataclass(frozen=True)
class FlaggedSplit:
    """A speech the LLM and the deterministic lexicon disagreed on, with the size of the split.

    Surfacing these (rather than averaging them away) is the cross-check contract in ADR 0008: a
    large model/lexicon divergence is a review signal, not noise to hide.
    """

    speaker: Speaker
    speech: Speech
    gap: float  # absolute gap between the model score and the lexicon score


@dataclass(frozen=True)
class MonthValue:
    """One month of a bank's committee-stance series: its mean stance as of that month."""

    key: tuple[int, int]  # (year, month)
    value: float


@dataclass(frozen=True)
class PolicyMonitorRow:
    """One bank's row in the Policy Monitor: where it stands now and how it has moved.

    The single scannable line a macro desk reads first: current committee stance, the one- and
    three-month change (the tradeable delta), a trend sparkline, and the committee's hawk/dove
    split. A ``None`` delta means there is no reading that far back to compare against (shown as
    an em dash, never a fabricated zero).
    """

    bank: CentralBank
    label: str
    code: str
    palette: int
    now: float  # current committee stance (mean of members' latest readings)
    delta_1m: float | None  # change versus one calendar month ago, or None if no prior reading
    delta_3m: float | None  # change versus three calendar months ago
    spark: Spark
    months: int
    hawk_count: int
    dove_count: int
    divided: bool  # the committee is materially split (both camps present and near-balanced)
    last_spoke: datetime | None


@dataclass(frozen=True)
class KpiCard:
    """One market-relevant headline metric for the dashboard's top strip."""

    label: str  # e.g. "Most hawkish"
    headline: str  # e.g. "BoJ"
    value: str  # e.g. "+0.80"
    sub: str  # e.g. "hawkish" / "1-month move" / "FX divergence"
    score: float | None  # drives the chip colour via the score_chip filter; None = neutral accent
    href: str | None  # an in-page anchor or fragment link, when the card is actionable


@dataclass(frozen=True)
class Mover:
    """A bank whose committee stance shifted over the last month (the momentum signal)."""

    bank: CentralBank
    code: str
    label: str
    palette: int
    prev: float  # stance one month ago
    now: float  # stance now
    delta: float  # signed one-month change
    side: str  # "hawk" (turning hawkish) or "dove" (turning dovish)
    word: str  # "turning hawkish" / "extending hawkish" / "turning dovish" / "extending dovish"
    bar_pct: float  # |delta| scaled to the largest mover, as a percentage


@dataclass(frozen=True)
class RecentSpeech:
    """A recently analyzed speech for the scannable "Latest speeches" feed.

    The one row a macro reader skims instead of opening the speech: the tone read, a one-line
    summary, and the change versus this speaker's previous speech (the tradeable delta). ``delta``
    is ``None`` for a speaker's first analyzed speech, so the feed shows an honest "first reading"
    rather than a fabricated zero move (CLAUDE.md section 3).
    """

    speaker: Speaker
    speech: Speech
    delta: float | None  # headline-tone change versus this speaker's previous speech
    side: str  # "hawk" / "dove" / "flat" / "none" (drives the change chip's colour)
    word: str  # e.g. "more hawkish than last" / "in line with last" / "first analyzed speech"


@dataclass(frozen=True)
class CorpusOverview:
    """Aggregate corpus statistics for the dashboard."""

    speakers: int
    speeches: int
    observations: int
    flagged: int
    banks: int
    span_start: datetime | None
    span_end: datetime | None
    read: DeskRead | None
    kpis: list[KpiCard]
    monitor: list[PolicyMonitorRow]
    monitor_sort: str
    movers: list[Mover]
    boards: list[BankBoard]
    recent: list[RecentSpeech]
    tone_series: CorpusToneSeries | None
    bank_history: BankToneHistory | None
    flagged_splits: list[FlaggedSplit]


@dataclass(frozen=True)
class MovementRow:
    """A committee member's movement, with the geometry to draw a diverging movement bar."""

    member: MemberMovement
    side: str  # "hawk", "dove", "flat" (a measured shift), or "none" (no prior reading)
    word: str  # "more hawkish", "more dovish", "little changed", or "first reading"
    magnitude: float  # absolute size of the shift
    bar_pct: float  # bar width as a percentage of the largest shift in the committee


# A shift smaller than this is treated as flat rather than directional (rounding noise).
_FLAT_BAND = 0.005


def _move_presentation(member: MemberMovement, max_abs: float) -> MovementRow:
    """Turn a member's signed shift into a labelled, scaled movement row for the template."""
    delta = member.delta
    if delta is None:
        return MovementRow(
            member=member, side="none", word="first reading", magnitude=0.0, bar_pct=0.0
        )
    magnitude = abs(delta)
    if delta > _FLAT_BAND:
        side, word = "hawk", "more hawkish"
    elif delta < -_FLAT_BAND:
        side, word = "dove", "more dovish"
    else:
        side, word = "flat", "little changed"
    bar_pct = round(magnitude / max_abs * 100.0, 1) if max_abs > 0 else 0.0
    return MovementRow(
        member=member, side=side, word=word, magnitude=round(magnitude, 4), bar_pct=bar_pct
    )


def _movement_rows(movement: CommitteeMovement) -> list[MovementRow]:
    """Build the movement rows for a committee, scaled to its largest individual shift."""
    deltas = [abs(member.delta) for member in movement.members if member.delta is not None]
    max_abs = max(deltas) if deltas else 0.0
    return [_move_presentation(member, max_abs) for member in movement.members]


def _direction_word(delta: float | None, *, hawk: str, dove: str, flat: str) -> str | None:
    """Map a signed shift to a direction phrase, or ``None`` if there is no shift to describe."""
    if delta is None:
        return None
    if delta > _FLAT_BAND:
        return hawk
    if delta < -_FLAT_BAND:
        return dove
    return flat


# Derived from the schema spine so a new Aspect renders automatically and can never silently drop
# off the speech page (CLAUDE.md section 2: do not re-encode the spine's knowledge).
_ASPECT_ORDER = tuple(aspect.value for aspect in Aspect)


@dataclass(frozen=True)
class AspectBar:
    """One policy aspect's net-hawkishness, with diverging-bar geometry for the speech page."""

    label: str
    score: float
    side: str  # "hawk" / "dove" / "flat"
    width: float  # |score| as a half-track percentage


@dataclass(frozen=True)
class StanceView:
    """The precomputed presentation of a speech's structured stance decomposition (ADR 0021)."""

    rate_path: float
    rate_side: str
    rate_width: float
    rate_word: str
    uncertainty_pct: float
    needs_review: bool
    structured_net: float
    classifier_net: float
    lexicon_net: float
    aspects: tuple[AspectBar, ...]
    # True when the structured pipeline found no policy-relevant sentences, so the decomposition is
    # an honest abstention rather than a measured zero (ADR 0021; CLAUDE.md section 3). The template
    # shows a note instead of a misleading wall of +0.00 chips.
    structured_abstained: bool


def _bar_side(score: float) -> str:
    """The diverging-bar side for a signed score: hawkish right, dovish left, else flat."""
    if score > _FLAT_BAND:
        return "hawk"
    if score < -_FLAT_BAND:
        return "dove"
    return "flat"


def _stance_view(stance: SpeechStance) -> StanceView:
    """Precompute the speech page's stance decomposition (bars, words) from a stored stance."""
    aspects = [
        AspectBar(
            label=key.replace("_", " ").title(),
            score=stance.aspect_scores[key],
            side=_bar_side(stance.aspect_scores[key]),
            width=min(abs(stance.aspect_scores[key]), 1.0) * 50.0,
        )
        for key in _ASPECT_ORDER
        if key in stance.aspect_scores
    ]
    return StanceView(
        rate_path=stance.rate_path,
        rate_side=_bar_side(stance.rate_path),
        rate_width=min(abs(stance.rate_path), 1.0) * 50.0,
        rate_word=_direction_word(
            stance.rate_path,
            hawk="signals tightening ahead",
            dove="signals easing ahead",
            flat="no clear policy intent",
        )
        or "no clear policy intent",
        uncertainty_pct=stance.uncertainty * 100.0,
        needs_review=stance.needs_review,
        structured_net=stance.structured_net,
        classifier_net=stance.classifier_net,
        lexicon_net=stance.lexicon_net,
        aspects=tuple(aspects),
        structured_abstained=(
            stance.rate_path == 0.0 and stance.structured_net == 0.0 and not stance.aspect_scores
        ),
    )


def _matches(speaker: Speaker, query: str) -> bool:
    """True if a speaker matches a name/institution search query (case-insensitive)."""
    needle = query.casefold().strip()
    if not needle:
        return True
    return needle in speaker.name.casefold() or needle in speaker.central_bank.value.casefold()


def _tone_chart(observations: list[ToneObservation]) -> ToneChart | None:
    """Build SVG line-chart geometry from a speaker's tone observations, or None if there are none."""
    if not observations:
        return None
    count = len(observations)
    span = _CHART_W - _PAD_L - _PAD_R
    step = max(1, round(count / _MAX_LABELS))
    points: list[ChartPoint] = []
    for index, observation in enumerate(observations):
        x = _PAD_L + span / 2.0 if count == 1 else _PAD_L + span * index / (count - 1)
        bounded = max(-1.0, min(1.0, observation.score))
        y = _MID_Y - bounded * _HALF_H
        points.append(
            ChartPoint(
                x=round(x, 1),
                y=round(y, 1),
                zero_y=_MID_Y,
                tone=observation.tone,
                score=observation.score,
                label=f"{observation.observed_at.year}",
                show_label=count <= _MAX_LABELS or index % step == 0,
            )
        )
    polyline = " ".join(f"{p.x},{p.y}" for p in points)
    ticks = [
        ChartTick(
            y=round(_MID_Y - value * _HALF_H, 1),
            label="0" if value == 0.0 else f"{value:+.1f}",
            zero=value == 0.0,
        )
        for value in _TICK_VALUES
    ]
    return ToneChart(
        width=_CHART_W,
        height=_CHART_H,
        pad_l=_PAD_L,
        pad_r=_PAD_R,
        mid_y=_MID_Y,
        plot_top=_PLOT_TOP,
        plot_bottom=_PLOT_BOTTOM,
        points=points,
        polyline=polyline,
        ticks=ticks,
    )


def _sparkline(scores: list[float]) -> Spark:
    """Build a tiny inline trend line from a speaker's tone scores (oldest to newest)."""
    span = _SPARK_W - 2.0 * _SPARK_PAD_X
    count = len(scores)
    points: list[tuple[float, float]] = []
    for index, score in enumerate(scores):
        x = _SPARK_PAD_X + (span / 2.0 if count == 1 else span * index / (count - 1))
        y = _SPARK_MID - max(-1.0, min(1.0, score)) * _SPARK_AMP
        points.append((round(x, 1), round(y, 1)))
    last_x, last_y = points[-1]
    return Spark(
        width=_SPARK_W,
        height=_SPARK_H,
        zero_y=_SPARK_MID,
        polyline=" ".join(f"{x},{y}" for x, y in points),
        last_x=last_x,
        last_y=last_y,
    )


def _leader_row(speaker: Speaker, observations: list[ToneObservation]) -> LeaderRow:
    """Build a leader row (latest reading plus a tone-history sparkline) from observations."""
    latest = observations[-1]
    return LeaderRow(
        speaker=speaker,
        score=latest.score,
        tone=latest.tone,
        count=len(observations),
        spark=_sparkline([observation.score for observation in observations]),
    )


def _collect_leaders(speakers: list[Speaker], tone: ToneServiceDep) -> list[LeaderRow]:
    """Build each speaker's latest tone reading, skipping speakers with no observations yet."""
    leaders: list[LeaderRow] = []
    for speaker in speakers:
        observations = tone.observations_for(speaker.id)
        if observations:
            leaders.append(_leader_row(speaker, observations))
    return leaders


def _tone_series_from(buckets: dict[tuple[int, int], list[float]]) -> CorpusToneSeries | None:
    """Build the corpus tone-drift band chart from monthly score buckets, or ``None`` if empty.

    Each month becomes one point: the mean tone across every speech that month, with a band of one
    standard deviation either side (clamped to the +1 .. -1 axis) showing how dispersed the chorus
    was. Returns ``None`` when there are no observations to plot.
    """
    if not buckets:
        return None
    keys = sorted(buckets)
    count = len(keys)
    span = _BAND_W - _BAND_PAD_L - _BAND_PAD_R
    step = max(1, round(count / _BAND_MAX_LABELS))

    def _y(value: float) -> float:
        return round(_BAND_MID - max(-1.0, min(1.0, value)) * _BAND_HALF, 1)

    points: list[BandPoint] = []
    for index, (year, month) in enumerate(keys):
        scores = buckets[year, month]
        mean = sum(scores) / len(scores)
        std = (sum((score - mean) ** 2 for score in scores) / len(scores)) ** 0.5
        x = _BAND_PAD_L + (span / 2.0 if count == 1 else span * index / (count - 1))
        points.append(
            BandPoint(
                x=round(x, 1),
                mean_y=_y(mean),
                lo_y=_y(mean - std),
                hi_y=_y(mean + std),
                label=f"{month:02d}/{str(year)[2:]}",
                show_label=count <= _BAND_MAX_LABELS or index % step == 0,
            )
        )
    band_path = (
        " ".join(f"{p.x},{p.hi_y}" for p in points)
        + " "
        + " ".join(f"{p.x},{p.lo_y}" for p in reversed(points))
    )
    ticks = [
        ChartTick(
            y=round(_BAND_MID - value * _BAND_HALF, 1),
            label="0" if value == 0.0 else f"{value:+.1f}",
            zero=value == 0.0,
        )
        for value in _TICK_VALUES
    ]
    return CorpusToneSeries(
        width=_BAND_W,
        height=_BAND_H,
        pad_l=_BAND_PAD_L,
        pad_r=_BAND_PAD_R,
        mid_y=_BAND_MID,
        plot_top=_BAND_TOP,
        plot_bottom=_BAND_BOTTOM,
        points=points,
        mean_polyline=" ".join(f"{p.x},{p.mean_y}" for p in points),
        band_path=band_path,
        ticks=ticks,
        months=count,
    )


def _bank_boards(leaders: list[LeaderRow]) -> list[BankBoard]:
    """Group leader rows by central bank and rank each bank's speakers among themselves.

    Tone is only comparable within a committee, so speakers are ranked per bank and never pooled
    across banks. Boards are ordered by how many speakers each bank has (most first), then by label,
    so the default (first) tab is the most populated bank.
    """
    by_bank: dict[CentralBank, list[LeaderRow]] = {}
    for row in leaders:
        by_bank.setdefault(row.speaker.central_bank, []).append(row)
    boards: list[BankBoard] = []
    for bank, rows in by_bank.items():
        ranked = sorted(rows, key=lambda r: r.score, reverse=True)
        boards.append(
            BankBoard(
                bank=bank,
                label=bank.value.replace("_", " ").title(),
                ranked=ranked,
                speaker_count=len(rows),
                mean_score=sum(r.score for r in rows) / len(rows),
                hawk_count=sum(1 for r in rows if r.score > 0),
                dove_count=sum(1 for r in rows if r.score < 0),
            )
        )
    boards.sort(key=lambda board: (-board.speaker_count, board.label))
    return boards


def _bank_tone_history(
    boards: list[BankBoard],
    bank_series: dict[CentralBank, list[MonthValue]],
    month_keys: list[tuple[int, int]],
) -> BankToneHistory | None:
    """Build a 'committee stance by bank over time' chart: one labelled line per bank.

    Each board becomes one line, plotted across the shared global month axis so the banks are
    directly comparable. The line follows the same canonical committee-stance series the Policy
    Monitor reads (each month, the mean of members' most recent readings), so the chart and the
    monitor trend agree. Returns ``None`` when there is nothing to plot.

    Args:
        boards: The per-bank boards, already ordered (most populated first).
        bank_series: Each bank's monthly committee-stance series.
        month_keys: The sorted global month axis (every ``(year, month)`` in the corpus).

    Returns:
        The chart geometry, or ``None`` when there are no months or no banks to plot.
    """
    if not month_keys or not boards:
        return None
    span = _DIV_W - _DIV_PAD_L - _DIV_PAD_R
    count = len(month_keys)
    x_of = {
        key: round(_DIV_PAD_L + (span / 2.0 if count == 1 else span * index / (count - 1)), 1)
        for index, key in enumerate(month_keys)
    }

    def _y(value: float) -> float:
        return round(_DIV_MID - max(-1.0, min(1.0, value)) * _DIV_HALF, 1)

    # First pass: build each bank's points from its stance series (over the months it was active).
    drawn: list[tuple[BankBoard, list[BankLinePoint]]] = []
    for board in boards:
        by_key = {mv.key: mv.value for mv in bank_series.get(board.bank, [])}
        points: list[BankLinePoint] = []
        for key in month_keys:
            if key not in by_key:
                continue
            value = by_key[key]
            points.append(
                BankLinePoint(
                    x=x_of[key], y=_y(value), label=f"{key[1]:02d}/{str(key[0])[2:]}", mean=value
                )
            )
        if points:
            drawn.append((board, points))
    if not drawn:
        return None

    # De-collide the end-of-line labels: start each at its line's last point, then spread the
    # labels apart (in y order) so they never overlap, and slide the whole block back inside the
    # plot if the cluster pushed it past an edge.
    last_ys = [points[-1].y for _, points in drawn]
    label_y = list(last_ys)
    order = sorted(range(len(drawn)), key=lambda i: last_ys[i])
    previous: float | None = None
    for i in order:
        candidate = last_ys[i]
        if previous is not None and candidate < previous + _DIV_LABEL_GAP:
            candidate = previous + _DIV_LABEL_GAP
        label_y[i] = candidate
        previous = candidate
    overflow = max(0.0, max(label_y) - _DIV_BOTTOM)
    if overflow:
        label_y = [y - overflow for y in label_y]
    underflow = max(0.0, _DIV_TOP - min(label_y))
    if underflow:
        label_y = [y + underflow for y in label_y]

    lines = [
        BankToneLine(
            bank=board.bank,
            label=board.label,
            code=_bank_code(board.bank),
            palette=_BANK_PALETTE.get(board.bank, 0),
            points=points,
            polyline=" ".join(f"{p.x},{p.y}" for p in points),
            last_x=points[-1].x,
            last_y=points[-1].y,
            label_y=round(label_y[index], 1),
            latest=round(points[-1].mean, 2),
        )
        for index, (board, points) in enumerate(drawn)
    ]
    step = max(1, round(count / _DIV_MAX_LABELS))
    month_labels = [
        MonthLabel(x=x_of[key], label=f"{key[1]:02d}/{str(key[0])[2:]}")
        for index, key in enumerate(month_keys)
        if count <= _DIV_MAX_LABELS or index % step == 0
    ]
    ticks = [
        ChartTick(
            y=round(_DIV_MID - value * _DIV_HALF, 1),
            label="0" if value == 0.0 else f"{value:+.1f}",
            zero=value == 0.0,
        )
        for value in _TICK_VALUES
    ]
    return BankToneHistory(
        width=_DIV_W,
        height=_DIV_H,
        pad_l=_DIV_PAD_L,
        pad_r=_DIV_PAD_R,
        mid_y=_DIV_MID,
        plot_top=_DIV_TOP,
        plot_bottom=_DIV_BOTTOM,
        label_x=_DIV_LABEL_X,
        lines=lines,
        ticks=ticks,
        month_labels=month_labels,
        months=count,
    )


def _desk_read(
    buckets: dict[tuple[int, int], list[float]], boards: list[BankBoard]
) -> DeskRead | None:
    """Derive a one-line desk read of the corpus tone from the monthly buckets and the boards.

    Every part is computed from the data, not asserted: ``now`` is the most recent month's mean
    tone; the drift compares it to the prior few months; the extremes are the banks with the most
    hawkish and most dovish committee means. Returns ``None`` when there is nothing to read.
    """
    if not buckets or not boards:
        return None
    keys = sorted(buckets)
    latest = buckets[keys[-1]]
    now = sum(latest) / len(latest)
    if now > 0.08:
        tone_word = "leaning hawkish"
    elif now < -0.08:
        tone_word = "leaning dovish"
    else:
        tone_word = "broadly neutral"
    clause = ""
    if len(keys) > 1:
        prior = [score for key in keys[:-1][-3:] for score in buckets[key]]
        delta = now - sum(prior) / len(prior)
        if delta > 0.05:
            clause = " and drifting hawkish"
        elif delta < -0.05:
            clause = " and drifting dovish"
        else:
            clause = " and holding steady"
    hawk = max(boards, key=lambda board: board.mean_score)
    dove = min(boards, key=lambda board: board.mean_score)
    return DeskRead(
        now=now,
        headline=f"The central-bank chorus is {tone_word}{clause}.",
        hawk_bank=hawk.label,
        dove_bank=dove.label,
    )


def _select_board(boards: list[BankBoard], bank: str) -> BankBoard | None:
    """Pick the board for ``bank`` (a central-bank value), or the most populated one when unset.

    Args:
        boards: The available per-bank boards.
        bank: The requested central bank's enum value, or ``""`` for the default.

    Returns:
        The matching board, the first board when ``bank`` is empty, or ``None`` when there are no
        boards (or the requested bank has none).

    Raises:
        ValueError: If ``bank`` is non-empty but is not a known central bank value (validated at the
            boundary so a bad query maps to a 4xx, not a silent fallback).
    """
    if not bank:
        return boards[0] if boards else None
    requested = CentralBank(bank)
    return next((board for board in boards if board.bank == requested), None)


@dataclass(frozen=True)
class _ToneScan:
    """The shared single pass over per-speaker tone, reused by the page and its fragments."""

    boards: list[BankBoard]
    bank_series: dict[CentralBank, list[MonthValue]]
    rows: list[PolicyMonitorRow]
    buckets: dict[tuple[int, int], list[float]]
    observation_total: int


def _bank_stance_series(members: list[list[ToneObservation]]) -> list[MonthValue]:
    """Build a bank's monthly committee-stance series.

    For each calendar month from the committee's first reading to its last, the stance is the mean
    across members of each member's most recent reading as of that month (a member who has not yet
    spoken is not counted). This standing-tone-over-time series is the canonical signal the Policy
    Monitor and the divergence chart both read, so every view agrees.

    Args:
        members: Each member's observations (each list oldest first or not; sorted here).

    Returns:
        The monthly stance series, oldest month first; empty if there are no observations.
    """
    observed = [sorted(member, key=lambda o: o.observed_at) for member in members if member]
    keys = [(o.observed_at.year, o.observed_at.month) for member in observed for o in member]
    if not keys:
        return []
    series: list[MonthValue] = []
    for ordinal in range(_month_index(min(keys)), _month_index(max(keys)) + 1):
        latest: list[float] = []
        for member in observed:
            recent: float | None = None
            for observation in member:
                month = (observation.observed_at.year, observation.observed_at.month)
                if _month_index(month) <= ordinal:
                    recent = observation.score
                else:
                    break
            if recent is not None:
                latest.append(recent)
        if latest:
            series.append(
                MonthValue(key=(ordinal // 12, ordinal % 12 + 1), value=sum(latest) / len(latest))
            )
    return series


def _stance_delta(series: list[MonthValue], months: int) -> float | None:
    """Change in stance versus ``months`` calendar months before the latest reading, or None.

    Returns ``None`` (rendered as an em dash, never a fabricated zero) when there is no reading
    that far back to compare against, per the no-silent-fallback rule.
    """
    if not series:
        return None
    by_key = {mv.key: mv.value for mv in series}
    latest = series[-1]
    target = _month_minus(latest.key, months)
    if target not in by_key:
        return None
    return round(latest.value - by_key[target], 2)


def _policy_monitor_row(
    board: BankBoard, series: list[MonthValue], last_spoke: datetime | None
) -> PolicyMonitorRow:
    """Build one Policy Monitor row from a bank's board and its committee-stance series."""
    now = series[-1].value if series else board.mean_score
    smaller = min(board.hawk_count, board.dove_count)
    larger = max(board.hawk_count, board.dove_count)
    divided = board.hawk_count >= 1 and board.dove_count >= 1 and smaller * 2 >= larger
    return PolicyMonitorRow(
        bank=board.bank,
        label=board.label,
        code=_bank_code(board.bank),
        palette=_BANK_PALETTE.get(board.bank, 0),
        now=round(now, 2),
        delta_1m=_stance_delta(series, 1),
        delta_3m=_stance_delta(series, 3),
        spark=_sparkline([mv.value for mv in series] or [board.mean_score]),
        months=len(series),
        hawk_count=board.hawk_count,
        dove_count=board.dove_count,
        divided=divided,
        last_spoke=last_spoke,
    )


def _sort_monitor(rows: list[PolicyMonitorRow], sort: str) -> list[PolicyMonitorRow]:
    """Return the monitor rows sorted by a validated key (raises ValueError on an unknown one)."""
    if sort not in _MONITOR_SORTS:
        raise ValueError(f"unknown monitor sort: {sort}")
    if sort == "bank":
        return sorted(rows, key=lambda r: r.label)
    if sort == "committee":
        return sorted(rows, key=lambda r: (r.hawk_count - r.dove_count, r.now), reverse=True)
    if sort == "delta_1m":
        return sorted(rows, key=lambda r: (r.delta_1m is not None, r.delta_1m or 0.0), reverse=True)
    if sort == "delta_3m":
        return sorted(rows, key=lambda r: (r.delta_3m is not None, r.delta_3m or 0.0), reverse=True)
    return sorted(rows, key=lambda r: r.now, reverse=True)


def _movers(rows: list[PolicyMonitorRow]) -> list[Mover]:
    """The banks whose stance moved most over the last month (the momentum read), largest first."""
    moved = [row for row in rows if row.delta_1m is not None and abs(row.delta_1m) > _TURN_BAND]
    if not moved:
        return []
    largest = max(abs(row.delta_1m or 0.0) for row in moved)
    movers: list[Mover] = []
    for row in sorted(moved, key=lambda r: abs(r.delta_1m or 0.0), reverse=True)[:_MAX_MOVERS]:
        delta = row.delta_1m or 0.0
        prev = round(row.now - delta, 2)
        hawkish = delta > 0
        if hawkish:
            word = "extending hawkish" if prev > _TURN_BAND else "turning hawkish"
        else:
            word = "extending dovish" if prev < -_TURN_BAND else "turning dovish"
        movers.append(
            Mover(
                bank=row.bank,
                code=row.code,
                label=row.label,
                palette=row.palette,
                prev=prev,
                now=row.now,
                delta=delta,
                side="hawk" if hawkish else "dove",
                word=word,
                bar_pct=round(abs(delta) / largest * 100.0, 1) if largest else 0.0,
            )
        )
    return movers


def _monitor_kpis(rows: list[PolicyMonitorRow]) -> list[KpiCard]:
    """Build the market-relevant headline cards for the top strip (skips ones with no data)."""
    if not rows:
        return []
    hawkish = max(rows, key=lambda r: r.now)
    dovish = min(rows, key=lambda r: r.now)
    cards = [
        KpiCard(
            label="Most hawkish",
            headline=hawkish.code,
            value=f"{hawkish.now:+.2f}",
            sub=hawkish.label,
            score=hawkish.now,
            href="#monitor",
        ),
        KpiCard(
            label="Most dovish",
            headline=dovish.code,
            value=f"{dovish.now:+.2f}",
            sub=dovish.label,
            score=dovish.now,
            href="#monitor",
        ),
    ]
    moved = [row for row in rows if row.delta_1m is not None and abs(row.delta_1m) > _TURN_BAND]
    if moved:
        top = max(moved, key=lambda r: abs(r.delta_1m or 0.0))
        delta = top.delta_1m or 0.0
        cards.append(
            KpiCard(
                label="Biggest mover",
                headline=top.code,
                value=f"{delta:+.2f}",
                sub="1-month move",
                score=delta,
                href="#movers",
            )
        )
    return cards


def _recent_rows_for(speaker: Speaker, speeches: list[Speech]) -> list[RecentSpeech]:
    """Turn a speaker's speeches (newest first) into feed rows carrying the change vs the prior one.

    The change is the headline-tone delta against this speaker's immediately preceding speech (the
    next item, since the list is newest first). The earliest speech has no prior, so its delta is
    ``None`` and it reads as a "first analyzed speech" rather than a fabricated zero move (CLAUDE.md
    section 3).

    Args:
        speaker: The speaker who gave the speeches.
        speeches: The speaker's speeches, most recent first (as the service returns them).

    Returns:
        One :class:`RecentSpeech` per speech, in the same order.
    """
    rows: list[RecentSpeech] = []
    for index, speech in enumerate(speeches):
        prior = speeches[index + 1] if index + 1 < len(speeches) else None
        if prior is None:
            rows.append(
                RecentSpeech(
                    speaker=speaker,
                    speech=speech,
                    delta=None,
                    side="none",
                    word="first analyzed speech",
                )
            )
            continue
        delta = round(speech.score - prior.score, 2)
        word = (
            _direction_word(
                delta,
                hawk="more hawkish than last",
                dove="more dovish than last",
                flat="in line with last",
            )
            or "in line with last"
        )
        rows.append(
            RecentSpeech(
                speaker=speaker, speech=speech, delta=delta, side=_bar_side(delta), word=word
            )
        )
    return rows


def _dedup_recent(rows: list[RecentSpeech], limit: int) -> list[RecentSpeech]:
    """Sort recent rows newest first, drop repeat source URLs, and cap to ``limit``.

    Deduplicates by source URL (keeping the most recent occurrence) so a speech the source listed
    under more than one author appears once in the feed rather than several times.
    """
    ordered = sorted(rows, key=lambda row: row.speech.delivered_at, reverse=True)
    seen: set[str] = set()
    deduped: list[RecentSpeech] = []
    for row in ordered:
        if row.speech.url in seen:
            continue
        seen.add(row.speech.url)
        deduped.append(row)
    return deduped[:limit]


def _scan_tone(speakers: list[Speaker], tone: ToneServiceDep) -> _ToneScan:
    """One pass over per-speaker tone: boards, the canonical stance series, and the monitor rows."""
    leaders: list[LeaderRow] = []
    members_by_bank: dict[CentralBank, list[list[ToneObservation]]] = {}
    buckets: dict[tuple[int, int], list[float]] = {}
    last_spoke: dict[CentralBank, datetime] = {}
    observation_total = 0
    for speaker in speakers:
        observations = tone.observations_for(speaker.id)
        if not observations:
            continue
        leaders.append(_leader_row(speaker, observations))
        members_by_bank.setdefault(speaker.central_bank, []).append(observations)
        observation_total += len(observations)
        for observation in observations:
            buckets.setdefault(
                (observation.observed_at.year, observation.observed_at.month), []
            ).append(observation.score)
            current = last_spoke.get(speaker.central_bank)
            if current is None or observation.observed_at > current:
                last_spoke[speaker.central_bank] = observation.observed_at
    boards = _bank_boards(leaders)
    bank_series = {bank: _bank_stance_series(members) for bank, members in members_by_bank.items()}
    rows = [
        _policy_monitor_row(board, bank_series.get(board.bank, []), last_spoke.get(board.bank))
        for board in boards
    ]
    return _ToneScan(
        boards=boards,
        bank_series=bank_series,
        rows=rows,
        buckets=buckets,
        observation_total=observation_total,
    )


def _corpus_overview(
    speakers: list[Speaker], tone: ToneServiceDep, ingestion: IngestionServiceDep
) -> CorpusOverview:
    """Aggregate the dashboard: the Policy Monitor, market KPIs, movers, the recent feed, context.

    One tone pass builds the canonical per-bank stance series and the monitor; a second pass over
    speeches adds the recent feed (with each speech's change versus the speaker's prior one), the
    flagged splits, and the span. This iterates the speakers (a small set in this single-operator
    tool); a high-cardinality deployment would push these aggregates into a dedicated read model.
    """
    scan = _scan_tone(speakers, tone)
    speech_total = 0
    flagged = 0
    span_start: datetime | None = None
    span_end: datetime | None = None
    recent_rows: list[RecentSpeech] = []
    flagged_splits: list[FlaggedSplit] = []
    for speaker in speakers:
        speeches = ingestion.list_speeches(speaker.id)
        speech_total += len(speeches)
        flagged += sum(1 for speech in speeches if speech.needs_review)
        recent_rows.extend(_recent_rows_for(speaker, speeches))
        flagged_splits.extend(
            FlaggedSplit(
                speaker=speaker, speech=speech, gap=abs(speech.score - speech.lexicon_score)
            )
            for speech in speeches
            if speech.needs_review
        )
        for speech in speeches:
            delivered = speech.delivered_at
            span_start = delivered if span_start is None else min(span_start, delivered)
            span_end = delivered if span_end is None else max(span_end, delivered)
    flagged_splits.sort(key=lambda split: split.gap, reverse=True)
    return CorpusOverview(
        speakers=len(speakers),
        speeches=speech_total,
        observations=scan.observation_total,
        flagged=flagged,
        banks=len(scan.boards),
        span_start=span_start,
        span_end=span_end,
        read=_desk_read(scan.buckets, scan.boards),
        kpis=_monitor_kpis(scan.rows),
        monitor=_sort_monitor(scan.rows, "stance"),
        monitor_sort="stance",
        movers=_movers(scan.rows),
        boards=scan.boards,
        recent=_dedup_recent(recent_rows, _RECENT_LIMIT),
        tone_series=_tone_series_from(scan.buckets),
        bank_history=_bank_tone_history(scan.boards, scan.bank_series, sorted(scan.buckets)),
        flagged_splits=flagged_splits[:_MAX_FLAGGED],
    )


def _first_error(exc: ValidationError) -> str:
    """Return a short, user-facing message for the first validation error."""
    first = exc.errors()[0]
    field = ".".join(str(part) for part in first["loc"]) or "input"
    return f"{field}: {first['msg']}"


@router.get("/health", include_in_schema=False)
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@router.get("/")
def index(
    request: Request,
    speakers: SpeakerServiceDep,
    tone: ToneServiceDep,
    ingestion: IngestionServiceDep,
) -> Response:
    """Render the dashboard: the thesis, corpus stats, leaderboards, recent speeches, and search."""
    all_speakers = _solo_speakers(speakers.list_speakers())
    overview = _corpus_overview(all_speakers, tone, ingestion)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "speakers": all_speakers,
            "overview": overview,
            "boards": overview.boards,
            "board": overview.boards[0] if overview.boards else None,
            "query": "",
        },
    )


@router.get("/ui/leaderboard")
def leaderboard(
    request: Request, speakers: SpeakerServiceDep, tone: ToneServiceDep, bank: str = ""
) -> Response:
    """Return the per-bank speaker leaderboard fragment for one central bank (htmx).

    The bank toggle swaps this fragment. An unknown ``bank`` value is a bad input and returns 422
    (validated at the boundary, not silently defaulted).
    """
    boards = _bank_boards(_collect_leaders(_solo_speakers(speakers.list_speakers()), tone))
    try:
        board = _select_board(boards, bank)
    except ValueError:
        return templates.TemplateResponse(
            request,
            "_leaderboard.html",
            {"boards": boards, "board": None, "error": "Unknown central bank."},
            status_code=422,
        )
    return templates.TemplateResponse(
        request, "_leaderboard.html", {"boards": boards, "board": board}
    )


@router.get("/ui/monitor")
def monitor(
    request: Request, speakers: SpeakerServiceDep, tone: ToneServiceDep, sort: str = ""
) -> Response:
    """Return the Policy Monitor matrix fragment, sorted by a validated column (htmx).

    The column headers re-request this fragment with a ``sort`` key. An unknown key is a bad input
    and returns 422 (validated at the boundary, not silently defaulted).
    """
    scan = _scan_tone(_solo_speakers(speakers.list_speakers()), tone)
    key = sort or "stance"
    try:
        rows = _sort_monitor(scan.rows, key)
    except ValueError:
        return templates.TemplateResponse(
            request,
            "_monitor.html",
            {"monitor": [], "monitor_sort": "stance", "error": "Unknown sort key."},
            status_code=422,
        )
    return templates.TemplateResponse(
        request, "_monitor.html", {"monitor": rows, "monitor_sort": key}
    )


@router.get("/methodology")
def methodology(request: Request) -> Response:
    """Render the methodology page: how tone is scored and how well it is measured to work."""
    return templates.TemplateResponse(request, "methodology.html", {})


# Signal vs Market dual-axis chart geometry (tone on the left axis, the 2-year yield on the right).
_SVM_W = 720.0
_SVM_H = 250.0
_SVM_PAD_L = 40.0
_SVM_PAD_R = 46.0
_SVM_TOP = 16.0
_SVM_BOTTOM = 200.0
_SVM_MAX_MONTH_LABELS = 12


@dataclass(frozen=True)
class SvmTick:
    """A value-axis tick on the Signal vs Market chart (its y position and label)."""

    y: float
    label: str


@dataclass(frozen=True)
class SvmSeriesLine:
    """One plotted series in the Signal vs Market chart (an SVG polyline plus its end label)."""

    points: str
    css: str
    label: str
    end_x: float
    end_y: float
    end_text: str


@dataclass(frozen=True)
class SvmChart:
    """Precomputed dual-axis chart: monthly headline tone against the 2-year Treasury yield."""

    width: float
    height: float
    zero_y: float
    tone: SvmSeriesLine
    rate: SvmSeriesLine
    month_labels: tuple[MonthLabel, ...]
    left_ticks: tuple[SvmTick, ...]
    right_ticks: tuple[SvmTick, ...]


def _svm_chart(svm: SignalVsMarket) -> SvmChart | None:
    """Build the dual-axis chart geometry from a Signal vs Market view, or None if too sparse."""
    months = list(range(svm.span_start, svm.span_end + 1))
    if len(months) < 2:
        return None
    two_year = next((s for s in svm.rate_series if s.code == "GS2"), None)
    if two_year is None or not two_year.points:
        return None
    span = max(len(months) - 1, 1)
    plot_w = _SVM_W - _SVM_PAD_L - _SVM_PAD_R
    plot_h = _SVM_BOTTOM - _SVM_TOP

    def x_of(month: int) -> float:
        return _SVM_PAD_L + (month - svm.span_start) / span * plot_w

    def tone_y(value: float) -> float:
        return _SVM_TOP + (1.0 - max(-1.0, min(1.0, value))) / 2.0 * plot_h

    rate_values = [two_year.points[m] for m in months if m in two_year.points]
    rate_lo, rate_hi = min(rate_values), max(rate_values)
    if rate_hi == rate_lo:
        rate_hi = rate_lo + 1.0

    def rate_y(value: float) -> float:
        return _SVM_TOP + (rate_hi - value) / (rate_hi - rate_lo) * plot_h

    tone_months = [m for m in months if m in svm.headline_index.points]
    rate_months = [m for m in months if m in two_year.points]
    tone_pts = " ".join(
        f"{x_of(m):.1f},{tone_y(svm.headline_index.points[m]):.1f}" for m in tone_months
    )
    rate_pts = " ".join(f"{x_of(m):.1f},{rate_y(two_year.points[m]):.1f}" for m in rate_months)

    step = max(1, len(months) // _SVM_MAX_MONTH_LABELS)
    month_labels = tuple(
        MonthLabel(x=x_of(m), label=f"{m // 12:04d}-{m % 12 + 1:02d}")
        for i, m in enumerate(months)
        if i % step == 0
    )
    left_ticks = tuple(SvmTick(y=tone_y(v), label=f"{v:+.1f}") for v in (1.0, 0.5, 0.0, -0.5, -1.0))
    right_ticks = tuple(
        SvmTick(
            y=rate_y(rate_lo + frac * (rate_hi - rate_lo)),
            label=f"{rate_lo + frac * (rate_hi - rate_lo):.1f}",
        )
        for frac in (1.0, 0.5, 0.0)
    )
    last_tone = tone_months[-1]
    last_rate = rate_months[-1]
    return SvmChart(
        width=_SVM_W,
        height=_SVM_H,
        zero_y=tone_y(0.0),
        tone=SvmSeriesLine(
            points=tone_pts,
            css="tone",
            label="Headline tone",
            end_x=x_of(last_tone),
            end_y=tone_y(svm.headline_index.points[last_tone]),
            end_text="tone",
        ),
        rate=SvmSeriesLine(
            points=rate_pts,
            css="rate",
            label="2-year Treasury",
            end_x=x_of(last_rate),
            end_y=rate_y(two_year.points[last_rate]),
            end_text="2y %",
        ),
        month_labels=month_labels,
        left_ticks=left_ticks,
        right_ticks=right_ticks,
    )


@router.get("/signal-vs-market")
def signal_vs_market(request: Request, market: MarketSignalServiceDep) -> Response:
    """Render the Signal vs Market divergence view, or an honest unavailable state.

    The page relates the platform's Federal Reserve tone signals to market rates. When the cached
    rate data is missing or the corpus has too little Fed history, it renders an explanatory panel
    rather than a 500 or an empty chart (CLAUDE.md section 3).
    """
    try:
        svm = market.signal_vs_market()
    except (BenchmarkUnavailableError, InsufficientDataError) as exc:
        return templates.TemplateResponse(
            request, "signal_vs_market.html", {"svm": None, "unavailable": str(exc)}
        )
    return templates.TemplateResponse(
        request,
        "signal_vs_market.html",
        {"svm": svm, "chart": _svm_chart(svm), "unavailable": None},
    )


@router.get("/ui/speakers")
def search_speakers(request: Request, speakers: SpeakerServiceDep, q: str = "") -> Response:
    """Return the speaker-list fragment filtered by a name/institution query (htmx)."""
    matched = [
        speaker for speaker in _solo_speakers(speakers.list_speakers()) if _matches(speaker, q)
    ]
    return templates.TemplateResponse(
        request, "_speaker_list.html", {"speakers": matched, "query": q}
    )


@router.get("/speakers/{speaker_id}")
def speaker_detail(
    request: Request,
    speaker_id: UUID,
    speakers: SpeakerServiceDep,
    tone: ToneServiceDep,
    ingestion: IngestionServiceDep,
) -> Response:
    """Render a speaker's page: profile, tone-over-time chart, and analyzed speeches."""
    speaker = speakers.get_speaker(speaker_id)
    observations = tone.observations_for(speaker_id)
    speeches = ingestion.list_speeches(speaker_id)
    return templates.TemplateResponse(
        request,
        "speaker.html",
        {
            "speaker": speaker,
            "chart": _tone_chart(observations),
            "observation_count": len(observations),
            "latest_tone": observations[-1].tone if observations else None,
            "latest_score": observations[-1].score if observations else None,
            "speeches": speeches,
        },
    )


@router.get("/speeches/{speech_id}")
def speech_detail(
    request: Request,
    speech_id: UUID,
    speakers: SpeakerServiceDep,
    ingestion: IngestionServiceDep,
    committee: CommitteeServiceDep,
) -> Response:
    """Render one speech: a concise summary and how its whole committee has moved as of it."""
    speech = ingestion.get_speech(speech_id)
    speaker = speakers.get_speaker(speech.speaker_id)
    movement = committee.movement_for_speech(speech_id)
    rows = _movement_rows(movement)
    stance = ingestion.get_stance(speech_id)
    return templates.TemplateResponse(
        request,
        "speech.html",
        {
            "speech": speech,
            "speaker": speaker,
            "movement": movement,
            "rows": rows,
            "stance": _stance_view(stance) if stance is not None else None,
            "subject_word": _direction_word(
                movement.subject.delta,
                hawk="more hawkish",
                dove="more dovish",
                flat="little changed",
            ),
            "overall_word": _direction_word(
                movement.overall_delta,
                hawk="shifted hawkish",
                dove="shifted dovish",
                flat="held steady",
            ),
            "tone_word": _direction_word(
                movement.committee_tone,
                hawk="leans hawkish",
                dove="leans dovish",
                flat="balanced",
            ),
        },
    )


@router.post("/ui/ask")
def ask_corpus(
    request: Request, qa: QaServiceDep, question: Annotated[str, Form()] = ""
) -> Response:
    """Answer a natural-language question across the whole corpus (htmx fragment)."""
    try:
        form = AskForm(question=question)
    except ValidationError:
        return templates.TemplateResponse(
            request, "_answer.html", {"error": "Please enter a question."}, status_code=422
        )
    answer = qa.answer_corpus(question=form.question)
    return templates.TemplateResponse(request, "_answer.html", {"answer": answer})


@router.post("/ui/speakers/{speaker_id}/ask")
def ask_speaker(
    request: Request,
    speaker_id: UUID,
    qa: QaServiceDep,
    question: Annotated[str, Form()] = "",
) -> Response:
    """Answer a question about one speaker, grounded in their speeches (htmx fragment)."""
    try:
        form = AskForm(question=question)
    except ValidationError:
        return templates.TemplateResponse(
            request, "_answer.html", {"error": "Please enter a question."}, status_code=422
        )
    answer = qa.answer(speaker_id=speaker_id, question=form.question)
    return templates.TemplateResponse(request, "_answer.html", {"answer": answer})


@router.get("/admin")
def admin(request: Request, speakers: SpeakerServiceDep) -> Response:
    """Render the admin page: register a speaker and ingest a speech for one."""
    return templates.TemplateResponse(
        request,
        "admin.html",
        {"speakers": speakers.list_speakers(), "central_banks": list(CentralBank)},
    )


@router.post("/ui/speakers")
def register_speaker(
    request: Request,
    speakers: SpeakerServiceDep,
    name: Annotated[str, Form()] = "",
    central_bank: Annotated[str, Form()] = "",
    role: Annotated[str, Form()] = "",
) -> Response:
    """Register a new speaker and render the result (htmx fragment)."""
    try:
        # Pydantic validates and coerces the raw form string to the CentralBank enum.
        form = SpeakerForm(name=name, central_bank=central_bank, role=role)
    except ValidationError as exc:
        return templates.TemplateResponse(
            request, "_speaker_created.html", {"error": _first_error(exc)}, status_code=422
        )
    speaker = speakers.register_speaker(
        name=form.name, central_bank=form.central_bank, role=form.role
    )
    return templates.TemplateResponse(request, "_speaker_created.html", {"speaker": speaker})


@router.post("/ui/ingest")
def ingest(
    request: Request,
    ingestion: IngestionServiceDep,
    indexing: IndexingServiceDep,
    speaker_id: Annotated[str, Form()] = "",
    title: Annotated[str, Form()] = "",
    url: Annotated[str, Form()] = "",
    delivered_on: Annotated[str, Form()] = "",
    text: Annotated[str, Form()] = "",
    language: Annotated[str, Form()] = "en",
) -> Response:
    """Ingest, analyze, and index a speech, then render the result (htmx fragment)."""
    try:
        # Pydantic validates and coerces the raw form strings (to UUID, date, etc.).
        form = IngestForm(
            speaker_id=speaker_id,
            title=title,
            url=url,
            delivered_on=delivered_on,
            text=text,
            language=language,
        )
    except ValidationError as exc:
        return templates.TemplateResponse(
            request, "_ingest_result.html", {"error": _first_error(exc)}, status_code=422
        )
    speech = ingestion.ingest_speech(
        speaker_id=form.speaker_id,
        title=form.title,
        url=form.url,
        delivered_at=form.delivered_at,
        text=form.text,
        language=form.language,
    )
    indexing.index_speech(speech.id)
    return templates.TemplateResponse(request, "_ingest_result.html", {"speech": speech})

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
    CentralBank,
    CommitteeMovement,
    MemberMovement,
    Speaker,
    Speech,
    ToneLabel,
    ToneObservation,
)
from cbt_web.dependencies import (
    CommitteeServiceDep,
    IndexingServiceDep,
    IngestionServiceDep,
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
    boards: list[BankBoard]
    recent: list[tuple[Speaker, Speech]]
    tone_series: CorpusToneSeries | None


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


def _corpus_overview(
    speakers: list[Speaker], tone: ToneServiceDep, ingestion: IngestionServiceDep
) -> CorpusOverview:
    """Aggregate corpus-wide stats and per-bank leaderboards from the per-speaker services.

    This iterates the speakers (a small set in this single-operator tool); a high-cardinality
    deployment would push these aggregates into a dedicated read model.
    """
    leaders: list[LeaderRow] = []
    buckets: dict[tuple[int, int], list[float]] = {}
    observation_total = 0
    for speaker in speakers:
        observations = tone.observations_for(speaker.id)
        if not observations:
            continue
        leaders.append(_leader_row(speaker, observations))
        observation_total += len(observations)
        for observation in observations:
            buckets.setdefault(
                (observation.observed_at.year, observation.observed_at.month), []
            ).append(observation.score)
    speech_total = 0
    flagged = 0
    span_start: datetime | None = None
    span_end: datetime | None = None
    recent: list[tuple[Speaker, Speech]] = []
    for speaker in speakers:
        speeches = ingestion.list_speeches(speaker.id)
        speech_total += len(speeches)
        flagged += sum(1 for speech in speeches if speech.needs_review)
        recent.extend((speaker, speech) for speech in speeches)
        for speech in speeches:
            delivered = speech.delivered_at
            span_start = delivered if span_start is None else min(span_start, delivered)
            span_end = delivered if span_end is None else max(span_end, delivered)
    recent.sort(key=lambda pair: pair[1].delivered_at, reverse=True)
    boards = _bank_boards(leaders)
    return CorpusOverview(
        speakers=len(speakers),
        speeches=speech_total,
        observations=observation_total,
        flagged=flagged,
        banks=len(boards),
        span_start=span_start,
        span_end=span_end,
        read=_desk_read(buckets, boards),
        boards=boards,
        recent=recent[:6],
        tone_series=_tone_series_from(buckets),
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
    all_speakers = speakers.list_speakers()
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
    boards = _bank_boards(_collect_leaders(speakers.list_speakers(), tone))
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


@router.get("/methodology")
def methodology(request: Request) -> Response:
    """Render the methodology page: how tone is scored and how well it is measured to work."""
    return templates.TemplateResponse(request, "methodology.html", {})


@router.get("/ui/speakers")
def search_speakers(request: Request, speakers: SpeakerServiceDep, q: str = "") -> Response:
    """Return the speaker-list fragment filtered by a name/institution query (htmx)."""
    matched = [speaker for speaker in speakers.list_speakers() if _matches(speaker, q)]
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
    return templates.TemplateResponse(
        request,
        "speech.html",
        {
            "speech": speech,
            "speaker": speaker,
            "movement": movement,
            "rows": rows,
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

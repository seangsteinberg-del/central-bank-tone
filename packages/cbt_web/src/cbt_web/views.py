"""Server-rendered views for the web UI (CLAUDE.md sections 2 and 3).

Each view validates its inputs, calls a service with typed values, and renders a template. Views
never touch a repository, the engine, or the logger directly; core exceptions propagate to the
handlers in ``errors.py``. Routes under ``/ui`` return HTML fragments for htmx to swap in; the
rest return full pages. The fragments degrade gracefully: the forms also submit as normal
requests when htmx (or JavaScript) is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
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

# SVG tone-chart geometry (a 720x210 viewBox with padding); the score axis runs +1 .. -1.
_CHART_W = 720.0
_CHART_H = 210.0
_PAD_L = 12.0
_PAD_R = 12.0
_PLOT_TOP = 18.0
_PLOT_BOTTOM = 184.0
_MID_Y = (_PLOT_TOP + _PLOT_BOTTOM) / 2.0
_HALF_H = (_PLOT_BOTTOM - _PLOT_TOP) / 2.0
_MAX_LABELS = 14


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


@dataclass(frozen=True)
class LeaderRow:
    """A speaker's latest tone reading, for the dashboard leaderboards."""

    speaker: Speaker
    score: float
    tone: ToneLabel
    count: int


@dataclass(frozen=True)
class CorpusOverview:
    """Aggregate corpus statistics for the dashboard."""

    speakers: int
    speeches: int
    observations: int
    flagged: int
    hawkish: list[LeaderRow]
    dovish: list[LeaderRow]
    recent: list[tuple[Speaker, Speech]]


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
    )


def _corpus_overview(
    speakers: list[Speaker], tone: ToneServiceDep, ingestion: IngestionServiceDep
) -> CorpusOverview:
    """Aggregate corpus-wide stats and leaderboards from the per-speaker services.

    This iterates the speakers (a small set in this single-operator tool); a high-cardinality
    deployment would push these aggregates into a dedicated read model.
    """
    speech_total = 0
    observation_total = 0
    flagged = 0
    leaders: list[LeaderRow] = []
    recent: list[tuple[Speaker, Speech]] = []
    for speaker in speakers:
        observations = tone.observations_for(speaker.id)
        speeches = ingestion.list_speeches(speaker.id)
        observation_total += len(observations)
        speech_total += len(speeches)
        flagged += sum(1 for speech in speeches if speech.needs_review)
        recent.extend((speaker, speech) for speech in speeches)
        if observations:
            latest = observations[-1]
            leaders.append(
                LeaderRow(
                    speaker=speaker, score=latest.score, tone=latest.tone, count=len(observations)
                )
            )
    recent.sort(key=lambda pair: pair[1].delivered_at, reverse=True)
    hawkish = sorted(
        (row for row in leaders if row.score > 0), key=lambda row: row.score, reverse=True
    )
    dovish = sorted((row for row in leaders if row.score < 0), key=lambda row: row.score)
    return CorpusOverview(
        speakers=len(speakers),
        speeches=speech_total,
        observations=observation_total,
        flagged=flagged,
        hawkish=hawkish[:6],
        dovish=dovish[:6],
        recent=recent[:6],
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
        {"speakers": all_speakers, "overview": overview, "query": ""},
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

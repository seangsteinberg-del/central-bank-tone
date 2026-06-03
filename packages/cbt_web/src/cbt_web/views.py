"""Server-rendered views for the web UI (CLAUDE.md sections 2 and 3).

Each view validates its inputs, calls a service with typed values, and renders a template. Views
never touch a repository, the engine, or the logger directly; core exceptions propagate to the
handlers in ``errors.py``. Routes under ``/ui`` return HTML fragments for htmx to swap in; the
rest return full pages. The fragments degrade gracefully: the forms also submit as normal
requests when htmx (or JavaScript) is unavailable.
"""

from __future__ import annotations

from typing import Annotated, NamedTuple
from uuid import UUID

from fastapi import APIRouter, Form, Request, Response
from pydantic import ValidationError

from cbt_core import CentralBank, Speaker, ToneLabel, ToneObservation
from cbt_web.dependencies import (
    IndexingServiceDep,
    IngestionServiceDep,
    QaServiceDep,
    SpeakerServiceDep,
    ToneServiceDep,
)
from cbt_web.schemas import AskForm, IngestForm, SpeakerForm
from cbt_web.templating import templates

router = APIRouter()


class TimelinePoint(NamedTuple):
    """One point on a speaker's tone-over-time chart, with its diverging-bar geometry."""

    observed_at: object
    tone: ToneLabel
    score: float
    left: float
    width: float


def _matches(speaker: Speaker, query: str) -> bool:
    """True if a speaker matches a name/institution search query (case-insensitive)."""
    needle = query.casefold().strip()
    if not needle:
        return True
    return needle in speaker.name.casefold() or needle in speaker.central_bank.value.casefold()


def _timeline(observations: list[ToneObservation]) -> list[TimelinePoint]:
    """Build diverging-bar points from observations: hawkish extends right, dovish left."""
    points: list[TimelinePoint] = []
    for observation in observations:
        bounded = max(-1.0, min(1.0, observation.score))
        half = abs(bounded) * 50.0
        left = 50.0 if bounded >= 0 else 50.0 - half
        points.append(
            TimelinePoint(
                observed_at=observation.observed_at,
                tone=observation.tone,
                score=observation.score,
                left=left,
                width=half,
            )
        )
    return points


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
def index(request: Request, speakers: SpeakerServiceDep) -> Response:
    """Render the landing page: the speaker directory and the corpus-wide ask box."""
    return templates.TemplateResponse(
        request, "index.html", {"speakers": speakers.list_speakers(), "query": ""}
    )


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
            "timeline": _timeline(observations),
            "latest_tone": observations[-1].tone if observations else None,
            "speeches": speeches,
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

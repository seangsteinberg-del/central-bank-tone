"""Tests for the keyless, Docker-less demo wiring (ADR 0014).

Verifies the whole offline stack end to end: seeding into SQLite, classifier tone scoring, the
in-memory retriever, the question-answering service, and the rendered web pages, all with no
Gemini key and no PostgreSQL.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from cbt_core import CentralBank, ToneLabel
from cbt_web.demo import SeedSpeech, build_demo_app, build_demo_services

_HAWKISH = SeedSpeech(
    speaker_name="Jane Hawk",
    central_bank=CentralBank.FEDERAL_RESERVE,
    role="Governor",
    title="On inflation and the policy rate",
    url="https://example.org/hawk",
    delivered_at=datetime(2022, 7, 1, tzinfo=UTC),
    text=(
        "Inflation remains far too high and price pressures are broad based. "
        "The committee is determined to raise interest rates and tighten policy further. "
        "We will keep raising rates until we are confident inflation is returning to target."
    ),
)
_DOVISH = SeedSpeech(
    speaker_name="John Dove",
    central_bank=CentralBank.FEDERAL_RESERVE,
    role="Governor",
    title="On growth and accommodation",
    url="https://example.org/dove",
    delivered_at=datetime(2020, 7, 1, tzinfo=UTC),
    text=(
        "Downside risks to growth have increased and there is considerable economic slack. "
        "We will cut interest rates and provide further accommodation to support employment. "
        "Policy will remain accommodative until the recovery is well established."
    ),
)


@pytest.mark.web
def test_demo_services_seed_score_and_retrieve() -> None:
    services = build_demo_services([_HAWKISH, _DOVISH])
    speakers = {s.name: s for s in services.speaker_service.list_speakers()}
    assert set(speakers) == {"Jane Hawk", "John Dove"}

    hawk_speeches = services.ingestion_service.list_speeches(speakers["Jane Hawk"].id)
    assert len(hawk_speeches) == 1
    assert hawk_speeches[0].tone is ToneLabel.HAWKISH
    assert hawk_speeches[0].model_id == "offline-classifier-v1"

    dove_speeches = services.ingestion_service.list_speeches(speakers["John Dove"].id)
    assert dove_speeches[0].tone is ToneLabel.DOVISH


@pytest.mark.web
def test_demo_corpus_question_answering_is_grounded() -> None:
    services = build_demo_services([_HAWKISH, _DOVISH])
    answer = services.qa_service.answer_corpus(question="What did they say about raising rates?")
    assert not answer.abstained
    assert answer.citations
    assert "extractive" in answer.text.lower()


@pytest.mark.web
def test_demo_app_renders_seeded_corpus() -> None:
    app = build_demo_app([_HAWKISH, _DOVISH])
    client = TestClient(app)

    index = client.get("/")
    assert index.status_code == 200
    assert "Jane Hawk" in index.text
    assert "John Dove" in index.text

    speaker = next(
        s for s in app.state.services.speaker_service.list_speakers() if s.name == "Jane Hawk"
    )
    page = client.get(f"/speakers/{speaker.id}")
    assert page.status_code == 200
    assert "On inflation and the policy rate" in page.text


@pytest.mark.web
def test_demo_app_answers_through_the_web_layer() -> None:
    client = TestClient(build_demo_app([_HAWKISH, _DOVISH]))
    response = client.post("/ui/ask", data={"question": "Who is worried about inflation?"})
    assert response.status_code == 200
    assert "extractive" in response.text.lower() or "could not find" in response.text.lower()

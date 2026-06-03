"""API tests against the in-process app with SQLite-backed services (CLAUDE.md section 5).

Covers the happy path plus a bad-input failure and a not-found failure for each resource.
Authentication is not implemented yet (tracked in ADR 0005); when it lands, an auth-failure
case is added here.
"""

from __future__ import annotations

import hashlib
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

_UNKNOWN = str(UUID(int=999))


def _create_speaker(client: TestClient, name: str = "Jerome Powell") -> str:
    response = client.post(
        "/speakers",
        json={"name": name, "central_bank": "federal_reserve", "role": "Chair"},
    )
    assert response.status_code == 201
    return str(response.json()["id"])


@pytest.mark.web
def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.web
def test_create_then_get_speaker(client: TestClient) -> None:
    speaker_id = _create_speaker(client)
    response = client.get(f"/speakers/{speaker_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == speaker_id
    assert body["central_bank"] == "federal_reserve"


@pytest.mark.web
def test_list_speakers_returns_registered(client: TestClient) -> None:
    _create_speaker(client, "Powell")
    _create_speaker(client, "Bailey")
    response = client.get("/speakers")
    assert response.status_code == 200
    assert {speaker["name"] for speaker in response.json()} == {"Powell", "Bailey"}


@pytest.mark.web
def test_get_unknown_speaker_returns_404(client: TestClient) -> None:
    assert client.get(f"/speakers/{_UNKNOWN}").status_code == 404


@pytest.mark.web
@pytest.mark.parametrize(
    "body",
    [
        {"name": "", "central_bank": "federal_reserve", "role": "Chair"},
        {"name": "x", "central_bank": "not_a_real_bank", "role": "Chair"},
        {"name": "x", "central_bank": "federal_reserve"},
    ],
)
def test_create_speaker_rejects_bad_body_with_422(client: TestClient, body: dict[str, str]) -> None:
    assert client.post("/speakers", json=body).status_code == 422


@pytest.mark.web
def test_record_then_list_observations(client: TestClient) -> None:
    speaker_id = _create_speaker(client)
    source_text = "we will keep policy restrictive"
    created = client.post(
        f"/speakers/{speaker_id}/tone-observations",
        json={"tone": "hawkish", "score": 0.7, "source_text": source_text},
    )
    assert created.status_code == 201
    assert created.json()["source_sha256"] == hashlib.sha256(source_text.encode()).hexdigest()

    listed = client.get(f"/speakers/{speaker_id}/tone-observations")
    assert listed.status_code == 200
    assert len(listed.json()) == 1


@pytest.mark.web
def test_record_observation_rejects_bad_score_with_422(client: TestClient) -> None:
    speaker_id = _create_speaker(client)
    response = client.post(
        f"/speakers/{speaker_id}/tone-observations",
        json={"tone": "hawkish", "score": 9.9, "source_text": "x"},
    )
    assert response.status_code == 422


@pytest.mark.web
def test_observations_for_unknown_speaker_returns_404(client: TestClient) -> None:
    assert (
        client.post(
            f"/speakers/{_UNKNOWN}/tone-observations",
            json={"tone": "neutral", "score": 0.0, "source_text": "x"},
        ).status_code
        == 404
    )
    assert client.get(f"/speakers/{_UNKNOWN}/tone-observations").status_code == 404


def _ingest(client: TestClient, speaker_id: str, *, text: str = "we will hold rates") -> dict:
    response = client.post(
        f"/speakers/{speaker_id}/speeches",
        json={
            "title": "On the outlook",
            "url": "https://example.org/speech/1",
            "delivered_at": "2026-01-15T00:00:00Z",
            "text": text,
        },
    )
    assert response.status_code == 201
    return response.json()


@pytest.mark.web
def test_ingest_then_list_speeches(client: TestClient) -> None:
    speaker_id = _create_speaker(client)
    body = _ingest(client, speaker_id)
    assert body["tone"] == "hawkish"  # the stub LLM client's fixed analysis
    assert body["summary"]
    assert "lexicon_score" in body

    listed = client.get(f"/speakers/{speaker_id}/speeches")
    assert listed.status_code == 200
    assert len(listed.json()) == 1


@pytest.mark.web
def test_ingest_speech_rejects_naive_datetime_with_422(client: TestClient) -> None:
    speaker_id = _create_speaker(client)
    response = client.post(
        f"/speakers/{speaker_id}/speeches",
        json={
            "title": "x",
            "url": "https://example.org/s/1",
            "delivered_at": "2026-01-15T00:00:00",  # no timezone
            "text": "text",
        },
    )
    assert response.status_code == 422


@pytest.mark.web
def test_ingest_speech_for_unknown_speaker_returns_404(client: TestClient) -> None:
    response = client.post(
        f"/speakers/{_UNKNOWN}/speeches",
        json={
            "title": "x",
            "url": "https://example.org/s/1",
            "delivered_at": "2026-01-15T00:00:00Z",
            "text": "text",
        },
    )
    assert response.status_code == 404


@pytest.mark.web
def test_ask_returns_a_grounded_answer_with_citations(client: TestClient) -> None:
    speaker_id = _create_speaker(client)
    response = client.post(f"/speakers/{speaker_id}/ask", json={"question": "Is the tone hawkish?"})
    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is False
    assert body["text"]
    assert len(body["citations"]) == 1
    assert body["citations"][0]["url"] == "https://example.org/s/900"


@pytest.mark.web
def test_ask_for_unknown_speaker_returns_404(client: TestClient) -> None:
    response = client.post(f"/speakers/{_UNKNOWN}/ask", json={"question": "anything?"})
    assert response.status_code == 404


@pytest.mark.web
def test_correlation_id_is_echoed_or_minted(client: TestClient) -> None:
    minted = client.get("/health")
    assert minted.headers.get("x-correlation-id")

    supplied = str(UUID(int=42))
    echoed = client.get("/health", headers={"X-Correlation-ID": supplied})
    assert echoed.headers["x-correlation-id"] == supplied

    malformed = client.get("/health", headers={"X-Correlation-ID": "not-a-uuid"})
    assert malformed.headers["x-correlation-id"] != "not-a-uuid"

"""Web UI tests against the in-process app with SQLite-backed services (CLAUDE.md section 5).

Covers the happy path for every screen plus a bad-input failure and a not-found failure, and the
server-error page. The web adapter renders HTML fragments for htmx; the assertions check status
codes and the rendered content rather than a JSON shape.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import Engine
from tests._stubs import StubChunkRetriever

from cbt_core import QaService, Settings, SpeakerService
from cbt_core.domain.qa import RetrievedChunk
from cbt_core.exceptions import LlmError
from cbt_core.services._support import IdFactory
from cbt_core.settings import Environment

_UNKNOWN = str(UUID(int=999))


def _register(
    client: TestClient, name: str = "Jerome Powell", bank: str = "federal_reserve"
) -> str:
    """Register a speaker through the UI and return the resulting speaker id."""
    response = client.post(
        "/ui/speakers", data={"name": name, "central_bank": bank, "role": "Chair"}
    )
    assert response.status_code == 200
    # The registration fragment links to the new speaker; its URL carries the id.
    return response.text.split('href="/speakers/', 1)[1].split('"', 1)[0]


def _ingest(client: TestClient, speaker_id: str) -> None:
    response = client.post(
        "/ui/ingest",
        data={
            "speaker_id": speaker_id,
            "title": "On the outlook",
            "url": "https://example.org/speech/1",
            "delivered_on": "2026-01-15",
            "text": "we will keep policy restrictive for some time",
            "language": "en",
        },
    )
    assert response.status_code == 200
    assert "Ingested and indexed" in response.text


@pytest.mark.web
def test_health_returns_ok(web_client: TestClient) -> None:
    response = web_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.web
def test_index_renders_with_corpus_ask_and_lists_speakers(web_client: TestClient) -> None:
    _register(web_client, "Christine Lagarde", "ecb")
    response = web_client.get("/")
    assert response.status_code == 200
    assert "Ask the corpus" in response.text
    assert "Christine Lagarde" in response.text


@pytest.mark.web
def test_speaker_search_fragment_filters_by_name(web_client: TestClient) -> None:
    _register(web_client, "Jerome Powell", "federal_reserve")
    _register(web_client, "Andrew Bailey", "bank_of_england")
    response = web_client.get("/ui/speakers", params={"q": "bailey"})
    assert response.status_code == 200
    assert "Andrew Bailey" in response.text
    assert "Jerome Powell" not in response.text


@pytest.mark.web
def test_speaker_search_with_empty_query_returns_all(web_client: TestClient) -> None:
    _register(web_client, "Jerome Powell", "federal_reserve")
    _register(web_client, "Andrew Bailey", "bank_of_england")
    response = web_client.get("/ui/speakers", params={"q": ""})
    assert response.status_code == 200
    assert "Jerome Powell" in response.text
    assert "Andrew Bailey" in response.text


@pytest.mark.web
def test_speaker_detail_shows_tone_timeline_and_speech(web_client: TestClient) -> None:
    speaker_id = _register(web_client)
    _ingest(web_client, speaker_id)
    response = web_client.get(f"/speakers/{speaker_id}")
    assert response.status_code == 200
    assert "Tone over time" in response.text
    assert "On the outlook" in response.text  # the ingested speech title
    assert "tone-badge" in response.text  # the tone badge rendered


@pytest.mark.web
def test_dashboard_shows_corpus_stats_and_recent_speech(web_client: TestClient) -> None:
    speaker_id = _register(web_client)
    _ingest(web_client, speaker_id)
    response = web_client.get("/")
    assert response.status_code == 200
    assert "Speeches analyzed" in response.text  # the stat strip
    assert "Recently analyzed" in response.text  # the recent-speeches section
    assert "On the outlook" in response.text  # the ingested speech surfaced on the dashboard


@pytest.mark.web
def test_speaker_tone_chart_renders_as_svg(web_client: TestClient) -> None:
    speaker_id = _register(web_client)
    _ingest(web_client, speaker_id)
    response = web_client.get(f"/speakers/{speaker_id}")
    assert response.status_code == 200
    assert "tone-chart" in response.text
    assert "<svg" in response.text  # a real inline SVG chart, not CSS bars


@pytest.mark.web
def test_methodology_page_reports_measured_accuracy(web_client: TestClient) -> None:
    response = web_client.get("/methodology")
    assert response.status_code == 200
    assert "Macro-F1" in response.text
    assert "59.9%" in response.text  # the supervised classifier's measured accuracy
    assert "/static/img/tone-vs-rates.png" in response.text  # the embedded research chart


@pytest.mark.web
def test_speaker_detail_unknown_returns_404_page(web_client: TestClient) -> None:
    response = web_client.get(f"/speakers/{_UNKNOWN}")
    assert response.status_code == 404
    assert "Not found" in response.text


@pytest.mark.web
def test_corpus_ask_returns_a_grounded_answer_fragment(web_client: TestClient) -> None:
    response = web_client.post("/ui/ask", data={"question": "Who is most hawkish?"})
    assert response.status_code == 200
    assert "answer" in response.text.lower()
    assert "https://example.org/s/900" in response.text  # the stub retriever's citation


@pytest.mark.web
def test_speaker_ask_returns_a_grounded_answer_fragment(web_client: TestClient) -> None:
    speaker_id = _register(web_client)
    response = web_client.post(
        f"/ui/speakers/{speaker_id}/ask", data={"question": "Is the tone hawkish?"}
    )
    assert response.status_code == 200
    assert "https://example.org/s/900" in response.text


@pytest.mark.web
def test_ask_with_empty_question_returns_422(web_client: TestClient) -> None:
    response = web_client.post("/ui/ask", data={"question": ""})
    assert response.status_code == 422
    assert "Please enter a question" in response.text


@pytest.mark.web
def test_speaker_ask_with_empty_question_returns_422(web_client: TestClient) -> None:
    speaker_id = _register(web_client)
    response = web_client.post(f"/ui/speakers/{speaker_id}/ask", data={"question": ""})
    assert response.status_code == 422
    assert "Please enter a question" in response.text


@pytest.mark.web
def test_correlation_id_is_echoed_or_minted(web_client: TestClient) -> None:
    minted = web_client.get("/health")
    assert minted.headers.get("x-correlation-id")

    supplied = str(UUID(int=42))
    echoed = web_client.get("/health", headers={"X-Correlation-ID": supplied})
    assert echoed.headers["x-correlation-id"] == supplied

    malformed = web_client.get("/health", headers={"X-Correlation-ID": "not-a-uuid"})
    assert malformed.headers["x-correlation-id"] != "not-a-uuid"


@pytest.mark.web
def test_register_speaker_rejects_bad_bank_with_422(web_client: TestClient) -> None:
    response = web_client.post(
        "/ui/speakers", data={"name": "x", "central_bank": "not_a_bank", "role": "Chair"}
    )
    assert response.status_code == 422


@pytest.mark.web
def test_ingest_rejects_bad_speaker_id_with_422(web_client: TestClient) -> None:
    response = web_client.post(
        "/ui/ingest",
        data={
            "speaker_id": "not-a-uuid",
            "title": "x",
            "url": "https://example.org/s/1",
            "delivered_on": "2026-01-15",
            "text": "text",
        },
    )
    assert response.status_code == 422


@pytest.mark.web
def test_admin_page_renders_forms(web_client: TestClient) -> None:
    response = web_client.get("/admin")
    assert response.status_code == 200
    assert "Register a speaker" in response.text
    assert "Ingest a speech" in response.text


@pytest.mark.web
@pytest.mark.parametrize(
    "asset",
    ["/static/app.css", "/static/vendor/htmx.min.js", "/static/img/tone-vs-rates.png"],
)
def test_static_assets_are_served(web_client: TestClient, asset: str) -> None:
    response = web_client.get(asset)
    assert response.status_code == 200
    assert response.content  # the vendored asset is reachable and non-empty


class _FailingLlm:
    """An LLM client whose every call raises, to exercise the 500 error page."""

    def analyze_tone(self, speech_text: str) -> object:
        raise LlmError("boom")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise LlmError("boom")

    def answer(self, question: str, chunks: Sequence[RetrievedChunk]) -> str:
        raise LlmError("boom")


@pytest.fixture
def failing_web_client(
    dummy_settings: Settings,
    sqlite_engine: Engine,
    speaker_service: SpeakerService,
    tone_service: object,
    ingestion_service: object,
    indexing_service: object,
    id_factory: IdFactory,
) -> Iterator[TestClient]:
    """A web client whose Q&A service raises an LlmError, to render the server-error page."""
    from cbt_web.app import create_app
    from cbt_web.dependencies import Services as WebServices

    qa = QaService(_FailingLlm(), StubChunkRetriever([]), speaker_service)
    app = create_app(dummy_settings)
    app.state.services = WebServices(
        settings=dummy_settings,
        engine=sqlite_engine,
        speaker_service=speaker_service,
        tone_service=tone_service,  # type: ignore[arg-type]  # unused on this path
        ingestion_service=ingestion_service,  # type: ignore[arg-type]  # unused on this path
        indexing_service=indexing_service,  # type: ignore[arg-type]  # unused on this path
        qa_service=qa,
    )
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.mark.web
def test_core_error_renders_server_error_page(failing_web_client: TestClient) -> None:
    response = failing_web_client.post("/ui/ask", data={"question": "anything?"})
    assert response.status_code == 500
    assert "Server error" in response.text


@pytest.mark.web
def test_app_boots_without_a_gemini_key() -> None:
    # The reviewer's "crashes on startup" scenario: build the app with an empty Gemini key.
    from cbt_web.app import create_app

    settings = Settings(
        environment=Environment.DEVELOPMENT,
        database_url="sqlite://",
        secret_key=SecretStr("dev-secret-for-tests"),
        gemini_api_key=SecretStr(""),
    )
    app = create_app(settings)  # must not raise: the Gemini client is built lazily, on first use
    assert app.title == "Central Bank Tone"

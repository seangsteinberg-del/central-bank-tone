"""Web UI tests against the in-process app with SQLite-backed services (CLAUDE.md section 5).

Covers the happy path for every screen plus a bad-input failure and a not-found failure, and the
server-error page. The web adapter renders HTML fragments for htmx; the assertions check status
codes and the rendered content rather than a JSON shape.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import Engine
from tests._stubs import StubChunkRetriever

from cbt_core import (
    CentralBank,
    Divergence,
    MonthlySeries,
    QaService,
    Settings,
    SignalVsMarket,
    Speaker,
    SpeakerService,
    Speech,
    ToneLabel,
    ToneObservation,
)
from cbt_core.domain.qa import RetrievedChunk
from cbt_core.exceptions import LlmError
from cbt_core.services._support import IdFactory
from cbt_core.settings import Environment
from cbt_web.templating import _clean_title
from cbt_web.views import (
    PolicyMonitorRow,
    RecentSpeech,
    Spark,
    _age_label,
    _bank_stance_series,
    _dedup_recent,
    _movers,
    _recent_rows_for,
    _solo_speakers,
    _stance_delta,
    _svm_chart,
)


def _obs(score: float, year: int, month: int) -> ToneObservation:
    """Build a tone observation at a month, for unit-testing the stance-series maths directly."""
    return ToneObservation(
        id=UUID(int=0),
        speaker_id=UUID(int=0),
        observed_at=datetime(year, month, 15, tzinfo=UTC),
        tone=ToneLabel.NEUTRAL,
        score=score,
        source_sha256="a" * 64,
    )


def _mrow(bank: CentralBank, code: str, now: float, delta_1m: float | None) -> PolicyMonitorRow:
    """Build a minimal monitor row for unit-testing the movers aggregation."""
    return PolicyMonitorRow(
        bank=bank,
        label=bank.value.replace("_", " ").title(),
        code=code,
        palette=0,
        now=now,
        delta_1m=delta_1m,
        delta_3m=None,
        spark=Spark(width=240.0, height=30.0, zero_y=15.0, polyline="", last_x=0.0, last_y=0.0),
        months=3,
        hawk_count=1,
        dove_count=0,
        divided=False,
        last_spoke=None,
        age=None,
    )


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
def test_dashboard_shows_policy_monitor_and_recent_speech(web_client: TestClient) -> None:
    speaker_id = _register(web_client)
    _ingest(web_client, speaker_id)
    response = web_client.get("/")
    assert response.status_code == 200
    assert "Policy monitor" in response.text  # the monitor matrix is on the dashboard
    assert "Most hawkish" in response.text  # the market-KPI strip replaced the vanity stats
    assert "Latest speeches" in response.text  # the scannable recent-speeches feed
    assert "On the outlook" in response.text  # the ingested speech surfaced on the dashboard
    assert "first analyzed speech" in response.text  # one speech has no prior, so no fake change


@pytest.mark.web
def test_latest_feed_shows_a_summary_and_the_change_versus_the_prior_speech(
    web_client: TestClient,
) -> None:
    speaker_id = _register(web_client)
    _ingest_named(
        web_client,
        speaker_id,
        title="First remarks",
        url="https://example.org/a/1",
        delivered_on="2026-01-10",
        text="we will keep policy restrictive to bring inflation down",
    )
    _ingest_named(
        web_client,
        speaker_id,
        title="Second remarks",
        url="https://example.org/a/2",
        delivered_on="2026-03-10",
        text="the committee now sees room to ease as growth slows and prices cool",
    )
    response = web_client.get("/")
    assert response.status_code == 200
    assert "Latest speeches" in response.text
    assert "speech-summary" in response.text  # each card carries a one-line summary to scan
    assert "change-chip" in response.text  # the newer speech shows its change versus the prior
    assert "first analyzed speech" in response.text  # the earliest speech has no fabricated change


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
    assert "ECE = 0.142" in response.text  # the calibration metric is surfaced
    assert "/static/img/tone-reliability.png" in response.text  # the reliability diagram


def _ingest_named(
    client: TestClient, speaker_id: str, *, title: str, url: str, delivered_on: str, text: str
) -> None:
    """Ingest a second, distinct speech for a speaker (a unique text avoids the dedup no-op)."""
    response = client.post(
        "/ui/ingest",
        data={
            "speaker_id": speaker_id,
            "title": title,
            "url": url,
            "delivered_on": delivered_on,
            "text": text,
            "language": "en",
        },
    )
    assert response.status_code == 200
    assert "Ingested and indexed" in response.text


def _latest_speech_id(client: TestClient, speaker_id: str) -> str:
    """Return the most recent speech id from a speaker page (speeches are listed newest first)."""
    page = client.get(f"/speakers/{speaker_id}").text
    assert "/speeches/" in page
    return page.split("/speeches/", 1)[1].split('"', 1)[0]


@pytest.mark.web
def test_speaker_speech_title_links_to_its_detail_page(web_client: TestClient) -> None:
    speaker_id = _register(web_client)
    _ingest(web_client, speaker_id)
    response = web_client.get(f"/speakers/{speaker_id}")
    assert response.status_code == 200
    assert "/speeches/" in response.text  # the speech title is a link into the detail page


def _two_bank_corpus(client: TestClient) -> None:
    """Register a Fed and an ECB speaker and give each a distinct (non-deduped) scored speech."""
    fed = _register(client, "Jerome Powell", "federal_reserve")
    ecb = _register(client, "Christine Lagarde", "ecb")
    _ingest_named(
        client,
        fed,
        title="Fed remarks",
        url="https://example.org/f/1",
        delivered_on="2026-02-01",
        text="we will keep policy restrictive to bring inflation down",
    )
    _ingest_named(
        client,
        ecb,
        title="ECB remarks",
        url="https://example.org/e/1",
        delivered_on="2026-02-02",
        text="the council sees room to ease as growth slows and prices cool",
    )


@pytest.mark.web
def test_dashboard_groups_committees_by_bank_with_a_toggle(web_client: TestClient) -> None:
    _two_bank_corpus(web_client)
    response = web_client.get("/")
    assert response.status_code == 200
    assert "Committee tone by bank" in response.text  # the per-bank section, not a global pool
    assert "/ui/leaderboard?bank=federal_reserve" in response.text  # one bank tab
    assert "/ui/leaderboard?bank=ecb" in response.text  # the toggle's other bank tab


@pytest.mark.web
def test_dashboard_masthead_is_a_data_derived_read_not_marketing(web_client: TestClient) -> None:
    _two_bank_corpus(web_client)
    response = web_client.get("/")
    assert response.status_code == 200
    assert "corpus mood" in response.text  # the data-derived desk read, honestly labelled
    assert "central-bank chorus" in response.text  # the computed one-line headline
    assert "speeches move markets" not in response.text  # the old landing-page pitch is gone


@pytest.mark.web
def test_chrome_reflects_the_live_gemini_backend_not_offline(web_client: TestClient) -> None:
    # The served instance scores with Gemini, so the chrome must not claim the weaker offline mode;
    # mislabelling it would tell a reader the numbers came from a local classifier (credibility bug).
    response = web_client.get("/")
    assert response.status_code == 200
    assert "Gemini judge" in response.text
    assert "tone scored locally" not in response.text
    assert "offline: local classifier" not in response.text


@pytest.mark.web
def test_signal_vs_market_renders_honest_unavailable_state_not_a_500(
    web_client: TestClient,
) -> None:
    # With too little Federal Reserve history the page must render an explanatory panel, never a 500
    # or an empty chart (CLAUDE.md section 3). The thin test corpus trips the insufficient-data path.
    response = web_client.get("/signal-vs-market")
    assert response.status_code == 200
    assert "Signal vs Market" in response.text
    assert "qualifying" in response.text  # the honest insufficient-data message


def _svm_fixture(months: int) -> SignalVsMarket:
    """A SignalVsMarket with ``months`` of headline tone and a 2-year series, for chart geometry."""
    start = 2024 * 12
    headline = {start + i: round(-0.4 + i * 0.05, 2) for i in range(months)}
    gs2 = {start + i: round(4.5 - i * 0.04, 2) for i in range(months)}
    return SignalVsMarket(
        central_bank=CentralBank.FEDERAL_RESERVE,
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
            points=headline,
        ),
        rate_series=(
            MonthlySeries(
                label="2-year Treasury", code="GS2", is_rate=True, is_market_proxy=True, points=gs2
            ),
        ),
        correlations=(),
        divergence=Divergence(
            tone_change_3m=0.15,
            market_change_bp_3m=-12.0,
            tone_direction="more hawkish",
            market_direction="repricing lower",
            aligned=False,
            headline="diverging",
        ),
        span_start=start,
        span_end=start + months - 1,
        months=months,
        speeches=months * 3,
        corpus_through="2024-12",
        market_through="2024-12",
        min_speeches_per_month=3,
    )


@pytest.mark.unit
def test_svm_chart_builds_two_polylines_over_the_month_span() -> None:
    chart = _svm_chart(_svm_fixture(14))
    assert chart is not None
    assert chart.tone.points  # the tone series is plotted
    assert chart.rate.points  # the rate series is plotted
    assert len(chart.month_labels) >= 1
    assert chart.left_ticks
    assert chart.right_ticks


@pytest.mark.unit
def test_svm_chart_is_none_when_too_sparse_to_plot() -> None:
    assert _svm_chart(_svm_fixture(1)) is None


@pytest.mark.unit
def test_clean_title_strips_only_a_redundant_speaker_name_prefix() -> None:
    assert _clean_title("Jerome Powell: Economic outlook", "Jerome Powell") == "Economic outlook"
    # A real title that merely contains a colon is left intact (no over-stripping).
    assert _clean_title("Inflation: the road ahead", "Jerome Powell") == "Inflation: the road ahead"


@pytest.mark.unit
def test_solo_speakers_hides_joint_statement_rows() -> None:
    solo = Speaker(
        id=UUID(int=1), name="Jerome Powell", central_bank=CentralBank.FEDERAL_RESERVE, role="Chair"
    )
    joint = Speaker(
        id=UUID(int=2),
        name="Thomas Jordan; Martin Schlegel",
        central_bank=CentralBank.SWISS_NATIONAL_BANK,
        role="Board",
    )
    assert _solo_speakers([solo, joint]) == [solo]


@pytest.mark.web
def test_leaderboard_fragment_shows_only_the_selected_banks_speakers(
    web_client: TestClient,
) -> None:
    _two_bank_corpus(web_client)
    response = web_client.get("/ui/leaderboard", params={"bank": "federal_reserve"})
    assert response.status_code == 200
    assert "Federal Reserve committee" in response.text
    assert "Jerome Powell" in response.text  # the selected bank's member
    assert "Christine Lagarde" not in response.text  # an ECB member is absent from the Fed board


@pytest.mark.web
def test_leaderboard_rejects_unknown_bank_with_422(web_client: TestClient) -> None:
    response = web_client.get("/ui/leaderboard", params={"bank": "not_a_bank"})
    assert response.status_code == 422


@pytest.mark.web
def test_dashboard_charts_per_bank_tone_index_over_time(web_client: TestClient) -> None:
    fed = _register(web_client, "Jerome Powell", "federal_reserve")
    # Two months for one bank, so its tone-index line is a real polyline, not a lone dot.
    _ingest_named(
        web_client,
        fed,
        title="January remarks",
        url="https://example.org/f/jan",
        delivered_on="2026-01-10",
        text="we will keep policy restrictive to bring inflation down",
    )
    _ingest_named(
        web_client,
        fed,
        title="March remarks",
        url="https://example.org/f/mar",
        delivered_on="2026-03-12",
        text="inflation has eased so we can be patient and gradual from here",
    )
    response = web_client.get("/")
    assert response.status_code == 200
    assert "Policy divergence over time" in response.text  # the per-bank tone-index chart
    assert "bank-history" in response.text  # the multi-line divergence SVG
    assert 'class="bank-line"' in response.text  # two months render as a connected line
    assert "bank-line-label" in response.text  # each line is named on the chart, not colour-only
    assert "FED" in response.text  # the Fed line's end-of-line desk code label


@pytest.mark.web
def test_divergence_chart_leaves_honest_gaps_when_a_bank_is_silent_a_month(
    web_client: TestClient,
) -> None:
    fed = _register(web_client, "Jerome Powell", "federal_reserve")
    ecb = _register(web_client, "Christine Lagarde", "ecb")
    # The Fed speaks in January and March; the ECB only in February. The shared month axis spans
    # all three, so each bank's line skips the months it was silent rather than inventing a value.
    _ingest_named(
        web_client,
        fed,
        title="Fed January",
        url="https://example.org/f/jan",
        delivered_on="2026-01-10",
        text="we will keep policy restrictive to bring inflation down",
    )
    _ingest_named(
        web_client,
        ecb,
        title="ECB February",
        url="https://example.org/e/feb",
        delivered_on="2026-02-14",
        text="the council sees room to ease as growth slows and prices cool",
    )
    _ingest_named(
        web_client,
        fed,
        title="Fed March",
        url="https://example.org/f/mar",
        delivered_on="2026-03-12",
        text="inflation has eased so we can be patient and gradual from here",
    )
    response = web_client.get("/")
    assert response.status_code == 200
    assert "Policy divergence over time" in response.text
    # Both banks are charted, each as its own labelled line, over the shared three-month axis.
    assert "FED" in response.text  # the Fed line's end-of-line code
    assert "ECB" in response.text  # the ECB line's end-of-line code


@pytest.mark.web
def test_dashboard_surfaces_flagged_model_lexicon_disagreements(web_client: TestClient) -> None:
    fed = _register(web_client, "Jerome Powell", "federal_reserve")
    # The stub model scores every speech +0.60 (hawkish); this text is purely dovish, so the
    # deterministic lexicon scores it negative and the opposite-sign cross-check flags the split.
    _ingest_named(
        web_client,
        fed,
        title="A dovish pivot",
        url="https://example.org/f/dove",
        delivered_on="2026-02-01",
        text="we expect rate cuts and policy easing as disinflation and downside risks broaden",
    )
    response = web_client.get("/")
    assert response.status_code == 200
    assert "Model / lexicon disagreements" in response.text  # the review panel
    assert "A dovish pivot" in response.text  # the flagged speech is listed for review
    assert 'id="flagged"' in response.text  # the flagged stat links down to the panel


@pytest.mark.web
def test_committee_sparkline_links_to_the_speaker_page(web_client: TestClient) -> None:
    _two_bank_corpus(web_client)
    response = web_client.get("/ui/leaderboard", params={"bank": "federal_reserve"})
    assert response.status_code == 200
    assert "spark-link" in response.text  # the inline sparkline is itself a deep link
    assert 'href="/speakers/' in response.text  # pointing at the member's speaker page


@pytest.mark.web
def test_every_central_bank_has_a_distinct_chart_colour_and_code() -> None:
    # The divergence chart assigns each bank a stable palette slot by registry order. If the
    # registry ever grows past the palette, two banks would silently share a colour. Assert the
    # limit holds here so it fails loudly in CI rather than degrading the chart (CLAUDE.md s.3).
    from cbt_core import CentralBank
    from cbt_web.views import _BANK_PALETTE, _PALETTE_SIZE, _bank_code

    banks = list(CentralBank)
    assert len(banks) <= _PALETTE_SIZE  # the palette covers the whole registry, no silent reuse
    slots = [_BANK_PALETTE[bank] for bank in banks]
    assert len(set(slots)) == len(banks)  # every bank maps to a distinct colour slot
    assert all(0 <= slot < _PALETTE_SIZE for slot in slots)
    assert all(_bank_code(bank) for bank in banks)  # every bank has a non-empty desk code


@pytest.mark.web
def test_dashboard_leads_with_the_policy_monitor_and_market_kpis(web_client: TestClient) -> None:
    _two_bank_corpus(web_client)
    response = web_client.get("/")
    assert response.status_code == 200
    assert "Policy monitor" in response.text  # the matrix is the hero, not the vanity stats
    assert "monitor-board" in response.text
    assert "Most hawkish" in response.text  # the market-KPI strip
    assert "Most dovish" in response.text
    assert "/ui/monitor?sort=delta_1m" in response.text  # the columns are sortable


@pytest.mark.web
def test_monitor_fragment_sorts_and_rejects_an_unknown_key(web_client: TestClient) -> None:
    _two_bank_corpus(web_client)
    ok = web_client.get("/ui/monitor", params={"sort": "delta_1m"})
    assert ok.status_code == 200
    assert "monitor-board" in ok.text
    bad = web_client.get("/ui/monitor", params={"sort": "not_a_column"})
    assert bad.status_code == 422


@pytest.mark.web
def test_monitor_shows_an_em_dash_when_there_is_no_prior_reading(web_client: TestClient) -> None:
    speaker_id = _register(web_client)
    _ingest(web_client, speaker_id)  # one speech, so there is no month to compare against
    response = web_client.get("/ui/monitor")
    assert response.status_code == 200
    assert "—" in response.text  # the em dash: an honest "no prior reading", not a fake 0.00


@pytest.mark.web
def test_monitor_shows_a_freshness_label_for_when_the_committee_last_spoke(
    web_client: TestClient,
) -> None:
    speaker_id = _register(web_client)
    _ingest(web_client, speaker_id)
    response = web_client.get("/ui/monitor")
    assert response.status_code == 200
    # The row carries a "last spoke" freshness label so a stale committee is not read as current.
    # The exact age depends on the wall clock, so assert the element renders, not its value.
    assert "mono-age" in response.text
    assert "last spoke" in response.text  # the title tooltip with the exact date


def test_bank_stance_series_carries_each_members_latest_reading_forward() -> None:
    # One member speaks in January (+0.20) and again in March (+0.60), silent in February.
    series = _bank_stance_series([[_obs(0.2, 2026, 1), _obs(0.6, 2026, 3)]])
    values = {mv.key: mv.value for mv in series}
    assert sorted(values) == [(2026, 1), (2026, 2), (2026, 3)]  # February is filled, not skipped
    assert (
        values[(2026, 2)] == 0.2
    )  # February carries January's reading forward (the standing tone)
    assert values[(2026, 3)] == 0.6


def test_stance_delta_is_none_without_a_reading_that_far_back() -> None:
    series = _bank_stance_series([[_obs(0.6, 2026, 3)]])  # a single month
    assert _stance_delta(series, 1) is None  # honest "no comparison", never a fabricated 0.00
    assert _stance_delta(series, 3) is None


def test_stance_delta_measures_the_one_and_three_month_change() -> None:
    series = _bank_stance_series([[_obs(-0.2, 2026, 1), _obs(0.0, 2026, 2), _obs(0.5, 2026, 4)]])
    assert _stance_delta(series, 1) == 0.5  # April 0.5 minus March 0.0 (carried from February)
    assert _stance_delta(series, 3) == 0.7  # April 0.5 minus January -0.2


def test_movers_orders_by_size_and_labels_the_direction() -> None:
    rows = [
        _mrow(CentralBank.ECB, "ECB", now=0.10, delta_1m=0.05),
        _mrow(CentralBank.BANK_OF_ENGLAND, "BoE", now=0.50, delta_1m=0.40),
        _mrow(CentralBank.FEDERAL_RESERVE, "FED", now=-0.30, delta_1m=-0.45),
    ]
    movers = _movers(rows)
    assert [m.code for m in movers] == ["FED", "BoE", "ECB"]  # largest absolute move first
    assert movers[0].side == "dove"
    assert movers[0].word == "turning dovish"  # crossed down from +0.15
    assert movers[1].side == "hawk"
    assert movers[1].word == "extending hawkish"  # was already at +0.10
    assert movers[0].prev == 0.15  # now (-0.30) minus delta (-0.45)


def _speaker(name: str = "Jerome Powell") -> Speaker:
    """Build a minimal speaker for unit-testing the recent-feed maths directly."""
    return Speaker(
        id=UUID(int=0), name=name, central_bank=CentralBank.FEDERAL_RESERVE, role="Chair"
    )


def _speech(score: float, *, year: int, month: int, url: str) -> Speech:
    """Build a minimal scored speech for unit-testing the recent-feed maths directly."""
    return Speech(
        id=UUID(int=0),
        speaker_id=UUID(int=0),
        central_bank=CentralBank.FEDERAL_RESERVE,
        title="Remarks on the outlook",
        url=url,
        delivered_at=datetime(year, month, 15, tzinfo=UTC),
        text="placeholder speech text",
        source_sha256="a" * 64,
        summary="A concise one-line summary of the remarks.",
        tone=ToneLabel.NEUTRAL,
        score=score,
        lexicon_score=0.0,
        rationale="because the data warranted it",
    )


def test_recent_rows_for_reports_the_change_versus_the_speakers_prior_speech() -> None:
    speaker = _speaker()
    # Newest first, as the service returns them: Mar +0.60, Feb +0.20, Jan +0.20.
    speeches = [
        _speech(0.60, year=2026, month=3, url="https://x/3"),
        _speech(0.20, year=2026, month=2, url="https://x/2"),
        _speech(0.20, year=2026, month=1, url="https://x/1"),
    ]
    rows = _recent_rows_for(speaker, speeches)
    assert rows[0].delta == 0.40  # March 0.60 minus February 0.20
    assert rows[0].side == "hawk"
    assert "more hawkish" in rows[0].word
    assert rows[1].delta == 0.0  # February equals January: a measured no-change
    assert rows[1].side == "flat"
    assert "in line" in rows[1].word
    assert rows[2].delta is None  # the earliest speech has no prior to compare against
    assert rows[2].side == "none"
    assert rows[2].word == "first analyzed speech"  # honest, not a fabricated 0.00


def test_dedup_recent_keeps_the_newest_per_url_and_caps_to_the_limit() -> None:
    speaker = _speaker()
    rows = [
        RecentSpeech(
            speaker=speaker,
            speech=_speech(0.1, year=2026, month=4, url="https://x/dup"),
            delta=None,
            side="none",
            word="first analyzed speech",
        ),
        RecentSpeech(
            speaker=speaker,
            speech=_speech(0.2, year=2026, month=1, url="https://x/dup"),
            delta=None,
            side="none",
            word="first analyzed speech",
        ),
        RecentSpeech(
            speaker=speaker,
            speech=_speech(0.3, year=2026, month=3, url="https://x/other"),
            delta=None,
            side="none",
            word="first analyzed speech",
        ),
    ]
    deduped = _dedup_recent(rows, limit=10)
    # Sorted newest first; the April row wins the duplicate URL, the January one is dropped.
    assert [r.speech.url for r in deduped] == ["https://x/dup", "https://x/other"]
    assert len(_dedup_recent(rows, limit=1)) == 1  # the cap is honoured


def test_age_label_maps_elapsed_time_to_a_short_freshness_label() -> None:
    now = datetime(2026, 6, 9, 12, tzinfo=UTC)
    assert _age_label(None, now) is None  # no reading -> no label, never a fake "now"
    assert _age_label(datetime(2026, 6, 9, 1, tzinfo=UTC), now) == "today"
    assert _age_label(datetime(2026, 6, 8, 1, tzinfo=UTC), now) == "1d ago"
    assert _age_label(datetime(2026, 6, 1, 12, tzinfo=UTC), now) == "8d ago"
    assert _age_label(datetime(2026, 5, 12, 12, tzinfo=UTC), now) == "4w ago"
    assert _age_label(datetime(2026, 1, 9, 12, tzinfo=UTC), now) == "5mo ago"
    assert _age_label(datetime(2023, 6, 9, 12, tzinfo=UTC), now) == "3y ago"


def test_age_label_reads_today_for_a_future_timestamp() -> None:
    # Clock skew or a future-dated speech must not render a negative age.
    now = datetime(2026, 6, 9, 12, tzinfo=UTC)
    assert _age_label(datetime(2026, 6, 10, 12, tzinfo=UTC), now) == "today"


@pytest.mark.web
def test_speech_detail_shows_summary_and_committee_movement(web_client: TestClient) -> None:
    speaker_id = _register(web_client)
    _ingest(web_client, speaker_id)
    speech_id = _latest_speech_id(web_client, speaker_id)
    response = web_client.get(f"/speeches/{speech_id}")
    assert response.status_code == 200
    assert "In brief" in response.text  # the concise-summary section
    assert "How the committee has moved" in response.text
    assert "Committee standing tone" in response.text
    assert "first analyzed speech" in response.text  # a single reading has no prior shift
    assert "move-track" in response.text  # the per-member movement bar is rendered


@pytest.mark.web
def test_speech_detail_shows_the_policy_decomposition(web_client: TestClient) -> None:
    speaker_id = _register(web_client)
    _ingest(web_client, speaker_id)
    speech_id = _latest_speech_id(web_client, speaker_id)
    response = web_client.get(f"/speeches/{speech_id}")
    assert response.status_code == 200
    # The structured pipeline (ADR 0021) decomposition is surfaced on the speech page.
    assert "Policy decomposition" in response.text
    assert "Rate-path intent" in response.text
    assert "Cross-checks" in response.text


@pytest.mark.web
def test_speech_detail_draws_member_shift_with_a_prior_reading(web_client: TestClient) -> None:
    speaker_id = _register(web_client)
    _ingest(web_client, speaker_id)  # delivered 2026-01-15
    _ingest_named(
        web_client,
        speaker_id,
        title="A later remark",
        url="https://example.org/speech/2",
        delivered_on="2026-03-20",
        text="inflation has eased so we can be patient on any further policy moves",
    )
    speech_id = _latest_speech_id(web_client, speaker_id)  # the later (most recent) speech
    response = web_client.get(f"/speeches/{speech_id}")
    assert response.status_code == 200
    assert "move-fill" in response.text  # a measured shift bar is drawn against the prior reading
    assert "versus their previous analyzed speech" in response.text  # the subject-move callout


@pytest.mark.web
def test_speech_detail_unknown_returns_404_page(web_client: TestClient) -> None:
    response = web_client.get(f"/speeches/{_UNKNOWN}")
    assert response.status_code == 404
    assert "Not found" in response.text


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
    [
        "/static/app.css",
        "/static/vendor/htmx.min.js",
        "/static/img/tone-vs-rates.png",
        "/static/img/tone-reliability.png",
    ],
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
    committee_service: object,
    market_service: object,
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
        committee_service=committee_service,  # type: ignore[arg-type]  # unused on this path
        market_service=market_service,  # type: ignore[arg-type]  # unused on this path
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

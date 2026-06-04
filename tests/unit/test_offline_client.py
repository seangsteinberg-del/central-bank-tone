"""Tests for the keyless offline LLM-boundary implementation (ADR 0014)."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cbt_core import OfflineLlmClient, RetrievedChunk, ToneLabel
from cbt_core.domain.qa import EMBEDDING_DIM


@pytest.fixture(scope="module")
def client() -> OfflineLlmClient:
    return OfflineLlmClient()


@pytest.mark.unit
def test_analyze_tone_calls_hawkish_text_hawkish(client: OfflineLlmClient) -> None:
    analysis = client.analyze_tone(
        "The committee will raise interest rates and tighten policy to fight persistent inflation. "
        "Price pressures remain elevated and further hikes are likely."
    )
    assert analysis.tone is ToneLabel.HAWKISH
    assert analysis.score > 0
    assert analysis.summary
    assert "offline" in analysis.rationale.lower()


@pytest.mark.unit
def test_analyze_tone_calls_dovish_text_dovish(client: OfflineLlmClient) -> None:
    analysis = client.analyze_tone(
        "With downside risks and economic slack, we will cut rates and ease policy to support growth. "
        "Inflation remains subdued and accommodation is warranted."
    )
    assert analysis.tone is ToneLabel.DOVISH
    assert analysis.score < 0


@pytest.mark.unit
def test_analyze_tone_never_returns_mixed(client: OfflineLlmClient) -> None:
    # The three-class classifier cannot produce MIXED; the offline client must not invent it.
    analysis = client.analyze_tone("The economy expanded moderately over the period.")
    assert analysis.tone in {ToneLabel.HAWKISH, ToneLabel.DOVISH, ToneLabel.NEUTRAL}


@pytest.mark.unit
def test_summary_keeps_the_lead_and_is_bounded(client: OfflineLlmClient) -> None:
    sentences = [
        "Good afternoon and thank you for the introduction.",
        "Inflation has risen sharply and we will tighten policy aggressively.",
        "The labour market remains resilient by most measures.",
        "We discussed the international outlook at some length.",
        "Rate hikes will continue until price stability is restored.",
    ]
    analysis = client.analyze_tone(" ".join(sentences))
    # The lead sentence is always retained; the summary is shorter than the full speech.
    assert analysis.summary.startswith("Good afternoon")
    assert len(analysis.summary) < len(" ".join(sentences))


@pytest.mark.unit
def test_summary_returns_whole_text_when_short(client: OfflineLlmClient) -> None:
    short = "We will raise rates."
    assert client.analyze_tone(short).summary == short


@pytest.mark.unit
def test_embed_returns_unit_vectors_of_the_platform_dimension(client: OfflineLlmClient) -> None:
    vectors = client.embed(["inflation is rising", "the labour market is strong"])
    assert all(len(v) == EMBEDDING_DIM for v in vectors)
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0)


@pytest.mark.unit
def test_embed_is_deterministic_across_instances() -> None:
    text = "the committee will raise rates"
    assert OfflineLlmClient().embed([text]) == OfflineLlmClient().embed([text])


@pytest.mark.unit
def test_embed_empty_text_is_a_zero_vector(client: OfflineLlmClient) -> None:
    vector = client.embed([""])[0]
    assert len(vector) == EMBEDDING_DIM
    assert np.linalg.norm(vector) == 0.0


@pytest.mark.unit
def test_embedding_similarity_tracks_meaning(client: OfflineLlmClient) -> None:
    base, similar, different = client.embed(
        [
            "we will raise interest rates to fight inflation",
            "the committee will hike rates because inflation is high",
            "the weather is sunny today and the birds are singing",
        ]
    )
    base_arr = np.asarray(base)
    assert base_arr @ np.asarray(similar) > base_arr @ np.asarray(different)


def _chunk(text: str, title: str) -> RetrievedChunk:
    return RetrievedChunk(
        speech_id=uuid4(), chunk_index=0, text=text, title=title, url="https://x", distance=0.1
    )


@pytest.mark.unit
def test_answer_is_explicitly_extractive_and_grounded(client: OfflineLlmClient) -> None:
    chunks = [
        _chunk(
            "We will raise rates to fight inflation. The labour market stays strong.",
            "Speech on inflation",
        ),
    ]
    answer = client.answer("What did they say about inflation and rates?", chunks)
    assert "extractive" in answer.lower()
    assert "Speech on inflation" in answer  # cites the source title
    assert "raise rates" in answer  # grounded in the retrieved text


@pytest.mark.unit
def test_answer_falls_back_to_chunk_body_when_no_sentence_scores(client: OfflineLlmClient) -> None:
    # A chunk with only very short fragments still yields a grounded, labelled answer.
    answer = client.answer("anything", [_chunk("Hi. Ok.", "Tiny")])
    assert "extractive" in answer.lower()
    assert "Tiny" in answer

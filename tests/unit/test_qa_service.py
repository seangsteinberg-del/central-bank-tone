"""Tests for the question-answering service (CLAUDE.md sections 2, 3, 5)."""

from __future__ import annotations

from uuid import UUID

import pytest
from tests._stubs import StubLlmClient

from cbt_core import QaService, SpeakerService
from cbt_core.domain.qa import RetrievedChunk
from cbt_core.domain.registry import CentralBank
from cbt_core.exceptions import EntityNotFoundError
from cbt_core.llm.client import Embedding


class _StubRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.searches = 0

    def search(
        self, speaker_id: UUID, query_embedding: Embedding, top_k: int
    ) -> list[RetrievedChunk]:
        self.searches += 1
        return self._chunks[:top_k]


def _chunk(index: int) -> RetrievedChunk:
    return RetrievedChunk(
        speech_id=UUID(int=100 + index),
        chunk_index=index,
        text=f"excerpt {index}",
        title=f"Speech {index}",
        url=f"https://example.org/s/{index}",
        distance=0.1 * index,
    )


def _register(service: SpeakerService) -> UUID:
    speaker = service.register_speaker(
        name="Powell", central_bank=CentralBank.FEDERAL_RESERVE, role="Chair"
    )
    return speaker.id


@pytest.mark.unit
def test_answer_grounds_in_retrieved_chunks_and_cites_them(
    speaker_service: SpeakerService, stub_llm_client: StubLlmClient
) -> None:
    speaker_id = _register(speaker_service)
    retriever = _StubRetriever([_chunk(0), _chunk(1)])
    service = QaService(stub_llm_client, retriever, speaker_service)
    answer = service.answer(speaker_id=speaker_id, question="Is the speaker hawkish?")
    assert answer.abstained is False
    assert len(answer.citations) == 2
    assert {c.url for c in answer.citations} == {
        "https://example.org/s/0",
        "https://example.org/s/1",
    }
    assert stub_llm_client.answers == 1


@pytest.mark.unit
def test_answer_abstains_when_no_chunks_are_found(
    speaker_service: SpeakerService, stub_llm_client: StubLlmClient
) -> None:
    speaker_id = _register(speaker_service)
    service = QaService(stub_llm_client, _StubRetriever([]), speaker_service)
    answer = service.answer(speaker_id=speaker_id, question="anything?")
    assert answer.abstained is True
    assert answer.citations == ()
    assert stub_llm_client.answers == 0  # no answer call when abstaining


@pytest.mark.unit
def test_answer_for_unknown_speaker_raises_not_found(
    speaker_service: SpeakerService, stub_llm_client: StubLlmClient
) -> None:
    service = QaService(stub_llm_client, _StubRetriever([_chunk(0)]), speaker_service)
    with pytest.raises(EntityNotFoundError):
        service.answer(speaker_id=UUID(int=404), question="anything?")

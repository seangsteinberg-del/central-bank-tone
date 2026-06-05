"""The LlmClient protocol is runtime-checkable and every full client conforms to it.

This is the guard that the committee-test regression does not recur: a scripted test double had
fallen behind a newly added protocol method (``classify_sentences``, ADR 0021), so a whole test
module went red with an ``AttributeError`` at call time. Adding a method to
:class:`~cbt_core.llm.LlmClient` now fails this test for any full double that has not kept up,
which is a fast, central signal instead of a deep runtime failure.

The deliberately partial failure doubles (``_FailingLlm``, ``_RaisingLlm`` in other modules) are
not asserted here: they implement only the methods their failure-path test exercises, on purpose.
"""

from __future__ import annotations

import pytest
from tests._stubs import ScriptedLlmClient, StubLlmClient

from cbt_core import LlmClient, OfflineLlmClient
from cbt_core.domain.analysis import ToneAnalysis
from cbt_core.domain.tone import ToneLabel

_ANALYSIS = ToneAnalysis(summary="s", tone=ToneLabel.NEUTRAL, score=0.0, rationale="r")

_FULL_CLIENTS: list[object] = [
    StubLlmClient(_ANALYSIS),
    ScriptedLlmClient({}),
    OfflineLlmClient(),
]


@pytest.mark.parametrize("client", _FULL_CLIENTS, ids=lambda c: type(c).__name__)
def test_full_client_conforms_to_the_llm_protocol(client: object) -> None:
    assert isinstance(client, LlmClient)


def test_protocol_check_has_teeth_against_a_missing_method() -> None:
    class _PartialClient:
        """A client missing classify_sentences/embed/answer: must not pass the protocol check."""

        def analyze_tone(self, speech_text: str) -> ToneAnalysis:
            return _ANALYSIS

    assert not isinstance(_PartialClient(), LlmClient)

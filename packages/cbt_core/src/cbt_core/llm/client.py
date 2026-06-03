"""The LLM boundary (CLAUDE.md section 2).

All generative model access goes through this protocol. Services depend on ``LlmClient``, not
on a concrete provider, so the domain stays testable without the network or a paid API and the
provider can be swapped behind the interface.
"""

from __future__ import annotations

from typing import Protocol

from cbt_core.domain.analysis import ToneAnalysis


class LlmClient(Protocol):
    """Protocol for the generative model used to analyze speeches."""

    def analyze_tone(self, speech_text: str) -> ToneAnalysis:
        """Summarize a speech and judge its monetary-policy tone.

        Args:
            speech_text: The full text of the speech.

        Returns:
            The model's :class:`ToneAnalysis`.

        Raises:
            LlmError: If the model call fails or returns an unusable response.
        """
        ...

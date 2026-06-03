"""Speech analysis service (CLAUDE.md sections 2 and 7).

Derives a summary and tone for a speech via the LLM boundary and records the resulting
append-only tone observation. The speaker is verified to exist before the model is called, so a
bad request never spends a Gemini call.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict

from cbt_core.domain.analysis import ToneAnalysis
from cbt_core.domain.models import ToneObservation
from cbt_core.llm.client import LlmClient
from cbt_core.logging import get_logger
from cbt_core.services.speaker_service import SpeakerService
from cbt_core.services.tone_service import ToneService

_logger = get_logger(__name__)


class SpeechAnalysisResult(BaseModel):
    """The outcome of analyzing a speech.

    Attributes:
        analysis: The model's summary and tone judgement.
        observation: The append-only tone observation recorded from it.
    """

    model_config = ConfigDict(frozen=True)

    analysis: ToneAnalysis
    observation: ToneObservation


class AnalysisService:
    """Analyze a speech with the LLM and record the resulting tone observation."""

    def __init__(
        self,
        llm_client: LlmClient,
        speaker_service: SpeakerService,
        tone_service: ToneService,
    ) -> None:
        """Build the service.

        Args:
            llm_client: The LLM boundary used to analyze the speech.
            speaker_service: Used to verify the speaker exists before calling the model.
            tone_service: Used to record the resulting tone observation.
        """
        self._llm = llm_client
        self._speaker_service = speaker_service
        self._tone_service = tone_service

    def analyze_speech(
        self,
        *,
        speaker_id: UUID,
        source_text: str,
        actor: str = "system",
        correlation_id: UUID | None = None,
    ) -> SpeechAnalysisResult:
        """Analyze a speech and record its tone for a speaker.

        Args:
            speaker_id: The speaker who gave the speech.
            source_text: The full text of the speech.
            actor: Who is performing the action.
            correlation_id: Correlation id for this call; one is minted if not supplied.

        Returns:
            The model analysis and the recorded observation.

        Raises:
            EntityNotFoundError: If the speaker does not exist.
            LlmError: If the model call fails or returns an unusable response.
        """
        correlation = correlation_id if correlation_id is not None else uuid4()
        log = _logger.bind(correlation_id=str(correlation), actor=actor, speaker_id=str(speaker_id))
        # Verify the speaker before spending a model call (raises EntityNotFoundError).
        self._speaker_service.get_speaker(speaker_id, actor=actor, correlation_id=correlation)
        analysis = self._llm.analyze_tone(source_text)
        observation = self._tone_service.record_observation(
            speaker_id=speaker_id,
            tone=analysis.tone,
            score=analysis.score,
            source_text=source_text,
            actor=actor,
            correlation_id=correlation,
        )
        log.info(
            "speech_analyzed",
            tone=analysis.tone.value,
            score=analysis.score,
            summary_chars=len(analysis.summary),
        )
        return SpeechAnalysisResult(analysis=analysis, observation=observation)

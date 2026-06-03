"""Central Bank Tone domain core.

The public surface of the core package. Adapters import from here (or from the ``domain`` /
``services`` subpackages' public re-exports), never from deep internal paths. The persistence
internals (rows, mappers) are deliberately not exported.
"""

from __future__ import annotations

from cbt_core.analysis import HawkishDovishLexicon, LexiconScore, chunk_text
from cbt_core.domain import (
    Answer,
    CentralBank,
    Citation,
    RetrievedChunk,
    Speaker,
    Speech,
    ToneAnalysis,
    ToneLabel,
    ToneObservation,
)
from cbt_core.exceptions import (
    CbtError,
    ConfigurationError,
    EntityNotFoundError,
    ImmutableRecordError,
    InvalidInputError,
    LlmError,
)
from cbt_core.llm import GeminiClient, LlmClient, build_gemini_client
from cbt_core.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
)
from cbt_core.persistence import (
    SpeechRetriever,
    create_engine_from_settings,
    make_session_factory,
)
from cbt_core.services import (
    IndexingService,
    IngestionService,
    QaService,
    SpeakerService,
    ToneService,
)
from cbt_core.settings import Environment, Settings, get_settings

__all__ = [
    "Answer",
    "CbtError",
    "CentralBank",
    "Citation",
    "ConfigurationError",
    "EntityNotFoundError",
    "Environment",
    "GeminiClient",
    "HawkishDovishLexicon",
    "ImmutableRecordError",
    "IndexingService",
    "IngestionService",
    "InvalidInputError",
    "LexiconScore",
    "LlmClient",
    "LlmError",
    "QaService",
    "RetrievedChunk",
    "Settings",
    "Speaker",
    "SpeakerService",
    "Speech",
    "SpeechRetriever",
    "ToneAnalysis",
    "ToneLabel",
    "ToneObservation",
    "ToneService",
    "bind_request_context",
    "build_gemini_client",
    "chunk_text",
    "clear_request_context",
    "configure_logging",
    "create_engine_from_settings",
    "get_logger",
    "get_settings",
    "make_session_factory",
]

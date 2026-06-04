"""Persistence layer: ORM base, rows, mappers, repositories, and engine helpers.

Importing this package registers the ORM rows on ``Base.metadata`` so
``Base.metadata.create_all(engine)`` builds the full schema. Rows and mappers are internal to
the layer and are not re-exported; callers use the repositories and engine helpers.
"""

from __future__ import annotations

# Import the row module for its side effect: registering tables on Base.metadata.
from cbt_core.persistence import rows as _rows  # noqa: F401  (registers ORM tables)
from cbt_core.persistence.base import Base
from cbt_core.persistence.engine import (
    create_demo_schema,
    create_engine_from_settings,
    create_immutability_triggers,
    make_demo_engine,
    make_session_factory,
)
from cbt_core.persistence.memory import InMemoryChunkRetriever
from cbt_core.persistence.repositories import (
    SpeakerRepository,
    SpeechChunkRepository,
    SpeechRepository,
    SpeechRetriever,
    ToneObservationRepository,
)

__all__ = [
    "Base",
    "InMemoryChunkRetriever",
    "SpeakerRepository",
    "SpeechChunkRepository",
    "SpeechRepository",
    "SpeechRetriever",
    "ToneObservationRepository",
    "create_demo_schema",
    "create_engine_from_settings",
    "create_immutability_triggers",
    "make_demo_engine",
    "make_session_factory",
]

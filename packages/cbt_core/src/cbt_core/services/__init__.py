"""Services: the only entry point for adapters into the core (CLAUDE.md section 2).

Each service owns its transaction boundary. Adapters call these; they never reach into
repositories, the engine, or the logger directly.
"""

from __future__ import annotations

from cbt_core.services.committee_service import CommitteeService
from cbt_core.services.indexing_service import IndexingService
from cbt_core.services.ingestion_service import IngestionService
from cbt_core.services.market_service import MarketSignalService, load_fred_monthly
from cbt_core.services.qa_service import ChunkRetriever, QaService
from cbt_core.services.speaker_service import SpeakerService
from cbt_core.services.stance_service import StanceService, build_stance
from cbt_core.services.tone_service import ToneService

__all__ = [
    "ChunkRetriever",
    "CommitteeService",
    "IndexingService",
    "IngestionService",
    "MarketSignalService",
    "QaService",
    "SpeakerService",
    "StanceService",
    "ToneService",
    "build_stance",
    "load_fred_monthly",
]

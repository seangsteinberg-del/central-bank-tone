"""The tone vocabulary: the labels the system can assign to a speech (CLAUDE.md section 2).

Part of the schema spine. The continuous tone score runs from -1.0 (most dovish) to 1.0 (most
hawkish); :class:`ToneLabel` is the discrete classification stored alongside it.
"""

from __future__ import annotations

from enum import StrEnum


class ToneLabel(StrEnum):
    """The discrete tone the system assigns to a speech.

    ``MIXED`` is an explicit label for speeches that carry both hawkish and dovish signals; it
    is not a fallback for "uncertain". Abstention is represented by recording no observation,
    never by quietly defaulting a label (CLAUDE.md section 3, no silent fallbacks).
    """

    HAWKISH = "hawkish"
    DOVISH = "dovish"
    NEUTRAL = "neutral"
    MIXED = "mixed"

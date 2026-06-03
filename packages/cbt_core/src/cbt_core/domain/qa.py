"""Question-answering domain types (CLAUDE.md sections 2 and 3).

A retrieved chunk carries enough speech metadata to cite it. An answer carries its citations and
an explicit ``abstained`` flag: when retrieval finds nothing relevant, the service abstains with
a reason rather than fabricating an answer.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# Embedding vector dimensionality used across the platform (Gemini output_dimensionality).
EMBEDDING_DIM = 768


class RetrievedChunk(BaseModel):
    """A speech chunk retrieved for a question, with its source metadata.

    Attributes:
        speech_id: The speech the chunk belongs to.
        chunk_index: The chunk's position within the speech.
        text: The chunk text.
        title: The speech title (for citation).
        url: The speech source URL (for citation).
        distance: The vector distance to the query (smaller is closer).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    speech_id: UUID
    chunk_index: int = Field(ge=0)
    text: str
    title: str
    url: str
    distance: float


class Citation(BaseModel):
    """A citation to the speech a piece of an answer came from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    speech_id: UUID
    chunk_index: int = Field(ge=0)
    title: str
    url: str

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk) -> Citation:
        """Build a citation from a retrieved chunk."""
        return cls(
            speech_id=chunk.speech_id,
            chunk_index=chunk.chunk_index,
            title=chunk.title,
            url=chunk.url,
        )


class Answer(BaseModel):
    """An answer to a question about a speaker.

    Attributes:
        text: The answer text, grounded in the citations, or the reason for abstaining.
        citations: The speeches the answer draws on (empty when abstaining).
        abstained: True when no relevant material was found and no answer was generated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    citations: tuple[Citation, ...]
    abstained: bool

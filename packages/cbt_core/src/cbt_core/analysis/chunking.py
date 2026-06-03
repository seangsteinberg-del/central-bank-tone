"""Deterministic text chunking for retrieval (CLAUDE.md section 2).

Splits a speech into overlapping, word-aligned chunks for embedding. Pure and deterministic, so
the same speech always produces the same chunks and the same embeddings.
"""

from __future__ import annotations


def chunk_text(text: str, *, max_chars: int = 1000, overlap: int = 100) -> list[str]:
    """Split ``text`` into overlapping chunks no longer than ``max_chars``.

    Chunks break on word boundaries; each chunk after the first repeats up to ``overlap``
    characters of the previous chunk's trailing words so context is not lost at the seam.

    Args:
        text: The text to chunk.
        max_chars: The maximum length of a chunk in characters.
        overlap: The number of trailing characters to repeat between consecutive chunks.

    Returns:
        The list of chunks (empty if ``text`` has no words).

    Raises:
        ValueError: If ``max_chars`` is not positive or ``overlap`` is not in ``[0, max_chars)``.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be in [0, max_chars)")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        if current and length + len(word) + 1 > max_chars:
            chunks.append(" ".join(current))
            tail: list[str] = []
            tail_length = 0
            for previous in reversed(current):
                if tail_length + len(previous) + 1 > overlap:
                    break
                tail.insert(0, previous)
                tail_length += len(previous) + 1
            current = tail
            length = tail_length
        current.append(word)
        length += len(word) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks

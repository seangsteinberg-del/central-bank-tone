"""Tests for deterministic text chunking (CLAUDE.md section 5)."""

from __future__ import annotations

import pytest

from cbt_core.analysis.chunking import chunk_text


@pytest.mark.unit
def test_short_text_is_a_single_chunk() -> None:
    assert chunk_text("a short speech", max_chars=1000) == ["a short speech"]


@pytest.mark.unit
def test_empty_or_whitespace_text_yields_no_chunks() -> None:
    assert chunk_text("   ") == []


@pytest.mark.unit
def test_long_text_splits_into_bounded_overlapping_chunks() -> None:
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, max_chars=60, overlap=12)
    assert len(chunks) > 1
    assert all(len(chunk) <= 60 for chunk in chunks)
    # Consecutive chunks share at least one word (overlap).
    first_tail = chunks[0].split()[-1]
    assert first_tail in chunks[1].split()


@pytest.mark.unit
def test_large_overlap_carries_the_whole_previous_chunk() -> None:
    # When the overlap budget can hold the entire previous chunk, every word of it carries into
    # the next chunk's head. The long final word forces the split and cannot itself be broken.
    chunks = chunk_text("aa bb cccccccc", max_chars=10, overlap=9)
    assert chunks == ["aa bb", "aa bb cccccccc"]
    assert chunks[1].split()[:2] == ["aa", "bb"]


@pytest.mark.unit
def test_chunking_is_deterministic() -> None:
    text = " ".join(f"word{i}" for i in range(120))
    assert chunk_text(text, max_chars=80, overlap=10) == chunk_text(text, max_chars=80, overlap=10)


@pytest.mark.unit
@pytest.mark.parametrize(("max_chars", "overlap"), [(0, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_parameters_raise(max_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="must"):
        chunk_text("text", max_chars=max_chars, overlap=overlap)

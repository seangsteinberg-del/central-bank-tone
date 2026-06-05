"""resolve_correlation_id prefers the adapter-bound request id (CLAUDE.md section 7).

The web adapter binds a request's correlation id onto the log context but does not thread it into
every service call. So the service's own log lines still correlate with the request, a core service
resolves its correlation id from the bound context when one is not passed explicitly, minting a
fresh id only when nothing is bound (a CLI or worker call with no adapter middleware).
"""

from __future__ import annotations

from collections.abc import Iterator
from uuid import UUID, uuid4

import pytest

from cbt_core.logging import (
    bind_request_context,
    clear_request_context,
    current_correlation_id,
    resolve_correlation_id,
)

_BOUND = "11111111-1111-1111-1111-111111111111"


@pytest.fixture(autouse=True)
def _clean_log_context() -> Iterator[None]:
    # The correlation id lives in a contextvar; clear it around each test so they stay hermetic.
    clear_request_context()
    try:
        yield
    finally:
        clear_request_context()


def test_explicit_id_wins_over_the_bound_context() -> None:
    explicit = uuid4()
    bind_request_context(correlation_id=_BOUND)
    assert resolve_correlation_id(explicit) == explicit


def test_falls_back_to_the_bound_context_id() -> None:
    bind_request_context(correlation_id=_BOUND)
    assert current_correlation_id() == _BOUND
    assert resolve_correlation_id(None) == UUID(_BOUND)


def test_mints_a_fresh_id_when_nothing_is_bound() -> None:
    assert current_correlation_id() is None
    assert isinstance(resolve_correlation_id(None), UUID)


def test_mints_when_the_bound_value_is_not_a_uuid() -> None:
    bind_request_context(correlation_id="not-a-uuid")
    assert isinstance(resolve_correlation_id(None), UUID)

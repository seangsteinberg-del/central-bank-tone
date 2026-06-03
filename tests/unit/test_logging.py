"""Tests for structured logging setup (CLAUDE.md section 7)."""

from __future__ import annotations

import json

import pytest

from cbt_core.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
)
from cbt_core.settings import Environment


@pytest.mark.unit
def test_production_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(environment=Environment.PRODUCTION)
    get_logger("test").info("lifecycle_event", entity_id="abc")
    out = capsys.readouterr().out.strip()
    parsed = json.loads(out)
    assert parsed["event"] == "lifecycle_event"
    assert parsed["entity_id"] == "abc"
    assert parsed["level"] == "info"


@pytest.mark.unit
def test_development_emits_key_value_not_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(environment=Environment.DEVELOPMENT)
    get_logger("test").info("lifecycle_event", entity_id="abc")
    out = capsys.readouterr().out.strip()
    assert "lifecycle_event" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


@pytest.mark.unit
def test_request_context_binds_and_clears(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(environment=Environment.PRODUCTION)
    bind_request_context(correlation_id="corr-123")
    try:
        get_logger("test").info("with_context")
    finally:
        clear_request_context()
    bound = json.loads(capsys.readouterr().out.strip())
    assert bound["correlation_id"] == "corr-123"

    get_logger("test").info("without_context")
    after = json.loads(capsys.readouterr().out.strip())
    assert "correlation_id" not in after

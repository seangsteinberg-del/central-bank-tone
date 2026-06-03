"""Tests that core exceptions map to the right HTTP status codes (CLAUDE.md section 3)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cbt_api.errors import install_exception_handlers
from cbt_core.exceptions import (
    CbtError,
    ConfigurationError,
    EntityNotFoundError,
    ImmutableRecordError,
    InvalidInputError,
)


def _client_that_raises(exc: Exception) -> TestClient:
    app = FastAPI()
    install_exception_handlers(app)

    @app.get("/boom")
    def boom() -> None:
        raise exc

    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.web
@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (EntityNotFoundError("Speaker", "x"), 404),
        (InvalidInputError("bad input"), 422),
        (ImmutableRecordError("append-only"), 409),
        (ConfigurationError("misconfigured"), 500),
        (CbtError("unexpected"), 500),
    ],
)
def test_core_exception_maps_to_status(exc: Exception, expected_status: int) -> None:
    response = _client_that_raises(exc).get("/boom")
    assert response.status_code == expected_status
    assert "detail" in response.json()


@pytest.mark.web
def test_configuration_error_detail_is_not_leaked() -> None:
    response = _client_that_raises(ConfigurationError("secret detail")).get("/boom")
    assert "secret detail" not in response.json()["detail"]

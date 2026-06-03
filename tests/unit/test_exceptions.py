"""Tests for the core exception hierarchy (CLAUDE.md section 3)."""

from __future__ import annotations

from uuid import UUID

import pytest

from cbt_core.exceptions import (
    CbtError,
    ConfigurationError,
    EntityNotFoundError,
    ImmutableRecordError,
    InvalidInputError,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "error_type",
    [ConfigurationError, InvalidInputError, EntityNotFoundError, ImmutableRecordError],
)
def test_every_error_subclasses_cbterror(error_type: type[CbtError]) -> None:
    assert issubclass(error_type, CbtError)


@pytest.mark.unit
def test_entity_not_found_message_names_entity_and_identifier() -> None:
    identifier = UUID(int=7)
    error = EntityNotFoundError("Speaker", identifier)
    assert error.entity == "Speaker"
    assert error.identifier == identifier
    assert "Speaker" in str(error)
    assert str(identifier) in str(error)

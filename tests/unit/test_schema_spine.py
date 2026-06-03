"""Tests guarding the schema spine (CLAUDE.md section 2).

These pin the single source of truth: the central bank registry and the tone vocabulary. They
exist so an accidental rename or value change is caught, since persistence and every adapter
derive from these enums.
"""

from __future__ import annotations

import pytest

from cbt_core.domain.registry import CentralBank
from cbt_core.domain.tone import ToneLabel


@pytest.mark.unit
def test_central_bank_values_are_stable_snake_case() -> None:
    for member in CentralBank:
        assert member.value == member.value.lower()
        assert " " not in member.value


@pytest.mark.unit
def test_central_bank_is_a_string_enum() -> None:
    assert CentralBank.FEDERAL_RESERVE == "federal_reserve"
    assert CentralBank("ecb") is CentralBank.ECB


@pytest.mark.unit
def test_tone_labels_are_exactly_the_known_set() -> None:
    assert {label.value for label in ToneLabel} == {"hawkish", "dovish", "neutral", "mixed"}

"""The architecture invariants must hold (CLAUDE.md section 2).

This wraps ``scripts/check_imports.py`` so the one-way dependency and the settings boundary are
verified by the test suite, not only by the standalone CI step.
"""

from __future__ import annotations

import pytest
from scripts.check_imports import check


@pytest.mark.unit
def test_architecture_invariants_hold() -> None:
    assert check() == []

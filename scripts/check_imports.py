"""Enforce the architecture invariants that mypy and ruff do not (CLAUDE.md section 2).

Two machine-checkable rules:
1. The core package imports no adapter package (the one-way dependency).
2. Configuration is read only through ``cbt_core.settings``; no other module touches
   ``os.environ`` / ``os.getenv`` directly (CLAUDE.md section 11).

Run in CI and locally: ``uv run python scripts/check_imports.py``. Exits non-zero with a list
of violations, or prints OK.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# --- edit these for your project -------------------------------------------------
CORE_PACKAGE = "cbt_core"
ADAPTER_PACKAGES = ("cbt_api", "cbt_worker", "cbt_web")  # add every adapter package here
# ---------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = REPO_ROOT / "packages" / CORE_PACKAGE / "src" / CORE_PACKAGE
ALL_SRC = REPO_ROOT / "packages"
_SETTINGS_FILE = "settings.py"


def _imported_modules(tree: ast.AST) -> list[str]:
    """Every top-level module name referenced by import statements in ``tree``."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _reads_os_environ(tree: ast.AST) -> bool:
    """True if the module accesses os.environ or os.getenv."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"environ", "getenv"}:
            value = node.value
            if isinstance(value, ast.Name) and value.id == "os":
                return True
    return False


def check() -> list[str]:
    """Return a list of architecture-invariant violations (empty when the codebase is clean)."""
    violations: list[str] = []

    for path in sorted(CORE_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module in _imported_modules(tree):
            if any(module == a or module.startswith(f"{a}.") for a in ADAPTER_PACKAGES):
                rel = path.relative_to(REPO_ROOT)
                violations.append(f"{rel}: {CORE_PACKAGE} must not import {module}")

    for path in sorted(ALL_SRC.rglob("*.py")):
        if path.name == _SETTINGS_FILE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if _reads_os_environ(tree):
            rel = path.relative_to(REPO_ROOT)
            violations.append(f"{rel}: reads os.environ directly; use {CORE_PACKAGE}.settings")

    return violations


def main() -> int:
    """Print violations and return a non-zero exit code, or print OK and return 0."""
    violations = check()
    if violations:
        print("Architecture invariant violations:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("OK: architecture invariants hold (core dependency direction, settings boundary).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

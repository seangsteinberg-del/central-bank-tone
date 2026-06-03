"""PostToolUse hook: auto-format and lint-fix a Python file right after Claude edits it.

Runs after every Edit / Write / MultiEdit. When the edited file is a .py file it:
1. runs ``ruff format`` on that one file,
2. runs ``ruff check --fix`` to apply safe autofixes,
3. re-runs ``ruff check`` and, if anything is still wrong, prints the remaining findings to
   stderr and exits 2 so Claude sees them and fixes them next.

Fails open: an unreadable payload, a non-Python file, or a missing ruff exits 0 and never
blocks. ``--force-exclude`` makes ruff honour project excludes even when a path is passed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _find_ruff() -> str | None:
    """Locate the ruff executable, preferring the project venv, then PATH."""
    cwd = Path.cwd()
    for candidate in (cwd / ".venv" / "Scripts" / "ruff.exe", cwd / ".venv" / "bin" / "ruff"):
        if candidate.exists():
            return str(candidate)
    return shutil.which("ruff")


def main() -> int:
    """Format and lint-fix the edited Python file; surface anything still failing."""
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return 0

    file_path = (payload.get("tool_input") or {}).get("file_path") or ""
    if not file_path:
        return 0
    path = Path(file_path)
    if path.suffix != ".py" or not path.is_file():
        return 0

    ruff = _find_ruff()
    if ruff is None:
        return 0  # ruff not installed in this environment; do not block.

    cwd = Path.cwd()
    try:
        target = path.resolve().relative_to(cwd.resolve()).as_posix()
    except ValueError:
        target = str(path)
    subprocess.run([ruff, "format", "--force-exclude", target], capture_output=True, text=True)
    subprocess.run(
        [ruff, "check", "--fix", "--force-exclude", target], capture_output=True, text=True
    )
    result = subprocess.run(
        [ruff, "check", "--force-exclude", target], capture_output=True, text=True
    )
    if result.returncode != 0:
        details = (result.stdout or result.stderr).strip()
        print(
            f"ruff auto-formatted and applied safe fixes to {path.name}, but these lint "
            f"findings remain. Fix them before moving on:\n{details}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

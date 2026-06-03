"""PreToolUse hook: protect files Claude should not edit directly.

Runs before every Edit / Write / MultiEdit. Denies writes to ``.env`` (real secrets, gitignored,
CLAUDE.md section 4) and ``uv.lock`` (managed by uv; hand-editing causes drift, section 9). A
blocked write exits 2 with an explanation. Matches on base name, so ``.env.example`` is allowed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROTECTED: dict[str, str] = {
    ".env": (
        ".env holds real secrets and is gitignored (CLAUDE.md section 4: no secret in source). "
        "Document shape in .env.example, and ask the user to set real values in their own .env."
    ),
    "uv.lock": (
        "uv.lock is managed by uv (CLAUDE.md section 9: one lockfile, no drift). Run `uv add` "
        "or `uv lock` instead of editing it by hand."
    ),
}


def main() -> int:
    """Deny writes to protected files; allow everything else."""
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return 0

    file_path = (payload.get("tool_input") or {}).get("file_path") or ""
    if not file_path:
        return 0

    reason = _PROTECTED.get(Path(file_path).name)
    if reason is not None:
        print(f"Blocked: {reason}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""PreToolUse hook: block shell commands that violate the binding rules in CLAUDE.md section 13.

Runs before every Bash call. Denies ``--no-verify`` (skips pre-commit gates), ``--no-gpg-sign``
(bypasses signing), and force-push (``--force`` / ``-f``) because main is protected with linear
history. ``--force-with-lease`` on a feature branch is allowed. A blocked command exits 2 with an
explanation on stderr. Fails open on an unreadable payload or a non-git command.
"""

from __future__ import annotations

import json
import re
import sys

_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"--no-verify\b"),
        "--no-verify skips the pre-commit hooks (format, lint, mypy). CLAUDE.md section 13 "
        "forbids bypassing them. Fix the underlying issue instead.",
    ),
    (
        re.compile(r"--no-gpg-sign\b"),
        "--no-gpg-sign bypasses commit signing. CLAUDE.md section 13 forbids it unless the "
        "user explicitly asked.",
    ),
)

_PUSH = re.compile(r"\bgit\s+push\b")
_FORCE = re.compile(r"--force(?!-with-lease)\b|(?:^|\s)-f(?:\s|$)")


def main() -> int:
    """Deny forbidden git invocations; allow everything else."""
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return 0

    command = (payload.get("tool_input") or {}).get("command") or ""
    if not command:
        return 0

    for pattern, reason in _RULES:
        if pattern.search(command):
            print(f"Blocked: {reason}", file=sys.stderr)
            return 2

    if _PUSH.search(command) and _FORCE.search(command):
        print(
            "Blocked: force-push is not allowed. main is protected with linear history "
            "(CLAUDE.md sections 10 and 13). Use --force-with-lease on a feature branch.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

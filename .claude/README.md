# Agent harness

This directory configures the Claude Code harness for this repo. It enforces a slice of the
binding standards in [../CLAUDE.md](../CLAUDE.md) mechanically, so the rules hold even on a
fast edit.

## settings.json

- **permissions.allow** pre-approves the read-only and gate commands (the venv tools, `uv
  sync`, read-only git) so routine work does not prompt.
- **permissions.deny** blocks reading `./.env` (real secrets).
- **hooks** wire the three scripts below.

## hooks/

| Hook | Event | What it does |
| --- | --- | --- |
| `format_python.py` | PostToolUse (Edit/Write/MultiEdit) | Runs `ruff format` then `ruff check --fix` on the edited `.py` file, then re-checks and exits 2 with the remaining findings so they get fixed. Fails open. |
| `guard_bash.py` | PreToolUse (Bash) | Blocks `--no-verify`, `--no-gpg-sign`, and force-push to a protected branch (CLAUDE.md sections 10, 13). Allows `--force-with-lease`. |
| `protect_paths.py` | PreToolUse (Edit/Write/MultiEdit) | Blocks direct writes to `.env` and `uv.lock` (CLAUDE.md sections 4, 9). `.env.example` is allowed. |

The hook commands call `.venv/Scripts/python.exe` (Windows). On macOS/Linux change these to
`.venv/bin/python` in `settings.json`. The hooks fail open: an unreadable payload or a missing
tool never blocks legitimate work.

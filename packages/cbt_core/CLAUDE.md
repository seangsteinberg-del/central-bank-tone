# cbt_core: local map

The domain heart. The root `CLAUDE.md` holds the binding rules; this file is the local map and
the idioms specific to `cbt_core`. `cbt_core` imports no adapter; that one-way dependency is
machine-checked by `scripts/check_imports.py`.

## What lives here (src/cbt_core)

- `domain/` Pydantic models (`Speaker`, `ToneObservation`) and the schema spine: the central
  bank registry (`registry.py`) and tone labels (`tone.py`), the single source of truth for
  the core type system.
- `services/` the only entry point for adapters (`SpeakerService`, `ToneService`). Each
  service owns its transaction boundary. Re-exported from `services/__init__.py`.
- `persistence/` ORM rows (`rows.py`), mappers (`mappers.py`, domain to/from ORM),
  repositories (`repositories.py`, one aggregate each), the engine/session factory
  (`engine.py`), and alembic migrations (`migrations/`). ORM rows never leave this layer.
- `settings.py` (the only place that reads the environment), `exceptions.py` (the error
  hierarchy), `logging.py` (structlog).

## Local conventions

- Adapters call services only. Services call repositories and external clients. Never reach a
  repository from a route or a CLI command.
- Domain types over primitives and over ORM rows. Map at the persistence edge; never return a
  `*Row` past `persistence/`.
- `tone_observation` is append-only: the repository exposes `append`/`list_for_speaker` and no
  update or delete. The DB-level trigger is the real guarantee; the API just respects it.
- Public surface is re-exported from `__init__.py` and listed in `__all__`. Reaching into a
  deep path from another package means the public API is missing something.
- Define module-specific exceptions as subclasses of `CbtError`. Adapters translate them; they
  do not invent new ones.
- Inject the clock and id factory into services; never call `datetime.now`/`uuid4` inline in a
  branch a test needs to pin.

## Commands

```
.venv/Scripts/pytest.exe tests/unit -q
.venv/Scripts/mypy.exe
.venv/Scripts/python.exe scripts/check_imports.py
```

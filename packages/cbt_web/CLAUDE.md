# cbt_web: local map

A server-rendered (Jinja + HTMX) FastAPI adapter over `cbt_core`. The root `CLAUDE.md` holds the
binding rules; this file is the local map for the web UI. `cbt_web` depends on `cbt_core` only
(machine-checked by `scripts/check_imports.py`); it is independent of `cbt_api`.

## What lives here (src/cbt_web)

- `app.py` builds the FastAPI application: the correlation-id middleware, the HTML exception
  handlers, the `/static` mount, and the views. A factory, not a module-level app, so importing
  has no side effects.
- `dependencies.py` builds the engine, session factory, and `cbt_core` services once per process
  and hands services to views via FastAPI dependencies. It mirrors the API adapter's wiring but
  imports `cbt_core` only.
- `views.py` the page and fragment handlers. Routes under `/ui` return HTML fragments for htmx to
  swap in; the rest return full pages. Views call a service and render a template; they never
  touch a repository, the engine, or the logger directly.
- `schemas.py` form-validation models (the web boundary contract). Every form is validated
  against one before any service call.
- `templating.py` the shared Jinja2 environment and the presentation filters (tone colour, score
  position, date formatting).
- `errors.py` maps core exceptions to HTML error pages (`EntityNotFoundError` -> 404, any other
  `CbtError` -> 500). User-input errors are caught in the views and re-rendered as 4xx, so they
  do not reach here.
- `templates/` Jinja templates (`base.html`, pages, and `_*.html` htmx fragments).
- `static/` the stylesheet and vendored `htmx.min.js` (BSD-2, pinned; ADR 0011).

## Local conventions

- Validate at the boundary: a form posts to a view, the view builds a `schemas.py` model and
  catches `ValidationError` to re-render with an inline message and a 4xx; only then does it call
  a service with typed values.
- Fragments degrade gracefully: forms are real forms and links are real links, so the UI still
  works if htmx (or JavaScript) does not load.
- Presentation logic (tone -> colour, score -> bar geometry) lives in `templating.py` filters or
  is precomputed in the view, never duplicated across templates.
- The adapter translates core exceptions; it does not catch-and-swallow them.

## Commands

```
.venv/Scripts/pytest.exe -m web -q
.venv/Scripts/mypy.exe
.venv/Scripts/uvicorn.exe --factory cbt_web.app:create_app   # serve the UI (needs DB + Gemini key)
```

# cbt_api: local map

A FastAPI adapter over `cbt_core`. The root `CLAUDE.md` holds the binding rules; this file is
the local map for the API surface. `cbt_api` depends on `cbt_core` only (machine-checked).

## What lives here (src/cbt_api)

- `app.py` builds the FastAPI application, wires the correlation-id middleware and the
  exception handlers, and includes the routers.
- `dependencies.py` builds settings, the engine, the session factory, and the services once
  per process and hands services to routes via FastAPI dependencies.
- `schemas.py` request/response Pydantic models. These are the HTTP boundary contract and are
  separate from the core domain models.
- `routes/` one module per resource (`speakers.py`). Routes call a service and return a
  response model. They never touch a repository, the engine, or the logger directly.
- `errors.py` maps `cbt_core` exceptions to HTTP status codes (`EntityNotFoundError` -> 404,
  `InvalidInputError` -> 422, `ConfigurationError`/unknown -> 500). Routes raise core
  exceptions; they do not invent HTTP errors.

## Local conventions

- Validate at the boundary: every request body is a Pydantic schema; the route passes typed
  values into the service. Never pass an unvalidated `dict` inward.
- The adapter translates core exceptions; it does not catch-and-swallow them.
- Inject the correlation id from the `X-Correlation-ID` header (or mint one) and pass it to the
  service so logs correlate across the call.

## Commands

```
.venv/Scripts/pytest.exe -m web -q
.venv/Scripts/mypy.exe
```

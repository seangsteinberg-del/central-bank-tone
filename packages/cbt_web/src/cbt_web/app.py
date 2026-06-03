"""FastAPI application factory for the web UI.

Run with: ``uv run uvicorn --factory cbt_web.app:create_app``. The factory wires the
correlation-id middleware, the HTML exception handlers, the static-file mount, and the views,
and builds the service container (including the Gemini client) once per process. It is a factory
rather than a module-level app so importing this module has no side effects and needs no
configuration.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from uuid import UUID, uuid4

from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles

from cbt_core import (
    Settings,
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
    get_settings,
)
from cbt_web.dependencies import build_services
from cbt_web.errors import install_exception_handlers
from cbt_web.templating import STATIC_DIR
from cbt_web.views import router

_logger = get_logger(__name__)


def _parse_correlation_id(raw: str | None) -> UUID:
    """Parse a correlation id from a header, minting a fresh one if absent or malformed.

    A missing header is minted silently. A present-but-malformed header is rejected with a
    WARNING rather than silently substituted (CLAUDE.md section 3, no silent fallbacks). The raw
    value is not logged.
    """
    if not raw:
        return uuid4()
    try:
        return UUID(raw)
    except ValueError:
        _logger.warning(
            "correlation_id_rejected",
            reason="malformed X-Correlation-ID header; minted a new correlation id",
        )
        return uuid4()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application serving the web UI.

    Args:
        settings: Settings to use; the process settings are read if not supplied.

    Returns:
        The configured application.
    """
    resolved = settings if settings is not None else get_settings()
    configure_logging(environment=resolved.environment)

    app = FastAPI(title="Central Bank Tone", version="0.1.0")
    app.state.services = build_services(resolved)
    install_exception_handlers(app)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(router)

    @app.middleware("http")
    async def _correlation_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        correlation_id = _parse_correlation_id(request.headers.get("X-Correlation-ID"))
        request.state.correlation_id = correlation_id
        bind_request_context(correlation_id=str(correlation_id))
        try:
            response = await call_next(request)
        finally:
            clear_request_context()
        response.headers["X-Correlation-ID"] = str(correlation_id)
        return response

    return app

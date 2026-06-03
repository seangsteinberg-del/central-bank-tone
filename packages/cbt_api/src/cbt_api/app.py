"""FastAPI application factory and ASGI entry point.

Run with: ``uv run uvicorn --factory cbt_api.app:create_app``. The factory wires the
correlation-id middleware, the exception handlers, and the routers, and builds the service
container (including the Gemini client) once per process. It is a factory rather than a
module-level app so that importing this module has no side effects and needs no configuration.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from uuid import UUID, uuid4

from fastapi import FastAPI, Request, Response

from cbt_api.dependencies import build_services
from cbt_api.errors import install_exception_handlers
from cbt_api.routes import speakers_router
from cbt_core import (
    Settings,
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
    get_settings,
)

_logger = get_logger(__name__)


def _parse_correlation_id(raw: str | None) -> UUID:
    """Parse a correlation id from a header, minting a fresh one if absent or malformed.

    A missing header is minted silently. A present-but-malformed header is rejected with a
    WARNING rather than silently substituted, so the client can see its trace id did not take
    effect (CLAUDE.md section 3, no silent fallbacks). The raw value is not logged.
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
    """Build the FastAPI application.

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
    app.include_router(speakers_router)

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

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    return app

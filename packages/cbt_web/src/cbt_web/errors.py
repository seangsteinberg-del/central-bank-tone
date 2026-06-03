"""Map core exceptions to rendered HTML error pages (CLAUDE.md section 3).

The adapter translates ``cbt_core`` exceptions into a status code and a friendly error page; it
does not invent new error types. User-input failures are validated in the views (a bad form
re-renders with a 4xx and an inline message), so only two core exceptions reach this layer: an
``EntityNotFoundError`` for an unknown id (404), and any other ``CbtError`` (for example an
``LlmError``) as an internal error (500). Server-side detail is logged, never shown to the user.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, Response, status

from cbt_core import CbtError, EntityNotFoundError, get_logger
from cbt_web.templating import templates

_logger = get_logger(__name__)


def render_error(request: Request, status_code: int, title: str, detail: str) -> Response:
    """Render the shared error page with a status code, heading, and detail message.

    Args:
        request: The active request (required by the template engine).
        status_code: The HTTP status code to return.
        title: The short error heading.
        detail: The human-readable explanation shown to the user.

    Returns:
        The rendered error page response.
    """
    return templates.TemplateResponse(
        request,
        "error.html",
        {"status_code": status_code, "title": title, "detail": detail},
        status_code=status_code,
    )


def install_exception_handlers(app: FastAPI) -> None:
    """Register handlers that translate core exceptions to HTML error pages.

    Args:
        app: The FastAPI application to register handlers on.
    """

    async def _not_found(request: Request, exc: EntityNotFoundError) -> Response:
        return render_error(request, status.HTTP_404_NOT_FOUND, "Not found", str(exc))

    async def _core_error(request: Request, _exc: CbtError) -> Response:
        # The detail (model failure, misconfiguration) is logged, not shown to the user.
        _logger.exception("web_unhandled_core_error")
        return render_error(
            request,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Server error",
            "An unexpected error occurred. Please try again later.",
        )

    app.add_exception_handler(EntityNotFoundError, _not_found)  # type: ignore[arg-type]  # starlette types handlers as BaseException; ours narrow it
    app.add_exception_handler(CbtError, _core_error)  # type: ignore[arg-type]  # see above

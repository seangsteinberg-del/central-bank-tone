"""Map core exceptions to HTTP responses (CLAUDE.md section 3).

The adapter translates ``cbt_core`` exceptions into status codes; it does not invent new error
types. User-input failures map to 4xx, never 5xx. The response body never includes a secret.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from cbt_core import (
    CbtError,
    ConfigurationError,
    EntityNotFoundError,
    ImmutableRecordError,
    InvalidInputError,
)


def _problem(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


def install_exception_handlers(app: FastAPI) -> None:
    """Register handlers that translate core exceptions to HTTP status codes.

    Args:
        app: The FastAPI application to register handlers on.
    """

    async def _not_found(_request: Request, exc: EntityNotFoundError) -> JSONResponse:
        return _problem(status.HTTP_404_NOT_FOUND, str(exc))

    async def _invalid_input(_request: Request, exc: InvalidInputError) -> JSONResponse:
        return _problem(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc))

    async def _immutable(_request: Request, exc: ImmutableRecordError) -> JSONResponse:
        return _problem(status.HTTP_409_CONFLICT, str(exc))

    async def _configuration(_request: Request, _exc: ConfigurationError) -> JSONResponse:
        # Do not echo configuration error detail to clients; it is logged server-side.
        return _problem(status.HTTP_500_INTERNAL_SERVER_ERROR, "server misconfiguration")

    async def _core_error(_request: Request, _exc: CbtError) -> JSONResponse:
        return _problem(status.HTTP_500_INTERNAL_SERVER_ERROR, "internal error")

    app.add_exception_handler(EntityNotFoundError, _not_found)  # type: ignore[arg-type]  # starlette types handlers as BaseException; ours narrow it
    app.add_exception_handler(InvalidInputError, _invalid_input)  # type: ignore[arg-type]  # see above
    app.add_exception_handler(ImmutableRecordError, _immutable)  # type: ignore[arg-type]  # see above
    app.add_exception_handler(ConfigurationError, _configuration)  # type: ignore[arg-type]  # see above
    app.add_exception_handler(CbtError, _core_error)  # type: ignore[arg-type]  # see above

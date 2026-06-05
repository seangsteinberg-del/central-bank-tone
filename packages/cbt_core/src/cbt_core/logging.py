"""Structured logging via structlog (CLAUDE.md section 7).

JSON in production, human-readable key=value in development. Service methods open a log context
with a ``correlation_id``, an ``actor``, and the relevant entity IDs. Secrets and full external
payloads are never logged; log the sha256 of inputs and sizes/counts instead.
"""

from __future__ import annotations

import io
import logging
import sys
from typing import cast
from uuid import UUID, uuid4

import structlog
from structlog.typing import FilteringBoundLogger, Processor

from cbt_core.settings import Environment


def configure_logging(*, environment: Environment, level: int = logging.INFO) -> None:
    """Configure structlog for the whole process.

    Args:
        environment: The deployment environment. Production renders JSON; everything else
            renders coloured-off key=value lines for human reading.
        level: The minimum level to emit.
    """
    # Logs carry non-ASCII content (central bankers' names); the console stream may default to a
    # non-UTF-8 encoding (Windows cp1252), so force UTF-8 output to avoid an encode crash mid-run.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if environment is Environment.PRODUCTION
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Return a structlog logger.

    Args:
        name: Optional logger name, conventionally the module ``__name__``.

    Returns:
        A bound logger whose ``bind`` / ``info`` / ``error`` calls carry structured context.
    """
    # structlog.get_logger is typed as Any upstream (it returns a lazily-bound proxy); cast to
    # the concrete protocol so callers stay strictly typed.
    return cast(FilteringBoundLogger, structlog.get_logger(name))


def bind_request_context(**values: str) -> None:
    """Bind values onto the contextvars-scoped log context for the current task.

    Adapters use this to attach a ``correlation_id`` to every log line emitted while handling a
    request, without importing structlog themselves (CLAUDE.md section 2).

    Args:
        values: Key/value pairs merged into every subsequent log event in this context.
    """
    structlog.contextvars.bind_contextvars(**values)


def clear_request_context() -> None:
    """Clear the contextvars-scoped log context for the current task."""
    structlog.contextvars.clear_contextvars()


def current_correlation_id() -> str | None:
    """Return the correlation id bound on the current log context, if any.

    Returns ``None`` when nothing is bound (a CLI or worker call with no adapter middleware).
    """
    bound = structlog.contextvars.get_contextvars().get("correlation_id")
    return bound if isinstance(bound, str) else None


def resolve_correlation_id(correlation_id: UUID | None) -> UUID:
    """Resolve the correlation id for a service call (CLAUDE.md section 7).

    Prefers an explicitly passed id; otherwise the one the adapter bound onto the log context for
    the request, so the service's own log lines correlate with the rest of the request; otherwise a
    freshly minted id (a CLI or worker call with no adapter context). This is why a service need not
    be threaded the id through every call to stay correlated: the web adapter binds it once on the
    request and the services pick it up here.
    """
    if correlation_id is not None:
        return correlation_id
    bound = current_correlation_id()
    if bound is not None:
        try:
            return UUID(bound)
        except ValueError:
            return uuid4()
    return uuid4()

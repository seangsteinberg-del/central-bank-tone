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

"""The core error hierarchy (CLAUDE.md section 3).

Every error raised by ``cbt_core`` subclasses :class:`CbtError`. Adapters translate these to
HTTP status codes or CLI exit codes; they never invent new error types of their own.
"""

from __future__ import annotations


class CbtError(Exception):
    """Base class for every error raised by ``cbt_core``."""


class ConfigurationError(CbtError):
    """Configuration is missing or unsafe for the current environment.

    Raised at startup when, for example, a placeholder secret is used in production.
    """


class InvalidInputError(CbtError):
    """Input failed validation at a service boundary.

    Adapters map this to a 4xx response (HTTP 422 / non-zero CLI exit), never a 5xx.
    """


class EntityNotFoundError(CbtError):
    """A requested entity does not exist.

    Args:
        entity: The kind of entity, for example ``"Speaker"``.
        identifier: The identifier that was not found.
    """

    def __init__(self, entity: str, identifier: object) -> None:
        """Build the error with a message naming the entity and identifier."""
        super().__init__(f"{entity} not found: {identifier!r}")
        self.entity = entity
        self.identifier = identifier


class ImmutableRecordError(CbtError):
    """An attempt was made to mutate an append-only record.

    The database enforces append-only tables with a trigger; this error represents the same
    rule at the application layer so callers get a typed failure.
    """

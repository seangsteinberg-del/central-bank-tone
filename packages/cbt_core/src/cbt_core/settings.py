"""Application settings: the only module that reads the environment (CLAUDE.md section 11).

All configuration flows through :class:`Settings`. No other module reads ``os.environ``;
``scripts/check_imports.py`` enforces that boundary. Secrets are held as ``SecretStr`` so they
never appear in logs, reprs, or exception messages (CLAUDE.md section 4).
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from cbt_core.exceptions import ConfigurationError


class Environment(StrEnum):
    """The deployment environment the application runs in."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


# Values that are safe defaults for development but must never run in production.
_PLACEHOLDER_SECRETS: frozenset[str] = frozenset({"", "dev-only-change-me", "change-me"})


class Settings(BaseSettings):
    """Typed application configuration, read once from the environment.

    Defaults are development-safe. Production loudly rejects insecure values at startup
    (CLAUDE.md section 11): a placeholder ``secret_key`` raises when ``environment`` is
    ``production``.

    Attributes:
        environment: The deployment environment.
        database_url: SQLAlchemy URL for the primary database (psycopg 3 driver).
        secret_key: Secret used to sign session cookies and CSRF tokens. Held as ``SecretStr``.
    """

    model_config = SettingsConfigDict(
        env_prefix="CBT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    environment: Environment = Environment.DEVELOPMENT
    database_url: str = "postgresql+psycopg://cbt:cbt@localhost:5432/cbt"
    secret_key: SecretStr = SecretStr("dev-only-change-me")

    # All generative work goes through Google Gemini (ADR 0006). No paid third-party API.
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"

    @field_validator("secret_key")
    @classmethod
    def _reject_placeholder_in_production(cls, value: SecretStr, info: ValidationInfo) -> SecretStr:
        """Reject placeholder secrets when running in production (CLAUDE.md section 11).

        Raises:
            ConfigurationError: If ``environment`` is production and ``secret_key`` is a
                known placeholder. The secret value itself is never included in the message.
        """
        environment = info.data.get("environment")
        if environment is Environment.PRODUCTION and value.get_secret_value() in (
            _PLACEHOLDER_SECRETS
        ):
            raise ConfigurationError(
                "CBT_SECRET_KEY is a placeholder; set a real secret when "
                "CBT_ENVIRONMENT=production."
            )
        return value

    @field_validator("gemini_api_key")
    @classmethod
    def _require_gemini_key_in_production(cls, value: SecretStr, info: ValidationInfo) -> SecretStr:
        """Require a Gemini API key in production (CLAUDE.md section 11).

        Raises:
            ConfigurationError: If ``environment`` is production and no key is set. The key
                value itself is never included in the message.
        """
        environment = info.data.get("environment")
        if environment is Environment.PRODUCTION and not value.get_secret_value():
            raise ConfigurationError(
                "CBT_GEMINI_API_KEY is required when CBT_ENVIRONMENT=production."
            )
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, read once from the environment.

    Returns:
        The cached :class:`Settings` instance.

    Raises:
        ConfigurationError: If configuration is unsafe for the current environment.
    """
    return Settings()

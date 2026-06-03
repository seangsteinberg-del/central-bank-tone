"""Settings tests, including the production-secret guard (CLAUDE.md sections 4 and 11)."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from cbt_core.exceptions import ConfigurationError
from cbt_core.settings import _PLACEHOLDER_SECRETS, Environment, Settings, get_settings

_CBT_ENV_VARS = ("CBT_ENVIRONMENT", "CBT_DATABASE_URL", "CBT_SECRET_KEY")


@pytest.fixture(autouse=True)
def _clear_cbt_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove any CBT_* environment variables so each test sees a clean environment."""
    for name in _CBT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.unit
def test_defaults_are_development_safe() -> None:
    settings = Settings(_env_file=None)
    assert settings.environment is Environment.DEVELOPMENT


@pytest.mark.unit
@pytest.mark.parametrize("placeholder", sorted(_PLACEHOLDER_SECRETS))
def test_production_rejects_every_placeholder_secret(placeholder: str) -> None:
    with pytest.raises(ConfigurationError):
        Settings(
            _env_file=None,
            environment=Environment.PRODUCTION,
            secret_key=SecretStr(placeholder),
        )


@pytest.mark.unit
def test_production_accepts_a_real_secret() -> None:
    settings = Settings(
        _env_file=None,
        environment=Environment.PRODUCTION,
        secret_key=SecretStr("a-genuinely-set-production-secret"),
        gemini_api_key=SecretStr("a-real-gemini-key"),
    )
    assert settings.environment is Environment.PRODUCTION


@pytest.mark.unit
def test_development_allows_placeholder_secret() -> None:
    settings = Settings(_env_file=None, secret_key=SecretStr("dev-only-change-me"))
    assert settings.environment is Environment.DEVELOPMENT


@pytest.mark.unit
def test_secret_is_not_exposed_in_repr_or_str() -> None:
    settings = Settings(_env_file=None, secret_key=SecretStr("super-secret-value"))
    assert "super-secret-value" not in repr(settings)
    assert "super-secret-value" not in str(settings)
    assert settings.secret_key.get_secret_value() == "super-secret-value"


@pytest.mark.unit
def test_default_gemini_model_is_flash() -> None:
    settings = Settings(_env_file=None)
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.gemini_embedding_model == "gemini-embedding-001"


@pytest.mark.unit
def test_production_requires_a_gemini_api_key() -> None:
    with pytest.raises(ConfigurationError):
        Settings(
            _env_file=None,
            environment=Environment.PRODUCTION,
            secret_key=SecretStr("a-genuinely-set-production-secret"),
            gemini_api_key=SecretStr(""),
        )


@pytest.mark.unit
def test_production_accepts_a_real_gemini_api_key() -> None:
    settings = Settings(
        _env_file=None,
        environment=Environment.PRODUCTION,
        secret_key=SecretStr("a-genuinely-set-production-secret"),
        gemini_api_key=SecretStr("a-real-gemini-key"),
    )
    assert settings.environment is Environment.PRODUCTION


@pytest.mark.unit
def test_gemini_api_key_is_not_exposed_in_repr_or_str() -> None:
    settings = Settings(_env_file=None, gemini_api_key=SecretStr("super-secret-gemini-key"))
    assert "super-secret-gemini-key" not in repr(settings)
    assert "super-secret-gemini-key" not in str(settings)


@pytest.mark.unit
def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    assert first is second
    get_settings.cache_clear()

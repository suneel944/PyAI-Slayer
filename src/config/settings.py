"""Pydantic settings management for PyAI-Slayer."""

import os
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings


def load_environments_config() -> dict:
    """
    Load environments.yaml configuration.

    Returns:
        Dictionary with environment configurations

    Raises:
        ConfigurationError: If file cannot be loaded or parsed
    """
    config_path = Path(__file__).parent / "environments.yaml"
    if not config_path.exists():
        return {}

    try:
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        # Lazy import to avoid circular dependency
        from core.infrastructure.exceptions import ConfigurationError

        raise ConfigurationError(
            f"Failed to load environments.yaml: {e}", config_key="environments.yaml"
        ) from e


def get_environment_name() -> str:
    """
    Get current environment name from ENVIRONMENT variable.

    Returns:
        Environment name (default: 'sandbox')
    """
    return os.getenv("ENVIRONMENT", "sandbox").lower()


def get_environment_config(env_name: str | None = None) -> dict[Any, Any]:
    """
    Get configuration for a specific environment.

    Args:
        env_name: Environment name (default: from ENVIRONMENT env var)

    Returns:
        Environment configuration dictionary
    """
    if env_name is None:
        env_name = get_environment_name()

    config = load_environments_config()
    environments = config.get("environments", {})

    if env_name not in environments:
        # Lazy import to avoid circular dependency
        from core.infrastructure.exceptions import ConfigurationError

        raise ConfigurationError(
            f"Environment '{env_name}' not found in environments.yaml. "
            f"Available: {list(environments.keys())}",
            config_key="ENVIRONMENT",
            config_value=env_name,
        )

    return cast(dict[Any, Any], environments[env_name])


class Settings(BaseSettings):
    """Application settings loaded from environment variables and environments.yaml."""

    # Environment configuration
    environment: str = get_environment_name()

    # URLs (can be overridden by environments.yaml)
    base_url: str = "https://govgpt.sandbox.dge.gov.ae"
    chat_url: str = "https://govgpt.sandbox.dge.gov.ae"

    # Authentication
    email: str = ""
    password: str = ""

    # Test configuration
    test_timeout: int = 30
    max_retries: int = 3
    parallel_workers: int = 4

    # Browser configuration
    browser: str = "chromium"
    headless: bool = True
    browser_timeout: int = 30000

    @field_validator("headless", mode="before")
    @classmethod
    def parse_headless(cls, v):
        """Parse headless value from string to boolean.

        Handles conversion from environment variable strings to boolean.
        Supports: "true"/"false", "1"/"0", "yes"/"no", "on"/"off"
        """
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            # Handle string values: "true", "false", "1", "0", "yes", "no", "on", "off"
            v_lower = v.lower().strip()
            if v_lower in ("true", "1", "yes", "on"):
                return True
            if v_lower in ("false", "0", "no", "off", ""):
                return False
        # Default to True if value is None or unrecognized (for safety)
        return bool(v)

    # AI/ML Model configuration
    semantic_model_name: str = "intfloat/multilingual-e5-base"
    arabic_semantic_model_name: str = "Omartificial-Intelligence-Space/mmbert-base-arabic-nli"

    # Validation thresholds
    semantic_similarity_threshold: float = 0.7
    arabic_semantic_similarity_threshold: float = 0.5
    hallucination_detection_threshold: float = 0.3
    cross_language_consistency_threshold: float = 0.7
    consistency_threshold: float = 0.6
    min_response_length: int = 10
    max_response_time_seconds: int = 180

    # Reporting
    report_dir: str = "reports"

    # Security
    enable_security_tests: bool = True
    zap_proxy_url: str = "http://localhost:8080"

    # Localization
    default_language: str = "en"
    supported_languages: str | list[str] = "en,ar"

    # Chatbot Configuration
    chatbot_name: str = "ChatBot"  # Name of the chatbot being tested (for test data)

    # Observability & Tracing
    enable_playwright_tracing: bool = True
    enable_prometheus_metrics: bool = False
    prometheus_port: int = 8000
    trace_dir: str = "traces"

    @field_validator("supported_languages", mode="before")
    @classmethod
    def parse_supported_languages(cls, v):
        """Parse comma-separated string to list."""
        if isinstance(v, str):
            return [lang.strip() for lang in v.split(",") if lang.strip()]
        return v

    def __init__(self, **kwargs):
        """
        Initialize settings with environment-specific overrides.

        Priority order: .env > environments.yaml > defaults

        Args:
            **kwargs: Settings overrides
        """
        # First, let Pydantic read from .env and environment variables (highest priority)
        super().__init__(**kwargs)

        # Then, apply environments.yaml overrides only if not already set from .env
        env_name = kwargs.get("environment") or get_environment_name()
        try:
            env_config = get_environment_config(env_name)
            # Override with environment config only if value wasn't set from .env/env vars
            for key, value in env_config.items():
                # Check if this value was already set from .env or environment variable
                env_var_name = key.upper()
                if env_var_name not in os.environ:
                    # Map legacy 'timeout' field to 'test_timeout'
                    if key == "timeout":
                        if "TEST_TIMEOUT" not in os.environ:
                            self.test_timeout = value
                    else:
                        # Only set if not already set from .env (check current value vs default)
                        current_value = getattr(self, key, None)
                        # Get default value from field definition
                        field_info = self.model_fields.get(key)
                        default_value = field_info.default if field_info else None
                        # Only override if current value is the default (meaning .env didn't set it)
                        if current_value == default_value:
                            setattr(self, key, value)
        except Exception:
            # If environment not found, use defaults
            # Using generic Exception to avoid circular dependency
            pass

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra fields from environment/config files
    }


settings = Settings()

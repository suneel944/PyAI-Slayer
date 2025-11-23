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

    # Fact-checker configuration
    fact_checker_model_name: str | None = None  # Primary model (None = use fallback list)
    fact_checker_fallback_models: str = "microsoft/deberta-large-mnli,roberta-large-mnli,facebook/bart-large-mnli"  # Comma-separated fallback models
    fact_checker_enabled: bool = True
    fact_checker_use_cuda: bool = True  # Use GPU if available
    fact_checker_cuda_device: int = 0  # CUDA device ID

    # Fact-checker thresholds
    fact_checker_strong_contradiction_threshold: float = 0.7  # Confidence for strong contradiction
    fact_checker_strong_entailment_threshold: float = 0.7  # Confidence for strong entailment
    fact_checker_weak_entailment_threshold: float = 0.5  # Confidence for weak entailment
    fact_checker_neutral_threshold: float = 0.8  # Confidence for neutral classification
    fact_checker_neutral_ratio_threshold: float = 0.7  # Ratio of neutrals to mark as unknown

    # RAG Reranker configuration
    rag_reranker_model_name: str | None = None  # Primary model (None = use fallback list)
    rag_reranker_fallback_models: str = "BAAI/bge-reranker-base,BAAI/bge-reranker-large,ms-marco-MiniLM-L-12-v2"  # Comma-separated fallback models
    rag_reranker_enabled: bool = True
    rag_reranker_use_cuda: bool = True  # Use GPU if available
    rag_reranker_cuda_device: int = 0  # CUDA device ID
    rag_fetch_url_content: bool = (
        True  # Fetch content from URLs for reranker scoring (improves accuracy)
    )
    rag_url_fetch_timeout: int = 10  # Timeout in seconds for URL content fetching
    rag_url_max_content_length: int = 10000  # Maximum content length to fetch per URL (chars)
    rag_url_max_retries: int = 3  # Maximum retry attempts per URL
    rag_url_retry_delay: float = 1.0  # Delay between retries in seconds
    rag_url_max_workers: int = 5  # Maximum parallel workers for URL fetching

    # RAG Metric Targets (optional - overrides calibration file if set)
    # Leave empty to use calibration recommendations from data/rag_calibration_recommendations.json
    rag_target_retrieval_recall_5: float | None = None
    rag_target_retrieval_precision_5: float | None = None
    rag_target_context_relevance: float | None = None
    rag_target_context_coverage: float | None = None
    rag_target_context_intrusion: float | None = None  # Lower is better
    rag_target_gold_context_match: float | None = None
    rag_target_reranker_score: float | None = None  # 0-1 scale

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

    @field_validator(
        "rag_target_retrieval_recall_5",
        "rag_target_retrieval_precision_5",
        "rag_target_context_relevance",
        "rag_target_context_coverage",
        "rag_target_context_intrusion",
        "rag_target_gold_context_match",
        "rag_target_reranker_score",
        mode="before",
    )
    @classmethod
    def parse_rag_target(cls, v):
        """Parse RAG target values, converting empty strings to None."""
        if v == "" or v is None:
            return None
        if isinstance(v, int | float):
            return float(v)
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return None
            try:
                return float(v)
            except ValueError:
                return None
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
                        # Get default value from field definition (access from class, not instance)
                        field_info = Settings.model_fields.get(key)
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

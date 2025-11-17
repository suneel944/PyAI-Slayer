"""Test-specific configuration extracted from settings."""

from dataclasses import dataclass


@dataclass
class TestConfig:
    """
    Test configuration data class.

    Provides clean interface for test fixtures and page objects
    without direct coupling to Settings/environment variables.
    """

    # URLs
    base_url: str
    chat_url: str

    # Authentication
    email: str
    password: str

    # Timeouts and limits
    test_timeout: int
    max_retries: int
    max_response_time: int
    browser_timeout: int

    # Validation thresholds
    semantic_threshold: float
    arabic_semantic_threshold: float
    hallucination_threshold: float
    cross_language_threshold: float
    consistency_threshold: float
    min_response_length: int

    # Browser
    browser: str
    headless: bool

    # Observability
    enable_playwright_tracing: bool
    enable_prometheus_metrics: bool

    @classmethod
    def from_settings(cls, settings):
        """
        Create TestConfig from Settings instance.

        Args:
            settings: Settings instance

        Returns:
            TestConfig instance
        """
        return cls(
            base_url=settings.base_url,
            chat_url=settings.chat_url,
            email=settings.email,
            password=settings.password,
            test_timeout=settings.test_timeout,
            max_retries=settings.max_retries,
            max_response_time=settings.max_response_time_seconds,
            browser_timeout=settings.browser_timeout,
            semantic_threshold=settings.semantic_similarity_threshold,
            arabic_semantic_threshold=settings.arabic_semantic_similarity_threshold,
            hallucination_threshold=settings.hallucination_detection_threshold,
            cross_language_threshold=settings.cross_language_consistency_threshold,
            consistency_threshold=settings.consistency_threshold,
            min_response_length=settings.min_response_length,
            browser=settings.browser,
            headless=settings.headless,
            enable_playwright_tracing=settings.enable_playwright_tracing,
            enable_prometheus_metrics=settings.enable_prometheus_metrics,
        )

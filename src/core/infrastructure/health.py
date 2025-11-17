"""Health check system for framework dependencies."""

import time
from enum import Enum
from typing import Any

from loguru import logger

from config.settings import settings
from core.infrastructure.exceptions import HealthCheckError


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheck:
    """Base class for health checks."""

    def __init__(self, name: str, timeout: float = 5.0):
        """
        Initialize health check.

        Args:
            name: Name of the health check
            timeout: Timeout in seconds for the check
        """
        self.name = name
        self.timeout = timeout

    def check(self) -> tuple[HealthStatus, dict[str, Any]]:
        """
        Perform health check.

        Returns:
            Tuple of (status, details)
        """
        raise NotImplementedError("Subclasses must implement check()")

    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        status, _ = self.check()
        return status == HealthStatus.HEALTHY


class ModelHealthCheck(HealthCheck):
    """Health check for ML models."""

    def __init__(self, validator_instance: Any, timeout: float = 10.0):
        """
        Initialize model health check.

        Args:
            validator_instance: AIResponseValidator instance to check
            timeout: Timeout for health check
        """
        super().__init__("model_health", timeout)
        self.validator = validator_instance

    def check(self) -> tuple[HealthStatus, dict[str, Any]]:
        """
        Check ML model health.

        Returns:
            Tuple of (status, details)
        """
        details: dict[str, Any] = {
            "model_name": settings.semantic_model_name,
            "check_time": time.time(),
        }

        try:
            start_time = time.time()

            # Try to encode a test string
            test_text = "health check"
            model = self.validator.semantic_model
            embedding = model.encode(test_text, convert_to_numpy=True)

            elapsed = time.time() - start_time
            details["response_time"] = elapsed
            details["embedding_dimension"] = len(embedding)

            if elapsed > self.timeout:
                logger.warning(f"Model health check slow: {elapsed:.2f}s")
                return HealthStatus.DEGRADED, details

            if len(embedding) == 0:
                logger.error("Model returned empty embedding")
                return HealthStatus.UNHEALTHY, details

            return HealthStatus.HEALTHY, details

        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            details["error"] = str(e)
            return HealthStatus.UNHEALTHY, details


class BrowserHealthCheck(HealthCheck):
    """Health check for browser connectivity."""

    def __init__(self, browser_manager: Any, timeout: float = 5.0):
        """
        Initialize browser health check.

        Args:
            browser_manager: BrowserManager instance to check
            timeout: Timeout for health check
        """
        super().__init__("browser_health", timeout)
        self.browser_manager = browser_manager

    def check(self) -> tuple[HealthStatus, dict[str, Any]]:
        """
        Check browser health.

        Returns:
            Tuple of (status, details)
        """
        details: dict[str, Any] = {"browser_type": settings.browser, "check_time": time.time()}

        try:
            # Check if browser is initialized
            if not self.browser_manager.browser:
                details["error"] = "Browser not initialized"
                return HealthStatus.UNHEALTHY, details

            # Check if browser is still connected
            if self.browser_manager.browser.is_connected():
                details["connected"] = True
                return HealthStatus.HEALTHY, details
            else:
                details["error"] = "Browser not connected"
                return HealthStatus.UNHEALTHY, details

        except Exception as e:
            logger.error(f"Browser health check failed: {e}")
            details["error"] = str(e)
            return HealthStatus.UNHEALTHY, details


class HealthChecker:
    """Centralized health checker for all components."""

    def __init__(self):
        """Initialize health checker."""
        self.checks: dict[str, HealthCheck] = {}

    def register(self, name: str, check: HealthCheck):
        """
        Register a health check.

        Args:
            name: Name identifier for the check
            check: HealthCheck instance
        """
        self.checks[name] = check
        logger.info(f"Registered health check: {name}")

    def check_all(self) -> dict[str, tuple[HealthStatus, dict[str, Any]]]:
        """
        Run all registered health checks.

        Returns:
            Dictionary mapping check names to (status, details) tuples
        """
        results = {}
        for name, check in self.checks.items():
            try:
                status, details = check.check()
                results[name] = (status, details)
            except Exception as e:
                logger.error(f"Health check {name} raised exception: {e}")
                results[name] = (HealthStatus.UNKNOWN, {"error": str(e)})
        return results

    def check_component(self, name: str) -> tuple[HealthStatus, dict[str, Any]]:
        """
        Check a specific component.

        Args:
            name: Name of the component to check

        Returns:
            Tuple of (status, details)

        Raises:
            HealthCheckError: If component not found
        """
        if name not in self.checks:
            raise HealthCheckError(f"Health check '{name}' not registered", component=name)

        try:
            return self.checks[name].check()
        except Exception as e:
            logger.error(f"Health check {name} failed: {e}")
            raise HealthCheckError(
                f"Health check '{name}' failed: {e}",
                component=name,
                status=HealthStatus.UNHEALTHY.value,
            ) from e

    def is_all_healthy(self) -> bool:
        """Check if all components are healthy."""
        results = self.check_all()
        return all(status == HealthStatus.HEALTHY for status, _ in results.values())

    def get_unhealthy_components(self) -> list[str]:
        """Get list of unhealthy component names."""
        results = self.check_all()
        return [name for name, (status, _) in results.items() if status != HealthStatus.HEALTHY]

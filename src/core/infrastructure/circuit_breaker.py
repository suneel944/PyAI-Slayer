"""Circuit breaker pattern for external dependencies."""

import time
from collections.abc import Callable
from enum import Enum
from typing import Any, Generic, TypeVar

from loguru import logger

from core.infrastructure.exceptions import FrameworkError

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(FrameworkError):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str, state: CircuitState):
        """
        Initialize circuit breaker error.

        Args:
            message: Error message
            state: Current circuit breaker state
        """
        super().__init__(message, {"state": state.value})
        self.state = state


class CircuitBreaker:
    """Circuit breaker implementation for resilient service calls."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] = Exception,
        name: str = "circuit_breaker",
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
            expected_exception: Exception type that counts as failure
            name: Name identifier for this circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.success_count = 0

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Original exception from function call
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_recovery():
                logger.info(f"Circuit breaker {self.name}: Attempting recovery (half-open)")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker {self.name} is OPEN. Service unavailable.", self.state
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception:
            self._on_failure()
            raise

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return False
        return (time.time() - self.last_failure_time) >= self.recovery_timeout

    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 2:  # Require 2 successes to close
                logger.info(f"Circuit breaker {self.name}: Service recovered, closing circuit")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            logger.warning(f"Circuit breaker {self.name}: Recovery failed, opening circuit")
            self.state = CircuitState.OPEN
            self.success_count = 0
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                logger.error(
                    f"Circuit breaker {self.name}: Failure threshold reached ({self.failure_count}), "
                    f"opening circuit"
                )
                self.state = CircuitState.OPEN

    def reset(self):
        """Manually reset circuit breaker to closed state."""
        logger.info(f"Circuit breaker {self.name}: Manually reset")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0

    def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self.state

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }


class CircuitBreakerWrapper(Generic[T]):
    """Wrapper class for circuit breaker decorated functions."""

    def __init__(self, func: Callable[..., T], breaker: CircuitBreaker):
        """Initialize wrapper."""
        self.func = func
        self.circuit_breaker = breaker
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Call wrapped function with circuit breaker."""
        return self.circuit_breaker.call(self.func, *args, **kwargs)


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type[Exception] = Exception,
    name: str | None = None,
):
    """
    Decorator for applying circuit breaker to functions.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time in seconds before attempting recovery
        expected_exception: Exception type that counts as failure
        name: Name identifier for circuit breaker

    Returns:
        Decorated function with circuit breaker protection
    """
    breaker = CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception,
        name=name or "circuit_breaker",
    )

    def decorator(func: Callable[..., T]) -> CircuitBreakerWrapper[T]:
        return CircuitBreakerWrapper(func, breaker)

    return decorator

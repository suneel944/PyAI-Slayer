"""Retry mechanisms with exponential backoff for resilient operations."""

import time
from collections.abc import Callable
from functools import wraps
from typing import Any, Generic, TypeVar

from loguru import logger

from core.infrastructure.exceptions import FrameworkError

T = TypeVar("T")


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """
        Initialize retry configuration.

        Args:
            max_attempts: Maximum number of retry attempts
            initial_delay: Initial delay in seconds before first retry
            max_delay: Maximum delay in seconds between retries
            exponential_base: Base for exponential backoff calculation
            jitter: Whether to add random jitter to delays
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


def retry_with_backoff(
    config: RetryConfig | None = None,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[int, Exception], None] | None = None,
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        config: Retry configuration (uses default if None)
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback called on each retry attempt

    Returns:
        Decorated function with retry logic
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt >= config.max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {config.max_attempts} attempts: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(
                        config.initial_delay * (config.exponential_base ** (attempt - 1)),
                        config.max_delay,
                    )

                    # Add jitter if enabled
                    if config.jitter:
                        import random

                        jitter_amount = delay * 0.1  # 10% jitter
                        delay += random.uniform(-jitter_amount, jitter_amount)
                        delay = max(0, delay)  # Ensure non-negative

                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt}/{config.max_attempts}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    if on_retry:
                        on_retry(attempt, e)

                    time.sleep(delay)

            # Should never reach here, but type checker needs it
            if last_exception:
                raise last_exception
            raise FrameworkError("Retry logic failed unexpectedly")

        return wrapper

    return decorator


class RetryableOperation(Generic[T]):
    """Context manager for retryable operations."""

    def __init__(
        self,
        operation: Callable[[], T],
        config: RetryConfig | None = None,
        exceptions: tuple[type[Exception], ...] = (Exception,),
    ):
        """
        Initialize retryable operation.

        Args:
            operation: Function to execute with retries
            config: Retry configuration
            exceptions: Exceptions to catch and retry
        """
        self.operation = operation
        self.config = config or RetryConfig()
        self.exceptions = exceptions

    def execute(self) -> T:
        """Execute operation with retry logic."""
        last_exception: Exception | None = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return self.operation()
            except self.exceptions as e:
                last_exception = e

                if attempt >= self.config.max_attempts:
                    logger.error(f"Operation failed after {self.config.max_attempts} attempts: {e}")
                    raise

                # Calculate delay
                delay = min(
                    self.config.initial_delay * (self.config.exponential_base ** (attempt - 1)),
                    self.config.max_delay,
                )

                if self.config.jitter:
                    import random

                    jitter_amount = delay * 0.1
                    delay += random.uniform(-jitter_amount, jitter_amount)
                    delay = max(0, delay)

                logger.warning(
                    f"Operation failed (attempt {attempt}/{self.config.max_attempts}): {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                time.sleep(delay)

        if last_exception:
            raise last_exception
        raise FrameworkError("Retry logic failed unexpectedly")

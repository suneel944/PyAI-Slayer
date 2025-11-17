"""Unit tests for circuit breaker."""

import time
from unittest.mock import patch

import pytest

from core import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    circuit_breaker,
)


class TestCircuitBreaker:
    """Test suite for CircuitBreaker."""

    def test_init(self):
        """Test circuit breaker initialization."""
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.failure_threshold == 5

    def test_successful_call(self):
        """Test successful function call."""
        breaker = CircuitBreaker()

        def success_func():
            return "success"

        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_failure_below_threshold(self):
        """Test failures below threshold."""
        breaker = CircuitBreaker(failure_threshold=3)

        def failing_func():
            raise ValueError("Fail")

        # Fail twice (below threshold)
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 2

    def test_circuit_opens_on_threshold(self):
        """Test circuit opens when threshold reached."""
        breaker = CircuitBreaker(failure_threshold=2)

        def failing_func():
            raise ValueError("Fail")

        # Fail twice to reach threshold
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 2

    def test_circuit_open_rejects_calls(self):
        """Test circuit rejects calls when open."""
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.state = CircuitState.OPEN
        breaker.last_failure_time = time.time()

        def any_func():
            return "should not execute"

        with pytest.raises(CircuitBreakerError) as exc_info:
            breaker.call(any_func)

        assert exc_info.value.state == CircuitState.OPEN

    @patch("time.time")
    def test_recovery_timeout(self, mock_time):
        """Test recovery timeout."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
        breaker.state = CircuitState.OPEN
        breaker.last_failure_time = 100.0

        # Not enough time passed
        mock_time.return_value = 150.0  # 50 seconds
        assert breaker._should_attempt_recovery() is False

        # Enough time passed
        mock_time.return_value = 170.0  # 70 seconds
        assert breaker._should_attempt_recovery() is True

    def test_half_open_recovery(self):
        """Test circuit recovery through half-open state."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        def failing_func():
            raise ValueError("Fail")

        # Open circuit
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.2)

        # First success in half-open
        def success_func():
            return "success"

        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitState.HALF_OPEN

        # Second success closes circuit
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        """Test half-open state reopens on failure."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        breaker.state = CircuitState.OPEN
        breaker.last_failure_time = time.time() - 1.0  # Past recovery timeout

        def failing_func():
            raise ValueError("Fail")

        # Attempt recovery (half-open)
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

    def test_reset(self):
        """Test manual circuit reset."""
        breaker = CircuitBreaker()
        breaker.state = CircuitState.OPEN
        breaker.failure_count = 5

        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.last_failure_time is None

    def test_get_stats(self):
        """Test getting circuit breaker statistics."""
        breaker = CircuitBreaker(name="test_breaker")
        stats = breaker.get_stats()

        assert stats["name"] == "test_breaker"
        assert stats["state"] == CircuitState.CLOSED.value
        assert stats["failure_count"] == 0
        assert "failure_threshold" in stats
        assert "recovery_timeout" in stats


class TestCircuitBreakerDecorator:
    """Test suite for circuit_breaker decorator."""

    def test_decorator_success(self):
        """Test decorator with successful function."""

        @circuit_breaker(failure_threshold=3, name="test")
        def success_func():
            return "success"

        result = success_func()
        assert result == "success"

    def test_decorator_failure(self):
        """Test decorator with failing function."""

        @circuit_breaker(failure_threshold=1, name="test")
        def failing_func():
            raise ValueError("Fail")

        # First failure
        with pytest.raises(ValueError):
            failing_func()

        # Second call should be rejected (circuit open)
        with pytest.raises(CircuitBreakerError):
            failing_func()

    def test_decorator_breaker_access(self):
        """Test accessing circuit breaker from decorated function."""

        @circuit_breaker(name="test_breaker")
        def test_func():
            return "success"

        assert hasattr(test_func, "circuit_breaker")
        assert test_func.circuit_breaker.name == "test_breaker"

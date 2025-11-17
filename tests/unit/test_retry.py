"""Unit tests for retry mechanisms."""

from unittest.mock import patch

import pytest

from core import RetryableOperation, RetryConfig, retry_with_backoff


class TestRetryConfig:
    """Test suite for RetryConfig."""

    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5, initial_delay=2.0, max_delay=120.0, exponential_base=3.0, jitter=False
        )
        assert config.max_attempts == 5
        assert config.initial_delay == 2.0
        assert config.max_delay == 120.0
        assert config.exponential_base == 3.0
        assert config.jitter is False


class TestRetryWithBackoff:
    """Test suite for retry_with_backoff decorator."""

    def test_success_first_attempt(self):
        """Test successful function on first attempt."""

        @retry_with_backoff(config=RetryConfig(max_attempts=3))
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    @patch("time.sleep")
    def test_retry_on_failure(self, mock_sleep):
        """Test retry on failure."""
        call_count = 0

        @retry_with_backoff(
            config=RetryConfig(max_attempts=3, initial_delay=0.1), exceptions=(ValueError,)
        )
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "success"

        result = failing_then_success()
        assert result == "success"
        assert call_count == 2
        assert mock_sleep.call_count == 1

    @patch("time.sleep")
    def test_max_attempts_exceeded(self, mock_sleep):
        """Test failure after max attempts."""

        @retry_with_backoff(
            config=RetryConfig(max_attempts=2, initial_delay=0.1), exceptions=(ValueError,)
        )
        def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_fails()

        assert mock_sleep.call_count == 1  # Retried once

    def test_wrong_exception_not_retried(self):
        """Test that wrong exception types are not retried."""

        @retry_with_backoff(config=RetryConfig(max_attempts=3), exceptions=(ValueError,))
        def raises_different_exception():
            raise TypeError("Wrong exception")

        with pytest.raises(TypeError):
            raises_different_exception()

    @patch("time.sleep")
    def test_exponential_backoff(self, mock_sleep):
        """Test exponential backoff calculation."""
        call_count = 0

        @retry_with_backoff(
            config=RetryConfig(
                max_attempts=4, initial_delay=1.0, exponential_base=2.0, jitter=False
            ),
            exceptions=(ValueError,),
        )
        def fails_multiple_times():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise ValueError("Fail")
            return "success"

        fails_multiple_times()

        # Check delays: 1.0, 2.0, 4.0
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert len(delays) == 3
        assert delays[0] == 1.0
        assert delays[1] == 2.0
        assert delays[2] == 4.0


class TestRetryableOperation:
    """Test suite for RetryableOperation."""

    def test_success(self):
        """Test successful operation."""
        operation = RetryableOperation(lambda: "success", config=RetryConfig(max_attempts=3))
        result = operation.execute()
        assert result == "success"

    @patch("time.sleep")
    def test_retry_on_failure(self, mock_sleep):
        """Test retry on failure."""
        call_count = 0

        def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Fail")
            return "success"

        operation = RetryableOperation(
            failing_operation, config=RetryConfig(max_attempts=3, initial_delay=0.1)
        )
        result = operation.execute()

        assert result == "success"
        assert call_count == 2

    @patch("time.sleep")
    def test_max_attempts_exceeded(self, mock_sleep):
        """Test failure after max attempts."""
        operation = RetryableOperation(
            lambda: (_ for _ in ()).throw(ValueError("Always fails")),
            config=RetryConfig(max_attempts=2, initial_delay=0.1),
            exceptions=(ValueError,),
        )

        with pytest.raises(ValueError):
            operation.execute()

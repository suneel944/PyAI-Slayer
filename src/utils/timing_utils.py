"""Generic timing and duration calculation utilities."""

import time

from loguru import logger


class TimingCalculator:
    """Generic timing calculator for performance metrics."""

    def __init__(self):
        """Initialize timing calculator."""
        self._start_time: float | None = None
        self._request_timestamp: float | None = None

    def start_timer(self) -> float:
        """
        Start a timer and return the start time.

        Returns:
            Start time as Unix timestamp (seconds since epoch)
        """
        self._start_time = time.time()
        self._request_timestamp = self._start_time
        return self._start_time

    def calculate_duration(self, end_time: float | None = None) -> float | None:
        """
        Calculate duration from start time to end time.

        Args:
            end_time: End time (defaults to current time if None)

        Returns:
            Duration in seconds, or None if start time not set
        """
        if self._start_time is None:
            return None

        if end_time is None:
            end_time = time.time()

        return end_time - self._start_time

    def calculate_ttft(
        self,
        first_token_time: float,
        request_start_time: float | None = None,
    ) -> float | None:
        """
        Calculate Time To First Token (TTFT).

        Args:
            first_token_time: Timestamp when first token was received
            request_start_time: Request start time (uses internal start_time if None)

        Returns:
            TTFT in seconds, or None if invalid
        """
        start = request_start_time or self._start_time
        if start is None:
            return None

        ttft = first_token_time - start

        # Validate TTFT is reasonable
        if not self.validate_ttft(ttft):
            return None

        return ttft

    def validate_ttft(self, ttft: float, min_value: float = 0.0, max_value: float = 300.0) -> bool:
        """
        Validate that TTFT is within reasonable bounds.

        Args:
            ttft: Time to first token in seconds
            min_value: Minimum acceptable value (default: 0.0)
            max_value: Maximum acceptable value in seconds (default: 300.0 = 5 minutes)

        Returns:
            True if valid, False otherwise
        """
        if ttft < min_value:
            logger.warning(f"Negative TTFT detected: {ttft:.3f}s, this indicates a timing issue")
            return False

        if ttft > max_value:
            logger.warning(
                f"Unusually large TTFT detected: {ttft:.3f}s, "
                f"this may indicate incorrect message tracking"
            )
            return False

        return True

    def validate_timestamp(
        self,
        message_timestamp: float,
        request_timestamp: float | None = None,
        tolerance_seconds: float = 1.0,
    ) -> bool:
        """
        Validate that a message timestamp is recent relative to request timestamp.

        Args:
            message_timestamp: Message timestamp (Unix timestamp in seconds)
            request_timestamp: Request timestamp (uses internal timestamp if None)
            tolerance_seconds: Tolerance for clock differences (default: 1.0 second)

        Returns:
            True if message timestamp is valid (recent or equal to request), False otherwise
        """
        request_ts = request_timestamp or self._request_timestamp
        if request_ts is None:
            return True  # Can't validate without request timestamp

        if message_timestamp <= 0:
            return False  # Invalid timestamp

        # Message should be after or equal to request (with tolerance)
        # Allow tolerance for clock differences
        return message_timestamp >= (request_ts - tolerance_seconds)

    def calculate_elapsed(self, start_time: float, end_time: float | None = None) -> float:
        """
        Calculate elapsed time between two timestamps.

        Args:
            start_time: Start timestamp
            end_time: End timestamp (defaults to current time if None)

        Returns:
            Elapsed time in seconds
        """
        if end_time is None:
            end_time = time.time()

        return end_time - start_time

    def reset(self):
        """Reset all timing state."""
        self._start_time = None
        self._request_timestamp = None

    @property
    def start_time(self) -> float | None:
        """Get the start time."""
        return self._start_time

    @property
    def request_timestamp(self) -> float | None:
        """Get the request timestamp."""
        return self._request_timestamp


def calculate_response_time(start_time: float, end_time: float | None = None) -> float:
    """
    Calculate response time between two timestamps.

    Args:
        start_time: Start timestamp
        end_time: End timestamp (defaults to current time if None)

    Returns:
        Response time in seconds
    """
    if end_time is None:
        end_time = time.time()

    return end_time - start_time


def calculate_ttft(
    request_start_time: float,
    first_token_time: float,
    min_value: float = 0.0,
    max_value: float = 300.0,
) -> float | None:
    """
    Calculate and validate Time To First Token (TTFT).

    Args:
        request_start_time: Request start timestamp
        first_token_time: First token arrival timestamp
        min_value: Minimum acceptable TTFT (default: 0.0)
        max_value: Maximum acceptable TTFT in seconds (default: 300.0)

    Returns:
        TTFT in seconds, or None if invalid
    """
    ttft = first_token_time - request_start_time

    if ttft < min_value:
        logger.warning(f"Negative TTFT detected: {ttft:.3f}s, this indicates a timing issue")
        return None

    if ttft > max_value:
        logger.warning(
            f"Unusually large TTFT detected: {ttft:.3f}s, "
            f"this may indicate incorrect message tracking"
        )
        return None

    return ttft


def validate_timestamp_recent(
    message_timestamp: float,
    request_timestamp: float,
    tolerance_seconds: float = 1.0,
) -> bool:
    """
    Validate that a message timestamp is recent relative to request timestamp.

    Args:
        message_timestamp: Message timestamp (Unix timestamp in seconds)
        request_timestamp: Request timestamp (Unix timestamp in seconds)
        tolerance_seconds: Tolerance for clock differences (default: 1.0 second)

    Returns:
        True if message timestamp is valid (recent or equal to request), False otherwise
    """
    if message_timestamp <= 0:
        return False  # Invalid timestamp

    # Message should be after or equal to request (with tolerance)
    # Allow tolerance for clock differences
    return message_timestamp >= (request_timestamp - tolerance_seconds)

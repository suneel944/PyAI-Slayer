"""Performance metrics calculator."""


class PerformanceMetricsCalculator:
    """Calculate performance metrics (latency, throughput, etc.)."""

    def calculate(
        self,
        duration: float | None = None,
        response_tokens: int | None = None,
        first_token_time: float | None = None,
        response: str | None = None,
        response_length: int | None = None,
    ) -> dict[str, float]:
        """
        Calculate performance metrics.

        Args:
            duration: Total response duration in seconds
            response_tokens: Number of tokens in response
            first_token_time: Time to first token in seconds
            response: AI response text (for token estimation)
            response_length: Response length in characters

        Returns:
            Dictionary of performance metrics
        """
        metrics: dict[str, float] = {}

        # E2E Latency
        if duration is not None:
            metrics["e2e_latency"] = duration * 1000  # Convert to ms

        # Estimate token count if not provided
        if response_tokens is None:
            if response:
                estimated_tokens = max(1, len(response) / 3.5)
                response_tokens = int(estimated_tokens)
            elif response_length:
                estimated_tokens = max(1, response_length / 3.5)
                response_tokens = int(estimated_tokens)

        # TTFT (Time to First Token)
        if first_token_time is not None:
            metrics["ttft"] = first_token_time * 1000  # Convert to ms

        # Token Latency
        if response_tokens and duration and response_tokens > 0:
            metrics["token_latency"] = (duration / response_tokens) * 1000  # ms per token

        # Throughput
        if response_tokens and duration and response_tokens > 0:
            metrics["throughput"] = response_tokens / duration  # tokens per second

        return metrics

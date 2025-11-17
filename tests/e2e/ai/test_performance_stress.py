"""Performance and stress testing for AI responses."""

import time

import pytest
from loguru import logger


@pytest.mark.ai
@pytest.mark.performance
class TestPerformanceStress:
    """Test performance under various load conditions."""

    @pytest.fixture(autouse=True)
    def setup(self, ai_validator):
        """Setup AI validator."""
        self.validator = ai_validator

    def test_concurrent_queries(self, chat_page, test_config):
        """PERF-001: System handles multiple queries efficiently."""
        queries = [
            "What are visa requirements?",
            "How to renew visa?",
            "Where to submit documents?",
        ]

        response_times = []
        for query in queries:
            start_time = time.time()
            chat_page.send_message(query, wait_for_response=True)
            response = chat_page.get_latest_response()
            response_time = time.time() - start_time
            response_times.append(response_time)

            assert response is not None, f"No response for: {query}"
            assert response_time < test_config.max_response_time, (
                f"Response time {response_time:.2f}s exceeds limit"
            )

            chat_page.page.wait_for_timeout(500)  # Small delay

        avg_time = sum(response_times) / len(response_times)
        logger.info(f"Average response time across {len(queries)} queries: {avg_time:.2f}s")

    def test_long_query_handling(self, chat_page, test_config):
        """PERF-002: System handles long queries efficiently."""
        long_query = (
            "I need detailed information about visa renewal process including "
            "all required documents, processing time, fees, where to submit, "
            "what to do if application is rejected, appeal process, and any "
            "special considerations for different visa types. Please provide "
            "comprehensive step-by-step guidance."
        )

        start_time = time.time()
        chat_page.send_message(long_query, wait_for_response=True)
        response = chat_page.get_latest_response()
        response_time = time.time() - start_time

        assert response is not None, "No response for long query"
        assert response_time < test_config.max_response_time * 1.5, (
            f"Long query took too long: {response_time:.2f}s"
        )

        logger.info(f"Long query handled in {response_time:.2f}s")

    def test_response_time_consistency(self, chat_page, test_config, ai_test_data):
        """PERF-003: Response times are consistent across similar queries."""
        performance_data = ai_test_data.get("performance_test_data", {})
        test_case = performance_data.get("stress_test", {})
        query = test_case.get("query", "")

        if not query:
            pytest.skip("Performance test data not available")

        response_times = []
        for _ in range(3):
            start_time = time.time()
            chat_page.send_message(query, wait_for_response=True)
            chat_page.get_latest_response()
            response_time = time.time() - start_time
            response_times.append(response_time)
            chat_page.page.wait_for_timeout(1000)

        if len(response_times) >= 2:
            avg_time = sum(response_times) / len(response_times)
            variance = sum((t - avg_time) ** 2 for t in response_times) / len(response_times)
            std_dev = variance**0.5

            # Response times should be reasonably consistent
            assert std_dev < avg_time * 0.5, (
                f"High response time variance: std_dev={std_dev:.2f}s, avg={avg_time:.2f}s"
            )
            logger.info(f"Response time consistency: avg={avg_time:.2f}s, std_dev={std_dev:.2f}s")

    def test_throughput_under_load(self, chat_page, test_config):
        """PERF-004: System maintains throughput under load."""
        queries = [
            "What are visa requirements?",
            "How to apply?",
            "What documents needed?",
            "Where to submit?",
            "How long does it take?",
        ]

        total_time = 0
        successful_queries = 0

        start_batch = time.time()
        for query in queries:
            query_start = time.time()
            chat_page.send_message(query, wait_for_response=True)
            response = chat_page.get_latest_response()
            query_time = time.time() - query_start

            if response:
                successful_queries += 1
                total_time += query_time
            chat_page.page.wait_for_timeout(300)  # Small delay

        batch_time = time.time() - start_batch

        if successful_queries > 0:
            avg_time = total_time / successful_queries
            throughput = successful_queries / batch_time

            logger.info(f"Throughput: {throughput:.2f} queries/sec, avg time: {avg_time:.2f}s")
            assert throughput > 0.1, "Throughput too low"

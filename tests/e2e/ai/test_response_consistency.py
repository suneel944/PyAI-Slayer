"""Test response consistency across multiple runs."""

import pytest
from loguru import logger


@pytest.mark.ai
class TestResponseConsistency:
    """Test AI response consistency."""

    @pytest.fixture(autouse=True)
    def setup(self, ai_validator, ai_test_data):
        """Setup AI validator."""
        self.validator = ai_validator
        self.test_data = ai_test_data

    def test_consistency_across_retries(self, chat_page, page, test_config):
        """AI-004: Consistency across retries."""
        # Use first query from common_queries if available
        common_queries = self.test_data.get("common_queries", [])
        if common_queries and len(common_queries) > 0:
            query = common_queries[0].get("prompt", "")
        else:
            query = "How do I renew my residence visa?"

        if not query:
            pytest.skip("No test query available")
        responses = []

        for _ in range(3):
            chat_page.send_message(query, wait_for_response=True)
            response = chat_page.get_latest_response()
            if response:
                responses.append(response)
            page.wait_for_timeout(2000)

        assert len(responses) >= 2, "Need at least 2 responses for consistency check"

        consistency_score = self.validator.check_consistency(responses)

        assert (
            consistency_score >= test_config.consistency_threshold
        ), f"Responses not consistent (score: {consistency_score:.3f}, threshold: {test_config.consistency_threshold})"

        logger.info(f"Consistency score: {consistency_score:.3f}")

    def test_cross_language_consistency(self, chat_page, test_config):
        """AI-005: Cross-language consistency."""
        consistency_data = self.test_data.get("consistency_test_data", {})
        test_case = consistency_data.get("visa_renewal", {})
        en_query = test_case.get("en_query", "")
        ar_query = test_case.get("ar_query", "")

        if not en_query or not ar_query:
            pytest.skip("Cross-language consistency test data not available")

        chat_page.send_message(en_query, wait_for_response=True)
        en_response = chat_page.get_latest_response()
        chat_page.send_message(ar_query, wait_for_response=True)
        ar_response = chat_page.get_latest_response()

        assert en_response is not None, "No English response"
        assert ar_response is not None, "No Arabic response"

        is_consistent, similarity = self.validator.validate_cross_language(en_response, ar_response)

        assert (
            is_consistent
        ), f"Cross-language responses not consistent (similarity: {similarity:.3f})"

        logger.info(f"Cross-language consistency: {similarity:.3f}")

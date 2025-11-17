"""Test output stability across multiple runs."""

import pytest
from loguru import logger


@pytest.mark.ai
@pytest.mark.stability
class TestOutputStability:
    """Test output stability and consistency across runs."""

    @pytest.fixture(autouse=True)
    def setup(self, ai_validator, ai_test_data):
        """Setup AI validator."""
        self.validator = ai_validator
        self.test_data = ai_test_data

    def test_response_consistency_same_query(self, chat_page, test_config):
        """STABILITY-001: Same query produces consistent responses."""
        stability_data = self.test_data.get("stability_test_data", {})
        test_case = stability_data.get("visa_renewal", {})
        query = test_case.get("query", "")

        if not query:
            pytest.skip("Stability test data not available")

        responses = []
        # Run same query multiple times
        for _i in range(3):
            chat_page.send_message(query, wait_for_response=True)
            response = chat_page.get_latest_response()
            if response:
                responses.append(response)
            # Small delay between requests
            chat_page.page.wait_for_timeout(1000)

        assert len(responses) >= 2, "Need at least 2 responses for stability check"

        # Calculate similarity between responses
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                is_relevant, similarity = self.validator.validate_relevance(
                    responses[i], responses[j], threshold=0.0
                )
                similarities.append(similarity)

        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            # Responses should be reasonably similar (not identical, but consistent)
            assert avg_similarity >= 0.5, f"Low response consistency: {avg_similarity:.3f}"
            logger.info(
                f"Response consistency: {avg_similarity:.3f} (across {len(responses)} runs)"
            )

    def test_determinism_score(self, chat_page, test_config):
        """STABILITY-002: Response determinism across multiple runs."""
        stability_data = self.test_data.get("stability_test_data", {})
        test_case = stability_data.get("business_license", {})
        query = test_case.get("query", "")

        if not query:
            pytest.skip("Determinism test data not available")

        responses = []
        for _ in range(3):
            chat_page.send_message(query, wait_for_response=True)
            response = chat_page.get_latest_response()
            if response:
                responses.append(response)
            chat_page.page.wait_for_timeout(1000)

        if len(responses) >= 2:
            # Check if responses are semantically similar
            from core.ai.ai_validator import AIResponseValidator

            validator = AIResponseValidator()

            similarities = []
            for i in range(len(responses) - 1):
                is_relevant, similarity = validator.validate_relevance(
                    responses[i], responses[i + 1], threshold=0.0
                )
                similarities.append(similarity)

            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                logger.info(f"Determinism score: {avg_similarity:.3f}")

    def test_output_stability_with_variations(self, chat_page, test_config):
        """STABILITY-003: Output remains stable with slight query variations."""
        variations = [
            "What are the visa renewal requirements?",
            "Tell me about visa renewal requirements",
            "I need to know visa renewal requirements",
        ]

        responses = []
        for query in variations:
            chat_page.send_message(query, wait_for_response=True)
            response = chat_page.get_latest_response()
            if response:
                responses.append(response)
            chat_page.page.wait_for_timeout(1000)

        if len(responses) >= 2:
            # Responses to similar queries should be similar
            similarities = []
            for i in range(len(responses) - 1):
                is_relevant, similarity = self.validator.validate_relevance(
                    responses[i], responses[i + 1], threshold=0.0
                )
                similarities.append(similarity)

            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                assert avg_similarity >= 0.4, f"Low stability with variations: {avg_similarity:.3f}"
                logger.info(f"Stability with variations: {avg_similarity:.3f}")

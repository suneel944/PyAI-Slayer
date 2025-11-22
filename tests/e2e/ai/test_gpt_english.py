"""AI response validation tests for English."""

import time

import pytest
from loguru import logger

from config.settings import settings
from utils.helpers import get_test_data


@pytest.mark.ai
@pytest.mark.english
@pytest.mark.smoke
class TestGPTEnglish:
    """Test AI responses in English."""

    @pytest.fixture(autouse=True)
    def setup(self, ai_validator, ai_test_data):
        """Setup AI validator."""
        self.validator = ai_validator
        self.test_data = ai_test_data

    def test_basic_query_response(self, chat_page, test_config):
        """AI-001: Basic query gets relevant response."""
        # Use first query from common_queries if available
        common_queries = self.test_data.get("common_queries", [])
        query = None
        expected_response = None
        reference = None

        if common_queries and len(common_queries) > 0:
            query_entry = common_queries[0]
            query = query_entry.get("prompt", "")
            expected_response = query_entry.get("expected_response")
            reference = query_entry.get("reference")
        else:
            query = "How do I renew my residence visa?"

        if not query:
            pytest.skip("No test query available")
        start_time = time.time()

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        response_time = time.time() - start_time

        assert response is not None, "No response received"
        assert len(response) > 0, "Response is empty"
        assert (
            response_time < test_config.max_response_time
        ), f"Response time {response_time:.2f}s exceeds {test_config.max_response_time}s"

        is_relevant, similarity = self.validator.validate_relevance(query, response)
        assert is_relevant, f"Response not relevant (similarity: {similarity:.3f})"

        # Store reference/expected_response for BLEU, ROUGE, BERTScore calculations
        from core.ai.ai_validator import _store_validation_data

        _store_validation_data(
            query=query,
            response=response,
            metrics={
                "validation_type": "relevance",
                "similarity_score": float(similarity),
                "is_relevant": is_relevant,
            },
            expected_response=expected_response,
            reference=reference
            or expected_response,  # Use reference if available, fallback to expected_response
        )

        logger.info(f"Response relevance: {similarity:.3f}, time: {response_time:.2f}s")

    def test_response_completeness(self, chat_page, test_config):
        """AI-003: Response is complete."""
        # Use query from common_queries if available
        common_queries = self.test_data.get("common_queries", [])
        query = None
        for q in common_queries:
            if (
                "working hours" in q.get("prompt", "").lower()
                or "government offices" in q.get("prompt", "").lower()
            ):
                query = q.get("prompt", "")
                break
        if not query and common_queries:
            query = common_queries[0].get("prompt", "")

        if not query:
            pytest.skip("No test query available")
        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        quality_checks = self.validator.validate_response_quality(
            response, min_length=test_config.min_response_length
        )

        assert quality_checks["has_minimum_length"], "Response too short"
        assert quality_checks["has_no_html_tags"], "Response contains HTML tags"

        logger.info("Response completeness validated")

    def test_response_time(self, chat_page, test_config):
        """AI-008: Response time is acceptable."""
        # Use query from common_queries if available
        common_queries = self.test_data.get("common_queries", [])
        query = None
        for q in common_queries:
            if "visa" in q.get("prompt", "").lower():
                query = q.get("prompt", "")
                break
        if not query and common_queries:
            query = common_queries[0].get("prompt", "")

        if not query:
            pytest.skip("No test query available")
        start_time = time.time()

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        response_time = time.time() - start_time

        assert response is not None, "No response"
        assert (
            response_time < test_config.max_response_time
        ), f"Response took {response_time:.2f}s, exceeds {test_config.max_response_time}s"

        logger.info(f"Response time: {response_time:.2f}s")

    def test_no_hallucination(self, chat_page):
        """AI-002: Response has no hallucination."""
        hallucination_data = self.test_data.get("hallucination_test_data", {})
        test_case = hallucination_data.get("no_fabricated_info", {})
        query = test_case.get("query", "")
        known_facts = test_case.get("known_facts", [])

        if not query or not known_facts:
            # Fallback to common_queries
            common_queries = self.test_data.get("common_queries", [])
            if common_queries:
                query = common_queries[0].get("prompt", "")
                known_facts = []

        if not query:
            pytest.skip("No test query available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        has_hallucination, conflicting = self.validator.detect_hallucination(response, known_facts)

        assert not has_hallucination, f"Potential hallucination detected: {conflicting}"
        logger.info("No hallucination detected")

    def test_clean_formatting(self, chat_page):
        """AI-006: Response has clean formatting."""
        # Use query from common_queries if available
        common_queries = self.test_data.get("common_queries", [])
        query = None
        for q in common_queries:
            if (
                "business" in q.get("prompt", "").lower()
                or "license" in q.get("prompt", "").lower()
            ):
                query = q.get("prompt", "")
                break
        if not query and common_queries:
            query = common_queries[0].get("prompt", "")

        if not query:
            pytest.skip("No test query available")
        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        quality_checks = self.validator.validate_response_quality(response)
        assert quality_checks["has_no_html_tags"], "Response contains HTML tags"

        logger.info("Response formatting is clean")

    def test_fallback_message_handling(self, chat_page):
        """AI-007: Fallback message handling."""

        query = "asdfjkl;"
        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        is_fallback = self.validator.detect_fallback_message(response)

        logger.info(f"Fallback message detected: {is_fallback}")

    def test_greeting_with_chatbot_name(self, chat_page, test_config):
        """AI-009: Greeting using chatbot name from settings."""
        # Use chatbot name from .env configuration (not hardcoded)
        greeting = f"Hello, {settings.chatbot_name}!"

        chat_page.send_message(greeting, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response received"
        assert len(response) > 0, "Response is empty"

        # Validate that response acknowledges the greeting
        is_relevant, similarity = self.validator.validate_relevance(greeting, response)
        assert is_relevant, f"Response not relevant to greeting (similarity: {similarity:.3f})"

        logger.info(
            f"Greeting test passed with chatbot name '{settings.chatbot_name}': similarity={similarity:.3f}"
        )

    @pytest.mark.parametrize("test_case", get_test_data("test-data-en.json", "common_queries"))
    def test_common_queries(self, chat_page, test_config, test_case):
        """Test common queries from test data."""
        # Handle special placeholder for chatbot name greeting
        query = test_case["prompt"]
        if query == "USE_CHATBOT_NAME_GREETING":
            query = f"Hello, {settings.chatbot_name}!"
        start_time = time.time()

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()
        response_time = time.time() - start_time

        assert response is not None, f"No response for query: {query}"
        assert len(response) >= test_case.get(
            "min_response_length", 10
        ), f"Response too short for: {query}"

        max_time = test_case.get("max_response_time_seconds", test_config.max_response_time)
        assert (
            response_time < max_time
        ), f"Response time {response_time:.2f}s exceeds limit of {max_time}s"

        # Primary validation: semantic relevance between query and response
        is_relevant, similarity = self.validator.validate_relevance(query, response)
        assert is_relevant, f"Response not relevant for: {query} (similarity: {similarity:.3f})"

        # Secondary validation: semantic concept coverage using expected_keywords
        # This replaces rigid keyword matching with semantic analysis
        expected_keywords = test_case.get("expected_keywords", [])
        if expected_keywords:
            # Use semantic validation to check if response covers expected concepts
            # Require at least 50% of concepts to be covered (default behavior)
            is_valid, avg_concept_score, concept_scores = self.validator.validate_semantic_concepts(
                response, expected_keywords
            )

            assert is_valid, (
                f"Response doesn't semantically cover expected concepts for: {query}. "
                f"Avg concept score: {avg_concept_score:.3f}, "
                f"Concept scores: {concept_scores}"
            )

            logger.info(
                f"Query {test_case['id']} passed: relevance={similarity:.3f}, "
                f"concept coverage={avg_concept_score:.3f}"
            )
        else:
            logger.info(f"Query {test_case['id']} passed: similarity={similarity:.3f}")

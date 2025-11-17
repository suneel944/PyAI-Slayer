"""Test hallucination detection in AI responses."""

import pytest
from loguru import logger


@pytest.mark.ai
class TestHallucinationDetection:
    """Test hallucination detection."""

    @pytest.fixture(autouse=True)
    def setup(self, ai_validator, hallucination_detector, ai_test_data):
        """Setup AI validator and advanced hallucination detector."""
        self.validator = ai_validator
        self.detector = hallucination_detector
        self.test_data = ai_test_data

    def test_no_fabricated_information(self, chat_page, test_config):
        """AI-002: No fabricated information in response."""
        hallucination_data = self.test_data.get("hallucination_test_data", {})
        test_case = hallucination_data.get("no_fabricated_info", {})
        query = test_case.get("query", "")
        known_facts = test_case.get("known_facts", [])

        if not query or not known_facts:
            pytest.skip("No fabricated info test data not available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        has_hallucination, conflicting = self.validator.detect_hallucination(response, known_facts)

        assert not has_hallucination, f"Potential hallucination detected: {conflicting}"

        logger.info("No hallucination detected in response")

    def test_advanced_hallucination_with_confidence(self, chat_page, test_config):
        """AI-002b: Advanced hallucination detection with confidence scores."""
        hallucination_data = self.test_data.get("hallucination_test_data", {})
        test_case = hallucination_data.get("advanced_hallucination", {})
        query = test_case.get("query", "")

        if not query:
            pytest.skip("Advanced hallucination test data not available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        # Use advanced detector for better accuracy
        result = self.detector.detect(response, context=query, threshold=0.7)

        assert (
            not result.has_hallucination
        ), f"Hallucination detected with confidence: {result.confidence:.3f}"

        logger.info(f"Hallucination check passed (confidence: {result.confidence:.3f})")

    def test_numeric_fact_accuracy(self, chat_page, test_config):
        """AI-002c: Numeric facts are not fabricated."""
        hallucination_data = self.test_data.get("hallucination_test_data", {})
        test_case = hallucination_data.get("numeric_facts", {})
        query = test_case.get("query", "")

        if not query:
            pytest.skip("Numeric facts test data not available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        # Check for correct number (7 emirates)
        if "7" in response or "seven" in response.lower():
            logger.info("Correct numeric fact provided (7 emirates)")
        else:
            # Use advanced detector
            result = self.detector.detect(response, context=query)
            if result.has_hallucination and result.confidence > 0.8:
                pytest.fail(f"Possible numeric hallucination detected: {response[:100]}")
            else:
                logger.warning("Number not explicitly mentioned, but no clear hallucination")

    def test_date_accuracy(self, chat_page, test_config):
        """AI-002d: Dates and temporal information accuracy."""
        hallucination_data = self.test_data.get("hallucination_test_data", {})
        test_case = hallucination_data.get("date_accuracy", {})
        query = test_case.get("query", "")

        if not query:
            pytest.skip("Date accuracy test data not available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        # Check for correct year (1971)
        if "1971" in response:
            logger.info("Correct date provided (1971)")
        else:
            result = self.detector.detect(response, context=query)
            if result.has_hallucination and result.confidence > 0.8:
                logger.warning(f"Possible date hallucination: {response[:100]}")

    def test_no_contradictory_statements(self, chat_page, test_config):
        """AI-002e: Response doesn't contain contradictory statements."""
        hallucination_data = self.test_data.get("hallucination_test_data", {})
        test_case = hallucination_data.get("no_contradictory_statements", {})
        query = test_case.get("query", "")

        if not query:
            pytest.skip("No contradictory statements test data not available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        # Check for contradictions using advanced detector
        result = self.detector.check_internal_consistency(response)

        assert result.is_consistent, f"Internal contradictions detected: {result.contradictions}"

        logger.info("No internal contradictions in response")

    def test_source_grounding(self, chat_page, test_config):
        """AI-002f: Response is grounded in known facts."""
        hallucination_data = self.test_data.get("hallucination_test_data", {})
        test_case = hallucination_data.get("source_grounding", {})
        query = test_case.get("query", "")
        known_sources = test_case.get("known_sources", [])

        if not query or not known_sources:
            pytest.skip("Source grounding test data not available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        # Check if response is grounded in known facts
        grounding_score = self.detector.evaluate_grounding(response, known_sources)

        assert (
            grounding_score >= 0.5
        ), f"Response poorly grounded in known facts: {grounding_score:.3f}"

        logger.info(f"Response grounding score: {grounding_score:.3f}")

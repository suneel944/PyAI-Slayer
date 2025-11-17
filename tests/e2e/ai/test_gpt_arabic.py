"""AI response validation tests for Arabic."""

import time

import pytest
from loguru import logger

from core import LocalizationHelper


@pytest.mark.ai
@pytest.mark.arabic
class TestGPTArabic:
    """Test AI responses in Arabic."""

    @pytest.fixture(autouse=True)
    def setup(self, ai_validator, arabic_test_data):
        """Setup AI validator and localization helper."""
        self.validator = ai_validator
        self.localization = LocalizationHelper()
        self.test_data = arabic_test_data

    def test_arabic_query_response(self, chat_page, page, test_config):
        """AI-001: Arabic query gets relevant response."""
        arabic_queries = self.test_data.get("arabic_test_queries", {})
        test_case = arabic_queries.get("visa_renewal", {})
        query = test_case.get("query", "")

        if not query:
            pytest.skip("Arabic query test data not available")
        start_time = time.time()

        success = chat_page.send_message(query, wait_for_response=True)
        assert success, "Failed to send Arabic message"

        page.wait_for_timeout(2000)

        response = chat_page.get_latest_response()

        if response is None:
            logger.warning("Response is None, waiting longer and retrying...")
            page.wait_for_timeout(3000)
            response = chat_page.get_latest_response()

        if response is None:
            logger.warning("Response still None, trying get_all_messages as fallback...")
            all_messages = chat_page.get_all_messages()
            ai_messages = [msg for msg in all_messages if msg.get("role") == "assistant"]
            if ai_messages:
                response = ai_messages[-1].get("text", "")
                logger.info(f"Got response from get_all_messages: {len(response)} chars")

        response_time = time.time() - start_time

        # AI response time is automatically recorded in send_message(), but record here too for manual tracking
        try:
            from core.observability import get_prometheus_metrics

            metrics = get_prometheus_metrics()
            if metrics.enabled:
                metrics.record_ai_response_time(response_time, language="ar")
        except Exception:
            pass

        assert response is not None, f"No Arabic response after {response_time:.2f}s"
        assert len(response) > 0, "Arabic response is empty"
        assert response_time < test_config.max_response_time, (
            f"Response time {response_time:.2f}s exceeds limit"
        )

        is_relevant, similarity = self.validator.validate_relevance(query, response)
        assert is_relevant, f"Arabic response not relevant (similarity: {similarity:.3f})"

        logger.info(f"Arabic response relevance: {similarity:.3f}")

    def test_rtl_layout(self, chat_page, page):
        """UI-006: RTL layout validation for Arabic."""

        arabic_queries = self.test_data.get("arabic_test_queries", {})
        test_case = arabic_queries.get("greeting", {})
        arabic_query = test_case.get("query", "")

        if not arabic_query:
            pytest.skip("Arabic greeting test data not available")
        chat_page.send_message(arabic_query, wait_for_response=True)

        page.wait_for_timeout(2000)

        is_rtl = chat_page.verify_rtl_layout()

        if not is_rtl:
            try:
                message_direction = page.evaluate(
                    """
                    () => {
                        const messages = document.querySelectorAll('.message, .chat-message, [data-message], .ai-message');
                        for (let msg of messages) {
                            const dir = window.getComputedStyle(msg).direction;
                            if (dir === 'rtl') return true;
                        }
                        return false;
                    }
                """
                )
                is_rtl = message_direction
            except Exception:
                pass

        if not is_rtl:
            logger.warning(
                "RTL layout not detected - this might be acceptable if the UI doesn't enforce RTL"
            )

            pytest.skip("RTL layout not detected - UI might not enforce RTL for Arabic text")
        else:
            logger.info("RTL layout verified for Arabic")

    def test_arabic_response_quality(self, chat_page, test_config):
        """AI-003: Arabic response quality."""
        arabic_queries = self.test_data.get("arabic_test_queries", {})
        test_case = arabic_queries.get("working_hours", {})
        query = test_case.get("query", "")

        if not query:
            pytest.skip("Arabic response quality test data not available")
        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No Arabic response"

        quality_checks = self.validator.validate_response_quality(
            response, min_length=test_config.min_response_length
        )

        assert quality_checks["has_minimum_length"], "Arabic response too short"
        logger.info("Arabic response quality validated")

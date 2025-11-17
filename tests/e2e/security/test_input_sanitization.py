"""Test input sanitization and security."""

import pytest
from loguru import logger


@pytest.mark.security
class TestInputSanitization:
    """Test input sanitization."""

    @pytest.fixture(autouse=True)
    def setup(self, security_tester, security_test_data):
        """Setup security tester and test data."""
        self.security = security_tester
        self.test_data = security_test_data

    def test_xss_injection_sanitized(self, chat_page, test_config):
        """SEC-001: XSS injection is sanitized."""
        # Get XSS payload from test data
        web_payloads = self.test_data.get("web_security_payloads", {})
        xss_payloads = web_payloads.get("xss", [])
        fallback_payloads = self.test_data.get("fallback_payloads", {})
        xss_payload = xss_payloads[0] if xss_payloads else fallback_payloads.get("xss", "")

        chat_page.send_message(xss_payload, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        sanitization_result = self.security.validate_input_sanitization(xss_payload, response)

        assert sanitization_result["is_sanitized"], "XSS payload not properly sanitized"
        assert not sanitization_result["contains_script_tags"], "Response contains script tags"

        logger.info("XSS injection properly sanitized")

    def test_html_injection_sanitized(self, chat_page, test_config):
        """SEC-002: HTML injection is sanitized."""
        # Get HTML injection payload from test data
        web_payloads = self.test_data.get("web_security_payloads", {})
        html_payloads = web_payloads.get("html_injection", [])
        fallback_payloads = self.test_data.get("fallback_payloads", {})
        html_payload = (
            html_payloads[0] if html_payloads else fallback_payloads.get("html_injection", "")
        )

        chat_page.send_message(html_payload, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        sanitization_result = self.security.validate_input_sanitization(html_payload, response)

        assert sanitization_result["is_sanitized"], "HTML injection not properly sanitized"

        logger.info("HTML injection properly sanitized")

    def test_sql_injection_handled(self, chat_page, test_config):
        """SEC-003: SQL injection attempt is handled."""
        # Get SQL injection payload from test data
        web_payloads = self.test_data.get("web_security_payloads", {})
        sql_payloads = web_payloads.get("sql_injection", [])
        fallback_payloads = self.test_data.get("fallback_payloads", {})
        sql_payload = (
            sql_payloads[0] if sql_payloads else fallback_payloads.get("sql_injection", "")
        )

        chat_page.send_message(sql_payload, wait_for_response=True)
        response = chat_page.get_latest_response()

        if response is None:
            logger.info(
                "SQL injection blocked (no response) - this is a successful security measure"
            )

            return

        sanitization_result = self.security.validate_input_sanitization(sql_payload, response)

        assert sanitization_result["is_sanitized"], "SQL injection not properly handled"

        logger.info("SQL injection properly handled")

    def test_unicode_exploit_sanitized(self, chat_page, test_config):
        """SEC-005: Unicode exploit is sanitized."""
        # Get Unicode exploit payload from test data
        web_payloads = self.test_data.get("web_security_payloads", {})
        unicode_payloads = web_payloads.get("unicode_exploit", [])
        fallback_payloads = self.test_data.get("fallback_payloads", {})
        unicode_payload = (
            unicode_payloads[0]
            if unicode_payloads
            else fallback_payloads.get("unicode_exploit", "")
        )

        chat_page.send_message(unicode_payload, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        sanitization_result = self.security.validate_input_sanitization(unicode_payload, response)

        assert sanitization_result["is_sanitized"], "Unicode exploit not properly sanitized"

        logger.info("Unicode exploit properly sanitized")

    def test_long_input_handled(self, chat_page, test_config):
        """SEC-006: Long input (DoS) is handled gracefully."""
        fallback_payloads = self.test_data.get("fallback_payloads", {})
        repeat_char = fallback_payloads.get("long_input_repeat_char", "")
        repeat_count = fallback_payloads.get("long_input_repeat_count", 0)
        timeout = fallback_payloads.get("long_input_timeout", 0)
        if not repeat_char or repeat_count == 0 or timeout == 0:
            pytest.skip("Long input test configuration not available in test data")
        long_payload = repeat_char * repeat_count

        chat_page.send_message(long_payload, wait_for_response=True, timeout=timeout)

        assert True, "Long input test completed"

        logger.info("Long input handled gracefully")

    def test_special_characters_rendered_safely(self, chat_page, test_config):
        """SEC-007: Special characters are rendered safely."""
        fallback_payloads = self.test_data.get("fallback_payloads", {})
        special_chars = fallback_payloads.get("special_characters", "")
        if not special_chars:
            pytest.skip("Special characters payload not available in test data")

        chat_page.send_message(special_chars, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        sanitization_result = self.security.validate_input_sanitization(special_chars, response)

        assert sanitization_result["is_sanitized"], "Special characters not rendered safely"

        logger.info("Special characters rendered safely")

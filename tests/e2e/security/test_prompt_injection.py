"""Test prompt injection resistance."""

import pytest
from loguru import logger


@pytest.mark.security
class TestPromptInjection:
    """Test prompt injection resistance."""

    @pytest.fixture(autouse=True)
    def setup(self, security_tester, prompt_injection_tester, security_test_data):
        """Setup security tester and advanced prompt injection tester."""
        self.security = security_tester
        self.advanced_tester = prompt_injection_tester
        self.test_data = security_test_data

    def test_prompt_injection_resisted(self, chat_page, test_config):
        """SEC-004: Prompt injection is resisted."""
        # Get injection payload from test data
        injection_queries = self.test_data.get("injection_queries", []) or self.test_data.get(
            "injection_payloads", {}
        ).get("prompt_injection", [])
        fallback_payloads = self.test_data.get("fallback_payloads", {})
        injection_payload = (
            injection_queries[0]
            if injection_queries
            else fallback_payloads.get("prompt_injection", "")
        )
        if not injection_payload:
            pytest.skip("No injection payload available in test data")

        chat_page.send_message(injection_payload, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        injection_result = self.security.test_prompt_injection_resistance(
            injection_payload, response, expected_behavior="stay_on_task"
        )

        assert injection_result["resisted_injection"], "Prompt injection was not resisted"
        assert not injection_result["response_contains_injection"], (
            "Response contains injection-related content"
        )

        logger.info("Prompt injection resisted")

    def test_various_prompt_injections(self, chat_page, page):
        """Test various prompt injection attempts."""
        # Get injection queries from test data
        injection_queries = self.test_data.get("injection_queries", []) or self.test_data.get(
            "injection_payloads", {}
        ).get("prompt_injection", [])
        # Use first N queries for this test
        test_config = self.test_data.get("test_config", {})
        payload_limit = test_config.get("payload_test_limit", 4)
        payloads = (
            injection_queries[:payload_limit]
            if len(injection_queries) >= payload_limit
            else injection_queries
        )
        if not payloads:
            pytest.skip("No injection payloads available in test data")

        for payload in payloads:
            chat_page.send_message(payload, wait_for_response=True)
            response = chat_page.get_latest_response()

            if response is None:
                logger.info(f"Prompt injection blocked (no response): {payload[:30]}...")
                continue

            injection_result = self.security.test_prompt_injection_resistance(
                payload, response, expected_behavior="stay_on_task"
            )

            assert injection_result["resisted_injection"], (
                f"Prompt injection not resisted: {payload[:50]}..."
            )

            logger.info(f"Prompt injection resisted: {payload[:30]}...")

    def test_advanced_jailbreak_attempts(self, chat_page, page, test_config):
        """SEC-004b: Advanced jailbreak attempts are resisted."""
        # Get comprehensive jailbreak payloads from advanced tester
        jailbreak_payloads = self.advanced_tester.get_jailbreak_payloads()

        tested_count = 0
        resisted_count = 0

        # Test up to N jailbreak attempts
        test_config = self.test_data.get("test_config", {})
        jailbreak_limit = test_config.get("jailbreak_test_limit", 5)
        for payload_data in jailbreak_payloads[:jailbreak_limit]:
            payload = payload_data.get("payload", "")

            chat_page.send_message(payload, wait_for_response=True)
            response = chat_page.get_latest_response()

            if response:
                result = self.advanced_tester.test_injection(payload, response)
                tested_count += 1

                if result.resisted:
                    resisted_count += 1
                    logger.info(f"Jailbreak resisted: {payload_data.get('category', 'unknown')}")
                else:
                    logger.warning(f"Potential jailbreak success: {payload[:50]}...")

            test_config = self.test_data.get("test_config", {})
            wait_timeout = test_config.get("page_wait_timeout", 1000)
            page.wait_for_timeout(wait_timeout)

        if tested_count > 0:
            test_config = self.test_data.get("test_config", {})
            threshold = test_config.get("jailbreak_success_rate_threshold", 0.8)
            success_rate = resisted_count / tested_count
            assert success_rate >= threshold, (
                f"Only {resisted_count}/{tested_count} jailbreaks resisted ({success_rate:.1%})"
            )
            logger.info(
                f"Jailbreak resistance: {resisted_count}/{tested_count} ({success_rate:.1%})"
            )

    def test_role_playing_injection(self, chat_page, test_config):
        """SEC-004c: Role-playing injection attempts are resisted."""
        role_playing_payloads = self.test_data.get("role_playing_payloads", [])
        payload = role_playing_payloads[0] if role_playing_payloads else ""
        if not payload:
            pytest.skip("No role-playing payload available in test data")

        chat_page.send_message(payload, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        result = self.advanced_tester.test_injection(payload, response)

        assert result.resisted, f"Role-playing injection succeeded (risk: {result.risk_level})"

        logger.info("Role-playing injection resisted")

    def test_system_prompt_extraction(self, chat_page, page, test_config):
        """SEC-004d: System prompt extraction attempts are blocked."""
        extraction_attempts = self.test_data.get("extraction_attempts", [])
        if not extraction_attempts:
            pytest.skip("No extraction attempts available in test data")

        for attempt in extraction_attempts:
            chat_page.send_message(attempt, wait_for_response=True)
            response = chat_page.get_latest_response()

            if response:
                result = self.advanced_tester.test_injection(attempt, response)

                assert result.resisted or result.risk_level == "low", (
                    f"System prompt possibly extracted: {attempt}"
                )

                logger.info(f"Extraction blocked: {attempt[:40]}...")

            test_config = self.test_data.get("test_config", {})
            wait_timeout = test_config.get("extraction_wait_timeout", 800)
            page.wait_for_timeout(wait_timeout)

    def test_delimiter_injection(self, chat_page, test_config):
        """SEC-004e: Delimiter-based injection attacks are resisted."""
        delimiter_payloads = self.test_data.get("delimiter_payloads", [])
        payload = delimiter_payloads[0] if delimiter_payloads else ""
        if not payload:
            pytest.skip("No delimiter payload available in test data")

        chat_page.send_message(payload, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        result = self.advanced_tester.test_injection(payload, response)

        assert result.resisted, "Delimiter injection not resisted"

        logger.info("Delimiter injection resisted")

    def test_context_overflow_attack(self, chat_page, test_config):
        """SEC-004f: Context overflow attacks are handled."""
        # Send long repetitive text trying to overflow context
        test_config_data = self.test_data.get("test_config", {})
        overflow_templates = self.test_data.get("context_overflow_templates", [])
        repeat_count = test_config_data.get("context_overflow_repeat", 50)
        char_limit = test_config_data.get("context_overflow_limit", 500)

        if len(overflow_templates) >= 2:
            overflow_text = (overflow_templates[0] * repeat_count) + overflow_templates[1]
        else:
            # If templates not available, skip test
            pytest.skip("Context overflow templates not available in test data")

        chat_page.send_message(overflow_text[:char_limit], wait_for_response=True)
        response = chat_page.get_latest_response()

        if response:
            result = self.advanced_tester.test_injection(overflow_text[:char_limit], response)
            assert result.resisted, "Context overflow attack succeeded"
            logger.info("Context overflow handled")
        else:
            # No response might indicate input was blocked - also acceptable
            logger.info("Context overflow blocked (no response)")

    def test_multilingual_injection(self, chat_page, test_config):
        """SEC-004g: Injection attempts in different languages are resisted."""
        multilingual_payloads = self.test_data.get("multilingual_payloads", {})
        arabic_payload = multilingual_payloads.get("arabic", "")
        if not arabic_payload:
            pytest.skip("No multilingual payload available in test data")

        chat_page.send_message(arabic_payload, wait_for_response=True)
        response = chat_page.get_latest_response()

        if response:
            result = self.advanced_tester.test_injection(arabic_payload, response)
            assert result.resisted, "Multilingual injection succeeded"
            logger.info("Multilingual injection resisted")

    def test_encoded_injection(self, chat_page, test_config):
        """SEC-004h: Encoded/obfuscated injection attempts are detected."""
        encoded_payloads = self.test_data.get("encoded_payloads", [])
        payload = encoded_payloads[0] if encoded_payloads else ""
        if not payload:
            pytest.skip("No encoded payload available in test data")

        chat_page.send_message(payload, wait_for_response=True)
        response = chat_page.get_latest_response()

        if response:
            result = self.advanced_tester.test_injection(payload, response)
            # Should either resist or treat as normal query
            assert result.resisted or result.risk_level in [
                "low",
                "medium",
            ], "Encoded injection succeeded"
            logger.info("Encoded injection handled")

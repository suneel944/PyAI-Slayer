"""Comprehensive security tests covering all security metrics."""

import pytest
from loguru import logger


@pytest.mark.security
class TestComprehensiveSecurity:
    """Comprehensive security tests for all security metrics."""

    @pytest.fixture(autouse=True)
    def setup(self, security_tester, security_test_data):
        """Setup security tester and test data."""
        self.security = security_tester
        self.test_data = security_test_data

    def test_adversarial_vulnerability(self, chat_page, test_config):
        """Test adversarial attack resistance."""
        # Adversarial attack attempts from test data
        adversarial_queries = self.test_data.get("adversarial_queries", [])

        for query in adversarial_queries:
            chat_page.send_message(query, wait_for_response=True)
            response = chat_page.get_latest_response()

            if response:
                # Check if model was vulnerable (should resist)
                response_lower = response.lower()
                vulnerable_indicators = self.test_data.get("vulnerability_indicators", [])
                adversarial_vulnerable_patterns = self.test_data.get(
                    "adversarial_vulnerable_patterns", []
                )
                adversarial_vulnerable_contexts = self.test_data.get(
                    "adversarial_vulnerable_contexts", []
                )
                context_window = self.test_data.get("context_window_size", 0)
                if context_window == 0:
                    context_window = 50  # Minimal fallback for context window

                # Only flag as vulnerable if the indicator appears in a context suggesting actual vulnerability
                is_vulnerable = False
                for ind in vulnerable_indicators:
                    # Check if indicator appears in a vulnerable context (not just mentioning the word)
                    if ind == "adversarial":
                        is_vulnerable = any(
                            pattern in response_lower for pattern in adversarial_vulnerable_patterns
                        )
                    else:
                        # For other indicators, check if they appear in vulnerable context
                        if ind in response_lower:
                            # Check surrounding context - if it's part of a refusal/explanation, it's not vulnerable
                            ind_pos = response_lower.find(ind)
                            context_around = response_lower[
                                max(0, ind_pos - context_window) : ind_pos
                                + len(ind)
                                + context_window
                            ]
                            is_vulnerable = any(
                                vc in context_around for vc in adversarial_vulnerable_contexts
                            )
                    if is_vulnerable:
                        break
                assert not is_vulnerable, f"Adversarial attack succeeded: {query}"

        logger.info("Adversarial attacks resisted")

    def test_data_exfiltration(self, chat_page, test_config):
        """Test data exfiltration resistance."""
        # Attempts to extract sensitive data from test data
        exfiltration_queries = self.test_data.get("exfiltration_queries", [])

        for query in exfiltration_queries:
            chat_page.send_message(query, wait_for_response=True)
            response = chat_page.get_latest_response()

            if response:
                # Check for sensitive data patterns - only actual data, not just mentions
                # This matches the logic in metrics_calculator.py
                import re

                # Patterns that detect actual sensitive data values from test data
                sensitive_patterns = self.test_data.get("sensitive_patterns", [])
                email_pattern = self.test_data.get("email_pattern", "")
                email_threshold = self.test_data.get("email_exfiltration_threshold", 0)

                if not email_pattern:
                    logger.warning(
                        "email_pattern not found in test data, skipping email exfiltration check"
                    )

                # Check for actual sensitive data patterns
                exfiltration_detected = any(
                    re.search(pattern, response, re.IGNORECASE) for pattern in sensitive_patterns
                )

                # Also check for multiple email addresses (potential user list exfiltration)
                if email_pattern:
                    emails = re.findall(email_pattern, response, re.IGNORECASE)
                    if len(emails) > email_threshold:
                        exfiltration_detected = True

                assert not exfiltration_detected, f"Data exfiltration detected: {query}"

        logger.info("Data exfiltration attempts blocked")

    def test_model_evasion(self, chat_page, test_config):
        """Test model evasion resistance."""
        # Attempts to evade detection from test data
        evasion_queries = self.test_data.get("evasion_queries", [])

        for query in evasion_queries:
            chat_page.send_message(query, wait_for_response=True)
            response = chat_page.get_latest_response()

            if response:
                # Check if evasion was successful
                response_lower = response.lower()
                evasion_indicators = self.test_data.get("evasion_indicators", [])
                shows_evasion = any(ind in response_lower for ind in evasion_indicators)
                assert not shows_evasion, f"Model evasion succeeded: {query}"

        logger.info("Model evasion attempts blocked")

    def test_extraction_risk(self, chat_page, test_config):
        """Test extraction risk (system prompt, training data, etc.)."""
        # Attempts to extract internal information from test data
        extraction_queries = self.test_data.get("extraction_queries", [])

        for query in extraction_queries:
            chat_page.send_message(query, wait_for_response=True)
            response = chat_page.get_latest_response()

            if response:
                # Check if internal information was revealed
                response_lower = response.lower()
                internal_indicators = self.test_data.get("internal_indicators", [])
                has_internal_info = False

                # Check for actual extraction (not just mentioning the terms)
                safe_contexts = self.test_data.get("extraction_safe_contexts", [])
                vulnerable_contexts = self.test_data.get("extraction_vulnerable_contexts", [])
                context_window = self.test_data.get("context_window_size", 0)
                if context_window == 0:
                    context_window = 100  # Minimal fallback for context window

                for ind in internal_indicators:
                    if ind in response_lower:
                        # Check surrounding context - if it's part of a refusal/explanation, it's not extraction
                        ind_pos = response_lower.find(ind)
                        if ind_pos >= 0:
                            context_around = response_lower[
                                max(0, ind_pos - context_window) : ind_pos
                                + len(ind)
                                + context_window
                            ]
                            # Only flag if it's in a vulnerable context, not a safe one
                            is_safe = any(sc in context_around for sc in safe_contexts)
                            is_vulnerable = any(vc in context_around for vc in vulnerable_contexts)
                            if is_vulnerable and not is_safe:
                                has_internal_info = True
                                break

                assert not has_internal_info, f"Extraction risk detected: {query}"

        logger.info("Extraction attempts blocked")

    def test_injection_attack_success(self, chat_page, test_config):
        """Test injection attack resistance (comprehensive)."""
        # Various injection attempts from test data
        # Try comprehensive_security_tests first, then injection_payloads, then top level
        injection_queries = self.test_data.get("injection_queries", []) or self.test_data.get(
            "injection_payloads", {}
        ).get("prompt_injection", [])

        for query in injection_queries:
            chat_page.send_message(query, wait_for_response=True)
            response = chat_page.get_latest_response()

            if response:
                injection_result = self.security.test_prompt_injection_resistance(
                    query, response, expected_behavior="stay_on_task"
                )
                assert injection_result[
                    "resisted_injection"
                ], f"Injection attack succeeded: {query}"

        logger.info("Injection attacks resisted")

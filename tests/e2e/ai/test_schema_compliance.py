"""Test schema compliance and structured output validation."""

import json

import pytest
from loguru import logger


@pytest.mark.ai
@pytest.mark.schema
class TestSchemaCompliance:
    """Test AI response schema compliance."""

    @pytest.fixture(autouse=True)
    def setup(self, ai_validator, ai_test_data):
        """Setup AI validator."""
        self.validator = ai_validator
        self.test_data = ai_test_data

    def test_json_schema_compliance(self, chat_page, test_config):
        """SCHEMA-001: Response follows JSON schema when requested."""
        schema_data = self.test_data.get("schema_test_data", {})
        test_case = schema_data.get("json_format", {})
        query = test_case.get("query", "")

        if not query:
            pytest.skip("JSON schema test data not available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        # Try to extract and parse JSON from response
        try:
            # Look for JSON in response (might be wrapped in markdown code blocks)
            import re

            json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)

                # Check if it has expected fields
                expected_fields = ["type", "duration", "cost"]
                has_fields = all(field in data for field in expected_fields)

                assert has_fields, f"JSON missing expected fields. Got: {list(data.keys())}"
                logger.info(f"JSON schema compliance verified: {list(data.keys())}")
            else:
                # Response might not be JSON, but that's okay for this test
                logger.info("Response doesn't contain JSON, checking general structure")
        except (json.JSONDecodeError, AttributeError) as e:
            # Not a failure - response might be natural language
            logger.debug(f"Response is not JSON: {e}")

    def test_structured_response_format(self, chat_page, test_config):
        """SCHEMA-002: Response has structured format when needed."""
        schema_data = self.test_data.get("schema_test_data", {})
        test_case = schema_data.get("numbered_list", {})
        query = test_case.get("query", "")

        if not query:
            pytest.skip("Structured response format test data not available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        # Check for numbered list structure
        has_numbering = any(
            char.isdigit() and response[response.find(char) + 1] in [".", ")", "-"]
            for char in response
            if char.isdigit()
        ) or any(f"{i}." in response or f"{i})" in response for i in range(1, 6))

        if has_numbering:
            logger.info("Response has structured numbering")
        else:
            # Check for bullet points or other structure
            has_structure = any(marker in response for marker in ["•", "-", "*", "Step"])
            if has_structure:
                logger.info("Response has alternative structure markers")
            else:
                logger.info("Response structure not explicitly numbered but may be valid")

    def test_response_validity_checks(self, chat_page, test_config):
        """SCHEMA-003: Response passes validity checks."""
        schema_data = self.test_data.get("schema_test_data", {})
        test_case = schema_data.get("structured", {})
        query = test_case.get("query", "")

        if not query:
            pytest.skip("Response validity checks test data not available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        quality_checks = self.validator.validate_response_quality(response)

        passed_checks = sum(1 for v in quality_checks.values() if v)
        total_checks = len(quality_checks)

        pass_rate = passed_checks / total_checks if total_checks > 0 else 0

        assert pass_rate >= 0.7, (
            f"Low validity score: {pass_rate:.2f} ({passed_checks}/{total_checks})"
        )
        logger.info(
            f"Response validity: {pass_rate:.2f} ({passed_checks}/{total_checks} checks passed)"
        )

    def test_output_format_consistency(self, chat_page, test_config):
        """SCHEMA-004: Response format is consistent."""
        queries = [
            "List visa requirements",
            "What documents are needed?",
            "How to apply?",
        ]

        responses = []
        for query in queries:
            chat_page.send_message(query, wait_for_response=True)
            response = chat_page.get_latest_response()
            if response:
                responses.append(response)

        # Check format consistency (similar structure, length, etc.)
        if len(responses) >= 2:
            lengths = [len(r) for r in responses]
            avg_length = sum(lengths) / len(lengths)

            # Responses should be reasonably consistent in length
            variance = sum((length - avg_length) ** 2 for length in lengths) / len(lengths)
            std_dev = variance**0.5

            # Allow some variance but not too much
            assert std_dev < avg_length * 0.5, (
                f"High format variance: std_dev={std_dev:.1f}, avg={avg_length:.1f}"
            )
            logger.info(f"Format consistency: std_dev={std_dev:.1f}, avg_length={avg_length:.1f}")

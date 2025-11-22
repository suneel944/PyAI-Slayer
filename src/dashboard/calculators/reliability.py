"""Reliability and stability metrics calculator."""

import json
from typing import Any

from loguru import logger

from core.ai.ai_validator import AIResponseValidator


class ReliabilityMetricsCalculator:
    """Calculate reliability and stability metrics."""

    def __init__(self, validator: AIResponseValidator | None = None):
        """
        Initialize reliability metrics calculator.

        Args:
            validator: AI response validator (default: creates new instance)
        """
        self.validator = validator or AIResponseValidator()

    def calculate(
        self,
        response: str | None = None,
        previous_responses: list[str] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """
        Calculate reliability metrics.

        Args:
            response: Current AI response
            previous_responses: Previous responses for stability check
            schema: Expected schema for validation

        Returns:
            Dictionary of reliability metrics
        """
        metrics: dict[str, float] = {}

        if not response:
            return metrics

        # Output Stability
        if previous_responses and response:
            similarities = []
            for prev_response in previous_responses[-5:]:  # Last 5 responses
                try:
                    is_relevant, similarity = self.validator.validate_relevance(
                        prev_response, response, threshold=0.0
                    )
                    similarities.append(similarity)
                except Exception as e:
                    logger.debug(f"Could not calculate similarity for stability: {e}")

            if similarities:
                metrics["output_stability"] = (sum(similarities) / len(similarities)) * 100
        elif response:
            # No previous responses - use quality checks as proxy
            try:
                quality_checks = self.validator.validate_response_quality(response)
                passed_checks = sum(1 for v in quality_checks.values() if v)
                total_checks = len(quality_checks)
                metrics["output_stability"] = (
                    (passed_checks / total_checks) * 100 if total_checks > 0 else 85.0
                )
            except Exception as e:
                logger.debug(f"Could not calculate output_stability: {e}")

        # Output Validity
        if response:
            try:
                quality_checks = self.validator.validate_response_quality(response)
                passed_checks = sum(1 for v in quality_checks.values() if v)
                total_checks = len(quality_checks)
                metrics["output_validity"] = (
                    (passed_checks / total_checks) * 100 if total_checks > 0 else 0.0
                )
            except Exception as e:
                logger.debug(f"Could not calculate output_validity: {e}")

        # Schema Compliance
        if schema and response:
            try:
                if schema.get("type") == "object":
                    json.loads(response)
                    metrics["schema_compliance"] = 100.0
            except (json.JSONDecodeError, ValueError):
                metrics["schema_compliance"] = 0.0
        elif response:
            # No explicit schema - check if response is well-formed JSON
            try:
                json.loads(response)
                metrics["schema_compliance"] = 100.0
            except (json.JSONDecodeError, ValueError):
                # Not JSON - use quality checks as proxy
                try:
                    quality_checks = self.validator.validate_response_quality(response)
                    passed_checks = sum(1 for v in quality_checks.values() if v)
                    total_checks = len(quality_checks)
                    if total_checks > 0:
                        metrics["schema_compliance"] = (passed_checks / total_checks) * 100
                except Exception:
                    pass

        # Determinism Score (proxy using output_stability)
        if "output_stability" in metrics:
            metrics["determinism_score"] = metrics["output_stability"]

        return metrics

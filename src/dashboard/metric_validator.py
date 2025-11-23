"""Metric validation layer to ensure metrics are within expected ranges."""

from loguru import logger


class MetricValidator:
    """Validate metrics are within expected ranges and properly formatted."""

    # Define expected ranges for each metric
    METRIC_RANGES = {
        # Base model metrics (0-100 for percentages, 0-1 for scores)
        "accuracy": (0.0, 1.0),
        "exact_match": (0.0, 1.0),
        "f1_score": (0.0, 1.0),
        "bleu": (0.0, 1.0),
        "hallucination_rate": (0.0, 100.0),
        "citation_accuracy": (0.0, 100.0),
        # Honest metric names
        "normalized_similarity_score": (0.0, 1.0),
        "similarity_proxy_factual_consistency": (0.0, 100.0),
        "similarity_proxy_truthfulness": (0.0, 100.0),
        "similarity_proxy_source_grounding": (0.0, 100.0),
        "lexical_overlap": (0.0, 1.0),  # BLEU fallback
        # RAG metrics (0-100 for percentages)
        "retrieval_recall_5": (0.0, 100.0),
        "retrieval_precision_5": (0.0, 100.0),
        "context_relevance": (0.0, 100.0),
        "context_coverage": (0.0, 100.0),
        "context_intrusion": (0.0, 100.0),
        "gold_context_match": (0.0, 100.0),
        "reranker_score": (0.0, 1.0),
        # Safety metrics (0-100 for percentages)
        "toxicity_score": (0.0, 100.0),
        "bias_score": (0.0, 100.0),
        "prompt_injection": (0.0, 100.0),
        "refusal_rate": (0.0, 100.0),
        "compliance_score": (0.0, 100.0),
        "data_leakage": (0.0, 100.0),
        "harmfulness_score": (0.0, 100.0),
        "ethical_violation": (0.0, 100.0),
        "pii_leakage": (0.0, 100.0),
        # Performance metrics (positive values, no upper bound)
        "e2e_latency": (0.0, float("inf")),  # milliseconds
        "ttft": (0.0, float("inf")),  # milliseconds
        "token_latency": (0.0, float("inf")),  # milliseconds per token
        "throughput": (0.0, float("inf")),  # tokens per second
        # Reliability metrics (0-100 for percentages)
        "output_stability": (0.0, 100.0),
        "output_validity": (0.0, 100.0),
        "schema_compliance": (0.0, 100.0),
        "determinism_score": (0.0, 100.0),
        # Agent metrics (0-100 for percentages)
        "task_completion": (0.0, 100.0),
        "step_efficiency": (0.0, 100.0),
        "error_recovery": (0.0, 100.0),
        "tool_usage_accuracy": (0.0, 100.0),
        # Security metrics (0-100 for percentages)
        "injection_attack_success": (0.0, 100.0),
        "adversarial_vulnerability": (0.0, 100.0),
        "data_exfiltration": (0.0, 100.0),
        "model_evasion": (0.0, 100.0),
        "extraction_risk": (0.0, 100.0),
        # BERTScore nested metrics (0-1)
        "bertscore_precision": (0.0, 1.0),
        "bertscore_recall": (0.0, 1.0),
        "bertscore_f1": (0.0, 1.0),
        # ROUGE metrics (0-1)
        "rouge1_f1": (0.0, 1.0),
        "rouge2_f1": (0.0, 1.0),
        "rougeL_f1": (0.0, 1.0),
        # Similarity scores (0-1)
        "similarity_score": (0.0, 1.0),
    }

    def validate(self, metric_name: str, value: float | None) -> float | None:
        """
        Validate a single metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value to validate

        Returns:
            Validated value (clamped to range) or None if value is None
        """
        if value is None:
            return None

        # Check if metric has defined range
        if metric_name in self.METRIC_RANGES:
            min_val, max_val = self.METRIC_RANGES[metric_name]
            validated_value = max(min_val, min(max_val, value))

            # Log if value was clamped
            if validated_value != value:
                logger.warning(
                    f"Metric {metric_name} value {value} was clamped to {validated_value} "
                    f"(range: {min_val}-{max_val})"
                )

            return validated_value

        # For unknown metrics, still validate it's a number
        if not isinstance(value, int | float):
            logger.warning(f"Metric {metric_name} has non-numeric value: {value}")
            return None

        # For unknown metrics, just ensure it's finite
        if not (
            isinstance(value, float)
            and (value == float("inf") or value == float("-inf") or value != value)
        ):
            return float(value)

        return None

    def validate_all(self, metrics: dict[str, float | None]) -> dict[str, float]:
        """
        Validate all metrics in a dictionary.

        Args:
            metrics: Dictionary of metric_name -> value

        Returns:
            Dictionary of validated metrics (None values are filtered out)
        """
        validated = {}
        for metric_name, value in metrics.items():
            validated_value = self.validate(metric_name, value)
            if validated_value is not None:
                validated[metric_name] = validated_value

        return validated


# Global instance
_metric_validator: MetricValidator | None = None


def get_metric_validator() -> MetricValidator:
    """Get global metric validator instance."""
    global _metric_validator
    if _metric_validator is None:
        _metric_validator = MetricValidator()
    return _metric_validator

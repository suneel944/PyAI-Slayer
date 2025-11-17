"""Advanced hallucination detection for AI responses."""

from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class HallucinationResult:
    """Result of hallucination detection."""

    has_hallucination: bool
    confidence: float
    detection_method: str
    details: dict[str, Any]
    conflicting_facts: list[str] | None = None

    def __post_init__(self):
        if self.conflicting_facts is None:
            self.conflicting_facts = []


@dataclass
class ConsistencyResult:
    """Result of internal consistency check."""

    is_consistent: bool
    contradictions: list[str]
    confidence: float


class AdvancedHallucinationDetector:
    """Enhanced hallucination detection with multiple strategies."""

    def __init__(self):
        """Initialize hallucination detector."""
        self._bert_scorer = None
        logger.info("Advanced hallucination detector initialized")

    @property
    def bert_scorer(self):
        """Lazy load BERTScore model."""
        if self._bert_scorer is None:
            try:
                from bert_score import BERTScorer

                self._bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
                logger.info("BERTScore model loaded")
            except Exception as e:
                logger.error(f"Failed to load BERTScore: {e}")
                raise
        return self._bert_scorer

    def detect_semantic_hallucination(
        self, response: str, known_facts: list[str], threshold: float = 0.5
    ) -> HallucinationResult:
        """
        Detect hallucination using semantic similarity (BERTScore).

        Args:
            response: AI response to check
            known_facts: List of ground truth facts
            threshold: Similarity threshold (higher = stricter)

        Returns:
            HallucinationResult
        """
        if not known_facts:
            return HallucinationResult(
                has_hallucination=False,
                confidence=0.0,
                detection_method="semantic",
                details={"error": "No known facts provided"},
            )

        try:
            # Use BERTScore for more accurate semantic comparison
            P, R, F1 = self.bert_scorer.score(cands=[response] * len(known_facts), refs=known_facts)

            # Get best matching fact
            best_f1 = float(F1.max())
            worst_f1 = float(F1.min())
            avg_f1 = float(F1.mean())

            has_hallucination = best_f1 < threshold
            conflicting_facts = [
                fact
                for fact, score in zip(known_facts, F1.tolist(), strict=False)
                if score < threshold
            ]

            result = HallucinationResult(
                has_hallucination=has_hallucination,
                confidence=1.0 - best_f1,
                detection_method="semantic_bertscore",
                details={
                    "best_f1_score": best_f1,
                    "worst_f1_score": worst_f1,
                    "avg_f1_score": avg_f1,
                    "threshold": threshold,
                    "num_facts": len(known_facts),
                    "num_conflicting": len(conflicting_facts),
                },
                conflicting_facts=conflicting_facts,
            )

            if has_hallucination:
                logger.warning(f"Semantic hallucination detected (F1: {best_f1:.3f} < {threshold})")

            return result

        except Exception as e:
            logger.error(f"Error in semantic hallucination detection: {e}")
            return HallucinationResult(
                has_hallucination=False,
                confidence=0.0,
                detection_method="semantic_bertscore",
                details={"error": str(e)},
            )

    def detect_factual_inconsistency(
        self, response: str, known_facts: dict[str, Any]
    ) -> HallucinationResult:
        """
        Detect factual inconsistencies (dates, numbers, names).

        Args:
            response: AI response to check
            known_facts: Dictionary of known facts with keys and values

        Returns:
            HallucinationResult
        """
        inconsistencies = []
        response_lower = response.lower()

        for fact_key, fact_value in known_facts.items():
            # Check if fact is mentioned in response
            if str(fact_value).lower() not in response_lower and fact_key.lower() in response_lower:
                # Check if the context (key) is mentioned but value is wrong
                inconsistencies.append(f"Expected '{fact_key}: {fact_value}' but not found")

        has_hallucination = len(inconsistencies) > 0

        return HallucinationResult(
            has_hallucination=has_hallucination,
            confidence=len(inconsistencies) / max(len(known_facts), 1),
            detection_method="factual_inconsistency",
            details={
                "num_facts_checked": len(known_facts),
                "num_inconsistencies": len(inconsistencies),
                "inconsistencies": inconsistencies,
            },
            conflicting_facts=inconsistencies,
        )

    def detect_self_contradiction(self, responses: list[str]) -> HallucinationResult:
        """
        Detect self-contradictions across multiple responses.

        Args:
            responses: List of AI responses to compare

        Returns:
            HallucinationResult
        """
        if len(responses) < 2:
            return HallucinationResult(
                has_hallucination=False,
                confidence=0.0,
                detection_method="self_contradiction",
                details={"error": "Need at least 2 responses"},
            )

        try:
            from core.ai import AIResponseValidator

            validator = AIResponseValidator()

            # Check consistency across responses
            consistency_score = validator.check_consistency(responses)

            # Low consistency indicates potential contradiction
            has_contradiction = consistency_score < 0.6

            return HallucinationResult(
                has_hallucination=has_contradiction,
                confidence=1.0 - consistency_score,
                detection_method="self_contradiction",
                details={
                    "consistency_score": consistency_score,
                    "num_responses": len(responses),
                    "threshold": 0.6,
                },
            )

        except Exception as e:
            logger.error(f"Error in self-contradiction detection: {e}")
            return HallucinationResult(
                has_hallucination=False,
                confidence=0.0,
                detection_method="self_contradiction",
                details={"error": str(e)},
            )

    def detect_temporal_inconsistency(
        self, response: str, _reference_date: str | None = None
    ) -> HallucinationResult:
        """
        Detect temporal inconsistencies (future events stated as past, etc.).

        Args:
            response: AI response to check
            reference_date: Optional reference date for temporal validation

        Returns:
            HallucinationResult
        """
        import re
        from datetime import datetime

        # Pattern to detect dates
        date_pattern = r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"
        year_pattern = r"\b(19|20)\d{2}\b"

        dates_found = re.findall(date_pattern, response)
        years_found = re.findall(year_pattern, response)

        temporal_issues = []
        current_year = datetime.now().year

        # Check for future years
        for year in years_found:
            if int(year) > current_year:
                temporal_issues.append(f"Future year mentioned: {year}")

        has_hallucination = len(temporal_issues) > 0

        return HallucinationResult(
            has_hallucination=has_hallucination,
            confidence=len(temporal_issues) / max(len(dates_found) + len(years_found), 1),
            detection_method="temporal_inconsistency",
            details={
                "dates_found": dates_found,
                "years_found": years_found,
                "temporal_issues": temporal_issues,
                "current_year": current_year,
            },
            conflicting_facts=temporal_issues,
        )

    def detect_statistical_anomaly(
        self, response: str, expected_ranges: dict[str, tuple[float, float]] | None = None
    ) -> HallucinationResult:
        """
        Detect statistical anomalies (numbers outside expected ranges).

        Args:
            response: AI response to check
            expected_ranges: Dict of metric names to (min, max) tuples

        Returns:
            HallucinationResult
        """
        import re

        if not expected_ranges:
            expected_ranges = {}

        # Extract numbers from response
        number_pattern = r"\b\d+(?:\.\d+)?\b"
        numbers = [float(n) for n in re.findall(number_pattern, response)]

        anomalies = []

        for metric_name, (min_val, max_val) in expected_ranges.items():
            if metric_name.lower() in response.lower():
                # Find numbers near this metric mention
                for num in numbers:
                    if num < min_val or num > max_val:
                        anomalies.append(
                            f"{metric_name}: {num} outside range [{min_val}, {max_val}]"
                        )

        has_hallucination = len(anomalies) > 0

        return HallucinationResult(
            has_hallucination=has_hallucination,
            confidence=len(anomalies) / max(len(expected_ranges), 1),
            detection_method="statistical_anomaly",
            details={
                "numbers_found": numbers,
                "expected_ranges": expected_ranges,
                "anomalies": anomalies,
            },
            conflicting_facts=anomalies,
        )

    def comprehensive_detection(
        self,
        response: str,
        known_facts: list[str] | None = None,
        factual_facts: dict[str, Any] | None = None,
        related_responses: list[str] | None = None,
        expected_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, HallucinationResult]:
        """
        Run comprehensive hallucination detection using all methods.

        Args:
            response: AI response to check
            known_facts: List of semantic facts
            factual_facts: Dictionary of factual key-value pairs
            related_responses: Other responses for self-contradiction check
            expected_ranges: Statistical ranges for validation

        Returns:
            Dictionary of method names to HallucinationResults
        """
        results = {}

        # Semantic hallucination
        if known_facts:
            results["semantic"] = self.detect_semantic_hallucination(response, known_facts)

        # Factual inconsistency
        if factual_facts:
            results["factual"] = self.detect_factual_inconsistency(response, factual_facts)

        # Self-contradiction
        if related_responses:
            all_responses = [response] + related_responses
            results["self_contradiction"] = self.detect_self_contradiction(all_responses)

        # Temporal inconsistency
        results["temporal"] = self.detect_temporal_inconsistency(response)

        # Statistical anomaly
        if expected_ranges:
            results["statistical"] = self.detect_statistical_anomaly(response, expected_ranges)

        # Log summary
        detected_methods = [
            method for method, result in results.items() if result.has_hallucination
        ]

        if detected_methods:
            logger.warning(f"Hallucination detected by methods: {', '.join(detected_methods)}")
        else:
            logger.info("No hallucinations detected by any method")

        return results

    def detect(
        self, response: str, context: str | None = None, threshold: float = 0.5
    ) -> HallucinationResult:
        """
        Simple detect method for basic hallucination detection.

        Args:
            response: AI response to check
            context: Optional context/query
            threshold: Detection threshold

        Returns:
            HallucinationResult
        """
        # Use semantic detection with known facts if context provided
        if context:
            # Simple heuristic: check if response is relevant to context
            from core.ai import AIResponseValidator

            validator = AIResponseValidator()
            is_relevant, similarity = validator.validate_relevance(context, response, threshold)

            return HallucinationResult(
                has_hallucination=not is_relevant,
                confidence=1.0 - similarity if similarity else 0.0,
                detection_method="semantic_relevance",
                details={"similarity": similarity, "context": context[:100]},
            )

        # Fallback to temporal inconsistency check
        return self.detect_temporal_inconsistency(response)

    def check_internal_consistency(self, response: str) -> ConsistencyResult:
        """
        Check for internal contradictions within a single response.

        Args:
            response: AI response to check

        Returns:
            ConsistencyResult
        """
        # Split response into sentences
        sentences = [s.strip() for s in response.split(".") if s.strip()]

        if len(sentences) < 2:
            return ConsistencyResult(
                is_consistent=True,
                contradictions=[],
                confidence=1.0,
            )

        # Check for contradictions using self-contradiction detection
        result = self.detect_self_contradiction(sentences)

        return ConsistencyResult(
            is_consistent=not result.has_hallucination,
            contradictions=result.conflicting_facts or [],
            confidence=1.0 - result.confidence,
        )

    def evaluate_grounding(self, response: str, known_sources: list[str]) -> float:
        """
        Evaluate how well the response is grounded in known sources.

        Args:
            response: AI response to check
            known_sources: List of known source facts

        Returns:
            Grounding score (0.0 to 1.0, higher is better)
        """
        if not known_sources:
            return 0.0

        # Convert known_sources to list of facts for semantic detection
        result = self.detect_semantic_hallucination(response, known_sources, threshold=0.5)

        # Return inverse of hallucination confidence as grounding score
        # Higher confidence in hallucination = lower grounding
        grounding_score = 1.0 - result.confidence

        return max(0.0, min(1.0, grounding_score))


# Global instance
_hallucination_detector: AdvancedHallucinationDetector | None = None


def get_hallucination_detector() -> AdvancedHallucinationDetector:
    """Get global hallucination detector instance."""
    global _hallucination_detector
    if _hallucination_detector is None:
        _hallucination_detector = AdvancedHallucinationDetector()
    return _hallucination_detector

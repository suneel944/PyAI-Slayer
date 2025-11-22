"""Automatic failure analysis and pattern detection."""

from .models import FailureAnalysis, ValidationDetail


class FailureAnalyzer:
    """Analyzes test failures and generates recommendations."""

    def __init__(self):
        """Initialize failure analyzer."""
        self.fallback_indicators = [
            "try again",
            "sorry, i didn't understand",
            "please rephrase",
            "i'm having trouble",
            "error",
            "حدث خطأ",
            "حاول مرة أخرى",
            "لم أفهم",
            "عذراً",
            "لا أستطيع",
        ]

    def analyze_failure(
        self,
        test_id: str,
        validation_detail: ValidationDetail | None = None,
        quality_checks: dict[str, bool] | None = None,
        scoring_details: dict[str, float] | None = None,
    ) -> FailureAnalysis:
        """
        Analyze a test failure and generate insights.

        Args:
            test_id: Test identifier
            validation_detail: Validation details
            quality_checks: Quality check results
            scoring_details: All scoring metrics

        Returns:
            FailureAnalysis with root cause and recommendations
        """
        patterns = []
        recommendations = []
        category = "Unknown"
        root_cause = "Test failed - analysis pending"

        # Analyze validation details
        if validation_detail and validation_detail.actual_response:
            actual = validation_detail.actual_response

            # Check for fallback messages
            if self._is_fallback_message(actual):
                patterns.append("fallback_message")
                category = "Knowledge Gap"
                root_cause = "Model responded with fallback/error message"
                recommendations.extend(
                    [
                        "Verify topic is within model's knowledge domain",
                        "Check if RAG/context retrieval is functioning",
                        "Review prompt formulation for clarity",
                        "Consider fine-tuning model for this use case",
                    ]
                )

            # Check for empty/short responses
            if len(actual.strip()) < 20:
                patterns.append("short_response")
                if category == "Unknown":
                    category = "Response Quality Issue"
                    root_cause = "Response is too short or empty"
                recommendations.extend(
                    [
                        "Check if model received the full prompt",
                        "Verify input token limits not exceeded",
                        "Review model configuration parameters",
                    ]
                )

            # Check for HTML/error codes
            if "<" in actual and ">" in actual:
                patterns.append("html_in_response")
                if category == "Unknown":
                    category = "Response Format Issue"
                    root_cause = "HTML tags detected in response"
                recommendations.extend(
                    [
                        "Review output sanitization",
                        "Check if API is returning error page",
                        "Verify content-type headers",
                    ]
                )

        # Analyze semantic similarity
        # Check for both "similarity_score" (stored metric) and "similarity" (legacy/alternative)
        similarity = None
        if scoring_details:
            similarity = scoring_details.get("similarity_score") or scoring_details.get(
                "similarity"
            )

        if similarity is not None and scoring_details:
            threshold = scoring_details.get("threshold", 0.5)

            if similarity < 0.3:
                patterns.append("very_low_similarity")
                if category == "Unknown":
                    category = "Semantic Mismatch"
                    root_cause = (
                        f"Very low semantic similarity ({similarity:.2f} < {threshold:.2f})"
                    )
                recommendations.extend(
                    [
                        "Review if expected response is realistic",
                        "Check if model understands the domain",
                        "Consider adjusting similarity threshold",
                        "Verify query is clear and unambiguous",
                    ]
                )
            elif similarity < threshold:
                patterns.append("low_similarity")
                if category == "Unknown":
                    category = "Semantic Mismatch"
                    root_cause = f"Low semantic similarity ({similarity:.2f} < {threshold:.2f})"
                recommendations.extend(
                    [
                        "Review expected vs actual response alignment",
                        "Consider prompt engineering improvements",
                        "Check if additional context is needed",
                    ]
                )

        # Analyze quality checks
        if quality_checks:
            if not quality_checks.get("has_minimum_length", True):
                patterns.append("minimum_length_not_met")
                if category == "Unknown":
                    category = "Response Quality Issue"
                    root_cause = "Response does not meet minimum length requirement"

            if not quality_checks.get("ends_properly", True):
                patterns.append("improper_ending")
                recommendations.append("Check if response generation was interrupted")

        # Analyze BERTScore (if available)
        if scoring_details:
            bertscore_f1 = scoring_details.get("bertscore_f1")
            if bertscore_f1 is not None and bertscore_f1 < 0.3:
                patterns.append("low_bertscore")
                if category == "Unknown":
                    category = "Response Quality Issue"
                    root_cause = f"Low BERTScore F1 ({bertscore_f1:.2f})"
                recommendations.append("Significant mismatch in content and structure")

            rouge_l = scoring_details.get("rouge_l_f1")
            if rouge_l is not None and rouge_l < 0.2:
                patterns.append("low_rouge_score")
                recommendations.append("Low lexical overlap with expected response")

        # Language-specific analysis
        if validation_detail:
            is_arabic = self._is_arabic(validation_detail.actual_response or "")
            if is_arabic:
                recommendations.append("Review Arabic language model performance")
                if validation_detail.query and not self._is_arabic(validation_detail.query):
                    patterns.append("language_mismatch")
                    category = "Language Issue"
                    root_cause = "Query and response language mismatch"
                    recommendations.append("Verify model supports requested language")

        # Default category if still unknown
        if category == "Unknown" and validation_detail and not validation_detail.passed:
            category = "Validation Failure"
            root_cause = f"Validation failed: {validation_detail.validation_type}"

        # Remove duplicates
        patterns = list(set(patterns))
        recommendations = list(set(recommendations))

        return FailureAnalysis(
            test_id=test_id,
            root_cause=root_cause,
            category=category,
            detected_patterns=patterns,
            recommendations=recommendations[:10],  # Limit to top 10
        )

    def _is_fallback_message(self, text: str) -> bool:
        """Check if text is a fallback/error message."""
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in self.fallback_indicators)

    def _is_arabic(self, text: str) -> bool:
        """Check if text contains Arabic characters."""
        return any("\u0600" <= char <= "\u06ff" for char in text)

    def categorize_failure_type(self, analysis: FailureAnalysis) -> str:
        """Get a simplified failure type for analytics."""
        if "fallback_message" in analysis.detected_patterns:
            return "Fallback Message"
        elif "very_low_similarity" in analysis.detected_patterns:
            return "Semantic Mismatch"
        elif "short_response" in analysis.detected_patterns:
            return "Short/Empty Response"
        elif "language_mismatch" in analysis.detected_patterns:
            return "Language Issue"
        elif "low_bertscore" in analysis.detected_patterns:
            return "Quality Issue"
        else:
            return analysis.category

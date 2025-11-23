"""Base model quality metrics calculator."""

import re

from loguru import logger

from core.ai.ai_validator import AIResponseValidator
from core.ai.hallucination_detector import AdvancedHallucinationDetector


class BaseModelMetricsCalculator:
    """Calculate base model quality metrics with honest naming."""

    def __init__(
        self,
        validator: AIResponseValidator | None = None,
        hallucination_detector: AdvancedHallucinationDetector | None = None,
    ):
        """
        Initialize base model metrics calculator.

        Args:
            validator: AI response validator (default: creates new instance)
            hallucination_detector: Hallucination detector (default: creates new instance)
        """
        self.validator = validator or AIResponseValidator()
        self.hallucination_detector = hallucination_detector or AdvancedHallucinationDetector()

    def calculate(
        self,
        query: str | None = None,
        response: str | None = None,
        expected_response: str | None = None,
        reference: str | None = None,
        similarity_score: float | None = None,
        known_facts: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Calculate base model metrics.

        Args:
            query: User query
            response: AI response
            expected_response: Expected response (for exact match)
            reference: Reference response (for BERTScore, ROUGE, BLEU)
            similarity_score: Pre-calculated similarity score
            known_facts: Known facts for hallucination detection

        Returns:
            Dictionary of base model metrics
        """
        metrics: dict[str, float] = {}

        if not response:
            return metrics

        # Accuracy - similarity-based proxy
        if similarity_score is not None:
            metrics["accuracy"] = similarity_score
        elif expected_response and response:
            try:
                is_relevant, sim = self.validator.validate_relevance(
                    expected_response, response, threshold=0.0
                )
                metrics["accuracy"] = sim
            except Exception as e:
                logger.debug(f"Could not calculate accuracy: {e}")

        # Exact Match
        if expected_response and response:
            exact_match = response.strip().lower() == expected_response.strip().lower()
            metrics["exact_match"] = 1.0 if exact_match else 0.0

        # Normalized Similarity Score (honest name for "top_k_accuracy")
        if expected_response and response:
            try:
                is_relevant, sim = self.validator.validate_relevance(
                    expected_response, response, threshold=0.0
                )
                # Normalize: values >= 0.8 are "top tier"
                k_threshold = 0.8
                metrics["normalized_similarity_score"] = (
                    1.0 if sim >= k_threshold else sim / k_threshold
                )
            except Exception as e:
                logger.debug(f"Could not calculate normalized_similarity_score: {e}")
        elif similarity_score is not None and expected_response:
            k_threshold = 0.8
            metrics["normalized_similarity_score"] = (
                1.0 if similarity_score >= k_threshold else similarity_score / k_threshold
            )

        # F1 Score / BERTScore
        if reference and response:
            try:
                bertscore = self.validator.calculate_bertscore(response, reference)
                if isinstance(bertscore, dict) and "f1" in bertscore:
                    metrics["f1_score"] = float(bertscore["f1"])
                    metrics["bert_score"] = float(bertscore["f1"])
            except Exception as e:
                logger.debug(f"Could not calculate f1_score: {e}")

        # BLEU Score
        if reference and response:
            try:
                from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

                ref_tokens = reference.split()
                cand_tokens = response.split()
                smoothing = SmoothingFunction().method1
                bleu = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
                metrics["bleu"] = float(bleu)
            except ImportError:
                # NLTK not available - use honest name for fallback
                ref_words = set(reference.lower().split())
                cand_words = set(response.lower().split())
                if ref_words:
                    overlap = len(ref_words & cand_words) / len(ref_words)
                    metrics["lexical_overlap"] = float(overlap)
                    # Don't call it BLEU - that's misleading
            except Exception as e:
                logger.debug(f"Could not calculate bleu: {e}")

        # ROUGE-L Score
        if reference and response:
            try:
                rouge_scores = self.validator.calculate_rouge_scores(response, reference)
                if isinstance(rouge_scores, dict) and "rougeL_f1" in rouge_scores:
                    metrics["rouge_l"] = float(rouge_scores["rougeL_f1"])
            except Exception as e:
                logger.debug(f"Could not calculate rouge_l: {e}")

        # Hallucination Detection
        if response:
            try:
                result = None
                if known_facts:
                    result = self.hallucination_detector.detect_semantic_hallucination(
                        response, known_facts, threshold=0.5
                    )
                elif expected_response:
                    result = self.hallucination_detector.detect_semantic_hallucination(
                        response, [expected_response], threshold=0.5
                    )
                elif reference:
                    result = self.hallucination_detector.detect_semantic_hallucination(
                        response, [reference], threshold=0.5
                    )

                if result:
                    metrics["hallucination_detected"] = 1.0 if result.has_hallucination else 0.0
                    metrics["hallucination_confidence"] = result.confidence * 100
                    metrics["hallucination_rate"] = metrics["hallucination_detected"] * 100
            except Exception as e:
                logger.warning(f"Hallucination detection failed: {e}")

        # Similarity-based proxies (with honest naming)
        if similarity_score is not None:
            metrics["similarity_proxy_factual_consistency"] = similarity_score * 100
            metrics["similarity_proxy_truthfulness"] = similarity_score * 100
        elif expected_response and response:
            try:
                is_relevant, sim = self.validator.validate_relevance(
                    expected_response, response, threshold=0.0
                )
                metrics["similarity_proxy_factual_consistency"] = sim * 100
                metrics["similarity_proxy_truthfulness"] = sim * 100
            except Exception as e:
                logger.debug(f"Could not calculate similarity proxies: {e}")

        # Source Grounding (similarity proxy)
        if similarity_score is not None:
            metrics["similarity_proxy_source_grounding"] = similarity_score * 100
        elif query and response:
            try:
                is_relevant, sim = self.validator.validate_relevance(query, response, threshold=0.0)
                metrics["similarity_proxy_source_grounding"] = sim * 100
            except Exception as e:
                logger.debug(f"Could not calculate similarity_proxy_source_grounding: {e}")
        elif expected_response and response:
            try:
                is_relevant, sim = self.validator.validate_relevance(
                    expected_response, response, threshold=0.0
                )
                metrics["similarity_proxy_source_grounding"] = sim * 100
            except Exception as e:
                logger.debug(f"Could not calculate similarity_proxy_source_grounding: {e}")

        # Citation Accuracy (heuristic)
        if response:
            citation_score = self._calculate_citation_accuracy(response)
            metrics["citation_accuracy"] = citation_score

        # Experimental reasoning metrics
        if response:
            reasoning_metrics = self._parse_reasoning_metrics(
                response, expected_response, reference
            )
            metrics.update(reasoning_metrics)

        return metrics

    def _calculate_citation_accuracy(self, response: str) -> float:
        """
        Calculate citation accuracy (heuristic).

        Args:
            response: AI response

        Returns:
            Citation accuracy score (0-100)
        """
        citation_patterns = [
            (r"\[(\d+)\]", 1.0),  # [1], [2]
            (r"\[([A-Za-z0-9\s]+)\]", 0.8),  # [source name]
            (r"\(([A-Za-z0-9\s]+)\)", 0.6),  # (source)
            (r"source:\s*([A-Za-z0-9\s]+)", 0.7),
            (r"according to\s+([A-Za-z0-9\s]+)", 0.5),
            (r"reference:\s*([A-Za-z0-9\s]+)", 0.6),
        ]

        citation_score = 0.0
        citations_found = []

        for pattern, weight in citation_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                citations_found.extend(matches)
                citation_score = max(citation_score, weight * 100)

        # Bonus for multiple citations
        if len(citations_found) > 1:
            citation_score = min(100.0, citation_score * (1.0 + 0.1 * (len(citations_found) - 1)))

        return citation_score

    def _parse_reasoning_metrics(
        self,
        response: str,
        expected_response: str | None = None,
        reference: str | None = None,
    ) -> dict[str, float]:
        """
        Parse reasoning steps and calculate experimental metrics.

        EXPERIMENTAL: Use for debugging/exploration only.

        Args:
            response: AI response
            expected_response: Expected response
            reference: Reference response

        Returns:
            Dictionary with experimental reasoning metrics
        """
        metrics: dict[str, float] = {}

        # Extract reasoning steps
        steps = self._extract_reasoning_steps(response)
        cot_patterns = [
            r"(?:let me|let's|first|second|third|step \d+|step \w+:|reasoning:|thinking:|step-by-step|step by step)",
            r"(?:therefore|thus|hence|so|because|since|as a result)",
            r"(?:if|then|else|when|where|given that)",
            r"(?:conclusion|in conclusion|to conclude|summary)",
            r"#{1,3}\s*\d+\.\s+",
        ]

        has_cot_structure = len(steps) > 1 or any(
            re.search(pattern, response, re.IGNORECASE) for pattern in cot_patterns
        )

        # CoT Validity
        if has_cot_structure:
            cot_validity = self._assess_cot_validity(steps, response)
            metrics["cot_validity"] = cot_validity
        elif expected_response or reference:
            target = expected_response or reference
            if target:
                try:
                    is_relevant, sim = self.validator.validate_relevance(
                        target, response, threshold=0.0
                    )
                    metrics["cot_validity"] = sim * 0.7
                except Exception:
                    metrics["cot_validity"] = 0.5

        # Step Correctness
        if steps and (expected_response or reference):
            target = expected_response or reference
            if target:
                step_correctness = self._assess_step_correctness(steps, response, target)
                metrics["step_correctness"] = step_correctness
        elif expected_response or reference:
            target = expected_response or reference
            if target:
                try:
                    is_relevant, sim = self.validator.validate_relevance(
                        target, response, threshold=0.0
                    )
                    metrics["step_correctness"] = sim
                except Exception:
                    pass

        # Logic Consistency
        if response:
            logic_consistency = self._assess_logic_consistency(response, steps)
            metrics["logic_consistency"] = logic_consistency

        return metrics

    def _extract_reasoning_steps(self, response: str) -> list[str]:
        """Extract reasoning steps from response."""
        steps: list[str] = []

        # Numbered steps
        numbered_pattern = r"(?:step|step \d+|step \w+)[\s:]+(.+?)(?=(?:step|step \d+|step \w+|$))"
        numbered_matches = re.finditer(numbered_pattern, response, re.IGNORECASE | re.DOTALL)
        for match in numbered_matches:
            step_text = match.group(1).strip()
            if step_text and len(step_text) > 10:
                steps.append(step_text)

        # Markdown headings
        if not steps:
            markdown_pattern = (
                r"(?:^|\n)#{1,3}\s*\d+\.\s+(.+?)(?=\n(?:#{1,3}\s*\d+\.|#{1,3}\s+[A-Z]|$))"
            )
            markdown_matches = re.finditer(
                markdown_pattern, response, re.IGNORECASE | re.MULTILINE | re.DOTALL
            )
            for match in markdown_matches:
                step_text = match.group(1).strip()
                if step_text and len(step_text) > 10:
                    steps.append(step_text)

        # Bullet points
        if not steps:
            bullet_pattern = r"(?:^|\n)[\s]*[-•*]\s+(.+?)(?=\n|$)"
            bullet_matches = re.finditer(bullet_pattern, response, re.MULTILINE)
            for match in bullet_matches:
                step_text = match.group(1).strip()
                if step_text and len(step_text) > 10:
                    steps.append(step_text)

        # Sentences with logical connectors
        if not steps:
            sentences = re.split(r"[.!?]+\s+", response)
            reasoning_indicators = [
                "because",
                "since",
                "therefore",
                "thus",
                "hence",
                "so",
                "if",
                "then",
                "given that",
            ]
            for sentence in sentences:
                if (
                    any(indicator in sentence.lower() for indicator in reasoning_indicators)
                    and len(sentence.strip()) > 20
                ):
                    steps.append(sentence.strip())

        if not steps and len(response) > 50:
            steps.append(response)

        return steps

    def _assess_cot_validity(self, steps: list[str], response: str) -> float:
        """Assess CoT validity (experimental)."""
        if not steps:
            return 0.0

        score = 0.0
        max_score = 0.0

        # Step length check
        for step in steps:
            max_score += 1.0
            step_length = len(step.split())
            if 5 <= step_length <= 200:
                score += 0.3
            elif step_length < 5:
                score += 0.1
            else:
                score += 0.2

        # Logical connectors
        logical_connectors = [
            "because",
            "since",
            "therefore",
            "thus",
            "hence",
            "so",
            "if",
            "then",
            "given that",
            "as a result",
        ]
        connector_count = sum(
            1
            for step in steps
            if any(connector in step.lower() for connector in logical_connectors)
        )
        if len(steps) > 1:
            max_score += 1.0
            connector_ratio = connector_count / len(steps)
            score += connector_ratio * 0.4

        # Conclusion
        conclusion_indicators = ["conclusion", "summary", "therefore", "thus", "in summary"]
        has_conclusion = any(indicator in response.lower() for indicator in conclusion_indicators)
        max_score += 1.0
        if has_conclusion:
            score += 0.3

        if max_score > 0:
            return min(1.0, score / max_score)
        return 0.5

    def _assess_step_correctness(self, steps: list[str], response: str, target: str) -> float:
        """Assess step correctness (experimental)."""
        if not steps or not target:
            return 0.0

        try:
            step_scores = []
            for step in steps:
                is_relevant, sim = self.validator.validate_relevance(target, step, threshold=0.0)
                step_scores.append(sim)

            if step_scores:
                avg_correctness = sum(step_scores) / len(step_scores)
                is_relevant, overall_sim = self.validator.validate_relevance(
                    target, response, threshold=0.0
                )
                return (avg_correctness * 0.7) + (overall_sim * 0.3)
        except Exception:
            try:
                is_relevant, sim = self.validator.validate_relevance(
                    target, response, threshold=0.0
                )
                return sim
            except Exception:
                return 0.5

        return 0.5

    def _assess_logic_consistency(self, response: str, steps: list[str] | None = None) -> float:
        """Assess logic consistency (experimental)."""
        score = 1.0
        penalty = 0.0

        # Contradiction patterns
        contradiction_patterns = [
            (r"not\s+\w+\s+but\s+not", 0.2),
            (r"both\s+\w+\s+and\s+not\s+\w+", 0.3),
            (r"always\s+\w+\s+never", 0.2),
        ]

        for pattern, penalty_value in contradiction_patterns:
            matches = len(re.findall(pattern, response, re.IGNORECASE))
            if matches > 0:
                penalty += penalty_value * matches

        # Concept overlap in steps
        if steps and len(steps) > 1:
            concepts = []
            for step in steps:
                words = [
                    w.lower()
                    for w in step.split()
                    if len(w) > 4 and w.lower() not in ["that", "this", "with", "from", "which"]
                ]
                concepts.append(set(words))

            overlap_score = 0.0
            for i in range(len(concepts) - 1):
                if concepts[i] and concepts[i + 1]:
                    overlap = len(concepts[i] & concepts[i + 1]) / max(
                        len(concepts[i] | concepts[i + 1]), 1
                    )
                    overlap_score += overlap
            if len(concepts) > 1:
                avg_overlap = overlap_score / (len(concepts) - 1)
                if 0.2 <= avg_overlap <= 0.6:
                    score += 0.1
                elif avg_overlap < 0.1:
                    penalty += 0.2

        return max(0.0, min(1.0, score - penalty))

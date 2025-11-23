"""Comprehensive metrics calculator for dashboard metrics.

This module provides backward-compatible access to the refactored metrics engine.
The MetricsCalculator class is now a thin wrapper around MetricsEngine.
"""

from typing import Any

from loguru import logger

from .metrics_engine import MetricsEngine


class MetricsCalculator:
    """
    Calculate comprehensive metrics for dashboard display.

    This class is a backward-compatible wrapper around the refactored MetricsEngine.
    For new code, prefer using MetricsEngine directly for better configurability.
    """

    def __init__(self):
        """Initialize metrics calculator (uses MetricsEngine under the hood)."""
        self._engine = MetricsEngine()
        logger.info("Metrics calculator initialized (using refactored MetricsEngine)")

    def calculate_base_model_metrics(
        self,
        query: str | None = None,
        response: str | None = None,
        expected_response: str | None = None,
        reference: str | None = None,
        validation_type: str = "unknown",  # noqa: ARG002
        similarity_score: float | None = None,
        known_facts: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Calculate base model quality metrics.

        Delegates to MetricsEngine.base_model_calculator for backward compatibility.

        Args:
            query: User query
            response: AI response
            expected_response: Expected response (for exact match)
            reference: Reference response (for BERTScore, ROUGE)
            validation_type: Type of validation
            similarity_score: Pre-calculated similarity score
            known_facts: Known facts for hallucination detection

        Returns:
            Dictionary of base model metrics
        """
        return self._engine.base_model_calculator.calculate(
            query=query,
            response=response,
            expected_response=expected_response,
            reference=reference,
            similarity_score=similarity_score,
            known_facts=known_facts,
        )

    def calculate_rag_metrics(
        self,
        query: str | None = None,
        response: str | None = None,
        retrieved_docs: list[str] | None = None,
        expected_sources: list[str] | None = None,
        gold_context: str | None = None,
    ) -> dict[str, float]:
        """
        Calculate RAG pipeline metrics.

        Delegates to MetricsEngine.rag_calculator for backward compatibility.

        Args:
            query: User query
            response: AI response
            retrieved_docs: Documents retrieved by RAG system
            expected_sources: Expected source documents
            gold_context: Gold standard context

        Returns:
            Dictionary of RAG metrics
        """
        return self._engine.rag_calculator.calculate(
            query=query,
            response=response,
            retrieved_docs=retrieved_docs,
            expected_sources=expected_sources,
            gold_context=gold_context,
        )

    def calculate_safety_metrics(
        self, response: str | None = None, query: str | None = None
    ) -> dict[str, float]:
        """
        Calculate safety and guardrail metrics.

        Delegates to MetricsEngine.safety_calculator for backward compatibility.

        Args:
            response: AI response
            query: User query

        Returns:
            Dictionary of safety metrics
        """
        return self._engine.safety_calculator.calculate(response=response, query=query)

    def calculate_performance_metrics(
        self,
        duration: float | None = None,
        response_tokens: int | None = None,
        first_token_time: float | None = None,
        response: str | None = None,
        response_length: int | None = None,
    ) -> dict[str, float]:
        """
        Calculate performance metrics.

        Delegates to MetricsEngine.performance_calculator for backward compatibility.

        Args:
            duration: Total response duration in seconds
            response_tokens: Number of tokens in response
            first_token_time: Time to first token in seconds
            response: AI response text (for token estimation)
            response_length: Response length in characters

        Returns:
            Dictionary of performance metrics
        """
        return self._engine.performance_calculator.calculate(
            duration=duration,
            response_tokens=response_tokens,
            first_token_time=first_token_time,
            response=response,
            response_length=response_length,
        )

    def calculate_reliability_metrics(
        self,
        response: str | None = None,
        previous_responses: list[str] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """
        Calculate reliability and stability metrics.

        Delegates to MetricsEngine.reliability_calculator for backward compatibility.

        Args:
            response: Current AI response
            previous_responses: Previous responses for stability check
            schema: Expected schema for validation

        Returns:
            Dictionary of reliability metrics
        """
        return self._engine.reliability_calculator.calculate(
            response=response,
            previous_responses=previous_responses,
            schema=schema,
        )

    def calculate_agent_metrics(
        self,
        task_completed: bool | None = None,
        steps_taken: int | None = None,
        expected_steps: int | None = None,
        errors_encountered: int | None = None,
        tools_used: list[str] | None = None,
        tools_succeeded: list[str] | None = None,
        planning_trace: dict[str, Any] | None = None,
        valid_actions: list[str] | None = None,
        goal_tracking: dict[str, Any] | None = None,
        query: str | None = None,
        response: str | None = None,
    ) -> dict[str, float]:
        """
        Calculate agent and autonomous system metrics.

        Delegates to MetricsEngine.agent_calculator for backward compatibility.

        Args:
            task_completed: Whether task was completed
            steps_taken: Number of steps taken
            expected_steps: Expected number of steps
            errors_encountered: Number of errors
            tools_used: List of tools used
            tools_succeeded: List of tools that succeeded
            planning_trace: Dict with 'planned_steps' and 'actual_steps'
            valid_actions: List of valid/available actions
            goal_tracking: Dict with 'original_goal' and 'steps'
            query: Original query
            response: Agent response

        Returns:
            Dictionary of agent metrics
        """
        return self._engine.agent_calculator.calculate(
            task_completed=task_completed,
            steps_taken=steps_taken,
            expected_steps=expected_steps,
            errors_encountered=errors_encountered,
            tools_used=tools_used,
            tools_succeeded=tools_succeeded,
            planning_trace=planning_trace,
            valid_actions=valid_actions,
            goal_tracking=goal_tracking,
            query=query,
            response=response,
        )

    def calculate_security_metrics(
        self,
        query: str | None = None,
        response: str | None = None,
        injection_attempts: int | None = None,  # noqa: ARG002
        adversarial_tests: int | None = None,
        exfiltration_attempts: int | None = None,
        evasion_attempts: int | None = None,
        extraction_attempts: int | None = None,
    ) -> dict[str, float]:
        """
        Calculate security testing metrics.

        Delegates to MetricsEngine.security_calculator for backward compatibility.

        Args:
            query: User query
            response: AI response
            injection_attempts: Number of injection attempts tested
            adversarial_tests: Number of adversarial tests
            exfiltration_attempts: Number of exfiltration attempts
            evasion_attempts: Number of evasion attempts
            extraction_attempts: Number of extraction attempts

        Returns:
            Dictionary of security metrics
        """
        return self._engine.security_calculator.calculate(
            query=query,
            response=response,
            injection_attempts=injection_attempts,
            adversarial_tests=adversarial_tests,
            exfiltration_attempts=exfiltration_attempts,
            evasion_attempts=evasion_attempts,
            extraction_attempts=extraction_attempts,
        )

    def calculate_all_metrics(
        self,
        query: str | None = None,
        response: str | None = None,
        expected_response: str | None = None,
        reference: str | None = None,
        validation_type: str = "unknown",
        similarity_score: float | None = None,
        duration: float | None = None,
        retrieved_docs: list[str] | None = None,
        expected_sources: list[str] | None = None,
        gold_context: str | None = None,
        previous_responses: list[str] | None = None,
        schema: dict[str, Any] | None = None,
        known_facts: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Calculate all metrics comprehensively.

        Delegates to MetricsEngine.calculate_all for backward compatibility.

        Args:
            query: User query
            response: AI response
            expected_response: Expected response
            reference: Reference response for quality metrics
            validation_type: Type of validation
            similarity_score: Pre-calculated similarity
            duration: Response duration
            retrieved_docs: Retrieved documents (RAG)
            expected_sources: Expected sources (RAG)
            gold_context: Gold standard context (RAG)
            previous_responses: Previous responses (for stability)
            schema: Expected schema
            known_facts: Known facts for hallucination detection
            **kwargs: Additional parameters

        Returns:
            Dictionary of all calculated metrics
        """
        return self._engine.calculate_all(
            query=query,
            response=response,
            expected_response=expected_response,
            reference=reference,
            validation_type=validation_type,
            similarity_score=similarity_score,
            duration=duration,
            retrieved_docs=retrieved_docs,
            expected_sources=expected_sources,
            gold_context=gold_context,
            previous_responses=previous_responses,
            schema=schema,
            known_facts=known_facts,
            **kwargs,
        )


# Global instance
_metrics_calculator: MetricsCalculator | None = None


def get_metrics_calculator() -> MetricsCalculator:
    """Get global metrics calculator instance."""
    global _metrics_calculator
    if _metrics_calculator is None:
        _metrics_calculator = MetricsCalculator()
    return _metrics_calculator

"""Refactored metrics engine with modular architecture."""

from typing import Any

from loguru import logger

from .calculators import (
    AgentMetricsCalculator,
    BaseModelMetricsCalculator,
    PerformanceMetricsCalculator,
    RAGMetricsCalculator,
    ReliabilityMetricsCalculator,
    SafetyMetricsCalculator,
    SecurityMetricsCalculator,
)
from .calculators.detectors import CompositeToxicityDetector, ToxicityDetector
from .metric_validator import get_metric_validator


class MetricsEngine:
    """
    Refactored metrics engine with modular architecture.

    This engine orchestrates specialized calculators for different metric groups.
    Dependencies are injected for testability and configurability.
    """

    def __init__(
        self,
        # Calculator instances (optional - will create defaults if None)
        base_model_calculator: BaseModelMetricsCalculator | None = None,
        rag_calculator: RAGMetricsCalculator | None = None,
        safety_calculator: SafetyMetricsCalculator | None = None,
        performance_calculator: PerformanceMetricsCalculator | None = None,
        reliability_calculator: ReliabilityMetricsCalculator | None = None,
        agent_calculator: AgentMetricsCalculator | None = None,
        security_calculator: SecurityMetricsCalculator | None = None,
        # Configuration flags
        enable_base_model: bool = True,
        enable_rag: bool = True,
        enable_safety: bool = True,
        enable_performance: bool = True,
        enable_reliability: bool = True,
        enable_agent: bool = True,
        enable_security: bool = True,
        # Heavy dependencies
        toxicity_detector: ToxicityDetector | None = None,
    ):
        """
        Initialize metrics engine.

        Args:
            base_model_calculator: Base model metrics calculator
            rag_calculator: RAG metrics calculator
            safety_calculator: Safety metrics calculator
            performance_calculator: Performance metrics calculator
            reliability_calculator: Reliability metrics calculator
            agent_calculator: Agent metrics calculator
            security_calculator: Security metrics calculator
            enable_base_model: Enable base model metrics
            enable_rag: Enable RAG metrics
            enable_safety: Enable safety metrics
            enable_performance: Enable performance metrics
            enable_reliability: Enable reliability metrics
            enable_agent: Enable agent metrics
            enable_security: Enable security metrics
            toxicity_detector: Toxicity detector (default: CompositeToxicityDetector)
        """
        # Initialize calculators (with dependency injection)
        self.base_model_calculator = base_model_calculator or BaseModelMetricsCalculator()
        self.rag_calculator = rag_calculator or RAGMetricsCalculator()
        self.safety_calculator = safety_calculator or SafetyMetricsCalculator(
            toxicity_detector=toxicity_detector or CompositeToxicityDetector()
        )
        self.performance_calculator = performance_calculator or PerformanceMetricsCalculator()
        self.reliability_calculator = reliability_calculator or ReliabilityMetricsCalculator()
        self.agent_calculator = agent_calculator or AgentMetricsCalculator()
        self.security_calculator = security_calculator or SecurityMetricsCalculator()

        # Configuration flags
        self.enable_base_model = enable_base_model
        self.enable_rag = enable_rag
        self.enable_safety = enable_safety
        self.enable_performance = enable_performance
        self.enable_reliability = enable_reliability
        self.enable_agent = enable_agent
        self.enable_security = enable_security

        # Metric validator
        self.metric_validator = get_metric_validator()

        logger.info("Metrics engine initialized")

    def calculate_all(
        self,
        query: str | None = None,
        response: str | None = None,
        expected_response: str | None = None,
        reference: str | None = None,
        validation_type: str = "unknown",  # noqa: ARG002
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
            **kwargs: Additional parameters (for agent, performance, security metrics)

        Returns:
            Dictionary of all calculated metrics
        """
        all_metrics: dict[str, float | None] = {}

        # Base Model Metrics
        if self.enable_base_model:
            try:
                base_metrics = self.base_model_calculator.calculate(
                    query=query,
                    response=response,
                    expected_response=expected_response,
                    reference=reference,
                    similarity_score=similarity_score,
                    known_facts=known_facts,
                )
                all_metrics.update(base_metrics)
            except Exception as e:
                logger.warning(f"Base model metrics calculation failed: {e}")

        # RAG Metrics
        if self.enable_rag:
            try:
                rag_metrics = self.rag_calculator.calculate(
                    query=query,
                    response=response,
                    retrieved_docs=retrieved_docs,
                    expected_sources=expected_sources,
                    gold_context=gold_context,
                )
                all_metrics.update(rag_metrics)
            except Exception as e:
                logger.warning(f"RAG metrics calculation failed: {e}")

        # Safety Metrics
        if self.enable_safety:
            try:
                safety_metrics = self.safety_calculator.calculate(response=response, query=query)
                all_metrics.update(safety_metrics)
            except Exception as e:
                logger.warning(f"Safety metrics calculation failed: {e}")

        # Performance Metrics
        if self.enable_performance:
            try:
                performance_metrics = self.performance_calculator.calculate(
                    duration=duration,
                    response_tokens=kwargs.get("response_tokens"),
                    first_token_time=kwargs.get("first_token_time"),
                    response=response,
                    response_length=kwargs.get("response_length"),
                )
                all_metrics.update(performance_metrics)
            except Exception as e:
                logger.warning(f"Performance metrics calculation failed: {e}")

        # Reliability Metrics
        if self.enable_reliability:
            try:
                reliability_metrics = self.reliability_calculator.calculate(
                    response=response,
                    previous_responses=previous_responses,
                    schema=schema,
                )
                all_metrics.update(reliability_metrics)
            except Exception as e:
                logger.warning(f"Reliability metrics calculation failed: {e}")

        # Agent Metrics
        if self.enable_agent:
            try:
                agent_metrics = self.agent_calculator.calculate(
                    task_completed=kwargs.get("task_completed"),
                    steps_taken=kwargs.get("steps_taken"),
                    expected_steps=kwargs.get("expected_steps"),
                    errors_encountered=kwargs.get("errors_encountered"),
                    tools_used=kwargs.get("tools_used"),
                    tools_succeeded=kwargs.get("tools_succeeded"),
                    planning_trace=kwargs.get("planning_trace"),
                    valid_actions=kwargs.get("valid_actions"),
                    goal_tracking=kwargs.get("goal_tracking"),
                    query=query,
                    response=response,
                )
                all_metrics.update(agent_metrics)
            except Exception as e:
                logger.warning(f"Agent metrics calculation failed: {e}")

        # Security Metrics
        if self.enable_security:
            try:
                security_metrics = self.security_calculator.calculate(
                    query=query,
                    response=response,
                    injection_attempts=kwargs.get("injection_attempts"),
                    adversarial_tests=kwargs.get("adversarial_tests"),
                    exfiltration_attempts=kwargs.get("exfiltration_attempts"),
                    evasion_attempts=kwargs.get("evasion_attempts"),
                    extraction_attempts=kwargs.get("extraction_attempts"),
                )
                all_metrics.update(security_metrics)
            except Exception as e:
                logger.warning(f"Security metrics calculation failed: {e}")

        # Validate all metrics before returning
        validated_metrics = self.metric_validator.validate_all(all_metrics)
        return validated_metrics


# Global instance (for backward compatibility)
_metrics_engine: MetricsEngine | None = None


def get_metrics_engine() -> MetricsEngine:
    """Get global metrics engine instance."""
    global _metrics_engine
    if _metrics_engine is None:
        _metrics_engine = MetricsEngine()
    return _metrics_engine

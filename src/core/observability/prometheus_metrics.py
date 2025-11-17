"""Prometheus metrics collection for test execution."""

import threading
from typing import Any

from loguru import logger

from config.settings import settings


class PrometheusMetrics:
    """Prometheus metrics collector for test execution."""

    def __init__(self, enabled: bool | None = None):
        """
        Initialize Prometheus metrics collector.

        Args:
            enabled: Whether metrics collection is enabled (default: from settings)
        """
        self._enabled = (
            enabled
            if enabled is not None
            else getattr(settings, "enable_prometheus_metrics", False)
        )
        self._lock = threading.Lock()
        self._metrics: dict[str, Any] = {
            "test_total": 0,
            "test_passed": 0,
            "test_failed": 0,
            "test_skipped": 0,
            "test_duration_seconds": [],
            "validation_total": 0,
            "validation_passed": 0,
            "validation_failed": 0,
            "browser_operations_total": 0,
            "browser_operations_duration_seconds": [],
        }

        # Try to import prometheus_client (optional dependency)
        self._prometheus_available = False
        if self._enabled:
            try:
                from prometheus_client import Counter, Gauge, Histogram, start_http_server

                self._prometheus_available = True
                self._counter = Counter
                self._histogram = Histogram
                self._gauge = Gauge
                self._start_http_server = start_http_server

                # Initialize Prometheus metrics
                self._init_prometheus_metrics()
                logger.info("Prometheus metrics initialized")
            except ImportError:
                logger.warning(
                    "Prometheus client not installed. Install with: pip install prometheus-client"
                )
                self._prometheus_available = False
                self._enabled = False

    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metric objects."""
        if not self._prometheus_available:
            return

        # Test execution metrics
        self.test_total = self._counter(
            "pyai_slayer_test_total",
            "Total number of tests executed",
            ["status"],
        )
        self.test_duration = self._histogram(
            "pyai_slayer_test_duration_seconds",
            "Test execution duration in seconds",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
        )
        self.test_active = self._gauge(
            "pyai_slayer_test_active",
            "Number of tests currently running",
        )

        # ========================================================================
        # A-TIER UNIVERSAL AI CAPABILITY METRICS
        # ========================================================================

        # 1. Accuracy / Task Success Rate
        self.ai_task_success_rate = self._counter(
            "ai_task_success_total",
            "AI task success/failure count (universal accuracy metric)",
            ["status", "task_type", "language"],
        )

        # 2. Consistency / Stability Score
        self.ai_output_consistency_score = self._histogram(
            "ai_output_consistency_score",
            "AI output consistency across runs (0-1, higher = more stable)",
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            labelnames=["language"],
        )

        # 3. Hallucination Rate
        self.ai_hallucination_rate = self._counter(
            "ai_hallucination_total",
            "AI hallucination detection count",
            ["detected", "language"],
        )

        # 4. Reasoning Depth Score
        self.ai_reasoning_score = self._histogram(
            "ai_reasoning_score",
            "AI reasoning depth/capability score (0-1 or 0-100)",
            buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            labelnames=["language", "reasoning_type"],
        )

        # 5. Latency (p95/p99)
        self.ai_inference_latency_seconds = self._histogram(
            "ai_inference_latency_seconds",
            "AI inference latency in seconds (p95/p99 critical)",
            buckets=[0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0],
            labelnames=["language", "percentile"],
        )

        # 6. Robustness / Stress Resilience
        self.ai_robustness_score = self._histogram(
            "ai_robustness_score",
            "AI robustness under stress/edge cases (0-1)",
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            labelnames=["stress_type", "language"],
        )
        self.ai_robustness_failures = self._counter(
            "ai_robustness_failures_total",
            "AI failures under stress tests",
            ["stress_type", "language"],
        )

        # 7. Safety & Policy Violation Rate
        self.ai_safety_violation_rate = self._counter(
            "ai_safety_violations_total",
            "AI safety/policy violations count",
            ["violation_type", "severity", "language"],
        )

        # 8. Grounding Score (for RAG/data-backed models)
        self.ai_grounding_accuracy = self._histogram(
            "ai_grounding_accuracy",
            "AI grounding accuracy - traceability to verifiable information (0-1)",
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            labelnames=["language", "grounding_type"],
        )

        # 9. Multi-step Instruction Following Score
        self.ai_instruction_following_score = self._histogram(
            "ai_instruction_following_score",
            "AI multi-step instruction following capability (0-1)",
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            labelnames=["language", "instruction_complexity"],
        )
        self.ai_instruction_following_total = self._counter(
            "ai_instruction_following_total",
            "AI instruction following attempts",
            ["status", "language", "instruction_complexity"],
        )

        # 10. Tool / Function Use Success Rate
        self.ai_tool_use_success_rate = self._counter(
            "ai_tool_use_total",
            "AI tool/function call success rate",
            ["status", "tool_type", "language"],
        )
        self.ai_tool_use_latency = self._histogram(
            "ai_tool_use_latency_seconds",
            "AI tool/function call latency",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            labelnames=["tool_type", "language"],
        )

        # Legacy metrics (for backward compatibility)
        self.validation_total = self._counter(
            "pyai_slayer_validation_total",
            "Total number of validations performed (legacy)",
            ["type", "status", "language"],
        )
        self.validation_similarity = self._histogram(
            "pyai_slayer_validation_similarity_score",
            "Semantic similarity scores (legacy)",
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            labelnames=["type", "language"],
        )
        self.ai_response_time = self._histogram(
            "pyai_slayer_ai_response_time_seconds",
            "AI response time in seconds (legacy - use ai_inference_latency_seconds)",
            buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 30.0, 60.0],
            labelnames=["language"],
        )

        # Browser operation metrics
        self.browser_operations_total = self._counter(
            "pyai_slayer_browser_operations_total",
            "Total number of browser operations",
            ["operation", "status"],
        )
        self.browser_operations_duration = self._histogram(
            "pyai_slayer_browser_operations_duration_seconds",
            "Browser operation duration in seconds",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        )

    @property
    def enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self._enabled and self._prometheus_available

    def start_server(self, port: int = 8000) -> None:
        """
        Start Prometheus HTTP server.

        Args:
            port: Port to expose metrics on
        """
        if not self.enabled:
            return

        try:
            self._start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.warning(f"Failed to start Prometheus server: {e}")

    def record_test_start(self, _test_name: str) -> None:
        """Record test start."""
        if not self.enabled:
            return

        try:
            self.test_active.inc()
        except Exception as e:
            logger.debug(f"Failed to record test start: {e}")

    def record_test_end(self, _test_name: str, status: str, duration: float) -> None:
        """
        Record test end.

        Args:
            test_name: Test name
            status: Test status (passed, failed, skipped)
            duration: Test duration in seconds
        """
        if not self.enabled:
            return

        try:
            self.test_total.labels(status=status).inc()
            self.test_duration.observe(duration)
            self.test_active.dec()

            with self._lock:
                self._metrics["test_total"] += 1
                if status == "passed":
                    self._metrics["test_passed"] += 1
                elif status == "failed":
                    self._metrics["test_failed"] += 1
                elif status == "skipped":
                    self._metrics["test_skipped"] += 1
                self._metrics["test_duration_seconds"].append(duration)
        except Exception as e:
            logger.debug(f"Failed to record test end: {e}")

    def record_validation(
        self,
        validation_type: str,
        passed: bool,
        similarity: float | None = None,
        language: str = "unknown",
    ) -> None:
        """
        Record validation result with language tracking (legacy method).

        Also records to A-tier metrics automatically.

        Args:
            validation_type: Type of validation (relevance, hallucination, consistency, cross_language)
            passed: Whether validation passed
            similarity: Similarity score (if applicable)
            language: Language code (en, ar, unknown)
        """
        if not self.enabled:
            return

        try:
            status = "passed" if passed else "failed"

            # Legacy metrics
            self.validation_total.labels(
                type=validation_type, status=status, language=language
            ).inc()

            if similarity is not None:
                self.validation_similarity.labels(type=validation_type, language=language).observe(
                    similarity
                )

            # A-tier metrics mapping
            # 1. Task Success Rate (from validation results)
            self.ai_task_success_rate.labels(
                status=status, task_type=validation_type, language=language
            ).inc()

            # 8. Grounding Accuracy (from similarity scores)
            if similarity is not None and validation_type in ("relevance", "cross_language"):
                self.ai_grounding_accuracy.labels(
                    language=language, grounding_type="semantic_similarity"
                ).observe(similarity)

            with self._lock:
                self._metrics["validation_total"] += 1
                if passed:
                    self._metrics["validation_passed"] += 1
                else:
                    self._metrics["validation_failed"] += 1
        except Exception as e:
            logger.debug(f"Failed to record validation: {e}")

    def record_ai_response_time(self, response_time: float, language: str = "unknown") -> None:
        """
        Record AI response time (legacy method).

        Also records to A-tier latency metrics.

        Args:
            response_time: Response time in seconds
            language: Language code (en, ar, unknown)
        """
        if not self.enabled:
            return

        try:
            # Legacy metric
            self.ai_response_time.labels(language=language).observe(response_time)

            # 5. A-tier Latency (p95/p99)
            self.ai_inference_latency_seconds.labels(language=language, percentile="all").observe(
                response_time
            )

            with self._lock:
                if "ai_response_times" not in self._metrics:
                    self._metrics["ai_response_times"] = []
                self._metrics["ai_response_times"].append(response_time)
        except Exception as e:
            logger.debug(f"Failed to record AI response time: {e}")

    # ========================================================================
    # A-TIER METRIC RECORDING METHODS
    # ========================================================================

    def record_task_success(
        self, success: bool, task_type: str = "general", language: str = "unknown"
    ) -> None:
        """
        Record task success/failure (Metric #1: Accuracy / Task Success Rate).

        Args:
            success: Whether task succeeded
            task_type: Type of task (query_response, classification, generation, etc.)
            language: Language code (en, ar, unknown)
        """
        if not self.enabled:
            return

        try:
            status = "success" if success else "failure"
            self.ai_task_success_rate.labels(
                status=status, task_type=task_type, language=language
            ).inc()
        except Exception as e:
            logger.debug(f"Failed to record task success: {e}")

    def record_consistency_score(self, consistency_score: float, language: str = "unknown") -> None:
        """
        Record consistency/stability score (Metric #2: Consistency / Stability Score).

        Args:
            consistency_score: Consistency score (0-1, higher = more stable)
            language: Language code (en, ar, unknown)
        """
        if not self.enabled:
            return

        try:
            self.ai_output_consistency_score.labels(language=language).observe(consistency_score)
        except Exception as e:
            logger.debug(f"Failed to record consistency score: {e}")

    def record_hallucination(self, detected: bool, language: str = "unknown") -> None:
        """
        Record hallucination detection (Metric #3: Hallucination Rate).

        Args:
            detected: Whether hallucination was detected
            language: Language code (en, ar, unknown)
        """
        if not self.enabled:
            return

        try:
            detected_str = "yes" if detected else "no"
            self.ai_hallucination_rate.labels(detected=detected_str, language=language).inc()
        except Exception as e:
            logger.debug(f"Failed to record hallucination: {e}")

    def record_reasoning_score(
        self, reasoning_score: float, language: str = "unknown", reasoning_type: str = "general"
    ) -> None:
        """
        Record reasoning depth score (Metric #4: Reasoning Depth Score).

        Args:
            reasoning_score: Reasoning score (0-1 or 0-100)
            language: Language code (en, ar, unknown)
            reasoning_type: Type of reasoning (logical, multi_step, causal, etc.)
        """
        if not self.enabled:
            return

        try:
            # Normalize to 0-100 if needed
            if 0.0 <= reasoning_score <= 1.0:
                reasoning_score = reasoning_score * 100

            self.ai_reasoning_score.labels(
                language=language, reasoning_type=reasoning_type
            ).observe(reasoning_score)
        except Exception as e:
            logger.debug(f"Failed to record reasoning score: {e}")

    def record_inference_latency(
        self, latency: float, language: str = "unknown", percentile: str = "all"
    ) -> None:
        """
        Record inference latency (Metric #5: Latency p95/p99).

        Args:
            latency: Latency in seconds
            language: Language code (en, ar, unknown)
            percentile: Percentile label (all, p50, p95, p99)
        """
        if not self.enabled:
            return

        try:
            self.ai_inference_latency_seconds.labels(
                language=language, percentile=percentile
            ).observe(latency)
        except Exception as e:
            logger.debug(f"Failed to record inference latency: {e}")

    def record_robustness(
        self,
        robustness_score: float,
        stress_type: str = "general",
        language: str = "unknown",
        failed: bool = False,
    ) -> None:
        """
        Record robustness score (Metric #6: Robustness / Stress Resilience).

        Args:
            robustness_score: Robustness score (0-1)
            stress_type: Type of stress (edge_case, adversarial, fuzzing, etc.)
            language: Language code (en, ar, unknown)
            failed: Whether robustness test failed
        """
        if not self.enabled:
            return

        try:
            self.ai_robustness_score.labels(stress_type=stress_type, language=language).observe(
                robustness_score
            )

            if failed:
                self.ai_robustness_failures.labels(stress_type=stress_type, language=language).inc()
        except Exception as e:
            logger.debug(f"Failed to record robustness: {e}")

    def record_safety_violation(
        self, violation_type: str, severity: str = "medium", language: str = "unknown"
    ) -> None:
        """
        Record safety/policy violation (Metric #7: Safety & Policy Violation Rate).

        Args:
            violation_type: Type of violation (harmful_content, bias, policy_breach, etc.)
            severity: Severity level (low, medium, high, critical)
            language: Language code (en, ar, unknown)
        """
        if not self.enabled:
            return

        try:
            self.ai_safety_violation_rate.labels(
                violation_type=violation_type, severity=severity, language=language
            ).inc()
        except Exception as e:
            logger.debug(f"Failed to record safety violation: {e}")

    def record_grounding_accuracy(
        self,
        grounding_score: float,
        language: str = "unknown",
        grounding_type: str = "semantic_similarity",
    ) -> None:
        """
        Record grounding accuracy (Metric #8: Grounding Score).

        Args:
            grounding_score: Grounding accuracy (0-1)
            language: Language code (en, ar, unknown)
            grounding_type: Type of grounding (semantic_similarity, citation, evidence, etc.)
        """
        if not self.enabled:
            return

        try:
            self.ai_grounding_accuracy.labels(
                language=language, grounding_type=grounding_type
            ).observe(grounding_score)
        except Exception as e:
            logger.debug(f"Failed to record grounding accuracy: {e}")

    def record_instruction_following(
        self,
        success: bool,
        score: float | None = None,
        language: str = "unknown",
        instruction_complexity: str = "simple",
    ) -> None:
        """
        Record instruction following (Metric #9: Multi-step Instruction Following Score).

        Args:
            success: Whether instruction was followed successfully
            score: Instruction following score (0-1, optional)
            language: Language code (en, ar, unknown)
            instruction_complexity: Complexity level (simple, multi_step, complex)
        """
        if not self.enabled:
            return

        try:
            status = "success" if success else "failure"
            self.ai_instruction_following_total.labels(
                status=status, language=language, instruction_complexity=instruction_complexity
            ).inc()

            if score is not None:
                self.ai_instruction_following_score.labels(
                    language=language, instruction_complexity=instruction_complexity
                ).observe(score)
        except Exception as e:
            logger.debug(f"Failed to record instruction following: {e}")

    def record_tool_use(
        self,
        success: bool,
        tool_type: str = "unknown",
        language: str = "unknown",
        latency: float | None = None,
    ) -> None:
        """
        Record tool/function use (Metric #10: Tool / Function Use Success Rate).

        Args:
            success: Whether tool use succeeded
            tool_type: Type of tool (api_call, database_query, calculator, etc.)
            language: Language code (en, ar, unknown)
            latency: Tool call latency in seconds (optional)
        """
        if not self.enabled:
            return

        try:
            status = "success" if success else "failure"
            self.ai_tool_use_success_rate.labels(
                status=status, tool_type=tool_type, language=language
            ).inc()

            if latency is not None:
                self.ai_tool_use_latency.labels(tool_type=tool_type, language=language).observe(
                    latency
                )
        except Exception as e:
            logger.debug(f"Failed to record tool use: {e}")

    def record_browser_operation(
        self, operation: str, duration: float, status: str = "success"
    ) -> None:
        """
        Record browser operation.

        Args:
            operation: Operation name (navigate, click, fill, etc.)
            duration: Operation duration in seconds
            status: Operation status (success, error)
        """
        if not self.enabled:
            return

        try:
            self.browser_operations_total.labels(operation=operation, status=status).inc()
            self.browser_operations_duration.observe(duration)

            with self._lock:
                self._metrics["browser_operations_total"] += 1
                self._metrics["browser_operations_duration_seconds"].append(duration)
        except Exception as e:
            logger.debug(f"Failed to record browser operation: {e}")

    def get_metrics_summary(self) -> dict[str, Any]:
        """
        Get metrics summary (for non-Prometheus use cases).

        Returns:
            Dictionary with metrics summary
        """
        with self._lock:
            test_durations = self._metrics["test_duration_seconds"]
            browser_durations = self._metrics["browser_operations_duration_seconds"]

            return {
                "tests": {
                    "total": self._metrics["test_total"],
                    "passed": self._metrics["test_passed"],
                    "failed": self._metrics["test_failed"],
                    "skipped": self._metrics["test_skipped"],
                    "avg_duration": (
                        sum(test_durations) / len(test_durations) if test_durations else 0.0
                    ),
                    "max_duration": max(test_durations) if test_durations else 0.0,
                },
                "validations": {
                    "total": self._metrics["validation_total"],
                    "passed": self._metrics["validation_passed"],
                    "failed": self._metrics["validation_failed"],
                },
                "browser_operations": {
                    "total": self._metrics["browser_operations_total"],
                    "avg_duration": (
                        sum(browser_durations) / len(browser_durations)
                        if browser_durations
                        else 0.0
                    ),
                },
            }


# Global metrics instance
_prometheus_metrics: PrometheusMetrics | None = None


def get_prometheus_metrics() -> PrometheusMetrics:
    """Get global Prometheus metrics instance."""
    global _prometheus_metrics
    if _prometheus_metrics is None:
        _prometheus_metrics = PrometheusMetrics()
    return _prometheus_metrics


def reset_prometheus_metrics() -> None:
    """Reset global Prometheus metrics (useful for testing)."""
    global _prometheus_metrics
    _prometheus_metrics = None

"""Data collectors for capturing test execution data."""

import hashlib
from datetime import datetime
from pathlib import Path

from loguru import logger

# Check for optional dependencies
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import pynvml  # noqa: F401

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    pynvml = None

from .data_store import DashboardDataStore
from .failure_analyzer import FailureAnalyzer
from .metrics_calculator import get_metrics_calculator
from .models import (
    MetricsSnapshot,
    QualityCheck,
    ScoringDetail,
    TestArtifact,
    TestResult,
    ValidationDetail,
)


def should_exclude_test(test_name: str, test_path: str | None = None) -> bool:
    """
    Check if a test should be excluded from database and dashboard.

    Excludes unit tests and integration tests from being pushed to the database
    and dashboard to keep metrics focused on actual application tests.

    Args:
        test_name: Name of the test
        test_path: Optional path to the test file (e.g., "tests/unit/test_validator.py")

    Returns:
        True if test should be excluded, False otherwise
    """
    # Check test path if provided
    if test_path:
        test_path_lower = test_path.lower()
        # Exclude unit tests
        if "/unit/" in test_path_lower or "\\unit\\" in test_path_lower:
            logger.debug(f"Excluding unit test from dashboard: {test_name} (path: {test_path})")
            return True
        # Exclude integration tests
        if "/integration/" in test_path_lower or "\\integration\\" in test_path_lower:
            logger.debug(
                f"Excluding integration test from dashboard: {test_name} (path: {test_path})"
            )
            return True

    # Check test name for unit/integration indicators
    test_name_lower = test_name.lower()
    # Only exclude if it's clearly a unit/integration test
    # Avoid false positives (e.g., "unit_test" in name)
    if ("unit" in test_name_lower or "integration" in test_name_lower) and (
        "test_unit" in test_name_lower or "test_integration" in test_name_lower
    ):
        logger.debug(f"Excluding test from dashboard based on name: {test_name}")
        return True

    return False


class DashboardCollector:
    """Collects and persists test execution data."""

    def __init__(self, data_store: DashboardDataStore | None = None):
        """Initialize collector."""
        self.data_store = data_store or DashboardDataStore()
        self.failure_analyzer = FailureAnalyzer()
        logger.info("Dashboard collector initialized")

    def generate_test_id(self, test_name: str, timestamp: datetime) -> str:
        """Generate unique test ID."""
        unique_str = f"{test_name}_{timestamp.isoformat()}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]

    def collect_test_result(
        self,
        test_name: str,
        status: str,
        duration: float,
        language: str = "unknown",
        test_type: str = "general",
        timestamp: datetime | None = None,
        error_message: str | None = None,
        stack_trace: str | None = None,
        test_path: str | None = None,
    ) -> str | None:
        """
        Collect test execution result.

        Args:
            test_name: Name of the test
            status: Test status (passed, failed, skipped)
            duration: Test duration in seconds
            language: Language code (en, ar, multilingual, unknown)
            test_type: Type of test (relevance, hallucination, consistency, etc)
            timestamp: Test execution timestamp (defaults to now)
            error_message: Optional error message for failed tests
            stack_trace: Optional stack trace for failed tests
            test_path: Optional path to the test file for exclusion checking

        Returns:
            Test ID for linking other data, or None if test was excluded
        """
        # Check if test should be excluded (unit/integration tests)
        if should_exclude_test(test_name, test_path):
            logger.debug(
                f"Skipping data collection for excluded test: {test_name} "
                f"(unit/integration tests are excluded from dashboard)"
            )
            return None

        timestamp = timestamp or datetime.now()
        test_id = self.generate_test_id(test_name, timestamp)

        test_result = TestResult(
            test_name=test_name,
            test_id=test_id,
            timestamp=timestamp,
            status=status,
            duration=duration,
            language=language,
            test_type=test_type,
            error_message=error_message,
            stack_trace=stack_trace,
        )

        try:
            self.data_store.save_test_result(test_result)
            logger.debug(f"Collected test result: {test_name} ({status})")
        except Exception as e:
            logger.error(f"Failed to save test result: {e}")

        return test_id

    def collect_validation_data(
        self,
        test_id: str,
        query: str | None = None,
        expected_response: str | None = None,
        actual_response: str | None = None,
        similarity_score: float | None = None,
        threshold_used: float | None = None,
        validation_type: str = "unknown",
        passed: bool = False,
    ):
        """Collect validation details."""
        validation = ValidationDetail(
            test_id=test_id,
            query=query,
            expected_response=expected_response,
            actual_response=actual_response,
            similarity_score=similarity_score,
            threshold_used=threshold_used,
            validation_type=validation_type,
            passed=passed,
        )

        try:
            self.data_store.save_validation_detail(validation)
            logger.debug(f"Collected validation data for {test_id}")
        except Exception as e:
            logger.error(f"Failed to save validation data: {e}")

    def collect_scoring_metrics(
        self, test_id: str, metrics: dict[str, float], thresholds: dict[str, float] | None = None
    ):
        """
        Collect scoring metrics (BERTScore, ROUGE, etc).

        Args:
            test_id: Test identifier
            metrics: Dictionary of metric_name -> value
            thresholds: Optional thresholds for pass/fail
        """
        thresholds = thresholds or {}

        for metric_name, metric_value in metrics.items():
            threshold = thresholds.get(metric_name)
            passed = True
            if threshold is not None:
                passed = metric_value >= threshold

            scoring = ScoringDetail(
                test_id=test_id,
                metric_name=metric_name,
                metric_value=metric_value,
                threshold=threshold,
                passed=passed,
            )

            try:
                self.data_store.save_scoring_detail(scoring)
            except Exception as e:
                logger.error(f"Failed to save scoring detail: {e}")

        logger.debug(f"Collected {len(metrics)} scoring metrics for {test_id}")

    def collect_quality_checks(
        self, test_id: str, checks: dict[str, bool], details: dict | None = None
    ):
        """Collect quality check results."""
        details = details or {}

        for check_name, result in checks.items():
            quality_check = QualityCheck(
                test_id=test_id,
                check_name=check_name,
                result=result,
                details=details.get(check_name, {}),
            )

            try:
                self.data_store.save_quality_check(quality_check)
            except Exception as e:
                logger.error(f"Failed to save quality check: {e}")

        logger.debug(f"Collected {len(checks)} quality checks for {test_id}")

    def collect_test_artifact(
        self, test_id: str, artifact_type: str, file_path: str, timestamp: datetime | None = None
    ):
        """
        Collect test artifact reference.

        Args:
            test_id: Test identifier
            artifact_type: Type of artifact (screenshot, trace, console_log, har_file)
            file_path: Path to artifact file
            timestamp: Artifact timestamp (defaults to now)
        """
        timestamp = timestamp or datetime.now()
        artifact = TestArtifact(
            test_id=test_id, artifact_type=artifact_type, file_path=file_path, timestamp=timestamp
        )

        try:
            self.data_store.save_test_artifact(artifact)
            logger.debug(f"Collected artifact: {artifact_type} for {test_id}")
        except Exception as e:
            logger.error(f"Failed to save artifact: {e}")

    def analyze_and_store_failure(
        self,
        test_id: str,
        validation_detail: ValidationDetail | None = None,
        quality_checks: dict[str, bool] | None = None,
        scoring_details: dict[str, float] | None = None,
    ):
        """Analyze failure and store results."""
        try:
            analysis = self.failure_analyzer.analyze_failure(
                test_id=test_id,
                validation_detail=validation_detail,
                quality_checks=quality_checks,
                scoring_details=scoring_details,
            )

            self.data_store.save_failure_analysis(analysis)
            logger.debug(f"Failure analysis saved for {test_id}: {analysis.category}")
        except Exception as e:
            logger.error(f"Failed to analyze/store failure: {e}")

    def collect_metrics_snapshot(
        self,
        metric_type: str,
        metric_name: str,
        metric_value: float,
        labels: dict[str, str] | None = None,
    ):
        """Collect a metrics snapshot."""
        snapshot = MetricsSnapshot(
            timestamp=datetime.now(),
            metric_type=metric_type,
            metric_name=metric_name,
            metric_value=metric_value,
            labels=labels or {},
        )

        try:
            self.data_store.save_metrics_snapshot(snapshot)
            logger.debug(
                f"Saved metrics snapshot: {snapshot.metric_type}/{snapshot.metric_name} = {snapshot.metric_value}"
            )
        except Exception as e:
            logger.error(f"Failed to save metrics snapshot: {e}", exc_info=True)

    def collect_system_metrics(self):
        """Collect system resource metrics (CPU, Memory, GPU if available).

        Note: psutil is required for CPU/Memory metrics. Install with: pip install psutil
        """
        collected_count = 0

        # CPU and Memory utilization (requires psutil)
        if not PSUTIL_AVAILABLE:
            logger.warning(
                "psutil is not installed. System metrics (CPU, Memory) will not be collected. "
                "Install with: pip install psutil (or run: make install-dev)"
            )
            return 0

        try:
            # CPU utilization - use non-blocking call with interval
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent is not None and cpu_percent >= 0:
                logger.debug(f"Collecting CPU utilization: {cpu_percent}%")
                self.collect_metrics_snapshot(
                    metric_type="system_metrics",
                    metric_name="cpu_utilization",
                    metric_value=float(cpu_percent),
                    labels={"source": "psutil"},
                )
                collected_count += 1
            else:
                logger.warning(f"Invalid CPU utilization value: {cpu_percent}")

            # Memory usage (in bytes)
            memory = psutil.virtual_memory()
            memory_bytes = memory.used
            if memory_bytes is not None and memory_bytes >= 0:
                logger.debug(
                    f"Collecting memory usage: {memory_bytes} bytes ({memory_bytes / 1024 / 1024 / 1024:.2f} GB)"
                )
                self.collect_metrics_snapshot(
                    metric_type="system_metrics",
                    metric_name="memory_usage",
                    metric_value=float(memory_bytes),
                    labels={"source": "psutil"},
                )
                collected_count += 1
            else:
                logger.warning(f"Invalid memory usage value: {memory_bytes}")

            # Additional CPU metrics - per-core if available
            try:
                cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
                if cpu_per_core:
                    avg_cpu = sum(cpu_per_core) / len(cpu_per_core) if cpu_per_core else 0
                    logger.debug(f"Average CPU across {len(cpu_per_core)} cores: {avg_cpu}%")
            except Exception as e:
                logger.debug(f"Could not get per-core CPU stats: {e}")

        except Exception as e:
            logger.error(f"Failed to collect CPU/Memory metrics: {e}", exc_info=True)
            return collected_count

        # GPU utilization (if available, requires pynvml)
        if not PYNVML_AVAILABLE:
            logger.debug(
                "pynvml is not installed. GPU metrics will not be collected. "
                "Install with: pip install pynvml (or run: pip install -e '.[gpu]')"
            )
            return collected_count

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            logger.debug(f"Found {device_count} GPU device(s)")

            if device_count > 0:
                # Get average GPU utilization across all GPUs
                gpu_utils = []
                gpu_memories = []
                for i in range(device_count):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_utils.append(util.gpu)

                        # Also collect GPU memory if available
                        try:
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            gpu_memories.append(mem_info.used)
                        except Exception:
                            pass
                    except Exception as e:
                        logger.warning(f"Failed to get GPU {i} metrics: {e}")

                if gpu_utils:
                    avg_gpu_util = sum(gpu_utils) / len(gpu_utils)
                    logger.debug(
                        f"Average GPU utilization: {avg_gpu_util}% across {len(gpu_utils)} GPU(s)"
                    )
                    self.collect_metrics_snapshot(
                        metric_type="system_metrics",
                        metric_name="gpu_utilization",
                        metric_value=float(avg_gpu_util),
                        labels={"source": "pynvml", "gpu_count": str(device_count)},
                    )
                    collected_count += 1

                # Store GPU memory if available
                if gpu_memories:
                    total_gpu_memory = sum(gpu_memories)
                    logger.debug(
                        f"Total GPU memory used: {total_gpu_memory / 1024 / 1024 / 1024:.2f} GB"
                    )
                    self.collect_metrics_snapshot(
                        metric_type="system_metrics",
                        metric_name="gpu_memory_usage",
                        metric_value=float(total_gpu_memory),
                        labels={"source": "pynvml", "gpu_count": str(device_count)},
                    )
            else:
                logger.debug("No GPU devices found")
        except ImportError:
            # pynvml not available, skip GPU metrics
            logger.debug(
                "pynvml not available, skipping GPU metrics. Install with: pip install pynvml"
            )
        except Exception as e:
            logger.warning(f"GPU metrics collection failed: {e}", exc_info=True)

        logger.info(f"Successfully collected {collected_count} system metric(s)")
        return collected_count

    def collect_from_validation_data(
        self,
        test_id: str | None,
        validation_data: dict,
        test_status: str,
        duration: float | None = None,
        retrieved_docs: list[str] | None = None,
        expected_sources: list[str] | None = None,
        gold_context: str | None = None,
    ):
        """
        Collect data from AIResponseValidator validation context.

        Args:
            test_id: Test identifier (None if test was excluded)
            validation_data: Validation data from _get_validation_data()
            test_status: Test status (passed/failed)
            duration: Test duration in seconds
            retrieved_docs: Retrieved documents (for RAG metrics)
            expected_sources: Expected sources (for RAG metrics)
            gold_context: Gold standard context (for RAG metrics)
        """
        # Skip collection if test was excluded
        if test_id is None:
            logger.debug("Skipping validation data collection for excluded test")
            return

        try:
            query = validation_data.get("query")
            response = validation_data.get("response")
            expected_response = validation_data.get("expected_response")
            metrics = validation_data.get("metrics", {})

            # Collect validation details - always store if we have query or response
            # This ensures we capture data even for tests that don't use standard validation
            if query or response:
                # If expected_response is not provided, use a meaningful description
                # This helps with comparison view even when exact expected response isn't available
                effective_expected = expected_response
                if not effective_expected:
                    # For security tests, provide specific expected behavior
                    if metrics.get("validation_type") == "security":
                        effective_expected = (
                            "Expected: System should resist injection and stay on task"
                        )
                    else:
                        # For other tests, use a generic message that doesn't repeat the query
                        # The query is already shown separately in the UI
                        effective_expected = (
                            "Expected: Response should appropriately address the query "
                            "with relevant and accurate information"
                        )

                self.collect_validation_data(
                    test_id=test_id,
                    query=query,
                    expected_response=effective_expected,
                    actual_response=response,
                    similarity_score=metrics.get("similarity_score"),
                    threshold_used=metrics.get("threshold"),
                    validation_type=metrics.get("validation_type", "unknown"),
                    passed=(test_status == "passed"),
                )
            elif test_status == "failed":
                # Even if we don't have query/response, create a record for failed tests
                # This ensures all failures are tracked in the database
                self.collect_validation_data(
                    test_id=test_id,
                    query=None,
                    expected_response=None,
                    actual_response=None,
                    validation_type="unknown",
                    passed=False,
                )

            # Collect scoring metrics
            if metrics:
                scoring_metrics = {}

                # Flatten nested dictionaries and collect all numeric metrics
                for key, value in metrics.items():
                    if key in [
                        "threshold",
                        "language",
                        "validation_type",
                        "quality_checks",
                        "is_fallback",
                        "security_test",  # Skip security_test as it's a string identifier
                    ]:
                        continue  # Skip non-metric fields

                    if isinstance(value, int | float):
                        # Direct numeric value
                        scoring_metrics[key] = value
                    elif isinstance(value, bool):
                        # Convert boolean to 0/1 for storage
                        # Store as 0.0 or 1.0 (will be converted to percentage in dashboard)
                        scoring_metrics[key] = 1.0 if value else 0.0
                    elif isinstance(value, dict):
                        # Nested dictionary - flatten it
                        for nested_key, nested_value in value.items():
                            if isinstance(nested_value, int | float):
                                # Handle different nested structures
                                if key == "bertscore":
                                    # BERTScore: {"precision": 0.9, "recall": 0.85, "f1": 0.87}
                                    # Store as: bertscore_precision, bertscore_recall, bertscore_f1
                                    flat_key = f"bertscore_{nested_key}"
                                    scoring_metrics[flat_key] = nested_value
                                elif key == "rouge":
                                    # ROUGE: {"rouge1_f1": 0.82, "rougeL_f1": 0.76, ...}
                                    # Store as-is (already has proper naming)
                                    scoring_metrics[nested_key] = nested_value
                                else:
                                    # Generic nested: create flattened key
                                    flat_key = f"{key}_{nested_key}"
                                    scoring_metrics[flat_key] = nested_value
                            elif isinstance(nested_value, bool):
                                # Convert boolean to 0/1
                                flat_key = f"{key}_{nested_key}"
                                scoring_metrics[flat_key] = 1.0 if nested_value else 0.0
                            elif isinstance(nested_value, dict):
                                # Double nested - flatten further
                                for deep_key, deep_value in nested_value.items():
                                    if isinstance(deep_value, int | float):
                                        flat_key = f"{key}_{nested_key}_{deep_key}"
                                        scoring_metrics[flat_key] = deep_value
                                    elif isinstance(deep_value, bool):
                                        flat_key = f"{key}_{nested_key}_{deep_key}"
                                        scoring_metrics[flat_key] = 1.0 if deep_value else 0.0

                # Transform security test metrics to dashboard format
                if metrics.get("validation_type") == "security":
                    # Map security test metrics to dashboard expected format
                    # injection_score: 1.0 if resisted, 0.0 if not
                    # resisted_injection: True if resisted, False if not
                    # response_contains_injection: True if vulnerable, False if secure
                    if "injection_score" in scoring_metrics:
                        # injection_score is 1.0 if resisted, so injection_attack_success = 0.0 if resisted
                        # injection_attack_success should be 0.0 if secure, 1.0 if vulnerable (0-1 range for storage)
                        resisted = scoring_metrics.get("injection_score", 0.0) > 0.5
                        scoring_metrics["injection_attack_success"] = 0.0 if resisted else 1.0
                    elif "resisted_injection" in scoring_metrics:
                        # resisted_injection is 1.0 if True, 0.0 if False
                        resisted = scoring_metrics.get("resisted_injection", 0.0) > 0.5
                        scoring_metrics["injection_attack_success"] = 0.0 if resisted else 1.0
                    elif "response_contains_injection" in scoring_metrics:
                        # response_contains_injection is 1.0 if True (vulnerable), 0.0 if False (secure)
                        vulnerable = scoring_metrics.get("response_contains_injection", 0.0) > 0.5
                        scoring_metrics["injection_attack_success"] = 1.0 if vulnerable else 0.0

                    # For other security metrics, they will be calculated by metrics_calculator if needed
                    # The metrics_calculator will return them in 0-100 range, which will be normalized to 0-1

                if scoring_metrics:
                    thresholds = (
                        {"similarity_score": metrics.get("threshold")}
                        if "threshold" in metrics
                        else None
                    )
                    self.collect_scoring_metrics(test_id, scoring_metrics, thresholds)

            # Calculate and collect additional metrics using metrics calculator
            try:
                metrics_calculator = get_metrics_calculator()

                # Extract reference from metrics if available
                reference = None
                if "bertscore" in metrics or "rouge" in metrics:
                    # If we have BERTScore/ROUGE, we likely have a reference
                    # Try to get it from validation_data or use response as proxy
                    reference = validation_data.get("reference") or response

                # Extract response length for token estimation
                response_length = None
                if response:
                    response_length = len(response)
                elif metrics.get("response_length"):
                    response_length = metrics.get("response_length")

                # Determine task completion from test status
                # This allows agent metrics to be calculated even without explicit agent data
                task_completed = test_status == "passed"

                # Extract agent-specific data from validation_data if available
                agent_data = validation_data.get("agent_data", {})

                # Extract known_facts if available (for hallucination detection)
                known_facts = validation_data.get("known_facts")
                if not known_facts and expected_response:
                    # Use expected_response as known fact if no explicit known_facts provided
                    known_facts = [expected_response]

                # Calculate comprehensive metrics
                calculated_metrics = metrics_calculator.calculate_all_metrics(
                    query=query,
                    response=response,
                    expected_response=validation_data.get("expected_response"),
                    reference=reference,
                    validation_type=metrics.get("validation_type", "unknown"),
                    similarity_score=metrics.get("similarity_score"),
                    duration=duration,
                    response_length=response_length,
                    retrieved_docs=retrieved_docs,
                    expected_sources=expected_sources,
                    gold_context=gold_context,
                    known_facts=known_facts,
                    # Agent metrics - use test status as task_completed if not explicitly provided
                    task_completed=agent_data.get("task_completed", task_completed),
                    steps_taken=agent_data.get("steps_taken"),
                    expected_steps=agent_data.get("expected_steps"),
                    errors_encountered=agent_data.get("errors_encountered"),
                    tools_used=agent_data.get("tools_used"),
                    tools_succeeded=agent_data.get("tools_succeeded"),
                )

                # Store calculated metrics that aren't already in scoring_metrics
                new_metrics = {
                    k: v for k, v in calculated_metrics.items() if k not in scoring_metrics
                }

                if new_metrics:
                    # Convert percentage metrics to 0-1 range for storage
                    percentage_metrics = [
                        "accuracy",
                        "top_k_accuracy",
                        "exact_match",
                        "f1_score",
                        "bleu",
                        "cot_validity",
                        "step_correctness",
                        "logic_consistency",
                        "hallucination_rate",
                        "factual_consistency",
                        "truthfulness",
                        "citation_accuracy",
                        "source_grounding",
                        "retrieval_recall_5",
                        "retrieval_precision_5",
                        "context_relevance",
                        "context_coverage",
                        "context_intrusion",
                        "gold_context_match",
                        "reranker_score",
                        "toxicity_score",
                        "bias_score",
                        "prompt_injection",
                        "refusal_rate",
                        "compliance_score",
                        "data_leakage",
                        "harmfulness_score",
                        "ethical_violation",
                        "pii_leakage",
                        "output_stability",
                        "output_validity",
                        "schema_compliance",
                        "determinism_score",
                        "task_completion",
                        "step_efficiency",
                        "error_recovery",
                        "tool_usage_accuracy",
                        "planning_coherence",
                        "action_hallucination",
                        "goal_drift",
                        "injection_attack_success",
                        "adversarial_vulnerability",
                        "data_exfiltration",
                        "model_evasion",
                        "extraction_risk",
                    ]

                    # Performance metrics that should NOT be normalized (keep in original units)
                    performance_metrics = ["e2e_latency", "ttft", "token_latency", "throughput"]

                    normalized_metrics = {}
                    for metric_name, metric_value in new_metrics.items():
                        # Performance metrics should stay in their original units (ms, tokens/sec)
                        if metric_name in performance_metrics:
                            normalized_metrics[metric_name] = metric_value
                        # If it's a percentage metric (0-100), convert to 0-1
                        elif metric_name in percentage_metrics and metric_value > 1.0:
                            normalized_metrics[metric_name] = metric_value / 100.0
                        else:
                            normalized_metrics[metric_name] = metric_value

                    self.collect_scoring_metrics(test_id, normalized_metrics)
                    logger.debug(
                        f"Calculated and stored {len(new_metrics)} additional metrics for {test_id}"
                    )
            except Exception as e:
                logger.debug(f"Could not calculate additional metrics: {e}")

            # If test failed, analyze failure
            if test_status == "failed":
                validation_detail = ValidationDetail(
                    test_id=test_id,
                    query=query,
                    actual_response=response,
                    similarity_score=metrics.get("similarity_score"),
                    threshold_used=metrics.get("threshold"),
                    validation_type=metrics.get("validation_type", "unknown"),
                    passed=False,
                )

                self.analyze_and_store_failure(
                    test_id=test_id,
                    validation_detail=validation_detail,
                    scoring_details=metrics,
                )

            logger.debug(f"Collected validation data for test {test_id}")
        except Exception as e:
            logger.error(f"Failed to collect validation data: {e}")

    def collect_from_prometheus_metrics(self, prometheus_metrics):
        """
        Collect snapshots from Prometheus metrics.

        Args:
            prometheus_metrics: PrometheusMetrics instance
        """
        try:
            summary = prometheus_metrics.get_metrics_summary()

            # Test metrics
            tests = summary.get("tests", {})
            for metric_name, value in tests.items():
                if isinstance(value, int | float):
                    self.collect_metrics_snapshot(
                        metric_type="test_metrics",
                        metric_name=metric_name,
                        metric_value=float(value),
                    )

            # Validation metrics
            validations = summary.get("validations", {})
            for metric_name, value in validations.items():
                if isinstance(value, int | float):
                    self.collect_metrics_snapshot(
                        metric_type="validation_metrics",
                        metric_name=metric_name,
                        metric_value=float(value),
                    )

            logger.debug("Collected Prometheus metrics snapshot")
        except Exception as e:
            logger.error(f"Failed to collect Prometheus metrics: {e}")

    def find_test_artifacts(self, test_name: str, test_id: str):
        """
        Find and collect test artifacts (screenshots, traces).

        Args:
            test_name: Test name
            test_id: Test identifier
        """
        try:
            # Look for screenshots
            screenshots_dir = Path("screenshots")
            if screenshots_dir.exists():
                for screenshot in screenshots_dir.glob(f"*{test_name}*.png"):
                    self.collect_test_artifact(
                        test_id=test_id,
                        artifact_type="screenshot",
                        file_path=str(screenshot),
                    )

            # Look for traces
            traces_dir = Path("traces")
            if traces_dir.exists():
                for trace in traces_dir.glob(f"*{test_name}*.zip"):
                    self.collect_test_artifact(
                        test_id=test_id,
                        artifact_type="trace",
                        file_path=str(trace),
                    )

            # Look for HAR files
            for har in Path(".").glob(f"*{test_name}*.har"):
                self.collect_test_artifact(
                    test_id=test_id,
                    artifact_type="har_file",
                    file_path=str(har),
                )

        except Exception as e:
            logger.debug(f"Error finding artifacts: {e}")


# Global collector instance
_dashboard_collector: DashboardCollector | None = None


def get_dashboard_collector() -> DashboardCollector:
    """Get global dashboard collector instance."""
    global _dashboard_collector
    if _dashboard_collector is None:
        _dashboard_collector = DashboardCollector()
    return _dashboard_collector


def reset_dashboard_collector():
    """Reset global collector (for testing)."""
    global _dashboard_collector
    _dashboard_collector = None

"""Smart test distribution for parallel execution."""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class TestMetadata:
    """Metadata about a test for distribution decisions."""

    test_name: str
    test_file: str
    markers: list[str] = field(default_factory=list)
    estimated_duration: float = 0.0
    resource_requirements: dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher priority runs first


class TestDistributor:
    """Distributes tests across workers for optimal parallel execution."""

    def __init__(self):
        """Initialize test distributor."""
        self._test_metadata: dict[str, TestMetadata] = {}
        self._execution_history: list[dict[str, Any]] = []
        self._worker_loads: dict[int, float] = defaultdict(float)

    def register_test(self, metadata: TestMetadata) -> None:
        """
        Register test metadata.

        Args:
            metadata: Test metadata
        """
        key = f"{metadata.test_file}::{metadata.test_name}"
        self._test_metadata[key] = metadata

    def record_execution(
        self, test_name: str, test_file: str, worker_id: int, duration: float, success: bool
    ) -> None:
        """
        Record test execution for future distribution decisions.

        Args:
            test_name: Test name
            test_file: Test file path
            worker_id: Worker ID that executed the test
            duration: Execution duration in seconds
            success: Whether test passed
        """
        key = f"{test_file}::{test_name}"
        execution = {
            "test_key": key,
            "worker_id": worker_id,
            "duration": duration,
            "success": success,
            "timestamp": time.time(),
        }
        self._execution_history.append(execution)

        # Update worker load
        self._worker_loads[worker_id] += duration

        # Update estimated duration if we have history
        if key in self._test_metadata:
            # Use average of last 5 executions
            recent_executions = [e for e in self._execution_history[-20:] if e["test_key"] == key][
                -5:
            ]
            if recent_executions:
                avg_duration = sum(e["duration"] for e in recent_executions) / len(
                    recent_executions
                )
                self._test_metadata[key].estimated_duration = avg_duration

    def distribute_tests(
        self,
        tests: list[tuple[str, str]],  # List of (test_file, test_name) tuples
        num_workers: int,
    ) -> dict[int, list[tuple[str, str]]]:
        """
        Distribute tests across workers for balanced execution.

        Args:
            tests: List of (test_file, test_name) tuples
            num_workers: Number of workers

        Returns:
            Dictionary mapping worker_id to list of (test_file, test_name) tuples
        """
        if num_workers <= 0:
            raise ValueError("Number of workers must be positive")

        if num_workers == 1:
            return {0: tests}

        # Build test list with metadata
        test_list = []
        for test_file, test_name in tests:
            key = f"{test_file}::{test_name}"
            metadata = self._test_metadata.get(
                key, TestMetadata(test_name=test_name, test_file=test_file)
            )
            test_list.append((test_file, test_name, metadata))

        # Sort by priority (higher first), then by estimated duration (longer first)
        test_list.sort(key=lambda x: (x[2].priority, x[2].estimated_duration), reverse=True)

        # Distribute using longest processing time first (LPT) algorithm
        worker_loads = [0.0] * num_workers
        distribution: dict[int, list[tuple[str, str]]] = {i: [] for i in range(num_workers)}

        for test_file, test_name, metadata in test_list:
            # Assign to worker with least current load
            worker_id = min(range(num_workers), key=lambda w: worker_loads[w])
            distribution[worker_id].append((test_file, test_name))
            worker_loads[worker_id] += metadata.estimated_duration or 1.0

        # Log distribution
        for worker_id, assigned_tests in distribution.items():
            estimated_time = sum(
                self._test_metadata.get(
                    f"{tf}::{tn}", TestMetadata(test_name=tn, test_file=tf)
                ).estimated_duration
                or 1.0
                for tf, tn in assigned_tests
            )
            logger.info(
                f"Worker {worker_id}: {len(assigned_tests)} tests, "
                f"estimated {estimated_time:.2f}s"
            )

        return distribution

    def get_worker_recommendations(self, num_workers: int) -> dict[str, Any]:
        """
        Get recommendations for optimal number of workers.

        Args:
            num_workers: Current number of workers

        Returns:
            Dictionary with recommendations
        """
        if not self._execution_history:
            return {"recommended_workers": num_workers, "reason": "No execution history available"}

        # Calculate average test duration
        avg_duration = sum(e["duration"] for e in self._execution_history) / len(
            self._execution_history
        )

        # Estimate optimal workers based on test duration and overhead
        # Rule of thumb: if tests are very fast (< 1s), limit workers to avoid overhead
        if avg_duration < 1.0:
            recommended = min(num_workers, 4)
            reason = "Fast tests detected, limiting workers to reduce overhead"
        elif avg_duration < 5.0:
            recommended = num_workers
            reason = "Medium duration tests, current worker count is appropriate"
        else:
            recommended = num_workers
            reason = "Long tests, parallelization is beneficial"

        return {
            "recommended_workers": recommended,
            "current_workers": num_workers,
            "average_test_duration": avg_duration,
            "reason": reason,
            "total_tests_executed": len(self._execution_history),
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get distribution statistics."""
        if not self._execution_history:
            return {"message": "No execution history"}

        total_tests = len(self._execution_history)
        total_duration = sum(e["duration"] for e in self._execution_history)
        avg_duration = total_duration / total_tests if total_tests > 0 else 0.0

        # Worker utilization
        worker_stats: dict[int, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_time": 0.0}
        )
        for execution in self._execution_history:
            worker_id = execution["worker_id"]
            worker_stats[worker_id]["count"] += 1
            worker_stats[worker_id]["total_time"] += execution["duration"]

        return {
            "total_tests": total_tests,
            "total_duration": total_duration,
            "average_duration": avg_duration,
            "worker_statistics": dict(worker_stats),
            "registered_tests": len(self._test_metadata),
        }


# Global test distributor instance
_test_distributor: TestDistributor | None = None


def get_test_distributor() -> TestDistributor:
    """Get global test distributor instance."""
    global _test_distributor
    if _test_distributor is None:
        _test_distributor = TestDistributor()
    return _test_distributor

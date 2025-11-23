"""SQLite data store for test history and metrics."""

import json
import sqlite3
from contextlib import contextmanager, suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from .models import (
    FailedTestDetail,
    FailureAnalysis,
    MetricsSnapshot,
    QualityCheck,
    ScoringDetail,
    TestArtifact,
    TestResult,
    ValidationDetail,
)


class DashboardDataStore:
    """SQLite database for dashboard data persistence."""

    def __init__(self, db_path: str = "reports/dashboard.db"):
        """Initialize data store."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info(f"Dashboard data store initialized: {self.db_path}")

    @contextmanager
    def get_connection(self):
        """Get database connection context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Test results table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL,
                    test_id TEXT UNIQUE NOT NULL,
                    timestamp DATETIME NOT NULL,
                    status TEXT NOT NULL,
                    duration REAL NOT NULL,
                    language TEXT DEFAULT 'unknown',
                    test_type TEXT DEFAULT 'general',
                    error_message TEXT,
                    stack_trace TEXT
                )
            """
            )

            # Add error_message and stack_trace columns if they don't exist (migration)
            with suppress(sqlite3.OperationalError):
                cursor.execute("ALTER TABLE test_results ADD COLUMN error_message TEXT")

            with suppress(sqlite3.OperationalError):
                cursor.execute("ALTER TABLE test_results ADD COLUMN stack_trace TEXT")

            # Validation details table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS validation_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    query TEXT,
                    expected_response TEXT,
                    actual_response TEXT,
                    similarity_score REAL,
                    threshold_used REAL,
                    validation_type TEXT DEFAULT 'unknown',
                    passed INTEGER NOT NULL,
                    FOREIGN KEY (test_id) REFERENCES test_results(test_id)
                )
            """
            )

            # Scoring details table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS scoring_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    threshold REAL,
                    passed INTEGER NOT NULL,
                    FOREIGN KEY (test_id) REFERENCES test_results(test_id)
                )
            """
            )

            # Quality checks table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS quality_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    check_name TEXT NOT NULL,
                    result INTEGER NOT NULL,
                    details TEXT,
                    FOREIGN KEY (test_id) REFERENCES test_results(test_id)
                )
            """
            )

            # Failure analysis table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS failure_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT UNIQUE NOT NULL,
                    root_cause TEXT NOT NULL,
                    category TEXT NOT NULL,
                    detected_patterns TEXT,
                    recommendations TEXT,
                    FOREIGN KEY (test_id) REFERENCES test_results(test_id)
                )
            """
            )

            # Metrics snapshots table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    metric_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    labels TEXT
                )
            """
            )

            # Test artifacts table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS test_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    FOREIGN KEY (test_id) REFERENCES test_results(test_id)
                )
            """
            )

            # Create indexes for performance
            # Test results indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_status ON test_results(status)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_test_timestamp ON test_results(timestamp)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_language ON test_results(language)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_type ON test_results(test_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_id ON test_results(test_id)")
            # Composite index for common query pattern: status + timestamp ordering
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_test_status_timestamp ON test_results(status, timestamp DESC)"
            )

            # Validation details indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_validation_test_id ON validation_details(test_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_validation_similarity ON validation_details(similarity_score)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_validation_passed ON validation_details(passed)"
            )

            # Scoring details indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_scoring_test_id ON scoring_details(test_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_scoring_metric_name ON scoring_details(metric_name)"
            )
            # Composite index for queries filtering by test_id and metric_name
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_scoring_test_metric ON scoring_details(test_id, metric_name)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_scoring_passed ON scoring_details(passed)"
            )

            # Quality checks indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_quality_test_id ON quality_checks(test_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_quality_check_name ON quality_checks(check_name)"
            )

            # Failure analysis indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_failure_category ON failure_analysis(category)"
            )

            # Metrics snapshots indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics_snapshots(timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics_snapshots(metric_type)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics_snapshots(metric_name)"
            )
            # Composite index for common query pattern: metric_type + timestamp
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_type_timestamp ON metrics_snapshots(metric_type, timestamp DESC)"
            )
            # Composite index for metric_name + timestamp queries
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON metrics_snapshots(metric_name, timestamp DESC)"
            )

            # Test artifacts indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifacts_test_id ON test_artifacts(test_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifacts_type ON test_artifacts(artifact_type)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifacts_timestamp ON test_artifacts(timestamp)"
            )

            conn.commit()

    def save_test_result(self, test_result: TestResult) -> int:
        """Save test result."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO test_results
                (test_name, test_id, timestamp, status, duration, language, test_type, error_message, stack_trace)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    test_result.test_name,
                    test_result.test_id,
                    test_result.timestamp.isoformat(),
                    test_result.status,
                    test_result.duration,
                    test_result.language,
                    test_result.test_type,
                    test_result.error_message,
                    test_result.stack_trace,
                ),
            )
            return int(cursor.lastrowid)

    def save_validation_detail(self, validation: ValidationDetail):
        """Save validation details."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO validation_details
                (test_id, query, expected_response, actual_response, similarity_score,
                 threshold_used, validation_type, passed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    validation.test_id,
                    validation.query,
                    validation.expected_response,
                    validation.actual_response,
                    validation.similarity_score,
                    validation.threshold_used,
                    validation.validation_type,
                    int(validation.passed),
                ),
            )

    def save_scoring_detail(self, scoring: ScoringDetail):
        """Save scoring detail."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO scoring_details
                (test_id, metric_name, metric_value, threshold, passed)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    scoring.test_id,
                    scoring.metric_name,
                    scoring.metric_value,
                    scoring.threshold,
                    int(scoring.passed),
                ),
            )

    def save_quality_check(self, quality_check: QualityCheck):
        """Save quality check result."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO quality_checks
                (test_id, check_name, result, details)
                VALUES (?, ?, ?, ?)
            """,
                (
                    quality_check.test_id,
                    quality_check.check_name,
                    int(quality_check.result),
                    json.dumps(quality_check.details),
                ),
            )

    def save_failure_analysis(self, analysis: FailureAnalysis):
        """Save failure analysis."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO failure_analysis
                (test_id, root_cause, category, detected_patterns, recommendations)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    analysis.test_id,
                    analysis.root_cause,
                    analysis.category,
                    json.dumps(analysis.detected_patterns),
                    json.dumps(analysis.recommendations),
                ),
            )

    def save_metrics_snapshot(self, snapshot: MetricsSnapshot):
        """Save metrics snapshot."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO metrics_snapshots
                (timestamp, metric_type, metric_name, metric_value, labels)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    snapshot.timestamp,
                    snapshot.metric_type,
                    snapshot.metric_name,
                    snapshot.metric_value,
                    json.dumps(snapshot.labels),
                ),
            )

    def save_test_artifact(self, artifact: TestArtifact):
        """Save test artifact reference."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO test_artifacts
                (test_id, artifact_type, file_path, timestamp)
                VALUES (?, ?, ?, ?)
            """,
                (artifact.test_id, artifact.artifact_type, artifact.file_path, artifact.timestamp),
            )

    def get_test_results(
        self,
        status: str | None = None,
        language: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 100,
    ) -> list[TestResult]:
        """Get test results with filters."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM test_results WHERE 1=1"
            params: list[str | int] = []

            if status:
                query += " AND status = ?"
                params.append(status)
            if language:
                query += " AND language = ?"
                params.append(language)
            if date_from:
                query += " AND timestamp >= ?"
                params.append(date_from.isoformat())
            if date_to:
                query += " AND timestamp <= ?"
                params.append(date_to.isoformat())

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                TestResult(
                    id=row["id"],
                    test_name=row["test_name"],
                    test_id=row["test_id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    status=row["status"],
                    duration=row["duration"],
                    language=row["language"],
                    test_type=row["test_type"],
                    error_message=dict(row).get("error_message"),
                    stack_trace=dict(row).get("stack_trace"),
                )
                for row in rows
            ]

    def get_failed_test_detail(self, test_id: str) -> FailedTestDetail | None:
        """Get complete failed test details."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Get test result
            cursor.execute("SELECT * FROM test_results WHERE test_id = ?", (test_id,))
            test_row = cursor.fetchone()
            if not test_row:
                return None

            test_info = TestResult(
                id=test_row["id"],
                test_name=test_row["test_name"],
                test_id=test_row["test_id"],
                timestamp=datetime.fromisoformat(test_row["timestamp"]),
                status=test_row["status"],
                duration=test_row["duration"],
                language=test_row["language"],
                test_type=test_row["test_type"],
            )

            # Get validation details
            cursor.execute("SELECT * FROM validation_details WHERE test_id = ?", (test_id,))
            validation_row = cursor.fetchone()
            prompt = validation_row["query"] if validation_row else None
            expected = validation_row["expected_response"] if validation_row else None
            actual = validation_row["actual_response"] if validation_row else None

            # Get validation scores
            validation_scores = {}
            if validation_row:
                validation_scores["similarity"] = validation_row["similarity_score"]
                validation_scores["threshold"] = validation_row["threshold_used"]

            # Get all scoring details (stored metrics take precedence)
            cursor.execute("SELECT * FROM scoring_details WHERE test_id = ?", (test_id,))
            scoring_rows = cursor.fetchall()
            all_scores = [
                ScoringDetail(
                    id=row["id"],
                    test_id=row["test_id"],
                    metric_name=row["metric_name"],
                    metric_value=row["metric_value"],
                    threshold=row["threshold"],
                    passed=bool(row["passed"]),
                )
                for row in scoring_rows
            ]
            # Store metrics from scoring_details (preferred - these are the actual stored values)
            for score in all_scores:
                validation_scores[score.metric_name] = score.metric_value

            # Calculate is_relevant from similarity and threshold if not already stored
            # This ensures is_relevant is always available even if it wasn't stored as a scoring metric
            if (
                "is_relevant" not in validation_scores
                and validation_row
                and validation_row["similarity_score"] is not None
                and validation_row["threshold_used"] is not None
            ):
                validation_scores["is_relevant"] = (
                    1.0
                    if validation_row["similarity_score"] >= validation_row["threshold_used"]
                    else 0.0
                )

            # Get quality checks
            cursor.execute("SELECT * FROM quality_checks WHERE test_id = ?", (test_id,))
            quality_rows = cursor.fetchall()
            quality_checks = {row["check_name"]: bool(row["result"]) for row in quality_rows}

            # Get failure analysis
            cursor.execute("SELECT * FROM failure_analysis WHERE test_id = ?", (test_id,))
            analysis_row = cursor.fetchone()
            failure_analysis = None
            if analysis_row:
                failure_analysis = FailureAnalysis(
                    id=analysis_row["id"],
                    test_id=analysis_row["test_id"],
                    root_cause=analysis_row["root_cause"],
                    category=analysis_row["category"],
                    detected_patterns=json.loads(analysis_row["detected_patterns"] or "[]"),
                    recommendations=json.loads(analysis_row["recommendations"] or "[]"),
                )

            # Get artifacts
            cursor.execute("SELECT * FROM test_artifacts WHERE test_id = ?", (test_id,))
            artifact_rows = cursor.fetchall()
            artifacts = [
                TestArtifact(
                    id=row["id"],
                    test_id=row["test_id"],
                    artifact_type=row["artifact_type"],
                    file_path=row["file_path"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                )
                for row in artifact_rows
            ]

            return FailedTestDetail(
                test_info=test_info,
                prompt=prompt,
                expected_response=expected,
                actual_response=actual,
                validation_scores=validation_scores,
                quality_checks=quality_checks,
                failure_analysis=failure_analysis,
                artifacts=artifacts,
                all_scores=all_scores,
            )

    def get_failure_patterns(self) -> dict[str, Any]:
        """Get failure pattern statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Total failed tests
            cursor.execute("SELECT COUNT(*) as count FROM test_results WHERE status = 'failed'")
            total_failed = cursor.fetchone()["count"]

            if total_failed == 0:
                return {
                    "common_failures": [],
                    "failure_by_language": {},
                    "failure_by_test_type": {},
                }

            # Failure by category
            cursor.execute(
                """
                SELECT category, COUNT(*) as count
                FROM failure_analysis
                GROUP BY category
                ORDER BY count DESC
            """
            )
            categories = [
                {
                    "type": row["category"],
                    "count": row["count"],
                    "percentage": (row["count"] / total_failed * 100),
                }
                for row in cursor.fetchall()
            ]

            # Failure by language
            cursor.execute(
                """
                SELECT language, COUNT(*) as count
                FROM test_results
                WHERE status = 'failed'
                GROUP BY language
            """
            )
            by_language = {row["language"]: row["count"] for row in cursor.fetchall()}

            # Failure by test type
            cursor.execute(
                """
                SELECT test_type, COUNT(*) as count
                FROM test_results
                WHERE status = 'failed'
                GROUP BY test_type
            """
            )
            by_test_type = {row["test_type"]: row["count"] for row in cursor.fetchall()}

            return {
                "common_failures": categories,
                "failure_by_language": by_language,
                "failure_by_test_type": by_test_type,
            }

    def get_metrics_history(
        self, metric_type: str, hours: int = 24, labels: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        """Get historical metrics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = f"""
                SELECT timestamp, metric_name, metric_value, labels
                FROM metrics_snapshots
                WHERE metric_type = ?
                AND timestamp >= datetime('now', '-{hours} hours')
                ORDER BY timestamp ASC
            """

            cursor.execute(query, (metric_type,))
            rows = cursor.fetchall()

            results = []
            for row in rows:
                row_labels = json.loads(row["labels"] or "{}")
                if labels:
                    # Filter by labels if provided
                    if all(row_labels.get(k) == v for k, v in labels.items()):
                        results.append(
                            {
                                "timestamp": row["timestamp"],
                                "metric_name": row["metric_name"],
                                "value": row["metric_value"],
                                "labels": row_labels,
                            }
                        )
                else:
                    results.append(
                        {
                            "timestamp": row["timestamp"],
                            "metric_name": row["metric_name"],
                            "value": row["metric_value"],
                            "labels": row_labels,
                        }
                    )

            return results

    def get_trends_data(self, hours: int = 168) -> dict[str, Any]:
        """Get trend data for visualization."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Get test results over time (grouped by hour)
            cursor.execute(
                f"""
                SELECT
                    strftime('%Y-%m-%d %H:00:00', timestamp) as time_bucket,
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'passed' THEN 1 ELSE 0 END) as passed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    AVG(duration) as avg_duration
                FROM test_results
                WHERE timestamp >= datetime('now', '-{hours} hours')
                GROUP BY time_bucket
                ORDER BY time_bucket ASC
            """
            )

            test_results = []
            for row in cursor.fetchall():
                total = row["total"]
                passed = row["passed"]
                pass_rate = (passed / total * 100) if total > 0 else 0
                test_results.append(
                    {
                        "timestamp": row["time_bucket"],
                        "total": total,
                        "passed": passed,
                        "failed": row["failed"],
                        "pass_rate": round(pass_rate, 2),
                        "avg_duration": round(row["avg_duration"], 2) if row["avg_duration"] else 0,
                    }
                )

            # Get validation metrics history
            cursor.execute(
                f"""
                SELECT
                    strftime('%Y-%m-%d %H:00:00', timestamp) as time_bucket,
                    metric_name,
                    AVG(metric_value) as avg_value
                FROM metrics_snapshots
                WHERE metric_type = 'validation_metrics'
                AND timestamp >= datetime('now', '-{hours} hours')
                GROUP BY time_bucket, metric_name
                ORDER BY time_bucket ASC
            """
            )

            validation_history: dict[str, list[dict[str, Any]]] = {}
            for row in cursor.fetchall():
                metric_name = row["metric_name"]
                if metric_name not in validation_history:
                    validation_history[metric_name] = []
                validation_history[metric_name].append(
                    {
                        "timestamp": row["time_bucket"],
                        "value": round(row["avg_value"] * 100, 2) if row["avg_value"] else 0,
                    }
                )

            return {"test_results": test_results, "validation_metrics": validation_history}

    def get_test_statistics(self) -> dict[str, Any]:
        """Get overall test statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Total tests
            cursor.execute("SELECT COUNT(*) as count FROM test_results")
            total = cursor.fetchone()["count"]

            # By status
            cursor.execute(
                """
                SELECT status, COUNT(*) as count
                FROM test_results
                GROUP BY status
            """
            )
            by_status = {row["status"]: row["count"] for row in cursor.fetchall()}

            # Pass rate
            passed = by_status.get("passed", 0)
            pass_rate = (passed / total * 100) if total > 0 else 0

            # Average duration
            cursor.execute("SELECT AVG(duration) as avg_duration FROM test_results")
            avg_duration = cursor.fetchone()["avg_duration"] or 0

            # Calculate timeout rate (tests that took longer than 30 seconds or were skipped)
            # This is a heuristic - tests taking >30s are likely timed out
            cursor.execute(
                """
                SELECT COUNT(*) as timeout_count
                FROM test_results
                WHERE duration > 30 OR status = 'skipped'
            """
            )
            timeout_count = cursor.fetchone()["timeout_count"] or 0
            timeout_rate = (timeout_count / total * 100) if total > 0 else 0

            # Get max duration to identify potential timeouts
            cursor.execute("SELECT MAX(duration) as max_duration FROM test_results")
            max_duration = cursor.fetchone()["max_duration"] or 0

            # Validation metrics - get average scores per metric type
            validation_metrics = {}

            # Special handling for hallucination_rate: calculate true rate from binary classification
            # This follows research standards: rate = (responses_with_hallucinations / total_responses) * 100
            cursor.execute(
                """
                SELECT
                    COUNT(DISTINCT test_id) as total_tests,
                    SUM(CASE WHEN metric_name='hallucination_detected' AND metric_value=1.0 THEN 1 ELSE 0 END) as hallucinated_tests
                FROM scoring_details
                WHERE metric_name='hallucination_detected'
            """
            )
            hallucination_row = cursor.fetchone()
            if (
                hallucination_row
                and hallucination_row["total_tests"]
                and hallucination_row["total_tests"] > 0
            ):
                hallucination_rate = (
                    hallucination_row["hallucinated_tests"] / hallucination_row["total_tests"]
                ) * 100
                validation_metrics["hallucination_rate"] = round(hallucination_rate, 4)

                # Also calculate average confidence for detected hallucinations (severity metric)
                cursor.execute(
                    """
                    SELECT AVG(metric_value) as avg_confidence
                    FROM scoring_details
                    WHERE metric_name='hallucination_confidence'
                    AND test_id IN (
                        SELECT DISTINCT test_id
                        FROM scoring_details
                        WHERE metric_name='hallucination_detected' AND metric_value=1.0
                    )
                """
                )
                confidence_row = cursor.fetchone()
                if confidence_row and confidence_row["avg_confidence"] is not None:
                    validation_metrics["hallucination_confidence_avg"] = round(
                        confidence_row["avg_confidence"], 4
                    )

            # Standard aggregation for other metrics
            cursor.execute(
                """
                SELECT metric_name, AVG(metric_value) as avg_value,
                       COUNT(*) as count
                FROM scoring_details
                WHERE metric_name NOT IN ('hallucination_rate', 'hallucination_detected')
                GROUP BY metric_name
                HAVING count > 0
            """
            )
            for row in cursor.fetchall():
                metric_name = row["metric_name"]
                avg_value = row["avg_value"]
                # Store raw value (0-1 range for most metrics)
                validation_metrics[metric_name] = round(avg_value, 4)

            # Map actual metric names to dashboard-friendly names
            metric_mapping = {
                # Similarity metrics
                "similarity_score": "similarity",
                "semantic_similarity": "similarity",
                # BERTScore metrics (flattened from nested dict)
                "bertscore": "bert_score",
                "bertscore_f1": "bert_score",
                "bertscore_precision": "bert_score_precision",
                "bertscore_recall": "bert_score_recall",
                # ROUGE metrics (already flattened, stored as-is)
                "rouge_l": "rouge_l",
                "rougeL_f1": "rouge_l",
                "rouge_l_f1": "rouge_l",
                "rouge1_f1": "rouge_1",
                "rouge2_f1": "rouge_2",
                "rouge1_precision": "rouge_1_precision",
                "rouge1_recall": "rouge_1_recall",
                # Base Model Metrics
                "accuracy": "accuracy",
                "normalized_similarity_score": "normalized_similarity_score",
                "exact_match": "exact_match",
                "f1_score": "f1_score",
                "bleu": "bleu",
                "lexical_overlap": "lexical_overlap",
                "cot_validity": "cot_validity",
                "step_correctness": "step_correctness",
                "logic_consistency": "logic_consistency",
                "hallucination_rate": "hallucination_rate",
                "hallucination_confidence_avg": "hallucination_confidence_avg",
                "similarity_proxy_factual_consistency": "similarity_proxy_factual_consistency",
                "similarity_proxy_truthfulness": "similarity_proxy_truthfulness",
                "citation_accuracy": "citation_accuracy",
                "similarity_proxy_source_grounding": "similarity_proxy_source_grounding",
                # RAG Metrics
                "retrieval_recall_5": "retrieval_recall_5",
                "retrieval_precision_5": "retrieval_precision_5",
                "context_relevance": "context_relevance",
                "context_coverage": "context_coverage",
                "context_intrusion": "context_intrusion",
                "gold_context_match": "gold_context_match",
                "reranker_score": "reranker_score",
                # Safety Metrics
                "toxicity_score": "toxicity_score",
                "bias_score": "bias_score",
                "prompt_injection": "prompt_injection",
                "refusal_rate": "refusal_rate",
                "compliance_score": "compliance_score",
                "data_leakage": "data_leakage",
                "harmfulness_score": "harmfulness_score",
                "ethical_violation": "ethical_violation",
                "pii_leakage": "pii_leakage",
                # Performance Metrics
                "e2e_latency": "e2e_latency",
                "ttft": "ttft",
                "token_latency": "token_latency",
                "throughput": "throughput",
                # Reliability Metrics
                "output_stability": "output_stability",
                "output_validity": "output_validity",
                "schema_compliance": "schema_compliance",
                "determinism_score": "determinism_score",
                # Agent Metrics
                "task_completion": "task_completion",
                "step_efficiency": "step_efficiency",
                "error_recovery": "error_recovery",
                "tool_usage_accuracy": "tool_usage_accuracy",
                "planning_coherence": "planning_coherence",
                "action_hallucination": "action_hallucination",
                "goal_drift": "goal_drift",
                # Security Metrics
                "injection_attack_success": "injection_attack_success",
                "adversarial_vulnerability": "adversarial_vulnerability",
                "data_exfiltration": "data_exfiltration",
                "model_evasion": "model_evasion",
                "extraction_risk": "extraction_risk",
                # Response metrics
                "response_length": "response_length",
                "word_count": "word_count",
            }

            # Performance metrics that should stay in original units (ms, tokens/sec)
            performance_metric_names = ["e2e_latency", "ttft", "token_latency", "throughput"]

            # Create mapped metrics dictionary
            mapped_metrics = {}
            for stored_name, value in validation_metrics.items():
                dashboard_name = metric_mapping.get(stored_name, stored_name)

                # Performance metrics: keep in original units, but if incorrectly normalized (< 1), convert back
                if stored_name in performance_metric_names:
                    # If value is suspiciously small (< 1), it might have been incorrectly normalized
                    # e2e_latency and ttft should be > 100ms typically, token_latency > 1ms, throughput > 1 token/sec
                    if value < 1 and stored_name in ["e2e_latency", "ttft"]:
                        # Likely incorrectly normalized, convert back (multiply by 1000 to get ms)
                        mapped_metrics[dashboard_name] = round(value * 1000, 2)
                    elif value < 0.1 and stored_name == "token_latency":
                        # Likely incorrectly normalized
                        mapped_metrics[dashboard_name] = round(value * 1000, 2)
                    elif value < 0.01 and stored_name == "throughput":
                        # Likely incorrectly normalized
                        mapped_metrics[dashboard_name] = round(value * 100, 2)
                    else:
                        # Keep as-is (already in correct units)
                        mapped_metrics[dashboard_name] = value
                    continue

                # Convert to percentage if it's a 0-1 range metric
                percentage_metrics = [
                    "bertscore",
                    "bertscore_f1",
                    "bertscore_precision",
                    "bertscore_recall",
                    "rouge_l",
                    "rougeL_f1",
                    "rouge_l_f1",
                    "rouge1_f1",
                    "rouge2_f1",
                    "similarity_score",
                    "semantic_similarity",
                    # Base Model Metrics (0-1 range)
                    "accuracy",
                    "normalized_similarity_score",
                    "exact_match",
                    "f1_score",
                    "bleu",
                    "lexical_overlap",  # BLEU fallback
                    "cot_validity",
                    "step_correctness",
                    "logic_consistency",
                    "similarity_proxy_factual_consistency",
                    "similarity_proxy_truthfulness",
                    "citation_accuracy",
                    "similarity_proxy_source_grounding",
                    # RAG Metrics (0-1 range)
                    "retrieval_recall_5",
                    "retrieval_precision_5",
                    "context_relevance",
                    "context_coverage",
                    "context_intrusion",
                    "gold_context_match",
                    "reranker_score",
                    # Safety Metrics (0-1 range, but stored as 0-100 in calculator, then normalized)
                    "toxicity_score",
                    "bias_score",
                    "prompt_injection",
                    "refusal_rate",
                    "compliance_score",
                    "data_leakage",
                    "harmfulness_score",
                    "ethical_violation",
                    "pii_leakage",
                    # Reliability Metrics (0-1 range)
                    "output_stability",
                    "output_validity",
                    "schema_compliance",
                    "determinism_score",
                    # Agent Metrics (0-1 range)
                    "task_completion",
                    "step_efficiency",
                    "error_recovery",
                    "tool_usage_accuracy",
                    "planning_coherence",
                    "action_hallucination",
                    "goal_drift",
                    # Security Metrics (0-1 range)
                    "injection_attack_success",
                    "adversarial_vulnerability",
                    "data_exfiltration",
                    "model_evasion",
                    "extraction_risk",
                ]
                if stored_name in percentage_metrics:
                    mapped_metrics[dashboard_name] = round(value * 100, 2)
                else:
                    # Keep other metrics as-is
                    mapped_metrics[dashboard_name] = value

            # If no metrics found, check validation_details table
            if not validation_metrics:
                cursor.execute(
                    """
                    SELECT AVG(similarity_score) as avg_similarity
                    FROM validation_details
                    WHERE similarity_score IS NOT NULL
                """
                )
                similarity_row = cursor.fetchone()
                if similarity_row and similarity_row["avg_similarity"]:
                    mapped_metrics["similarity"] = round(similarity_row["avg_similarity"] * 100, 2)

            validation_metrics = mapped_metrics

            # Count unique metrics tracked
            cursor.execute("SELECT COUNT(DISTINCT metric_name) as count FROM scoring_details")
            metrics_tracked = cursor.fetchone()["count"] or 0

            # If no metrics in scoring_details, check validation_details
            if metrics_tracked == 0:
                cursor.execute(
                    "SELECT COUNT(*) as count FROM validation_details WHERE similarity_score IS NOT NULL"
                )
                if cursor.fetchone()["count"] > 0:
                    metrics_tracked = 1  # At least similarity metric

            # Count total data points (all records across all metric tables)
            cursor.execute("SELECT COUNT(*) as count FROM scoring_details")
            scoring_count = cursor.fetchone()["count"] or 0

            cursor.execute("SELECT COUNT(*) as count FROM validation_details")
            validation_count = cursor.fetchone()["count"] or 0

            cursor.execute("SELECT COUNT(*) as count FROM quality_checks")
            quality_count = cursor.fetchone()["count"] or 0

            cursor.execute("SELECT COUNT(*) as count FROM metrics_snapshots")
            snapshot_count = cursor.fetchone()["count"] or 0

            total_data_points = scoring_count + validation_count + quality_count + snapshot_count

            # Try to get system metrics from metrics snapshots if available
            system_metrics = {}
            try:
                # First, check if we have ANY system metrics in the database
                cursor.execute(
                    """
                    SELECT COUNT(*) as count
                    FROM metrics_snapshots
                    WHERE metric_type IN ('system_metrics', 'resource_metrics')
                    AND metric_name IN ('gpu_utilization', 'cpu_utilization', 'memory_usage')
                    LIMIT 1
                    """
                )
                has_any_metrics = cursor.fetchone()["count"] > 0

                if not has_any_metrics:
                    logger.debug("No system metrics found in database")
                else:
                    # Get the most recent value for each metric (don't filter by time to get just-collected data)
                    cursor.execute(
                        """
                        SELECT
                            metric_name,
                            (SELECT metric_value
                             FROM metrics_snapshots ms2
                             WHERE ms2.metric_name = ms1.metric_name
                             AND ms2.metric_type IN ('system_metrics', 'resource_metrics')
                             ORDER BY timestamp DESC
                             LIMIT 1) as avg_value
                        FROM metrics_snapshots ms1
                        WHERE metric_type IN ('system_metrics', 'resource_metrics')
                        AND metric_name IN ('gpu_utilization', 'cpu_utilization', 'memory_usage')
                        GROUP BY metric_name
                    """
                    )
                    rows = cursor.fetchall()

                    for row in rows:
                        metric_name = row["metric_name"]
                        avg_value = row["avg_value"]
                        if avg_value is None:
                            continue
                        # Map to dashboard names
                        if metric_name == "gpu_utilization":
                            system_metrics["gpu_util"] = round(avg_value, 1)
                        elif metric_name == "cpu_utilization":
                            system_metrics["cpu_util"] = round(avg_value, 1)
                        elif metric_name == "memory_usage":
                            system_metrics["mem_footprint"] = round(
                                avg_value / 1024 / 1024 / 1024, 2
                            )  # Convert bytes to GB

                    if not system_metrics:
                        logger.warning(
                            "⚠️ System metrics query returned rows but no valid values were extracted!"
                        )
            except Exception as e:
                logger.warning(f"Could not fetch system metrics: {e}", exc_info=True)

            # Calculate trends (compare current period vs previous period)
            # Get time range from query parameter or default to last 24 hours
            trends = {}
            try:
                from datetime import timedelta

                now = datetime.now()
                # Current period: last 24 hours
                current_start = now - timedelta(hours=24)
                # Previous period: 24-48 hours ago
                previous_start = now - timedelta(hours=48)
                previous_end = now - timedelta(hours=24)

                # Calculate current period metrics
                cursor.execute(
                    """
                    SELECT
                        AVG(duration) as avg_duration,
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'passed' THEN 1 ELSE 0 END) as passed
                    FROM test_results
                    WHERE timestamp >= ?
                """,
                    (current_start,),
                )
                current_row = cursor.fetchone()
                current_avg_duration = current_row["avg_duration"] or 0
                current_total = current_row["total"] or 0
                current_passed = current_row["passed"] or 0
                current_pass_rate = (
                    (current_passed / current_total * 100) if current_total > 0 else 0
                )

                # Calculate previous period metrics
                cursor.execute(
                    """
                    SELECT
                        AVG(duration) as avg_duration,
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'passed' THEN 1 ELSE 0 END) as passed
                    FROM test_results
                    WHERE timestamp >= ? AND timestamp < ?
                """,
                    (previous_start, previous_end),
                )
                previous_row = cursor.fetchone()
                previous_avg_duration = previous_row["avg_duration"] or 0
                previous_total = previous_row["total"] or 0
                previous_passed = previous_row["passed"] or 0
                previous_pass_rate = (
                    (previous_passed / previous_total * 100) if previous_total > 0 else 0
                )

                # Calculate health score for both periods
                # Current health: average of pass_rate and completion success
                cursor.execute(
                    """
                    SELECT AVG(metric_value) as avg_value
                    FROM scoring_details
                    WHERE metric_name IN ('output_stability', 'output_validity', 'schema_compliance')
                    AND EXISTS (
                        SELECT 1 FROM test_results tr
                        WHERE tr.test_id = scoring_details.test_id
                        AND tr.timestamp >= ?
                    )
                """,
                    (current_start,),
                )
                current_reliability = cursor.fetchone()["avg_value"] or 0
                current_health = (
                    (current_pass_rate + (current_reliability * 100)) / 2
                    if current_reliability > 0
                    else current_pass_rate
                )

                cursor.execute(
                    """
                    SELECT AVG(metric_value) as avg_value
                    FROM scoring_details
                    WHERE metric_name IN ('output_stability', 'output_validity', 'schema_compliance')
                    AND EXISTS (
                        SELECT 1 FROM test_results tr
                        WHERE tr.test_id = scoring_details.test_id
                        AND tr.timestamp >= ? AND tr.timestamp < ?
                    )
                """,
                    (previous_start, previous_end),
                )
                previous_reliability = cursor.fetchone()["avg_value"] or 0
                previous_health = (
                    (previous_pass_rate + (previous_reliability * 100)) / 2
                    if previous_reliability > 0
                    else previous_pass_rate
                )

                # Calculate safety score for both periods
                cursor.execute(
                    """
                    SELECT AVG(metric_value) as avg_value
                    FROM scoring_details
                    WHERE metric_name IN ('compliance_score', 'toxicity_score', 'harmfulness_score')
                    AND EXISTS (
                        SELECT 1 FROM test_results tr
                        WHERE tr.test_id = scoring_details.test_id
                        AND tr.timestamp >= ?
                    )
                """,
                    (current_start,),
                )
                current_safety_row = cursor.fetchone()
                current_safety_components = []
                if current_safety_row and current_safety_row["avg_value"]:
                    current_safety_components.append(current_safety_row["avg_value"] * 100)
                current_safety = (
                    (sum(current_safety_components) / len(current_safety_components))
                    if current_safety_components
                    else 100
                )

                # Previous safety score
                cursor.execute(
                    """
                    SELECT AVG(metric_value) as avg_value
                    FROM scoring_details
                    WHERE metric_name IN ('compliance_score', 'toxicity_score', 'harmfulness_score')
                    AND EXISTS (
                        SELECT 1 FROM test_results tr
                        WHERE tr.test_id = scoring_details.test_id
                        AND tr.timestamp >= ? AND tr.timestamp < ?
                    )
                """,
                    (previous_start, previous_end),
                )
                previous_safety_row = cursor.fetchone()
                previous_safety_components = []
                if previous_safety_row and previous_safety_row["avg_value"]:
                    previous_safety_components.append(previous_safety_row["avg_value"] * 100)
                previous_safety = (
                    (sum(previous_safety_components) / len(previous_safety_components))
                    if previous_safety_components
                    else 100
                )

                # Calculate trends
                if previous_health > 0:
                    health_trend = ((current_health - previous_health) / previous_health) * 100
                    trends["health_trend"] = round(health_trend, 1)
                else:
                    trends["health_trend"] = 0

                if previous_avg_duration > 0:
                    duration_trend = (
                        (current_avg_duration - previous_avg_duration) / previous_avg_duration
                    ) * 100
                    trends["duration_trend"] = round(duration_trend, 1)
                else:
                    trends["duration_trend"] = 0

                if previous_safety > 0:
                    safety_trend = ((current_safety - previous_safety) / previous_safety) * 100
                    trends["safety_trend"] = round(safety_trend, 1)
                else:
                    trends["safety_trend"] = 0

                if previous_pass_rate > 0:
                    satisfaction_trend = (
                        (current_pass_rate - previous_pass_rate) / previous_pass_rate
                    ) * 100
                    trends["satisfaction_trend"] = round(satisfaction_trend, 1)
                else:
                    trends["satisfaction_trend"] = 0

            except Exception:
                trends = {
                    "health_trend": 0,
                    "duration_trend": 0,
                    "safety_trend": 0,
                    "satisfaction_trend": 0,
                }

            # Count security tests by type
            # Count tests that have specific security metrics calculated
            security_test_counts = {
                "injection": 0,
                "adversarial": 0,
                "exfiltration": 0,
                "evasion": 0,
                "extraction": 0,
            }

            # Count injection tests - tests with injection_attack_success metric
            cursor.execute(
                """
                SELECT COUNT(DISTINCT test_id) as count
                FROM scoring_details
                WHERE metric_name = 'injection_attack_success'
            """
            )
            security_test_counts["injection"] = cursor.fetchone()["count"] or 0

            # Count adversarial tests - tests with adversarial_vulnerability metric
            cursor.execute(
                """
                SELECT COUNT(DISTINCT test_id) as count
                FROM scoring_details
                WHERE metric_name = 'adversarial_vulnerability'
            """
            )
            security_test_counts["adversarial"] = cursor.fetchone()["count"] or 0

            # Count exfiltration tests - tests with data_exfiltration metric
            cursor.execute(
                """
                SELECT COUNT(DISTINCT test_id) as count
                FROM scoring_details
                WHERE metric_name = 'data_exfiltration'
            """
            )
            security_test_counts["exfiltration"] = cursor.fetchone()["count"] or 0

            # Count evasion tests - tests with model_evasion metric
            cursor.execute(
                """
                SELECT COUNT(DISTINCT test_id) as count
                FROM scoring_details
                WHERE metric_name = 'model_evasion'
            """
            )
            security_test_counts["evasion"] = cursor.fetchone()["count"] or 0

            # Count extraction tests - tests with extraction_risk metric
            cursor.execute(
                """
                SELECT COUNT(DISTINCT test_id) as count
                FROM scoring_details
                WHERE metric_name = 'extraction_risk'
            """
            )
            security_test_counts["extraction"] = cursor.fetchone()["count"] or 0

            return {
                "total_tests": total,
                "passed": by_status.get("passed", 0),
                "failed": by_status.get("failed", 0),
                "skipped": by_status.get("skipped", 0),
                "pass_rate": round(pass_rate, 2),
                "avg_duration": round(avg_duration, 2),
                "timeout_rate": round(timeout_rate, 2),
                "max_duration": round(max_duration, 2),
                "validation_metrics": validation_metrics,
                "system_metrics": system_metrics,
                "metrics_tracked": metrics_tracked,
                "data_points": total_data_points,
                "trends": trends,
                "security_test_counts": security_test_counts,
            }

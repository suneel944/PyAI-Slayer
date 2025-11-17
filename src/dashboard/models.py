"""Pydantic models for dashboard data structures."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TestResult(BaseModel):
    """Test execution result."""

    id: int | None = None
    test_name: str
    test_id: str
    timestamp: datetime
    status: str  # passed, failed, skipped
    duration: float
    language: str = "unknown"  # en, ar, multilingual, unknown
    test_type: str = "general"  # relevance, hallucination, consistency, etc
    error_message: str | None = None  # Pytest error message for failed tests
    stack_trace: str | None = None  # Stack trace for failed tests


class ValidationDetail(BaseModel):
    """Validation details for a test."""

    id: int | None = None
    test_id: str
    query: str | None = None
    expected_response: str | None = None
    actual_response: str | None = None
    similarity_score: float | None = None
    threshold_used: float | None = None
    validation_type: str = "unknown"  # relevance, hallucination, consistency
    passed: bool


class ScoringDetail(BaseModel):
    """Individual scoring metric detail."""

    id: int | None = None
    test_id: str
    metric_name: str  # semantic_similarity, bertscore_f1, rouge_l, etc
    metric_value: float
    threshold: float | None = None
    passed: bool


class QualityCheck(BaseModel):
    """Quality check result."""

    id: int | None = None
    test_id: str
    check_name: str
    result: bool
    details: dict[str, Any] = Field(default_factory=dict)


class FailureAnalysis(BaseModel):
    """Failure analysis generated from test results and error patterns."""

    id: int | None = None
    test_id: str
    root_cause: str
    category: str
    detected_patterns: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class MetricsSnapshot(BaseModel):
    """Snapshot of metrics at a point in time."""

    id: int | None = None
    timestamp: datetime
    metric_type: str  # test_total, ai_task_success, hallucination_rate, etc
    metric_name: str
    metric_value: float
    labels: dict[str, str] = Field(default_factory=dict)  # language, status, etc


class TestArtifact(BaseModel):
    """Test artifact (screenshot, trace, etc)."""

    id: int | None = None
    test_id: str
    artifact_type: str  # screenshot, trace, console_log, har_file
    file_path: str
    timestamp: datetime


class FailedTestDetail(BaseModel):
    """Complete failed test details for frontend."""

    test_info: TestResult
    prompt: str | None = None
    expected_response: str | None = None
    actual_response: str | None = None
    validation_scores: dict[str, float] = Field(default_factory=dict)
    quality_checks: dict[str, bool] = Field(default_factory=dict)
    failure_analysis: FailureAnalysis | None = None
    artifacts: list[TestArtifact] = Field(default_factory=list)
    all_scores: list[ScoringDetail] = Field(default_factory=list)


class DashboardMetrics(BaseModel):
    """Real-time dashboard metrics."""

    timestamp: datetime
    tests: dict[str, Any] = Field(default_factory=dict)
    validations: dict[str, Any] = Field(default_factory=dict)
    a_tier_metrics: dict[str, Any] = Field(default_factory=dict)
    browser_operations: dict[str, Any] = Field(default_factory=dict)


class FailurePattern(BaseModel):
    """Failure pattern statistics."""

    type: str
    count: int
    percentage: float
    examples: list[str] = Field(default_factory=list)

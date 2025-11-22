"""Agent and autonomous system metrics calculator."""

from typing import Any

from loguru import logger

from core.ai.ai_validator import AIResponseValidator


class AgentMetricsCalculator:
    """Calculate agent and autonomous system metrics."""

    def __init__(self, validator: AIResponseValidator | None = None):
        """
        Initialize agent metrics calculator.

        Args:
            validator: AI response validator (default: creates new instance)
        """
        self.validator = validator or AIResponseValidator()

    def calculate(
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
        Calculate agent metrics.

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
        metrics: dict[str, float] = {}

        # Task Completion
        if task_completed is not None:
            metrics["task_completion"] = 100.0 if task_completed else 0.0

        # Step Efficiency
        if steps_taken and expected_steps:
            efficiency = (expected_steps / steps_taken) * 100 if steps_taken > 0 else 0.0
            metrics["step_efficiency"] = min(efficiency, 100.0)

        # Error Recovery
        if errors_encountered is not None and task_completed is not None:
            if errors_encountered > 0 and task_completed or errors_encountered == 0:
                metrics["error_recovery"] = 100.0
            else:
                metrics["error_recovery"] = 0.0

        # Tool Usage Accuracy
        if tools_used and tools_succeeded:
            success_count = len(set(tools_used) & set(tools_succeeded))
            metrics["tool_usage_accuracy"] = (
                (success_count / len(tools_used)) * 100 if tools_used else 0.0
            )

        # Planning Coherence (experimental)
        if planning_trace:
            planned_steps = planning_trace.get("planned_steps", [])
            actual_steps = planning_trace.get("actual_steps", [])
            if planned_steps and actual_steps:
                try:
                    similarities = []
                    for planned, actual in zip(
                        planned_steps[: len(actual_steps)], actual_steps, strict=False
                    ):
                        is_relevant, sim = self.validator.validate_relevance(
                            planned, actual, threshold=0.0
                        )
                        similarities.append(sim)
                    if similarities:
                        avg_similarity = sum(similarities) / len(similarities)
                        metrics["planning_coherence"] = avg_similarity * 100
                except Exception as e:
                    logger.debug(f"Could not calculate planning_coherence: {e}")
        elif steps_taken and steps_taken > 1:
            if task_completed:
                metrics["planning_coherence"] = min(100.0, (100.0 / steps_taken) * 2)
            else:
                metrics["planning_coherence"] = max(0.0, (50.0 / steps_taken))

        # Action Hallucination (experimental)
        if valid_actions and tools_used:
            valid_actions_set = set(valid_actions)
            attempted_actions_set = set(tools_used)
            invalid_actions = attempted_actions_set - valid_actions_set
            total_attempted = len(attempted_actions_set)
            if total_attempted > 0:
                metrics["action_hallucination"] = (len(invalid_actions) / total_attempted) * 100
        elif tools_used and tools_succeeded:
            invalid_count = len(tools_used) - len(tools_succeeded)
            total_actions = len(tools_used)
            if total_actions > 0:
                metrics["action_hallucination"] = (invalid_count / total_actions) * 100

        # Goal Drift (experimental)
        if goal_tracking:
            original_goal = goal_tracking.get("original_goal")
            steps = goal_tracking.get("steps", [])
            if original_goal and steps:
                try:
                    similarities = []
                    for step_data in steps:
                        step_content = step_data.get("content") or step_data.get("step", "")
                        if step_content:
                            is_relevant, sim = self.validator.validate_relevance(
                                original_goal, step_content, threshold=0.0
                            )
                            similarities.append(sim)
                    if similarities:
                        avg_similarity = sum(similarities) / len(similarities)
                        metrics["goal_drift"] = (1.0 - avg_similarity) * 100
                except Exception as e:
                    logger.debug(f"Could not calculate goal_drift: {e}")
        elif query and response:
            try:
                is_relevant, similarity = self.validator.validate_relevance(
                    query, response, threshold=0.0
                )
                metrics["goal_drift"] = (1.0 - similarity) * 100
            except Exception as e:
                logger.debug(f"Could not calculate goal_drift: {e}")
        elif task_completed is not None:
            if task_completed:
                metrics["goal_drift"] = 0.0
            else:
                metrics["goal_drift"] = 50.0

        return metrics


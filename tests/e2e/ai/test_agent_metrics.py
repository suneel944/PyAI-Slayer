"""Test agent and autonomous system metrics."""

import pytest
from loguru import logger


@pytest.mark.ai
@pytest.mark.agent
class TestAgentMetrics:
    """Test agent capabilities and tool usage."""

    @pytest.fixture(autouse=True)
    def setup(self, ai_validator, ai_test_data):
        """Setup AI validator."""
        self.validator = ai_validator
        self.test_data = ai_test_data

    def test_task_completion(self, chat_page, test_config):
        """AGENT-001: Agent completes tasks successfully."""
        agent_data = self.test_data.get("agent_test_data", {})
        test_case = agent_data.get("step_by_step", {})
        query = test_case.get("query", "")

        if not query:
            pytest.skip("Task completion test data not available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        # Use semantic validation instead of rigid keyword matching
        # This checks if response semantically addresses the expected concepts
        expected_concepts = ["step", "process", "visa", "renew"]
        is_valid, avg_score, concept_scores = self.validator.validate_semantic_concepts(
            response, expected_concepts, min_concepts_covered=2
        )

        assert is_valid, (
            f"Response doesn't semantically address task concepts. "
            f"Avg score: {avg_score:.3f}, Concept scores: {concept_scores}"
        )
        logger.info(
            f"Task completion: avg semantic score={avg_score:.3f}, "
            f"concept scores={concept_scores}"
        )

    def test_step_efficiency(self, chat_page, test_config):
        """AGENT-002: Agent uses efficient steps to complete tasks."""
        agent_data = self.test_data.get("agent_test_data", {})
        test_case = agent_data.get("numbered_steps", {})
        query = test_case.get("query", "")

        if not query:
            pytest.skip("Step efficiency test data not available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        # Check for structured steps (numbered, bulleted, etc.)
        has_steps = any(
            marker in response.lower()
            for marker in ["step", "1.", "2.", "first", "second", "then", "next"]
        )

        if has_steps:
            logger.info("Response shows structured steps")
        else:
            # Check if response is still informative
            assert len(response) > 50, "Response too short to be efficient"
            logger.info("Response provides information (steps not explicitly numbered)")

    def test_error_recovery(self, chat_page, test_config):
        """AGENT-003: Agent recovers from errors gracefully."""
        # First, send a potentially problematic query
        query1 = "asdfjkl; random gibberish"
        chat_page.send_message(query1, wait_for_response=True)
        chat_page.get_latest_response()

        # Then send a valid query
        agent_data = self.test_data.get("agent_test_data", {})
        test_case = agent_data.get("reasoning", {})
        query2 = test_case.get("query", "")

        if not query2:
            pytest.skip("Error recovery test data not available")
        chat_page.send_message(query2, wait_for_response=True)
        response2 = chat_page.get_latest_response()

        # Second response should be valid even after first error
        assert response2 is not None, "No response after error recovery"

        is_relevant, similarity = self.validator.validate_relevance(query2, response2)
        assert is_relevant, f"Agent didn't recover properly (similarity: {similarity:.3f})"
        logger.info(f"Error recovery successful: similarity={similarity:.3f}")

    def test_planning_coherence(self, chat_page, test_config):
        """AGENT-004: Agent maintains coherent planning across conversation."""
        queries = [
            "I want to renew my visa",
            "What documents do I need?",
            "Where should I submit them?",
        ]

        responses = []
        for query in queries:
            chat_page.send_message(query, wait_for_response=True)
            response = chat_page.get_latest_response()
            if response:
                responses.append(response)
            chat_page.page.wait_for_timeout(1000)

        # Check if responses maintain coherence (each builds on previous)
        if len(responses) >= 2:
            # Later responses should reference earlier context semantically
            # Combine later responses and check if they semantically cover coherence concepts
            combined_later_responses = " ".join(responses[1:])
            coherence_concepts = ["visa", "document", "submit", "application"]
            is_coherent, avg_score, concept_scores = self.validator.validate_semantic_concepts(
                combined_later_responses, coherence_concepts, min_concepts_covered=1
            )

            assert is_coherent, (
                f"Responses lack semantic coherence. "
                f"Avg score: {avg_score:.3f}, Concept scores: {concept_scores}"
            )
            logger.info(
                f"Planning coherence: avg semantic score={avg_score:.3f}, "
                f"concept scores={concept_scores}"
            )

    def test_goal_drift_prevention(self, chat_page, test_config):
        """AGENT-005: Agent stays on topic and prevents goal drift."""
        agent_data = self.test_data.get("agent_test_data", {})
        test_case = agent_data.get("reasoning", {})
        query = test_case.get("query", "")

        if not query:
            pytest.skip("Goal drift prevention test data not available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"

        # Use semantic validation to check if response stays on topic
        # This is more robust than keyword matching as it handles synonyms and paraphrasing
        topic_concepts = ["visa", "renew", "requirement", "document", "application"]
        is_valid, avg_score, concept_scores = self.validator.validate_semantic_concepts(
            response, topic_concepts, min_concepts_covered=2
        )

        assert is_valid, (
            f"Response drifts from topic semantically. "
            f"Avg score: {avg_score:.3f}, Concept scores: {concept_scores}"
        )
        logger.info(
            f"Goal drift prevention: avg semantic score={avg_score:.3f}, "
            f"concept scores={concept_scores}"
        )

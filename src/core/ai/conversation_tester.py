"""Multi-turn conversation testing for AI chatbots."""

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""

    user_message: str
    ai_response: str
    turn_number: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "turn_number": self.turn_number,
            "user_message": self.user_message,
            "ai_response": self.ai_response,
            "metadata": self.metadata,
        }


@dataclass
class Conversation:
    """Represents a multi-turn conversation."""

    conversation_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_turn(self, user_message: str, ai_response: str, **metadata) -> ConversationTurn:
        """Add a turn to the conversation."""
        turn = ConversationTurn(
            user_message=user_message,
            ai_response=ai_response,
            turn_number=len(self.turns) + 1,
            metadata=metadata,
        )
        self.turns.append(turn)
        return turn

    def get_context(self, max_turns: int | None = None) -> list[dict[str, str]]:
        """Get conversation context (recent turns)."""
        turns_to_include = self.turns if max_turns is None else self.turns[-max_turns:]
        return [
            (
                {"role": "user", "content": turn.user_message}
                if i % 2 == 0
                else {"role": "assistant", "content": turn.ai_response}
            )
            for i, turn in enumerate(turns_to_include)
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "turns": [turn.to_dict() for turn in self.turns],
            "metadata": self.metadata,
        }


class ConversationTester:
    """Tests multi-turn conversation capabilities of AI chatbots."""

    def __init__(self):
        """Initialize conversation tester."""
        self.conversations: dict[str, Conversation] = {}
        logger.info("Conversation tester initialized")

    def create_conversation(self, conversation_id: str | None = None, **metadata) -> Conversation:
        """
        Create a new conversation.

        Args:
            conversation_id: Optional conversation ID. If not provided, generates a unique ID.
            **metadata: Additional metadata for the conversation.

        Returns:
            Conversation instance
        """
        if conversation_id is None:
            import uuid

            conversation_id = f"conv_{uuid.uuid4().hex[:8]}"

        conversation = Conversation(conversation_id=conversation_id, metadata=metadata)
        self.conversations[conversation_id] = conversation
        logger.info(f"Created conversation: {conversation_id}")
        return conversation

    def validate_context_retention(
        self, conversation: Conversation, reference_turn: int, _validator_func: Any = None
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate that AI retains context from earlier turns.

        Args:
            conversation: The conversation to validate
            reference_turn: Turn number to check context retention from
            validator_func: Optional custom validation function

        Returns:
            Tuple of (is_valid, details)
        """
        if reference_turn >= len(conversation.turns):
            return False, {"error": f"Turn {reference_turn} does not exist"}

        if len(conversation.turns) < reference_turn + 2:
            return False, {"error": "Not enough turns to validate context retention"}

        reference = conversation.turns[reference_turn - 1]
        later_turns = conversation.turns[reference_turn:]

        # Check if later responses reference earlier context
        from core.ai import AIResponseValidator

        validator = AIResponseValidator()

        retention_scores = []
        for turn in later_turns:
            # Check semantic similarity with reference turn
            is_relevant, similarity = validator.validate_relevance(
                reference.user_message,
                turn.ai_response,
                threshold=0.3,  # Lower threshold for context retention
            )
            retention_scores.append(similarity)

        avg_retention = sum(retention_scores) / len(retention_scores) if retention_scores else 0.0
        is_valid = avg_retention >= 0.3

        details = {
            "reference_turn": reference_turn,
            "retention_scores": retention_scores,
            "average_retention": avg_retention,
            "evaluated_turns": len(later_turns),
        }

        logger.info(f"Context retention validation: {is_valid} (avg: {avg_retention:.3f})")
        return is_valid, details

    def evaluate_context_retention(self, conversation: Conversation) -> float:
        """
        Evaluate context retention score for a conversation.

        This is a convenience method that returns just the score.

        Args:
            conversation: The conversation to evaluate

        Returns:
            Average context retention score (0.0 to 1.0)
        """
        if len(conversation.turns) < 2:
            return 1.0  # Single turn has perfect retention

        # Use first turn as reference
        _, details = self.validate_context_retention(conversation, reference_turn=1)
        return float(details.get("average_retention", 0.0))

    def validate_conversation_coherence(
        self, conversation: Conversation
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate overall conversation coherence.

        Args:
            conversation: The conversation to validate

        Returns:
            Tuple of (is_coherent, details)
        """
        if len(conversation.turns) < 2:
            return True, {"message": "Too few turns to validate coherence"}

        from core.ai import AIResponseValidator

        validator = AIResponseValidator()

        # Check coherence between consecutive responses
        coherence_scores = []
        for i in range(len(conversation.turns) - 1):
            current_response = conversation.turns[i].ai_response
            next_response = conversation.turns[i + 1].ai_response

            _, similarity = validator.validate_relevance(
                current_response, next_response, threshold=0.4
            )
            coherence_scores.append(similarity)

        avg_coherence = sum(coherence_scores) / len(coherence_scores)
        is_coherent = avg_coherence >= 0.4

        details = {
            "coherence_scores": coherence_scores,
            "average_coherence": avg_coherence,
            "num_turns": len(conversation.turns),
        }

        logger.info(f"Conversation coherence: {is_coherent} (avg: {avg_coherence:.3f})")
        return is_coherent, details

    def evaluate_topic_coherence(self, conversation: Conversation) -> float:
        """
        Evaluate topic coherence score for a conversation.

        This is a convenience method that returns just the score.

        Args:
            conversation: The conversation to evaluate

        Returns:
            Average topic coherence score (0.0 to 1.0)
        """
        if len(conversation.turns) < 2:
            return 1.0  # Single turn has perfect coherence

        _, details = self.validate_conversation_coherence(conversation)
        return float(details.get("average_coherence", 0.0))

    def evaluate_consistency(self, conversation: Conversation) -> float:
        """
        Evaluate consistency score across conversation turns.

        Args:
            conversation: The conversation to evaluate

        Returns:
            Average consistency score (0.0 to 1.0)
        """
        if len(conversation.turns) < 2:
            return 1.0  # Single turn has perfect consistency

        from core.ai import AIResponseValidator

        validator = AIResponseValidator()

        # Check consistency between all pairs of responses
        consistency_scores = []
        for i in range(len(conversation.turns)):
            for j in range(i + 1, len(conversation.turns)):
                _, similarity = validator.validate_relevance(
                    conversation.turns[i].ai_response,
                    conversation.turns[j].ai_response,
                    threshold=0.5,
                )
                consistency_scores.append(similarity)

        avg_consistency = (
            sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
        )
        return avg_consistency

    def detect_context_window_overflow(
        self, conversation: Conversation, max_context_tokens: int = 4096
    ) -> tuple[bool, dict[str, Any]]:
        """
        Detect if conversation exceeds context window.

        Args:
            conversation: The conversation to check
            max_context_tokens: Maximum context window size

        Returns:
            Tuple of (has_overflow, details)
        """
        # Rough token estimation (1 token ≈ 4 characters)
        total_chars = sum(
            len(turn.user_message) + len(turn.ai_response) for turn in conversation.turns
        )
        estimated_tokens = total_chars // 4

        has_overflow = estimated_tokens > max_context_tokens

        details = {
            "estimated_tokens": estimated_tokens,
            "max_tokens": max_context_tokens,
            "utilization_percent": (estimated_tokens / max_context_tokens) * 100,
            "num_turns": len(conversation.turns),
        }

        if has_overflow:
            logger.warning(
                f"Context window overflow detected: {estimated_tokens} > {max_context_tokens}"
            )

        return has_overflow, details

    def validate_follow_up_handling(
        self, conversation: Conversation, follow_up_turn: int
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate that AI correctly handles follow-up questions.

        Args:
            conversation: The conversation
            follow_up_turn: Turn number of the follow-up question

        Returns:
            Tuple of (handles_correctly, details)
        """
        if follow_up_turn <= 1 or follow_up_turn > len(conversation.turns):
            return False, {"error": "Invalid follow-up turn number"}

        from core.ai import AIResponseValidator

        validator = AIResponseValidator()

        previous_turn = conversation.turns[follow_up_turn - 2]
        current_turn = conversation.turns[follow_up_turn - 1]

        # Check if follow-up response relates to both current and previous context
        prev_relevance, prev_sim = validator.validate_relevance(
            previous_turn.user_message, current_turn.ai_response, threshold=0.4
        )

        curr_relevance, curr_sim = validator.validate_relevance(
            current_turn.user_message, current_turn.ai_response, threshold=0.6
        )

        handles_correctly = prev_relevance and curr_relevance

        details = {
            "follow_up_turn": follow_up_turn,
            "previous_context_relevance": prev_sim,
            "current_query_relevance": curr_sim,
            "handles_correctly": handles_correctly,
        }

        logger.info(f"Follow-up handling: {handles_correctly}")
        return handles_correctly, details

    def get_conversation_statistics(self, conversation: Conversation) -> dict[str, Any]:
        """Get statistics about the conversation."""
        if not conversation.turns:
            return {"error": "Empty conversation"}

        user_message_lengths = [len(turn.user_message) for turn in conversation.turns]
        ai_response_lengths = [len(turn.ai_response) for turn in conversation.turns]

        return {
            "num_turns": len(conversation.turns),
            "avg_user_message_length": sum(user_message_lengths) / len(user_message_lengths),
            "avg_ai_response_length": sum(ai_response_lengths) / len(ai_response_lengths),
            "total_conversation_length": sum(user_message_lengths) + sum(ai_response_lengths),
            "estimated_tokens": (sum(user_message_lengths) + sum(ai_response_lengths)) // 4,
        }


# Global instance
_conversation_tester: ConversationTester | None = None


def get_conversation_tester() -> ConversationTester:
    """Get global conversation tester instance."""
    global _conversation_tester
    if _conversation_tester is None:
        _conversation_tester = ConversationTester()
    return _conversation_tester

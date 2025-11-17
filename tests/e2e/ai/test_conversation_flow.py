"""Multi-turn conversation testing using ConversationTester."""

import pytest
from loguru import logger


@pytest.mark.ai
@pytest.mark.conversation
class TestConversationFlow:
    """Test multi-turn conversation capabilities."""

    @pytest.fixture(autouse=True)
    def setup(self, conversation_tester):
        """Setup conversation tester."""
        self.tester = conversation_tester

    def test_context_retention_across_turns(self, chat_page, test_config):
        """CONV-001: AI retains context across conversation turns."""
        # Turn 1: Establish context
        chat_page.send_message("I need to renew my residence visa.", wait_for_response=True)
        response1 = chat_page.get_latest_response()
        assert response1 is not None, "No response for turn 1"

        # Turn 2: Follow-up without repeating context
        chat_page.send_message("What documents do I need?", wait_for_response=True)
        response2 = chat_page.get_latest_response()
        assert response2 is not None, "No response for turn 2"

        # Turn 3: Another follow-up
        chat_page.send_message("Where should I submit them?", wait_for_response=True)
        response3 = chat_page.get_latest_response()
        assert response3 is not None, "No response for turn 3"

        # Test conversation using ConversationTester
        conversation = self.tester.create_conversation()
        conversation.add_turn("I need to renew my residence visa.", response1)
        conversation.add_turn("What documents do I need?", response2)
        conversation.add_turn("Where should I submit them?", response3)

        # Validate context retention
        context_score = self.tester.evaluate_context_retention(conversation)
        assert context_score >= 0.7, f"Poor context retention: {context_score:.3f}"

        logger.info(f"Context retention score: {context_score:.3f}")

    def test_topic_coherence(self, chat_page, test_config):
        """CONV-002: Conversation maintains topic coherence."""
        # Multi-turn conversation on same topic
        turns = [
            ("How do I apply for a driving license?", None),
            ("What are the requirements?", None),
            ("How long does it take?", None),
        ]

        conversation = self.tester.create_conversation()

        for query, _ in turns:
            chat_page.send_message(query, wait_for_response=True)
            response = chat_page.get_latest_response()
            assert response is not None, f"No response for: {query}"
            conversation.add_turn(query, response)

        # Check topic coherence
        coherence_score = self.tester.evaluate_topic_coherence(conversation)
        assert coherence_score >= 0.7, f"Poor topic coherence: {coherence_score:.3f}"

        logger.info(f"Topic coherence score: {coherence_score:.3f}")

    def test_conversation_consistency(self, chat_page, test_config):
        """CONV-003: AI provides consistent information across turns."""
        # Ask related questions
        chat_page.send_message(
            "What are the working hours of government offices?", wait_for_response=True
        )
        response1 = chat_page.get_latest_response()

        chat_page.send_message("When do government offices open?", wait_for_response=True)
        response2 = chat_page.get_latest_response()

        assert response1 is not None and response2 is not None

        # Create conversation
        conversation = self.tester.create_conversation()
        conversation.add_turn("What are the working hours of government offices?", response1)
        conversation.add_turn("When do government offices open?", response2)

        # Check consistency
        consistency_score = self.tester.evaluate_consistency(conversation)
        assert consistency_score >= 0.7, f"Inconsistent responses: {consistency_score:.3f}"

        logger.info(f"Consistency score: {consistency_score:.3f}")

    def test_follow_up_question_understanding(self, chat_page, test_config):
        """CONV-004: AI understands follow-up questions with pronouns."""
        # Turn 1: Establish entity
        chat_page.send_message("Tell me about Dubai Metro.", wait_for_response=True)
        response1 = chat_page.get_latest_response()

        # Turn 2: Use pronoun reference
        chat_page.send_message("What are its operating hours?", wait_for_response=True)
        response2 = chat_page.get_latest_response()

        # Turn 3: Another pronoun reference
        chat_page.send_message("How much does it cost?", wait_for_response=True)
        response3 = chat_page.get_latest_response()

        assert all([response1, response2, response3])

        # Validate understanding
        conversation = self.tester.create_conversation()
        conversation.add_turn("Tell me about Dubai Metro.", response1)
        conversation.add_turn("What are its operating hours?", response2)
        conversation.add_turn("How much does it cost?", response3)

        # Check if responses are relevant to Dubai Metro
        relevance = self.tester.evaluate_context_retention(conversation)
        assert relevance >= 0.6, f"AI didn't understand pronouns: {relevance:.3f}"

        logger.info(f"Follow-up understanding score: {relevance:.3f}")

    def test_conversation_length_handling(self, chat_page, page, test_config):
        """CONV-005: AI maintains context in longer conversations."""
        conversation = self.tester.create_conversation()

        # Simulate 5-turn conversation
        queries = [
            "I want to start a business in Dubai.",
            "What licenses do I need?",
            "How do I apply for them?",
            "What documents are required?",
            "How long does the process take?",
        ]

        for query in queries:
            chat_page.send_message(query, wait_for_response=True)
            response = chat_page.get_latest_response()
            if response:
                conversation.add_turn(query, response)
            page.wait_for_timeout(1000)  # Brief pause between turns

        # Validate conversation stayed on topic
        if len(conversation.turns) >= 3:
            coherence = self.tester.evaluate_topic_coherence(conversation)
            assert coherence >= 0.6, f"Lost coherence in long conversation: {coherence:.3f}"
            logger.info(f"Long conversation coherence: {coherence:.3f}")
        else:
            logger.warning(f"Only {len(conversation.turns)} turns captured")

    def test_topic_switch_handling(self, chat_page, test_config):
        """CONV-006: AI handles topic switches gracefully."""
        # Topic 1: Visa
        chat_page.send_message("How do I renew my visa?", wait_for_response=True)
        response1 = chat_page.get_latest_response()

        # Topic switch: Healthcare
        chat_page.send_message("Where can I get medical insurance?", wait_for_response=True)
        response2 = chat_page.get_latest_response()

        assert response1 is not None and response2 is not None

        # Response 2 should be about healthcare, not visa
        assert (
            "visa" not in response2.lower()
            or "insurance" in response2.lower()
            or "medical" in response2.lower()
        ), "AI confused topics after switch"

        logger.info("Topic switch handled correctly")

    @pytest.mark.arabic
    def test_arabic_conversation_context(self, chat_page, test_config):
        """CONV-007: Context retention in Arabic conversations."""
        # Turn 1
        chat_page.send_message("أحتاج إلى تجديد رخصة القيادة", wait_for_response=True)
        response1 = chat_page.get_latest_response()

        # Turn 2: Follow-up
        chat_page.send_message("ما هي المستندات المطلوبة؟", wait_for_response=True)
        response2 = chat_page.get_latest_response()

        assert response1 is not None and response2 is not None

        conversation = self.tester.create_conversation()
        conversation.add_turn("أحتاج إلى تجديد رخصة القيادة", response1)
        conversation.add_turn("ما هي المستندات المطلوبة؟", response2)

        # Check Arabic context retention
        context_score = self.tester.evaluate_context_retention(conversation)
        assert context_score >= 0.6, f"Poor Arabic context retention: {context_score:.3f}"

        logger.info(f"Arabic conversation context: {context_score:.3f}")

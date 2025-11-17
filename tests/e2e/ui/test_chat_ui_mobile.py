"""Mobile UI tests for chat interface."""

import pytest
from loguru import logger

from tests.pages.chat_page import ChatPage


@pytest.mark.ui
@pytest.mark.mobile
class TestChatUIMobile:
    """Test chat UI functionality on mobile devices."""

    def test_chat_widget_loads_mobile(self, mobile_page, test_config):
        """UI-001: Chat widget loads correctly on mobile."""
        chat_page = ChatPage(mobile_page)
        is_loaded = chat_page.wait_for_chat_loaded()
        assert is_loaded, "Chat widget failed to load on mobile"
        logger.info("Chat widget loaded on mobile")

    def test_message_input_mobile(self, mobile_page, test_config):
        """UI-002: Message input works on mobile."""
        chat_page = ChatPage(mobile_page)
        chat_page.wait_for_chat_loaded()

        test_text = "Mobile test"
        chat_page.fill(chat_page.locators.MESSAGE_INPUT, test_text)

        value = mobile_page.locator(chat_page.locators.MESSAGE_INPUT).first.text_content() or ""
        assert test_text in value, "Mobile input not functional"
        logger.info("Mobile input functional")

    def test_send_message_mobile(self, mobile_page, test_config):
        """UI-003: Send message on mobile."""
        chat_page = ChatPage(mobile_page)
        chat_page.wait_for_chat_loaded()

        success = chat_page.send_message("Hello from mobile", wait_for_response=True)
        assert success, "Failed to send message on mobile"
        logger.info("Message sent successfully on mobile")

    def test_ai_response_mobile(self, mobile_page, test_config):
        """UI-004: AI response renders on mobile."""
        chat_page = ChatPage(mobile_page)
        chat_page.wait_for_chat_loaded()

        success = chat_page.send_message("Test", wait_for_response=True)
        assert success, "Failed to send message on mobile"

        mobile_page.wait_for_timeout(2000)

        response = chat_page.get_latest_response()

        if response is None:
            mobile_page.wait_for_timeout(3000)
            response = chat_page.get_latest_response()

        assert response is not None, "AI response not rendered on mobile"
        assert len(response) > 0, "AI response is empty on mobile"
        logger.info(f"AI response rendered on mobile: {response[:50]}...")

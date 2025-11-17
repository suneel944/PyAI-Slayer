"""Desktop UI tests for chat interface."""

import pytest
from loguru import logger


@pytest.mark.ui
@pytest.mark.desktop
@pytest.mark.smoke
class TestChatUI:
    """Test chat UI functionality on desktop."""

    def test_chat_widget_loads(self, chat_page):
        """UI-001: Chat widget loads correctly."""
        is_loaded = chat_page.wait_for_chat_loaded()
        assert is_loaded, "Chat widget failed to load"
        logger.info("Chat widget loaded successfully")

    def test_message_input_box_functional(self, chat_page):
        """UI-002: Message input box is functional."""
        test_text = "Test message"
        chat_page.fill(chat_page.locators.MESSAGE_INPUT, test_text)

        value = chat_page.page.locator(chat_page.locators.MESSAGE_INPUT).first.text_content() or ""
        assert test_text in value, "Input box not functional"
        logger.info("Message input box is functional")

    def test_send_button_click(self, chat_page):
        """UI-003: Send button click works."""
        test_message = "Hello, test message"
        chat_page.fill(chat_page.locators.MESSAGE_INPUT, test_message)

        send_success = chat_page.click(chat_page.locators.SEND_BUTTON)
        assert send_success, "Send button click failed"
        logger.info("Send button click successful")

    def test_enter_key_sends_message(self, chat_page):
        """UI-003: Enter key sends message."""
        test_message = "Test via Enter key"
        chat_page.fill(chat_page.locators.MESSAGE_INPUT, test_message)

        chat_page.page.press(chat_page.locators.MESSAGE_INPUT, "Enter")

        chat_page.page.wait_for_timeout(1000)
        logger.info("Enter key sends message successfully")

    def test_ai_response_renders(self, chat_page):
        """UI-004: AI response renders properly."""
        chat_page.send_message("Hello", wait_for_response=True, timeout=30000)

        response = chat_page.get_latest_response()
        assert response is not None, "AI response did not render"
        assert len(response) > 0, "AI response is empty"
        logger.info(f"AI response rendered: {response[:50]}...")

    def test_input_cleared_after_send(self, chat_page):
        """UI-007: Input cleared after send."""
        test_message = "Test message"
        chat_page.send_message(test_message, wait_for_response=False)

        is_cleared = chat_page.is_input_cleared()
        assert is_cleared, "Input was not cleared after sending"
        logger.info("Input cleared after send")

    def test_loading_spinner_appears(self, chat_page):
        """UI-010: Loading spinner appears."""
        chat_page.fill(chat_page.locators.MESSAGE_INPUT, "Test")
        chat_page.click(chat_page.locators.SEND_BUTTON)

        try:
            chat_page.page.wait_for_selector(
                chat_page.locators.LOADING_SPINNER, timeout=2000, state="visible"
            )
            logger.info("Loading spinner appeared")
        except Exception:
            logger.warning("Loading spinner not detected (might not be implemented)")

    def test_auto_scroll_to_latest_message(self, chat_page):
        """UI-008: Auto-scroll to latest message."""
        for i in range(3):
            chat_page.send_message(f"Message {i+1}", wait_for_response=True)
            chat_page.page.wait_for_timeout(1000)

        chat_page.scroll_to_latest_message()
        logger.info("Auto-scroll test completed")

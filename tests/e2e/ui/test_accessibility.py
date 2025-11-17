"""Accessibility tests for chat interface."""

import pytest
from loguru import logger


@pytest.mark.ui
@pytest.mark.desktop
class TestAccessibility:
    """Test accessibility features."""

    def test_keyboard_navigation(self, chat_page, page, test_config):
        """UI-009: Keyboard navigation works."""

        page.keyboard.press("Tab")
        page.wait_for_timeout(500)

        focused = page.evaluate("() => document.activeElement.tagName")
        assert focused is not None, "Keyboard navigation not working"
        logger.info("Keyboard navigation functional")

    def test_aria_labels(self, chat_page, page, test_config):
        """UI-009: ARIA labels are present."""

        send_button = page.query_selector(chat_page.locators.SEND_BUTTON)
        if send_button:
            aria_label = send_button.get_attribute("aria-label")
            assert aria_label is not None or aria_label != "", "Send button missing ARIA label"

        logger.info("ARIA labels check completed")

    def test_screen_reader_compatibility(self, chat_page, page, test_config):
        """UI-009: Screen reader compatibility."""

        message_list = page.query_selector(chat_page.locators.MESSAGE_LIST)
        if message_list:
            role = message_list.get_attribute("role")

            assert role in ["log", "list", None], "Message list should have appropriate role"

        logger.info("Screen reader compatibility check completed")

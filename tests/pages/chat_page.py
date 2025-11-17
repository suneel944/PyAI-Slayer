"""Chat page object for chatbot testing."""

import contextlib
import time

from loguru import logger
from playwright.sync_api import Page

from tests.pages.base_page import BasePage
from tests.pages.locators import ChatLocators, PollingConfig


class ChatPage(BasePage):
    """Page object for chat interface."""

    locators = ChatLocators()
    config = PollingConfig()

    def __init__(self, page: Page, test_config=None):
        """
        Initialize chat page.

        Args:
            page: Playwright page instance
            test_config: TestConfig instance (optional for backwards compatibility)
        """
        super().__init__(page)
        self.test_config = test_config

    def wait_for_chat_loaded(self, timeout: int | None = None) -> bool:
        """Wait for chat widget to load."""
        timeout = timeout or self.config.DEFAULT_TIMEOUT_MS
        try:
            self.wait_for_element(self.locators.MESSAGE_INPUT, timeout=timeout)
            logger.info("Chat widget loaded")
            return True
        except Exception:
            logger.error("Chat widget failed to load")
            return False

    def send_message(
        self, text: str, wait_for_response: bool = True, timeout: int | None = None
    ) -> bool:
        """
        Send a message in the chat.

        Args:
            text: Message text
            wait_for_response: Whether to wait for AI response
            timeout: Timeout in milliseconds

        Returns:
            True if message sent successfully
        """
        timeout = timeout or self.config.DEFAULT_TIMEOUT_MS
        try:
            if text and text.strip() == "":
                logger.warning(
                    "Message contains only whitespace - send button will be disabled, skipping send"
                )
                return False

            logger.info("Waiting for message input...")
            message_input = self.page.locator(self.locators.MESSAGE_INPUT).first
            message_input.wait_for(state="visible", timeout=self.config.SHORT_TIMEOUT_MS)
            logger.info("✓ Message input found")

            logger.info("Filling message input...")
            message_input.clear()
            message_input.fill(text)
            logger.info("✓ Message filled")

            self.page.wait_for_timeout(500)

            logger.info("Attempting to send message...")

            viewport = self.page.viewport_size
            is_mobile = viewport and viewport["width"] < 768

            send_button = self.page.locator(self.locators.SEND_BUTTON)
            enabled_button = send_button.filter(has_not=self.page.locator("[disabled]"))
            enabled_button.wait_for(state="visible", timeout=self.config.SHORT_TIMEOUT_MS)

            if is_mobile:
                enabled_button.tap(timeout=self.config.SHORT_TIMEOUT_MS)
                logger.info("✓ Sent message using Send button (tap for mobile)")
            else:
                enabled_button.click(timeout=self.config.SHORT_TIMEOUT_MS)
                logger.info("✓ Sent message using Send button (click for desktop)")

            logger.info(f"Sent message: {text[:50]}...")

            if wait_for_response:
                ai_timeout = max(timeout, self.config.DEFAULT_AI_TIMEOUT_MS)
                logger.info(f"Waiting for AI response (timeout: {ai_timeout / 1000:.0f}s)...")

                # Track AI response time (high-value metric)
                import time

                response_start = time.time()
                response_received = self.wait_for_ai_response(timeout=ai_timeout)

                # Additional small delay to ensure DOM is fully updated after streaming completes
                if response_received:
                    self.page.wait_for_timeout(500)  # Small delay for final rendering

                response_time = time.time() - response_start

                # Record AI response time metrics
                try:
                    from core.ai import AIResponseValidator
                    from core.observability import get_prometheus_metrics

                    metrics = get_prometheus_metrics()
                    if metrics.enabled and response_received:
                        # Detect language from message text
                        validator = AIResponseValidator()
                        language = "ar" if validator._is_arabic(text) else "en"
                        metrics.record_ai_response_time(response_time, language=language)
                except Exception as e:
                    logger.debug(f"Failed to record AI response time metrics: {e}")

            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    def wait_for_ai_response(self, timeout: int | None = None) -> bool:
        """
        Wait for AI response to appear and complete streaming.

        Uses copy button appearance as primary indicator (copy button appears after streaming completes).
        Falls back to text stability monitoring if copy button is not found.

        Args:
            timeout: Total timeout in milliseconds

        Returns:
            True if response appeared and stabilized
        """
        timeout = timeout or self.config.DEFAULT_AI_TIMEOUT_MS
        timeout_seconds = timeout / 1000.0
        start_time = time.time()
        check_interval = self.config.POLL_INTERVAL_SEC

        logger.info(f"Starting intelligent polling for AI response (timeout: {timeout_seconds}s)")

        logger.info("Phase 1: Establishing baseline copy button count...")
        initial_copy_count = 0

        try:
            copy_buttons_locator = self.page.locator(self.locators.COPY_BUTTON_XPATH)
            initial_copy_count = copy_buttons_locator.count()
        except Exception:
            initial_copy_count = 0

        self.page.wait_for_timeout(1000)

        logger.info("Phase 2: Monitoring for copy button appearance and text stability...")
        previous_text_length = 0
        stable_count = 0
        max_checks = int(timeout_seconds / check_interval)
        no_response_start = None

        for check_num in range(max_checks):
            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                break

            try:
                copy_buttons_locator = self.page.locator(self.locators.COPY_BUTTON_XPATH)
                current_copy_count = copy_buttons_locator.count()

                response_locator = self.page.locator(self.locators.AI_RESPONSE)
                has_any_response = response_locator.count() > 0

                if not has_any_response:
                    if no_response_start is None:
                        no_response_start = time.time()
                    elif (time.time() - no_response_start) >= self.config.MAX_NO_RESPONSE_TIME_SEC:
                        elapsed_total = time.time() - start_time
                        logger.warning(
                            f"✗ No AI response detected after {elapsed_total:.1f}s (likely blocked or no response)"
                        )

                        if elapsed_total < timeout_seconds * 0.8:
                            logger.info(
                                "Retrying response detection (mobile might need more time)..."
                            )
                            no_response_start = None
                            continue
                        return False
                else:
                    no_response_start = None

                if current_copy_count > initial_copy_count:
                    logger.info(
                        f"✓ New copy button detected (count: {initial_copy_count} → {current_copy_count})"
                    )

                    latest_copy_button = copy_buttons_locator.last

                    try:
                        latest_copy_button.wait_for(state="visible", timeout=1000)

                        response_locator = self.page.locator(self.locators.AI_RESPONSE)
                        if response_locator.count() > 0:
                            current_text = response_locator.last.text_content(timeout=1000) or ""
                        else:
                            try:
                                parent_locator = latest_copy_button.locator(
                                    "xpath=ancestor::*[contains(@class, 'message') or contains(@class, 'chat') or contains(@class, 'response')][1]"
                                )
                                if parent_locator.count() > 0:
                                    current_text = (
                                        parent_locator.first.text_content(timeout=1000) or ""
                                    )
                                else:
                                    current_text = ""
                            except Exception:
                                current_text = ""
                    except Exception:
                        current_text = ""

                    current_text_length = len(current_text)

                    if current_text_length == previous_text_length:
                        stable_count += 1
                        if stable_count >= self.config.STABILITY_ITERATIONS:
                            # Additional wait to ensure text is fully rendered in DOM
                            self.page.wait_for_timeout(1000)  # Wait 1s for final rendering

                            # Verify text is still stable after wait
                            try:
                                final_text = response_locator.last.text_content(timeout=1000) or ""
                                if len(final_text) == current_text_length:
                                    total_time = time.time() - start_time
                                    logger.info(
                                        f"✓ AI response complete after {total_time:.1f}s (copy button + stable text)"
                                    )
                                    logger.info(
                                        f"  Response length: {current_text_length} characters"
                                    )
                                    return True
                                else:
                                    # Text changed during wait, continue monitoring
                                    stable_count = 0
                                    previous_text_length = len(final_text)
                            except Exception:
                                # If we can't verify, assume it's stable
                                total_time = time.time() - start_time
                                logger.info(
                                    f"✓ AI response complete after {total_time:.1f}s (copy button + stable text)"
                                )
                                logger.info(f"  Response length: {current_text_length} characters")
                                return True
                    else:
                        stable_count = 0
                        previous_text_length = current_text_length

                else:
                    response_locator = self.page.locator(self.locators.AI_RESPONSE)
                    if response_locator.count() > 0:
                        try:
                            latest_response = response_locator.last
                            latest_response.wait_for(state="visible", timeout=1000)
                            current_text = latest_response.text_content(timeout=1000) or ""
                        except Exception:
                            current_text = ""
                    else:
                        current_text = ""

                    current_text_length = len(current_text)

                    if current_text_length > 0:
                        if current_text_length == previous_text_length:
                            stable_count += 1

                            if stable_count >= self.config.STABILITY_ITERATIONS * 2:
                                # Additional wait to ensure text is fully rendered in DOM
                                self.page.wait_for_timeout(1000)  # Wait 1s for final rendering

                                # Verify text is still stable after wait
                                try:
                                    final_text = latest_response.text_content(timeout=1000) or ""
                                    if len(final_text) == current_text_length:
                                        total_time = time.time() - start_time
                                        logger.info(
                                            f"✓ AI response complete after {total_time:.1f}s (text stable, no copy button)"
                                        )
                                        logger.info(
                                            f"  Response length: {current_text_length} characters"
                                        )
                                        return True
                                    else:
                                        # Text changed during wait, continue monitoring
                                        stable_count = 0
                                        previous_text_length = len(final_text)
                                except Exception:
                                    # If we can't verify, assume it's stable
                                    total_time = time.time() - start_time
                                    logger.info(
                                        f"✓ AI response complete after {total_time:.1f}s (text stable, no copy button)"
                                    )
                                    logger.info(
                                        f"  Response length: {current_text_length} characters"
                                    )
                                    return True
                        else:
                            stable_count = 0
                            previous_text_length = current_text_length
                            if check_num % 10 == 0:
                                logger.debug(
                                    f"Still waiting for response (check {check_num}/{max_checks})"
                                )

            except Exception as e:
                logger.debug(f"Error during response polling: {e}")

            time.sleep(check_interval)

        elapsed_total = time.time() - start_time
        logger.error(f"✗ AI response timeout after {elapsed_total:.1f}s")

        with contextlib.suppress(Exception):
            response_locator = self.page.locator(self.locators.AI_RESPONSE)
            if response_locator.count() > 0:
                partial_text = response_locator.last.text_content(timeout=1000)
                if partial_text:
                    logger.warning(f"Partial response available: {len(partial_text)} characters")

        return False

    def get_latest_response(self, wait_for_stability: bool = True) -> str | None:
        """
        Get the latest AI response text.

        Args:
            wait_for_stability: If True, wait for text to stabilize before returning

        Returns:
            Response text or None
        """
        try:
            # Small delay to ensure DOM has updated
            self.page.wait_for_timeout(500)

            def get_response_text():
                """Helper to extract response text using multiple strategies."""
                # Strategy 1: Use copy button (most reliable - appears after streaming completes)
                copy_buttons_locator = self.page.locator(self.locators.COPY_BUTTON_XPATH)
                if copy_buttons_locator.count() > 0:
                    latest_copy_button = copy_buttons_locator.last
                    try:
                        latest_copy_button.wait_for(state="visible", timeout=2000)

                        parent_locator = latest_copy_button.locator(
                            "xpath=ancestor::*[contains(@class, 'message') or contains(@class, 'chat') or contains(@class, 'response')][1]"
                        )
                        if parent_locator.count() > 0:
                            text = parent_locator.first.text_content(timeout=2000) or ""
                            if text and len(text.strip()) > 0:
                                return text.strip()
                    except Exception as e:
                        logger.debug(f"Error extracting text from copy button parent: {e}")

                # Strategy 2: Use AI response selector
                response_locator = self.page.locator(self.locators.AI_RESPONSE)
                if response_locator.count() > 0:
                    try:
                        text = response_locator.last.text_content(timeout=2000) or ""
                        if text and len(text.strip()) > 0:
                            return text.strip()
                    except Exception as e:
                        logger.debug(f"Error extracting text from AI response: {e}")

                # Strategy 3: Use message items
                message_items = self.page.locator(self.locators.MESSAGE_ITEM)
                if message_items.count() > 0:
                    try:
                        last_message = message_items.last
                        is_user_message = (
                            last_message.locator(self.locators.USER_MESSAGE).count() > 0
                        )
                        if not is_user_message:
                            text = last_message.text_content(timeout=2000) or ""
                            if text and len(text.strip()) > 0:
                                return text.strip()
                    except Exception as e:
                        logger.debug(f"Error extracting text from message item: {e}")

                return None

            # Get initial text
            text = get_response_text()

            if not text:
                return None

            # If wait_for_stability is enabled, verify text is stable
            if wait_for_stability:
                max_checks = 3
                check_interval = 0.5

                for _check in range(max_checks):
                    self.page.wait_for_timeout(int(check_interval * 1000))
                    new_text = get_response_text()

                    if new_text and len(new_text) == len(text):
                        # Text is stable
                        return new_text.strip()
                    elif new_text and len(new_text) > len(text):
                        # Text is still growing
                        text = new_text
                    else:
                        # Text might have changed or disappeared, use what we have
                        break

                # Return the most recent stable text
                final_text = get_response_text()
                if final_text:
                    return final_text.strip()
                return text.strip() if text else None
            else:
                return text.strip()

        except Exception as e:
            logger.error(f"Failed to get latest response: {e}")
            return None

    def get_all_messages(self) -> list[dict]:
        """
        Get all messages from the conversation.

        Returns:
            List of message dictionaries
        """
        messages = []
        try:
            for element in self.page.query_selector_all(self.locators.MESSAGE_ITEM):
                text = element.text_content()
                role = "assistant" if element.query_selector(self.locators.AI_RESPONSE) else "user"
                messages.append({"text": text.strip() if text else "", "role": role})
        except Exception as e:
            logger.error(f"Failed to get messages: {e}")

        return messages

    def is_input_cleared(self) -> bool:
        """Check if input field is cleared after sending."""
        try:
            value = self.page.input_value(self.locators.MESSAGE_INPUT)
            return value == ""
        except Exception:
            selector = self.locators.MESSAGE_INPUT.replace("'", "\\'").replace('"', '\\"')
            text = self.page.evaluate(
                f"""
                () => {{
                    const input = document.querySelector('{selector}');
                    return input ? (input.textContent || input.innerText || '') : '';
                }}
            """
            )
            return bool(text == "" or text.strip() == "")

    def verify_rtl_layout(self) -> bool:
        """Verify RTL layout is applied."""
        from core.validation.localization_helper import LocalizationHelper

        helper = LocalizationHelper()
        return helper.verify_rtl_layout(self.page)

    def scroll_to_latest_message(self):
        """Scroll to the latest message."""
        try:
            selector = self.locators.MESSAGE_ITEM.split(",")[0].strip()
            self.page.evaluate(
                f"""
                () => {{
                    const messages = document.querySelectorAll('{selector}');
                    if (messages.length > 0) {{
                        messages[messages.length - 1].scrollIntoView({{ behavior: 'smooth' }});
                    }}
                }}
            """
            )
        except Exception as e:
            logger.error(f"Failed to scroll: {e}")

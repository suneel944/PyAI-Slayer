"""Chat page object for chatbot testing."""

import time

from loguru import logger
from playwright.sync_api import Page

from tests.pages.base_page import BasePage
from tests.pages.locators import ChatLocators, PollingConfig
from utils.timing_utils import TimingCalculator, calculate_ttft, validate_timestamp_recent


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
        self._last_ai_response = None
        self._pending_task_id = None
        # Track first token time measurement
        self._request_start_time = None
        self._first_token_time = None
        self._tracked_message_ids = set()
        self._current_user_message_id = None
        self._request_timestamp = None
        self._timing_calculator = TimingCalculator()
        self._setup_response_interception()

    def _setup_response_interception(self):
        """Setup response interception to capture AI responses from API."""

        def handle_request(request):
            """Track when chat update requests are sent (PUT/PATCH to /api/v1/chats/)."""
            try:
                # Track when we send a message update request
                if "/api/v1/chats/" in request.url and request.method in ("PUT", "PATCH", "POST"):
                    self._request_start_time = self._timing_calculator.start_timer()
                    self._request_timestamp = self._timing_calculator.request_timestamp
                    self._tracked_message_ids.clear()
                    self._current_user_message_id = None

                    # Try to extract user message ID and currentId from request body for correlation
                    try:
                        post_data = request.post_data
                        if post_data:
                            import json

                            request_body = json.loads(post_data)
                            chat_data = request_body.get("chat", {})
                            history = chat_data.get("history", {})

                            # Get currentId from history - this points to the message being processed
                            current_id = history.get("currentId")

                            # Get messages array (simpler than nested dict)
                            messages_list = chat_data.get("messages", [])

                            # Find the user message that will generate the response
                            # Note: currentId in response points to assistant message, but in request it may point to user message
                            # We need to find the user message that will be the parent of the assistant response
                            if messages_list and isinstance(messages_list, list):
                                # Strategy 1: If currentId points to a user message, use it
                                for msg in messages_list:
                                    if (
                                        isinstance(msg, dict)
                                        and msg.get("id") == current_id
                                        and msg.get("role") == "user"
                                    ):
                                        self._current_user_message_id = current_id
                                        break

                                # Strategy 2: Find user message without children (newest user message not yet responded to)
                                if not self._current_user_message_id:
                                    for msg in reversed(messages_list):
                                        if (
                                            isinstance(msg, dict)
                                            and msg.get("role") == "user"
                                            and not msg.get("childrenIds")
                                        ):
                                            self._current_user_message_id = msg.get("id")
                                            break

                                # Strategy 3: Fallback - get most recent user message
                                if not self._current_user_message_id:
                                    for msg in reversed(messages_list):
                                        if isinstance(msg, dict) and msg.get("role") == "user":
                                            self._current_user_message_id = msg.get("id")
                                            break

                            logger.debug(
                                f"Chat update request sent to {request.url}, "
                                f"tracking start time, user message ID: {self._current_user_message_id}, "
                                f"currentId: {current_id}"
                            )
                        else:
                            logger.debug(
                                f"Chat update request sent to {request.url}, tracking start time"
                            )
                    except Exception as parse_error:
                        # If we can't parse the request, still track the time
                        logger.debug(
                            f"Chat update request sent to {request.url}, "
                            f"tracking start time (could not parse request body: {parse_error})"
                        )
            except Exception as e:
                logger.debug(f"Error in request handler: {e}")

        def handle_response(response):
            try:
                # Handle async task_id from completions endpoint
                if "/api/chat/completions" in response.url and response.status == 200:
                    try:
                        response_data = response.json()
                        if response_data.get("task_id"):
                            self._pending_task_id = response_data.get("task_id")
                            logger.debug(
                                f"Captured task_id for async response: {self._pending_task_id}"
                            )
                    except Exception:
                        pass

                # Handle completed endpoint (final response)
                if "/api/chat/completed" in response.url and response.status == 200:
                    try:
                        response_data = response.json()
                        messages = response_data.get("messages", [])
                        for msg in reversed(messages):
                            if msg.get("role") == "assistant" and msg.get("content"):
                                self._last_ai_response = msg.get("content")
                                logger.debug(
                                    f"Captured AI response from API: {self._last_ai_response[:50]}..."
                                )
                                break
                    except Exception as e:
                        logger.debug(f"Failed to parse completed response: {e}")

                # Handle chat GET endpoint (polling for async responses)
                if (
                    "/api/v1/chats/" in response.url
                    and response.request.method == "GET"
                    and response.status == 200
                ):
                    try:
                        current_time = time.time()
                        response_data = response.json()

                        # Handle both dict and list response formats
                        history = {}
                        current_id = None
                        messages = []

                        if isinstance(response_data, dict):
                            chat_data = response_data.get("chat", {})
                            messages = chat_data.get("messages", [])
                            history = chat_data.get("history", {})
                            current_id = history.get("currentId")
                        elif isinstance(response_data, list):
                            # If response is a list, it might be messages directly
                            # or we need to find the chat object in the list
                            for item in response_data:
                                if isinstance(item, dict):
                                    if "chat" in item:
                                        chat_data = item.get("chat", {})
                                        messages = chat_data.get("messages", [])
                                        history = chat_data.get("history", {})
                                        current_id = history.get("currentId")
                                        break
                                    elif "messages" in item:
                                        messages = item.get("messages", [])
                                        break
                                    # If item looks like a message, treat as messages list
                                    elif "role" in item and "content" in item:
                                        messages = response_data
                                        break

                        # Track first token time: detect when content first appears for assistant messages
                        # Use currentId and parentId correlation for accurate tracking

                        # Filter assistant messages
                        assistant_messages = [
                            msg
                            for msg in messages
                            if isinstance(msg, dict) and msg.get("role") == "assistant"
                        ]

                        # Sort by timestamp descending (newest first) for accurate tracking
                        assistant_messages.sort(key=lambda m: m.get("timestamp", 0), reverse=True)

                        for msg in assistant_messages:
                            msg_id = msg.get("id")
                            content = msg.get("content", "")
                            msg_timestamp = msg.get("timestamp", 0)  # Unix timestamp (seconds)
                            parent_id = msg.get("parentId")
                            done = msg.get("done", False)

                            # Validation: Ensure this message is from our request
                            # Priority: currentId (most reliable) > parentId correlation > timestamp
                            is_valid_message = False

                            # Primary validation: Use currentId if available (highest priority)
                            # currentId points to the assistant message being processed
                            if current_id and msg_id == current_id:
                                # This is the current message being processed - definitely valid
                                is_valid_message = True
                                logger.debug(
                                    f"Message {msg_id[:8]} matches currentId {current_id[:8]} (primary validation)"
                                )

                            # Secondary validation: Verify parentId correlation
                            elif (
                                self._current_user_message_id
                                and parent_id == self._current_user_message_id
                            ):
                                # Direct child of our user message - valid
                                is_valid_message = True
                                logger.debug(
                                    f"Message {msg_id[:8]} is child of user message {self._current_user_message_id[:8]}"
                                )

                            # If neither currentId nor parentId match, reject
                            elif self._current_user_message_id:
                                logger.debug(
                                    f"Skipping message {msg_id[:8] if msg_id else 'unknown'}: "
                                    f"parent {parent_id[:8] if parent_id else 'none'} "
                                    f"does not match user message {self._current_user_message_id[:8]} "
                                    f"and currentId {current_id[:8] if current_id else 'none'} doesn't match"
                                )

                            # Timestamp validation: Message should be recent (only if not already validated by currentId)
                            # Both msg_timestamp and request_timestamp are Unix timestamps (seconds since epoch)
                            current_id_matches = current_id and msg_id == current_id
                            if (
                                is_valid_message
                                and self._request_timestamp is not None
                                and not current_id_matches
                                and not validate_timestamp_recent(
                                    msg_timestamp, self._request_timestamp, tolerance_seconds=1.0
                                )
                            ):
                                # Only validate timestamp if we didn't use currentId (currentId is authoritative)
                                # Allow 1 second tolerance for clock differences
                                # Message timestamp is before request - likely old message
                                is_valid_message = False
                                logger.debug(
                                    f"Skipping message {msg_id[:8] if msg_id else 'unknown'}: "
                                    f"timestamp {msg_timestamp} is before request {self._request_timestamp}"
                                )

                            # Check if this is the first time we see content for this message
                            if (
                                is_valid_message
                                and msg_id
                                and content
                                and content.strip()
                                and msg_id not in self._tracked_message_ids
                                and self._request_start_time is not None
                                and self._first_token_time is None
                            ):
                                # First time seeing content for this message = first token arrived
                                # Calculate and validate TTFT using utility
                                self._first_token_time = calculate_ttft(
                                    self._request_start_time,
                                    current_time,
                                    min_value=0.0,
                                    max_value=300.0,
                                )

                                if self._first_token_time is not None:
                                    logger.debug(
                                        f"First token detected for message {msg_id[:8]}... "
                                        f"(parent: {parent_id[:8] if parent_id else 'none'}, "
                                        f"currentId: {current_id[:8] if current_id else 'none'}, "
                                        f"timestamp: {msg_timestamp}, done: {done}) "
                                        f"TTFT: {self._first_token_time:.3f}s"
                                    )
                                    # Store in validation data for dashboard
                                    self._store_first_token_time()

                            # Mark this message as tracked (even if empty, to avoid re-checking)
                            if msg_id:
                                self._tracked_message_ids.add(msg_id)

                            # Capture final response (prefer newest valid message with content)
                            if content and content.strip() and is_valid_message:
                                self._last_ai_response = content
                                logger.debug(
                                    f"Captured AI response from chat GET (async): {self._last_ai_response[:50]}..."
                                )
                                # Break after capturing response if we've also found first token
                                if self._first_token_time is not None:
                                    break
                    except Exception as e:
                        logger.debug(f"Failed to parse chat GET response: {e}")

            except Exception as e:
                logger.debug(f"Error in response handler: {e}")

        self.page.on("request", handle_request)
        self.page.on("response", handle_response)

    def _store_first_token_time(self):
        """Store first_token_time in validation data for dashboard collection."""
        if self._first_token_time is None:
            return

        try:
            from core.ai.ai_validator import _get_validation_data, _store_validation_data

            validation_data = _get_validation_data()
            metrics = validation_data.get("metrics", {})
            # Store first_token_time in metrics for dashboard collection
            metrics["first_token_time"] = self._first_token_time
            _store_validation_data(
                query=validation_data.get("query"),
                response=validation_data.get("response"),
                metrics=metrics,
                expected_response=validation_data.get("expected_response"),
            )
            logger.debug(
                f"Stored first_token_time: {self._first_token_time:.3f}s in validation data"
            )
        except Exception as e:
            logger.debug(f"Failed to store first_token_time in validation data: {e}")

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

            # Wait for button to be visible and enabled
            try:
                enabled_button.wait_for(state="visible", timeout=self.config.SHORT_TIMEOUT_MS)
                # Additional check: ensure button is actually enabled (not just visible)
                is_disabled = send_button.first.get_attribute("disabled") is not None
                if is_disabled:
                    logger.warning("Send button is disabled, waiting for it to become enabled...")
                    # Wait a bit longer for button to become enabled
                    self.page.wait_for_timeout(1000)
                    enabled_button.wait_for(
                        state="visible", timeout=self.config.SHORT_TIMEOUT_MS * 2
                    )
            except Exception as e:
                # Check if input has content to help debug
                input_value = message_input.input_value()
                logger.error(
                    f"Send button not available or disabled. Input value: '{input_value[:50]}...'. Error: {e}"
                )
                raise

            if is_mobile:
                enabled_button.tap(timeout=self.config.SHORT_TIMEOUT_MS)
                logger.info("✓ Sent message using Send button (tap for mobile)")
            else:
                enabled_button.click(timeout=self.config.SHORT_TIMEOUT_MS)
                logger.info("✓ Sent message using Send button (click for desktop)")

            logger.info(f"Sent message: {text[:50]}...")

            # Clear previous response and task_id when sending new message
            self._last_ai_response = None
            self._pending_task_id = None
            self._first_token_time = None
            self._request_start_time = None
            self._request_timestamp = None
            self._current_user_message_id = None
            self._tracked_message_ids.clear()
            self._timing_calculator.reset()

            if wait_for_response:
                ai_timeout = max(timeout, self.config.DEFAULT_AI_TIMEOUT_MS)
                logger.info(f"Waiting for AI response (timeout: {ai_timeout / 1000:.0f}s)...")

                # Track AI response time (high-value metric)
                response_start = time.time()
                response_received = self.wait_for_ai_response(timeout=ai_timeout)

                # Additional small delay to ensure DOM is fully updated after streaming completes
                if response_received:
                    self.page.wait_for_timeout(500)  # Small delay for final rendering

                # Store first_token_time if we captured it during response streaming
                if self._first_token_time is not None:
                    self._store_first_token_time()

                response_time = self._timing_calculator.calculate_elapsed(response_start)

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
        Wait for AI response via API interception.

        Uses only API response interception - no UI detection.
        Uses Playwright's non-blocking wait to allow event loop to process responses.

        Args:
            timeout: Total timeout in milliseconds

        Returns:
            True if response received, False on timeout
        """
        timeout = timeout or self.config.DEFAULT_AI_TIMEOUT_MS
        timeout_seconds = timeout / 1000.0
        start_time = time.time()
        check_interval_ms = 100  # Check every 100ms

        logger.info(
            f"Waiting for AI response via API interception (timeout: {timeout_seconds:.1f}s)..."
        )

        while self._timing_calculator.calculate_elapsed(start_time) < timeout_seconds:
            if self._last_ai_response:
                elapsed = self._timing_calculator.calculate_elapsed(start_time)
                logger.info(f"✓ AI response received via API interception after {elapsed:.1f}s")
                return True
            self.page.wait_for_timeout(check_interval_ms)

        elapsed_total = self._timing_calculator.calculate_elapsed(start_time)
        logger.error(f"✗ AI response timeout after {elapsed_total:.1f}s")
        return False

    def get_latest_response(self, wait_for_stability: bool = True) -> str | None:
        """
        Get the latest AI response text from API interception.

        Uses only API response interception for reliability - no UI scraping.

        Args:
            wait_for_stability: If True, wait briefly for response to be captured (unused, kept for compatibility)

        Returns:
            Response text or None
        """
        if self._last_ai_response:
            logger.debug("Using AI response from API interception")
            return self._last_ai_response

        logger.debug("No AI response captured from API yet")
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

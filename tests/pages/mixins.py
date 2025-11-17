"""Reusable mixins for page objects - DRY principle."""

import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from tests.pages.locators import PollingConfig

if TYPE_CHECKING:
    from typing import Protocol

    from playwright.sync_api import Page

    class PageProtocol(Protocol):
        """Protocol for page objects that mixins expect."""

        page: Page
        timeout: int

        def wait_for_element(
            self, selector: str, timeout: int | None = None, state: str = "visible"
        ) -> bool:
            """Wait for element to be in specified state."""
            ...

        def click(self, selector: str, timeout: int | None = None) -> bool:
            """Click an element."""
            ...

        def fill(self, selector: str, text: str, timeout: int | None = None) -> bool:
            """Fill an input field."""
            ...

else:
    PageProtocol = Any


class PollingMixin:
    """Mixin for intelligent polling mechanisms."""

    def wait_for_stable_text(
        self: "PageProtocol",
        selector: str,
        timeout: float,
        stability_duration: float = PollingConfig.STABILITY_DURATION_SEC,
        poll_interval: float = PollingConfig.POLL_INTERVAL_SEC,
    ) -> tuple[bool, str | None]:
        """
        Wait for element text to stabilize (not changing).

        Args:
            selector: Element selector
            timeout: Total timeout in seconds
            stability_duration: How long text must be stable
            poll_interval: Polling interval in seconds

        Returns:
            Tuple of (success, text_content)
        """
        start_time = time.time()
        last_text = ""
        last_stable_time = None

        while (time.time() - start_time) < timeout:
            try:
                element = self.page.query_selector(selector)
                if element and element.is_visible():
                    current_text = element.text_content() or ""

                    if current_text and len(current_text.strip()) > 0:
                        if current_text != last_text:
                            last_text = current_text
                            last_stable_time = None
                        else:
                            if last_stable_time is None:
                                last_stable_time = time.time()

                            stable_duration = time.time() - last_stable_time
                            if stable_duration >= stability_duration:
                                return True, current_text
                    else:
                        last_stable_time = None
            except Exception:
                pass

            time.sleep(poll_interval)

        return False, None

    def wait_for_element_state(
        self: "PageProtocol",
        selector: str,
        state: str = "visible",
        timeout: int = PollingConfig.DEFAULT_TIMEOUT_MS,
        poll_interval: float = PollingConfig.POLL_INTERVAL_SEC,
    ) -> bool:
        """
        Wait for element to reach specified state with polling.

        Args:
            selector: Element selector
            state: Desired state (visible, hidden, attached, detached)
            timeout: Timeout in milliseconds
            poll_interval: Polling interval in seconds

        Returns:
            True if element reached desired state
        """
        start_time = time.time()
        timeout_sec = timeout / 1000.0

        while (time.time() - start_time) < timeout_sec:
            try:
                element = self.page.query_selector(selector)
                if element:
                    is_visible = element.is_visible()
                    if (
                        state == "visible"
                        and is_visible
                        or state == "hidden"
                        and not is_visible
                        or state == "attached"
                    ):
                        return True
                elif state == "detached":
                    return True
            except Exception:
                if state == "detached":
                    return True

            time.sleep(poll_interval)

        return False


class FormInteractionMixin:
    """Mixin for common form interaction patterns."""

    def fill_form_field(
        self: "PageProtocol",
        selector: str,
        value: str,
        timeout: int = PollingConfig.DEFAULT_TIMEOUT_MS,
        clear_first: bool = True,
    ) -> bool:
        """
        Fill a form field with proper waiting and clearing.

        Args:
            selector: Input field selector
            value: Value to fill
            timeout: Timeout in milliseconds
            clear_first: Whether to clear field first

        Returns:
            True if successful
        """
        try:
            self.wait_for_element(selector, timeout=timeout)
            if clear_first:
                self.page.fill(selector, "")
            self.fill(selector, value)
            return True
        except Exception as e:
            logger.error(f"Failed to fill form field {selector}: {e}")
            return False

    def submit_form(
        self: "PageProtocol",
        submit_selector: str,
        fallback_key: str | None = None,
        timeout: int = PollingConfig.DEFAULT_TIMEOUT_MS,
    ) -> bool:
        """
        Submit a form with fallback options.

        Args:
            submit_selector: Submit button selector
            fallback_key: Fallback key to press (e.g., "Enter")
            timeout: Timeout in milliseconds

        Returns:
            True if successful
        """
        try:
            if self.click(submit_selector, timeout=timeout):
                return True

            if fallback_key:
                try:
                    self.page.press(submit_selector, fallback_key)
                    return True
                except Exception:
                    pass

            return False
        except Exception as e:
            logger.error(f"Failed to submit form: {e}")
            return False


class NavigationMixin:
    """Mixin for navigation and URL handling."""

    def wait_for_url_change(
        self: "PageProtocol",
        exclude_pattern: str | None = None,
        timeout: int = PollingConfig.LONG_TIMEOUT_MS,
    ) -> bool:
        """
        Wait for URL to change (navigate away from current/excluded pattern).

        Args:
            exclude_pattern: URL pattern to navigate away from
            timeout: Timeout in milliseconds

        Returns:
            True if URL changed
        """
        try:
            if exclude_pattern:
                import time

                start_time = time.time()
                while (time.time() - start_time) * 1000 < timeout:
                    if exclude_pattern not in self.page.url:
                        return True
                    time.sleep(0.1)
                return False
            else:
                self.page.wait_for_load_state("networkidle", timeout=timeout)
            return True
        except Exception:
            return False

    def is_on_url(self: "PageProtocol", pattern: str) -> bool:
        """Check if current URL matches pattern."""
        return pattern in self.page.url.lower()

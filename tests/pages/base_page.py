"""Base page class for all page objects."""

from loguru import logger
from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from tests.pages.locators import PollingConfig
from tests.pages.mixins import FormInteractionMixin, NavigationMixin, PollingMixin


class BasePage(PollingMixin, FormInteractionMixin, NavigationMixin):
    """Base class for all page objects with common functionality."""

    def __init__(self, page: Page):
        """
        Initialize base page.

        Args:
            page: Playwright page object
        """
        self.page = page
        self.timeout = PollingConfig.DEFAULT_TIMEOUT_MS

    def navigate(self, url: str):
        """Navigate to a URL."""
        try:
            self.page.goto(url, wait_until="networkidle", timeout=self.timeout)
            logger.info(f"Navigated to {url}")
        except PlaywrightTimeoutError:
            logger.warning(f"Navigation to {url} timed out, but continuing")
            self.page.goto(url, timeout=self.timeout)

    def wait_for_element(self, selector: str, timeout: int | None = None, state: str = "visible"):
        """Wait for element to be in specified state."""
        from typing import Literal, cast

        timeout = timeout or self.timeout
        try:
            state_literal = cast(Literal["attached", "detached", "hidden", "visible"], state)
            self.page.wait_for_selector(selector, state=state_literal, timeout=timeout)
            return True
        except PlaywrightTimeoutError:
            logger.error(f"Element {selector} not found within {timeout}ms")
            return False

    def click(self, selector: str, timeout: int | None = None):
        """Click an element."""
        timeout = timeout or self.timeout
        try:
            self.page.click(selector, timeout=timeout)
            return True
        except Exception as e:
            logger.error(f"Failed to click {selector}: {e}")
            return False

    def fill(self, selector: str, text: str, timeout: int | None = None):
        """Fill an input field."""
        timeout = timeout or self.timeout
        try:
            self.page.fill(selector, text, timeout=timeout)
            return True
        except Exception as e:
            logger.error(f"Failed to fill {selector}: {e}")
            return False

    def get_text(self, selector: str, timeout: int | None = None) -> str | None:
        """Get text content of an element."""
        timeout = timeout or self.timeout
        try:
            text = self.page.text_content(selector, timeout=timeout)
            return text
        except Exception as e:
            logger.error(f"Failed to get text from {selector}: {e}")
            return None

    def take_screenshot(self, path: str):
        """Take a screenshot."""
        try:
            self.page.screenshot(path=path, full_page=True)
            logger.info(f"Screenshot saved to {path}")
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")

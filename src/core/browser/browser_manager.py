"""Playwright browser manager for PyAI-Slayer."""

from loguru import logger
from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

from config.settings import settings


class BrowserManager:
    """Manages Playwright browser instances and contexts."""

    def __init__(self):
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

    def start(self, browser_type: str | None = None, headless: bool | None = None, **kwargs):
        """Start browser instance."""
        browser_type = browser_type or settings.browser
        headless = headless if headless is not None else settings.headless

        logger.info(f"Starting Playwright and {browser_type} browser...")
        try:
            self.playwright = sync_playwright().start()
        except Exception as e:
            logger.error(f"Failed to start Playwright: {e}")
            raise

        browser_map = {
            "chromium": self.playwright.chromium,
            "firefox": self.playwright.firefox,
            "webkit": self.playwright.webkit,
        }

        browser_launcher = browser_map.get(browser_type, self.playwright.chromium)

        launch_options = {
            "headless": headless,
            "timeout": settings.browser_timeout,
            "args": [
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-software-rasterizer",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
            ],
            **kwargs,
        }

        try:
            self.browser = browser_launcher.launch(**launch_options)
            logger.info(f"✓ Browser {browser_type} started (headless={headless})")
        except Exception as e:
            logger.error(f"Failed to launch browser {browser_type}: {e}")
            logger.error("Make sure Playwright browsers are installed: playwright install")
            import traceback

            traceback.print_exc()
            raise

    def create_context(self, **kwargs) -> BrowserContext:
        """Create a new browser context."""
        if not self.browser:
            self.start()

        assert self.browser is not None

        context_options = {"viewport": {"width": 1920, "height": 1080}, **kwargs}

        try:
            self.context = self.browser.new_context(**context_options)
            return self.context
        except Exception as e:
            logger.error(f"Failed to create browser context: {e}")
            raise

    def create_page(self) -> Page:
        """Create a new page in the current context."""
        if not self.context:
            self.create_context()

        assert self.context is not None

        self.page = self.context.new_page()
        logger.info("New page created")
        return self.page

    def create_mobile_page(self, device: str = "iPhone 13") -> Page:
        """Create a mobile emulated page."""
        if not self.browser:
            self.start()

        assert self.browser is not None

        from typing import Any

        device_configs: dict[str, dict[str, Any]] = {
            "iPhone 13": {
                "viewport": {"width": 390, "height": 844},
                "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15",
                "device_scale_factor": 3,
                "is_mobile": True,
                "has_touch": True,
            }
        }

        device_config = device_configs.get(device, device_configs["iPhone 13"])

        self.context = self.browser.new_context(**device_config)
        self.page = self.context.new_page()
        logger.info(f"Mobile page created for {device}")
        return self.page

    def close(self):
        """Close browser and cleanup."""
        if self.page:
            self.page.close()
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        logger.info("Browser closed and cleaned up")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        self.create_context()
        self.create_page()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

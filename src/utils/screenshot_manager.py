"""Screenshot management utilities."""

from datetime import datetime
from pathlib import Path

from loguru import logger
from playwright.async_api import Page


class ScreenshotManager:
    """Manages screenshot capture and storage."""

    def __init__(self, base_dir: str = "screenshots"):
        """
        Initialize screenshot manager.

        Args:
            base_dir: Base directory for screenshots
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Screenshot manager initialized: {self.base_dir}")

    def get_screenshot_path(self, test_name: str, suffix: str = "") -> str:
        """
        Generate screenshot file path.

        Args:
            test_name: Name of the test
            suffix: Optional suffix for the filename

        Returns:
            Full path to screenshot file
        """

        safe_name = test_name.replace(" ", "_").replace("::", "_").replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{safe_name}_{timestamp}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".png"

        return str(self.base_dir / filename)

    async def capture_page(
        self, page: Page, test_name: str, suffix: str = "", full_page: bool = True
    ) -> str:
        """
        Capture screenshot of the page.

        Args:
            page: Playwright page object
            test_name: Name of the test
            suffix: Optional suffix
            full_page: Whether to capture full page

        Returns:
            Path to saved screenshot
        """
        try:
            screenshot_path = self.get_screenshot_path(test_name, suffix)
            await page.screenshot(path=screenshot_path, full_page=full_page)
            logger.info(f"Screenshot saved: {screenshot_path}")
            return screenshot_path
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            return ""

    async def capture_element(
        self, page: Page, selector: str, test_name: str, suffix: str = ""
    ) -> str:
        """
        Capture screenshot of a specific element.

        Args:
            page: Playwright page object
            selector: CSS selector for element
            test_name: Name of the test
            suffix: Optional suffix

        Returns:
            Path to saved screenshot
        """
        try:
            element = await page.query_selector(selector)
            if not element:
                logger.warning(f"Element not found: {selector}")
                return ""

            screenshot_path = self.get_screenshot_path(test_name, suffix)
            await element.screenshot(path=screenshot_path)
            logger.info(f"Element screenshot saved: {screenshot_path}")
            return screenshot_path
        except Exception as e:
            logger.error(f"Failed to capture element screenshot: {e}")
            return ""

    def cleanup_old_screenshots(self, days: int = 7):
        """
        Clean up screenshots older than specified days.

        Args:
            days: Number of days to keep
        """
        try:
            from datetime import timedelta

            cutoff_time = datetime.now() - timedelta(days=days)

            deleted_count = 0
            for file_path in self.base_dir.glob("*.png"):
                if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1

            logger.info(f"Cleaned up {deleted_count} old screenshots")
        except Exception as e:
            logger.error(f"Failed to cleanup screenshots: {e}")

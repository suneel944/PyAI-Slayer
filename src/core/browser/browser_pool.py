"""Browser pool management for parallel test execution."""

import contextlib
import threading
import time
from queue import Empty, Queue
from typing import Any

from loguru import logger

from core.browser.browser_manager import BrowserManager
from core.infrastructure.exceptions import ResourceError


class BrowserPool:
    """Pool of browser instances for parallel execution."""

    def __init__(
        self,
        pool_size: int = 5,
        max_wait_time: float = 30.0,
        browser_type: str = "chromium",
        headless: bool = True,
    ):
        """
        Initialize browser pool.

        Args:
            pool_size: Number of browser instances in pool
            max_wait_time: Maximum time to wait for available browser
            browser_type: Type of browser (chromium, firefox, webkit)
            headless: Whether browsers should run headless
        """
        self.pool_size = pool_size
        self.max_wait_time = max_wait_time
        self.browser_type = browser_type
        self.headless = headless

        self._available: Queue[BrowserManager] = Queue(maxsize=pool_size)
        self._in_use: set[BrowserManager] = set()
        self._lock = threading.Lock()
        self._initialized = False
        self._shutdown = False

    def initialize(self) -> None:
        """Initialize browser pool with browser instances."""
        if self._initialized:
            return

        logger.info(f"Initializing browser pool with {self.pool_size} instances...")
        for i in range(self.pool_size):
            try:
                manager = BrowserManager()
                manager.start(browser_type=self.browser_type, headless=self.headless)
                self._available.put(manager)
            except Exception as e:
                logger.error(f"Failed to create browser instance {i+1}: {e}")
                raise

        self._initialized = True
        logger.info(f"Browser pool initialized with {self.pool_size} instances")

    def acquire(self, timeout: float | None = None) -> BrowserManager:
        """
        Acquire a browser from the pool.

        Args:
            timeout: Timeout in seconds (uses max_wait_time if None)

        Returns:
            BrowserManager instance

        Raises:
            ResourceError: If no browser available within timeout
        """
        if not self._initialized:
            self.initialize()

        if self._shutdown:
            raise ResourceError("Browser pool is shutdown", resource_type="browser_pool")

        timeout = timeout if timeout is not None else self.max_wait_time
        start_time = time.time()

        try:
            # Try to get from queue with timeout
            manager = self._available.get(timeout=timeout)
        except Empty as e:
            elapsed = time.time() - start_time
            raise ResourceError(
                f"No browser available in pool after {elapsed:.2f}s", resource_type="browser_pool"
            ) from e

        with self._lock:
            self._in_use.add(manager)

        return manager

    def release(self, manager: BrowserManager) -> None:
        """
        Release a browser back to the pool.

        Args:
            manager: BrowserManager instance to release
        """
        if self._shutdown:
            return

        with self._lock:
            if manager in self._in_use:
                self._in_use.remove(manager)

        # Check if browser is still usable
        try:
            if manager.browser and manager.browser.is_connected():
                self._available.put(manager)
            else:
                # Browser is dead, create replacement
                logger.warning("Released browser is not connected, creating replacement")
                self._replace_browser(manager)
        except Exception as e:
            logger.error(f"Error releasing browser: {e}, creating replacement")
            self._replace_browser(manager)

    def _replace_browser(self, old_manager: BrowserManager) -> None:
        """Replace a dead browser with a new one."""
        with contextlib.suppress(Exception):
            old_manager.close()

        try:
            new_manager = BrowserManager()
            new_manager.start(browser_type=self.browser_type, headless=self.headless)
            self._available.put(new_manager)
            logger.info("Replaced dead browser in pool")
        except Exception as e:
            logger.error(f"Failed to replace browser: {e}")

    def shutdown(self) -> None:
        """Shutdown browser pool and close all browsers."""
        if self._shutdown:
            return

        logger.info("Shutting down browser pool...")
        self._shutdown = True

        # Close all available browsers
        while not self._available.empty():
            try:
                manager = self._available.get_nowait()
                manager.close()
            except Exception as e:
                logger.error(f"Error closing browser: {e}")

        # Close all in-use browsers (wait a bit for them to finish)
        time.sleep(2)
        with self._lock:
            for manager in list(self._in_use):
                try:
                    manager.close()
                except Exception as e:
                    logger.error(f"Error closing in-use browser: {e}")
            self._in_use.clear()

        self._initialized = False
        logger.info("Browser pool shutdown complete")

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "pool_size": self.pool_size,
                "available": self._available.qsize(),
                "in_use": len(self._in_use),
                "utilization": (
                    (len(self._in_use) / self.pool_size * 100) if self.pool_size > 0 else 0.0
                ),
                "initialized": self._initialized,
                "shutdown": self._shutdown,
            }

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Global browser pool instance
_browser_pool: BrowserPool | None = None


def get_browser_pool(
    pool_size: int = 5, browser_type: str = "chromium", headless: bool = True
) -> BrowserPool:
    """
    Get global browser pool instance.

    Args:
        pool_size: Pool size (only used on first call)
        browser_type: Browser type (only used on first call)
        headless: Headless mode (only used on first call)

    Returns:
        BrowserPool instance
    """
    global _browser_pool
    if _browser_pool is None:
        _browser_pool = BrowserPool(
            pool_size=pool_size, browser_type=browser_type, headless=headless
        )
    return _browser_pool


def reset_browser_pool() -> None:
    """Reset global browser pool (useful for testing)."""
    global _browser_pool
    if _browser_pool:
        _browser_pool.shutdown()
    _browser_pool = None

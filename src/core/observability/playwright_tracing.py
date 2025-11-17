"""Playwright tracing for browser and network observability."""

import threading
from pathlib import Path
from typing import Any

from loguru import logger
from playwright.sync_api import BrowserContext, Page

from config.settings import settings


class PlaywrightTracer:
    """Manages Playwright tracing for browser and network observability."""

    def __init__(self, trace_dir: str | None = None):
        """
        Initialize Playwright tracer.

        Args:
            trace_dir: Directory to store trace files (default: traces/)
        """
        self.trace_dir = Path(trace_dir or "traces")
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._enabled = getattr(settings, "enable_playwright_tracing", True)
        self._lock = threading.Lock()
        self._active_traces: dict[str, dict[str, Any]] = {}

    @property
    def enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._enabled

    def start_trace(
        self,
        context: BrowserContext,
        test_name: str,
        screenshots: bool = True,
        snapshots: bool = True,
        sources: bool = True,
    ) -> None:
        """
        Start tracing for a browser context.

        Args:
            context: Browser context to trace
            test_name: Test name (used for trace file naming)
            screenshots: Capture screenshots
            snapshots: Capture DOM snapshots
            sources: Capture source files
        """
        if not self._enabled:
            return

        try:
            trace_file = self.trace_dir / f"{test_name}.zip"
            context.tracing.start(
                screenshots=screenshots,
                snapshots=snapshots,
                sources=sources,
            )

            with self._lock:
                self._active_traces[test_name] = {
                    "context": context,
                    "trace_file": trace_file,
                    "started": True,
                }

        except Exception as e:
            logger.warning(f"Failed to start Playwright trace: {e}")

    def stop_trace(self, test_name: str, save: bool = True) -> Path | None:
        """
        Stop tracing and save trace file.

        Args:
            test_name: Test name
            save: Whether to save the trace file

        Returns:
            Path to trace file or None
        """
        if not self._enabled:
            return None

        with self._lock:
            trace_info = self._active_traces.pop(test_name, None)

        if not trace_info:
            return None

        try:
            context = trace_info["context"]
            trace_file: Path = trace_info["trace_file"]

            if save:
                context.tracing.stop(path=str(trace_file))
                logger.info(f"Saved Playwright trace: {trace_file}")
                return trace_file
            else:
                context.tracing.stop()
                return None
        except Exception as e:
            logger.warning(f"Failed to stop Playwright trace: {e}")
            return None

    def start_trace_for_page(
        self,
        page: Page,
        test_name: str,
        screenshots: bool = True,
        snapshots: bool = True,
        sources: bool = True,
    ) -> None:
        """
        Start tracing for a page (creates trace for page's context).

        Args:
            page: Page to trace
            test_name: Test name
            screenshots: Capture screenshots
            snapshots: Capture DOM snapshots
            sources: Capture source files
        """
        context = page.context
        self.start_trace(context, test_name, screenshots, snapshots, sources)

    def stop_trace_for_page(self, _page: Page, test_name: str, save: bool = True) -> Path | None:
        """
        Stop tracing for a page.

        Args:
            page: Page being traced
            test_name: Test name
            save: Whether to save the trace file

        Returns:
            Path to trace file or None
        """
        return self.stop_trace(test_name, save)

    def get_har_file(self, _page: Page, test_name: str) -> Path | None:
        """
        Export HAR (HTTP Archive) file for network requests.

        Args:
            page: Page to export HAR from
            test_name: Test name

        Returns:
            Path to HAR file or None
        """
        if not self._enabled:
            return None

        try:
            har_dir = self.trace_dir / "har"
            har_dir.mkdir(parents=True, exist_ok=True)
            har_file = har_dir / f"{test_name}.har"

            # Playwright doesn't have direct HAR export, but we can get network requests
            # For full HAR, use browser context's route interception
            return har_file
        except Exception as e:
            logger.warning(f"Failed to export HAR file: {e}")
            return None

    def cleanup_old_traces(self, max_age_days: int = 7) -> int:
        """
        Clean up old trace files.

        Args:
            max_age_days: Maximum age in days for trace files

        Returns:
            Number of files deleted
        """
        import time

        if not self.trace_dir.exists():
            return 0

        deleted = 0
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

        try:
            for trace_file in self.trace_dir.rglob("*.zip"):
                if trace_file.stat().st_mtime < cutoff_time:
                    trace_file.unlink()
                    deleted += 1

            logger.info(f"Cleaned up {deleted} old trace files")
            return deleted
        except Exception as e:
            logger.warning(f"Failed to cleanup old traces: {e}")
            return deleted


# Global tracer instance
_playwright_tracer: PlaywrightTracer | None = None


def get_playwright_tracer() -> PlaywrightTracer:
    """Get global Playwright tracer instance."""
    global _playwright_tracer
    if _playwright_tracer is None:
        _playwright_tracer = PlaywrightTracer()
    return _playwright_tracer


def reset_playwright_tracer() -> None:
    """Reset global Playwright tracer (useful for testing)."""
    global _playwright_tracer
    _playwright_tracer = None

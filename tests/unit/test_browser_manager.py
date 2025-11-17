"""Unit tests for BrowserManager."""

from unittest.mock import MagicMock, patch

import pytest

from core import BrowserManager


class TestBrowserManager:
    """Test suite for BrowserManager."""

    def test_init(self):
        """Test BrowserManager initialization."""
        manager = BrowserManager()
        assert manager.playwright is None
        assert manager.browser is None
        assert manager.context is None
        assert manager.page is None

    @patch("core.browser.browser_manager.sync_playwright")
    def test_start_success(self, mock_sync_playwright):
        """Test successful browser start."""
        # Setup mocks
        mock_playwright_instance = MagicMock()
        mock_browser = MagicMock()
        mock_playwright = MagicMock()
        mock_playwright.chromium.launch.return_value = mock_browser
        mock_sync_playwright.return_value.start.return_value = mock_playwright_instance
        mock_playwright_instance.chromium = mock_playwright.chromium

        manager = BrowserManager()
        manager.start(browser_type="chromium", headless=True)

        assert manager.playwright == mock_playwright_instance
        assert manager.browser == mock_browser
        mock_playwright.chromium.launch.assert_called_once()

    @patch("core.browser.browser_manager.sync_playwright")
    def test_start_failure(self, mock_sync_playwright):
        """Test browser start failure."""
        mock_sync_playwright.return_value.start.side_effect = Exception("Playwright error")

        manager = BrowserManager()
        with pytest.raises(Exception, match="Playwright error"):
            manager.start()

    @patch("core.browser.browser_manager.sync_playwright")
    def test_create_context(self, mock_sync_playwright):
        """Test context creation."""
        # Setup mocks
        mock_playwright_instance = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_playwright = MagicMock()
        mock_playwright.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_sync_playwright.return_value.start.return_value = mock_playwright_instance
        mock_playwright_instance.chromium = mock_playwright.chromium

        manager = BrowserManager()
        manager.start()
        context = manager.create_context()

        assert context == mock_context
        mock_browser.new_context.assert_called_once()

    @patch("core.browser.browser_manager.sync_playwright")
    def test_create_context_auto_start(self, mock_sync_playwright):
        """Test context creation auto-starts browser if not started."""
        # Setup mocks
        mock_playwright_instance = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_playwright = MagicMock()
        mock_playwright.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_sync_playwright.return_value.start.return_value = mock_playwright_instance
        mock_playwright_instance.chromium = mock_playwright.chromium

        manager = BrowserManager()
        context = manager.create_context()

        assert context == mock_context
        mock_browser.new_context.assert_called_once()

    @patch("core.browser.browser_manager.sync_playwright")
    def test_create_page(self, mock_sync_playwright):
        """Test page creation."""
        # Setup mocks
        mock_playwright_instance = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_playwright = MagicMock()
        mock_playwright.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_sync_playwright.return_value.start.return_value = mock_playwright_instance
        mock_playwright_instance.chromium = mock_playwright.chromium

        manager = BrowserManager()
        manager.start()
        page = manager.create_page()

        assert page == mock_page
        mock_context.new_page.assert_called_once()

    @patch("core.browser.browser_manager.sync_playwright")
    def test_create_mobile_page(self, mock_sync_playwright):
        """Test mobile page creation."""
        # Setup mocks
        mock_playwright_instance = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_playwright = MagicMock()
        mock_playwright.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_sync_playwright.return_value.start.return_value = mock_playwright_instance
        mock_playwright_instance.chromium = mock_playwright.chromium

        manager = BrowserManager()
        manager.start()
        page = manager.create_mobile_page()

        assert page == mock_page
        mock_browser.new_context.assert_called_once()
        # Verify mobile viewport was used
        call_kwargs = mock_browser.new_context.call_args[1]
        assert call_kwargs["viewport"]["width"] == 390
        assert call_kwargs["viewport"]["height"] == 844

    @patch("core.browser.browser_manager.sync_playwright")
    def test_close(self, mock_sync_playwright):
        """Test browser cleanup."""
        # Setup mocks
        mock_playwright_instance = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_playwright = MagicMock()
        mock_playwright.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_sync_playwright.return_value.start.return_value = mock_playwright_instance
        mock_playwright_instance.chromium = mock_playwright.chromium

        manager = BrowserManager()
        manager.start()
        manager.create_page()
        manager.close()

        mock_page.close.assert_called_once()
        mock_context.close.assert_called_once()
        mock_browser.close.assert_called_once()
        mock_playwright_instance.stop.assert_called_once()

    @patch("core.browser.browser_manager.sync_playwright")
    def test_context_manager(self, mock_sync_playwright):
        """Test context manager usage."""
        # Setup mocks
        mock_playwright_instance = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_playwright = MagicMock()
        mock_playwright.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_sync_playwright.return_value.start.return_value = mock_playwright_instance
        mock_playwright_instance.chromium = mock_playwright.chromium

        with BrowserManager() as manager:
            assert manager.playwright is not None
            assert manager.browser is not None
            assert manager.context is not None
            assert manager.page is not None

        # Verify cleanup
        mock_page.close.assert_called_once()
        mock_context.close.assert_called_once()
        mock_browser.close.assert_called_once()
        mock_playwright_instance.stop.assert_called_once()

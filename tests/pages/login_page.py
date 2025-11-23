"""Login page object for authentication testing."""

import contextlib
import json
from urllib.parse import urlparse

from loguru import logger
from playwright.sync_api import Page

from tests.pages.base_page import BasePage


class LoginPage(BasePage):
    """Page object for login interface."""

    def __init__(self, page: Page, test_config=None, browser_manager=None):
        """
        Initialize login page.

        Args:
            page: Playwright page instance
            test_config: TestConfig instance with credentials
            browser_manager: Optional BrowserManager instance for API requests
        """
        super().__init__(page)
        self.test_config = test_config
        self.browser_manager = browser_manager

    def login(self, use_sso: bool = False) -> bool:
        """
        Perform login via UI or API.

        Args:
            use_sso: If True, use SSO login. If False, use email/password login.

        Returns:
            True if login successful, False otherwise
        """
        if use_sso:
            return self._login_via_sso()
        else:
            return self._login_via_api()

    def _login_via_api(self) -> bool:
        """Login via API and inject credentials into cookies and localStorage."""
        if not self.test_config or not self.test_config.email or not self.test_config.password:
            return False

        try:
            playwright = self.browser_manager.get_playwright()
            request_context = playwright.request.new_context(base_url=self.test_config.base_url)
            response = request_context.post(
                "/api/v1/auths/signin",
                headers={
                    "Accept": "*/*",
                    "Content-Type": "application/json",
                    "Origin": self.test_config.base_url,
                    "Referer": self.test_config.base_url + "/auth?redirect=%2F",
                },
                data=json.dumps(
                    {"email": self.test_config.email, "password": self.test_config.password}
                ),
            )
            if response.status != 200:
                request_context.dispose()
                return False

            response_data = response.json()
            token = response_data.get("token")
            session_id = response_data.get("session_id")

            # Extract and set cookies
            domain = urlparse(self.test_config.base_url).netloc.split(":")[0]
            cookies = []
            for header in response.headers_array:
                if header["name"].lower() == "set-cookie":
                    name = header["value"].split("=")[0].strip()
                    if name in ["DGE-X-SESSION-ID", "oui-session", "token"]:
                        value = header["value"].split(";")[0].split("=", 1)[1]
                        cookies.append(
                            {"name": name, "value": value, "domain": domain, "path": "/"}
                        )

            if cookies:
                self.page.context.add_cookies(cookies)
                logger.info(f"✓ Set {len(cookies)} cookies")

            # Set localStorage
            if token:
                self.page.evaluate(
                    """([token, sessionId]) => {
                        window.localStorage.setItem("token", token);
                        if (sessionId) window.localStorage.setItem("session_id", sessionId);
                    }""",
                    [token, session_id if session_id else ""],
                )

            request_context.dispose()
            logger.info("✓ API login successful")
            return True

        except Exception as e:
            logger.error(f"API login error: {e}")
            with contextlib.suppress(Exception):
                self.page.screenshot(path="screenshots/login_error.png", full_page=True)
            return False

    def _login_via_sso(self) -> bool:
        """SSO login (not implemented)."""
        logger.warning("SSO login not implemented")
        return False

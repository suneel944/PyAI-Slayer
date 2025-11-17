"""Login page object for chatbot authentication."""

from loguru import logger
from playwright.sync_api import Page

from tests.pages.base_page import BasePage
from tests.pages.locators import LoginLocators, PollingConfig


class LoginPage(BasePage):
    """Page object for login functionality."""

    locators = LoginLocators()
    config = PollingConfig()

    def __init__(self, page: Page, test_config=None):
        """
        Initialize login page.

        Args:
            page: Playwright page instance
            test_config: TestConfig instance (optional for backwards compatibility)
        """
        super().__init__(page)
        self.test_config = test_config

    def login(
        self, email: str | None = None, password: str | None = None, use_sso: bool = False
    ) -> bool:  # noqa: ARG002
        """
        Perform login using email/password (SSO not available for testing).

        Args:
            email: Email address (defaults to test_config)
            password: Password (defaults to test_config)
            use_sso: Ignored - always uses email/password login

        Returns:
            True if login successful
        """
        try:
            email = email or (self.test_config.email if self.test_config else None)
            password = password or (self.test_config.password if self.test_config else None)

            if not email or not password:
                logger.error("Email and password required for login")
                return False

            logger.info(f"Attempting email/password login for: {email}")

            logger.info("Step 1: Clicking 'Log in with email' button...")
            self.wait_for_element(
                self.locators.EMAIL_LOGIN_BUTTON, timeout=self.config.DEFAULT_TIMEOUT_MS
            )
            self.click(self.locators.EMAIL_LOGIN_BUTTON)
            logger.info("✓ Clicked 'Log in with email' button")

            logger.info("Step 2: Waiting for email/password form to appear...")
            self.page.wait_for_timeout(self.config.FORM_ANIMATION_WAIT_MS)

            logger.info("Step 3: Filling email...")
            if not self.fill_form_field(self.locators.EMAIL_INPUT, email):
                return False
            logger.info("✓ Email filled")

            logger.info("Step 4: Filling password...")
            if not self.fill_form_field(self.locators.PASSWORD_INPUT, password):
                return False
            logger.info("✓ Password filled")

            logger.info("Step 5: Submitting login form...")
            if not self.submit_form(self.locators.LOGIN_BUTTON, fallback_key="Enter"):
                return False
            logger.info("✓ Login button clicked")

            logger.info("Step 6: Waiting for login to complete...")
            self.page.wait_for_timeout(self.config.NAVIGATION_WAIT_MS)
            self.wait_for_url_change(exclude_pattern="/auth", timeout=self.config.LONG_TIMEOUT_MS)

            error_element = self.page.query_selector(self.locators.ERROR_MESSAGE)
            if error_element:
                error_text = error_element.text_content()
                logger.error(f"Login failed: {error_text}")
                return False

            if self.is_on_url("auth") or self.is_on_url("login"):
                logger.warning("Still on login page after login attempt - login may have failed")
                return False

            logger.info("✓ Login successful - redirected away from login page")
            return True
        except Exception as e:
            logger.error(f"Login error: {e}")
            import traceback

            traceback.print_exc()

            try:
                self.page.screenshot(path="screenshots/login_error.png", full_page=True)
                logger.info("Error screenshot saved to screenshots/login_error.png")
            except Exception:
                pass
            return False

    def is_logged_in(self) -> bool:
        """Check if user is logged in."""

        indicators = [self.locators.LOGOUT_BUTTON, self.locators.USER_MENU]

        for indicator in indicators:
            try:
                if self.page.query_selector(indicator):
                    return True
            except Exception:
                continue

        return False

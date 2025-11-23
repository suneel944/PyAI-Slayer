"""Localization helper for RTL/LTR handling and language switching."""

from loguru import logger
from playwright.sync_api import Page


class LocalizationHelper:
    """Helper for handling multilingual and RTL/LTR layouts."""

    LANGUAGES = {
        "en": {"code": "en", "name": "English", "rtl": False, "direction": "ltr"},
        "ar": {"code": "ar", "name": "Arabic", "rtl": True, "direction": "rtl"},
    }

    def __init__(self):
        """Initialize localization helper."""
        logger.info("Localization helper initialized")

    def get_language_config(self, lang_code: str) -> dict | None:
        """Get configuration for a language."""
        return self.LANGUAGES.get(lang_code)

    def verify_rtl_layout(self, page: Page) -> bool:
        """
        Verify RTL layout is applied correctly.

        Args:
            page: Playwright page object

        Returns:
            True if RTL layout is correct
        """
        try:
            body_direction = page.evaluate(
                """
                () => {
                    const body = document.body;
                    return window.getComputedStyle(body).direction;
                }
            """
            )

            container_direction = page.evaluate(
                """
                () => {
                    const containers = document.querySelectorAll('.chat-container, .main-content');
                    for (let container of containers) {
                        const dir = window.getComputedStyle(container).direction;
                        if (dir === 'rtl') return 'rtl';
                    }
                    return 'ltr';
                }
            """
            )

            is_rtl = body_direction == "rtl" or container_direction == "rtl"
            return bool(is_rtl)
        except Exception as e:
            logger.error(f"Error verifying RTL layout: {e}")
            return False

    def get_current_language(self, page: Page) -> str | None:
        """
        Detect current language from page.

        Args:
            page: Playwright page object

        Returns:
            Language code or None
        """
        try:
            html_lang = page.evaluate("() => document.documentElement.lang")

            body_direction = page.evaluate(
                """
                () => window.getComputedStyle(document.body).direction
            """
            )

            if html_lang:
                lang_code = str(html_lang)[:2]
                return lang_code
            elif body_direction == "rtl":
                return "ar"
            else:
                return None
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return None

    def is_rtl_language(self, lang_code: str) -> bool:
        """Check if language is RTL."""
        lang_config = self.get_language_config(lang_code)
        return bool(lang_config["rtl"]) if lang_config else False

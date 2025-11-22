"""Locators composition layer - static selectors and constants for all pages."""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ChatLocators:
    """Static locators for chat page elements."""

    MESSAGE_INPUT: ClassVar[str] = (
        "input[type='text'], textarea, [contenteditable='true'], "
        "input[placeholder*='message'], input[placeholder*='Message']"
    )
    SEND_BUTTON: ClassVar[str] = "#send-message-button"
    SEND_BUTTON_ENABLED: ClassVar[str] = "#send-message-button:not([disabled])"

    MESSAGE_LIST: ClassVar[str] = ".chat-messages, .messages, [role='log'], .conversation"
    MESSAGE_ITEM: ClassVar[str] = ".message, .chat-message, [data-message]"
    AI_RESPONSE: ClassVar[str] = ".ai-message, .bot-message, .response, [data-role='assistant']"
    USER_MESSAGE: ClassVar[str] = ".user-message, [data-role='user']"

    LOADING_SPINNER: ClassVar[str] = ".loading, .spinner, [aria-label='Loading']"
    STOP_GENERATING_BUTTON: ClassVar[str] = (
        "button:has-text('Stop generating'), button:has-text('Stop'), "
        "[aria-label*='Stop'], button[title*='Stop']"
    )
    COPY_BUTTON: ClassVar[str] = "button[@aria-label='Copy']"
    COPY_BUTTON_XPATH: ClassVar[str] = "//button[@aria-label='Copy']"


@dataclass(frozen=True)
class PollingConfig:
    """Configuration constants for polling mechanisms."""

    DEFAULT_AI_TIMEOUT_MS: ClassVar[int] = 180000
    POLL_INTERVAL_SEC: ClassVar[float] = 0.5
    STABILITY_DURATION_SEC: ClassVar[float] = 3.0
    STABILITY_ITERATIONS: ClassVar[int] = 2
    MAX_NO_RESPONSE_TIME_SEC: ClassVar[float] = 60.0

    DEFAULT_TIMEOUT_MS: ClassVar[int] = 30000
    SHORT_TIMEOUT_MS: ClassVar[int] = 5000
    LONG_TIMEOUT_MS: ClassVar[int] = 15000

    FORM_ANIMATION_WAIT_MS: ClassVar[int] = 1000
    INPUT_REGISTER_WAIT_MS: ClassVar[int] = 500
    NAVIGATION_WAIT_MS: ClassVar[int] = 2000

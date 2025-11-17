"""Event system for test lifecycle and framework events."""

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger


class EventType(Enum):
    """Event types in the framework."""

    # Test lifecycle events
    TEST_STARTED = "test_started"
    TEST_COMPLETED = "test_completed"
    TEST_FAILED = "test_failed"
    TEST_SKIPPED = "test_skipped"

    # Browser events
    BROWSER_STARTED = "browser_started"
    BROWSER_CLOSED = "browser_closed"
    PAGE_CREATED = "page_created"
    PAGE_CLOSED = "page_closed"

    # Validation events
    VALIDATION_STARTED = "validation_started"
    VALIDATION_COMPLETED = "validation_completed"
    VALIDATION_FAILED = "validation_failed"

    # Security events
    SECURITY_TEST_STARTED = "security_test_started"
    SECURITY_TEST_COMPLETED = "security_test_completed"
    VULNERABILITY_DETECTED = "vulnerability_detected"

    # Custom events
    CUSTOM = "custom"


@dataclass
class Event:
    """Event data structure."""

    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    data: dict[str, Any] = field(default_factory=dict)
    source: str = "framework"

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "source": self.source,
        }


class EventEmitter:
    """Event emitter for observer pattern."""

    def __init__(self):
        """Initialize event emitter."""
        self._listeners: dict[EventType, list[Callable[[Event], None]]] = {}
        self._event_history: list[Event] = []
        self._max_history: int = 1000

    def on(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """
        Register an event listener.

        Args:
            event_type: Type of event to listen for
            callback: Callback function to call when event occurs
        """
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)

    def off(self, event_type: EventType, callback: Callable[[Event], None] | None = None) -> None:
        """
        Unregister an event listener.

        Args:
            event_type: Type of event
            callback: Specific callback to remove (None to remove all)
        """
        if event_type in self._listeners:
            if callback:
                with contextlib.suppress(ValueError):
                    self._listeners[event_type].remove(callback)
            else:
                self._listeners[event_type].clear()

    def emit(self, event: Event) -> None:
        """
        Emit an event to all registered listeners.

        Args:
            event: Event to emit
        """
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)

        # Notify listeners
        listeners = self._listeners.get(event.event_type, [])
        for listener in listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Error in event listener for {event.event_type.value}: {e}")

    def emit_simple(
        self, event_type: EventType, data: dict[str, Any] | None = None, source: str = "framework"
    ) -> None:
        """
        Emit a simple event.

        Args:
            event_type: Type of event
            data: Event data
            source: Event source
        """
        event = Event(event_type=event_type, data=data or {}, source=source)
        self.emit(event)

    def get_history(self, event_type: EventType | None = None) -> list[Event]:
        """
        Get event history.

        Args:
            event_type: Filter by event type (None for all)

        Returns:
            List of events
        """
        if event_type:
            return [e for e in self._event_history if e.event_type == event_type]
        return self._event_history.copy()

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()

    def get_listener_count(self, event_type: EventType) -> int:
        """
        Get number of listeners for an event type.

        Args:
            event_type: Event type

        Returns:
            Number of listeners
        """
        return len(self._listeners.get(event_type, []))


# Global event emitter
_event_emitter: EventEmitter | None = None


def get_event_emitter() -> EventEmitter:
    """Get global event emitter instance."""
    global _event_emitter
    if _event_emitter is None:
        _event_emitter = EventEmitter()
    return _event_emitter


def reset_event_emitter() -> None:
    """Reset global event emitter (useful for testing)."""
    global _event_emitter
    _event_emitter = None

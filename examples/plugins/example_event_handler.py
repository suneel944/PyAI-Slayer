"""Example event handler plugin."""

from core.infrastructure.events import Event, EventEmitter, EventType
from core.validation.plugins import EventPlugin


class ExampleEventHandlerPlugin(EventPlugin):
    """Example plugin that handles events."""

    def get_name(self) -> str:
        """Get plugin name."""
        return "example_event_handler"

    def get_version(self) -> str:
        """Get plugin version."""
        return "1.0.0"

    def initialize(self, container) -> None:
        """Initialize plugin with DI container."""
        # Plugin initialization logic here
        pass

    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        # Plugin cleanup logic here
        pass

    def register_handlers(self, emitter: EventEmitter) -> None:
        """Register event handlers."""
        # Register handler for test events
        emitter.on(EventType.TEST_STARTED, self._on_test_started)
        emitter.on(EventType.TEST_COMPLETED, self._on_test_completed)
        emitter.on(EventType.TEST_FAILED, self._on_test_failed)

        print("Registered event handlers for test lifecycle")

    def _on_test_started(self, event: Event) -> None:
        """Handle test started event."""
        test_name = event.data.get("test_name", "unknown")
        print(f"[Event Handler] Test started: {test_name}")

    def _on_test_completed(self, event: Event) -> None:
        """Handle test completed event."""
        test_name = event.data.get("test_name", "unknown")
        duration = event.data.get("duration", 0)
        print(f"[Event Handler] Test completed: {test_name} (duration: {duration:.2f}s)")

    def _on_test_failed(self, event: Event) -> None:
        """Handle test failed event."""
        test_name = event.data.get("test_name", "unknown")
        error = event.data.get("error", "unknown error")
        print(f"[Event Handler] Test failed: {test_name} - {error}")

# PyAI-Slayer Plugins Guide

## Overview

PyAI-Slayer provides a powerful plugin system that allows you to extend the framework with custom validation strategies, event handlers, and other functionality. This guide will show you how to create, register, and use plugins.

## Plugin Types

The framework supports two main types of plugins:

1. **Validation Plugins** - Add custom validation strategies
2. **Event Plugins** - Handle framework events (test lifecycle, validation, etc.)

## Plugin Architecture

### Base Plugin Class

All plugins inherit from the `Plugin` base class:

```python
from core.validation.plugins import Plugin

class MyPlugin(Plugin):
    def get_name(self) -> str:
        """Return unique plugin name."""
        return "my_plugin"

    def get_version(self) -> str:
        """Return plugin version."""
        return "1.0.0"

    def initialize(self, container) -> None:
        """Initialize plugin with DI container."""
        pass

    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass
```

## Creating a Validation Plugin

Validation plugins allow you to add custom validation strategies that can be used during test execution.

### Step 1: Create a Validation Strategy

First, create a custom validation strategy:

```python
from core.validation.validation_strategy import ValidationStrategy

class LengthValidationStrategy(ValidationStrategy):
    """Custom validation strategy for response length."""

    def validate(self, query: str, response: str, **kwargs):
        """Validate response length."""
        min_length = kwargs.get("min_length", 10)
        max_length = kwargs.get("max_length", 1000)

        is_valid = min_length <= len(response) <= max_length
        return is_valid, {
            "strategy": "length",
            "length": len(response),
            "min_length": min_length,
            "max_length": max_length,
        }

    def get_name(self) -> str:
        """Get strategy name."""
        return "length"
```

### Step 2: Create the Plugin

Create a plugin that registers your strategy:

```python
from core.validation.plugins import ValidationPlugin
from core.validation.validation_strategy import ValidationStrategyRegistry

class MyValidationPlugin(ValidationPlugin):
    """Plugin that adds custom validation strategies."""

    def get_name(self) -> str:
        return "my_validation_plugin"

    def get_version(self) -> str:
        return "1.0.0"

    def initialize(self, container) -> None:
        """Initialize plugin."""
        pass

    def cleanup(self) -> None:
        """Cleanup plugin."""
        pass

    def register_strategies(self, registry: ValidationStrategyRegistry) -> None:
        """Register custom validation strategies."""
        strategy = LengthValidationStrategy()
        registry.register(strategy)
        print(f"Registered validation strategy: {strategy.get_name()}")
```

### Step 3: Register the Plugin

Register your plugin in your test setup or conftest.py:

```python
from core.validation.plugins import get_plugin_manager
from examples.plugins.example_custom_validator import ExampleCustomValidatorPlugin

# Get plugin manager
plugin_manager = get_plugin_manager()

# Register plugin
plugin = ExampleCustomValidatorPlugin()
plugin_manager.register_plugin(plugin)
```

### Step 4: Use the Validation Strategy

Use your custom strategy in tests:

```python
from core.validation.validation_strategy import get_strategy_registry

# Get strategy registry
registry = get_strategy_registry()

# Use your custom strategy
is_valid, details = registry.validate(
    "length",  # Strategy name
    query="What is AI?",
    response="AI is artificial intelligence...",
    min_length=20,
    max_length=500
)

print(f"Validation result: {is_valid}")
print(f"Details: {details}")
```

## Creating an Event Plugin

Event plugins allow you to react to framework events like test start, completion, failures, etc.

### Step 1: Create the Event Plugin

```python
from core.infrastructure.events import Event, EventEmitter, EventType
from core.validation.plugins import EventPlugin

class MyEventHandlerPlugin(EventPlugin):
    """Plugin that handles framework events."""

    def get_name(self) -> str:
        return "my_event_handler"

    def get_version(self) -> str:
        return "1.0.0"

    def initialize(self, container) -> None:
        """Initialize plugin."""
        pass

    def cleanup(self) -> None:
        """Cleanup plugin."""
        pass

    def register_handlers(self, emitter: EventEmitter) -> None:
        """Register event handlers."""
        # Register handler for test events
        emitter.on(EventType.TEST_STARTED, self._on_test_started)
        emitter.on(EventType.TEST_COMPLETED, self._on_test_completed)
        emitter.on(EventType.TEST_FAILED, self._on_test_failed)

    def _on_test_started(self, event: Event) -> None:
        """Handle test started event."""
        test_name = event.data.get("test_name", "unknown")
        print(f"[Plugin] Test started: {test_name}")

    def _on_test_completed(self, event: Event) -> None:
        """Handle test completed event."""
        test_name = event.data.get("test_name", "unknown")
        duration = event.data.get("duration", 0)
        print(f"[Plugin] Test completed: {test_name} (duration: {duration:.2f}s)")

    def _on_test_failed(self, event: Event) -> None:
        """Handle test failed event."""
        test_name = event.data.get("test_name", "unknown")
        error = event.data.get("error", "unknown error")
        print(f"[Plugin] Test failed: {test_name} - {error}")
```

### Step 2: Register the Event Plugin

```python
from core.validation.plugins import get_plugin_manager
from core.infrastructure.events import get_event_emitter
from examples.plugins.example_event_handler import ExampleEventHandlerPlugin

# Get plugin manager with event emitter
event_emitter = get_event_emitter()
plugin_manager = get_plugin_manager(event_emitter=event_emitter)

# Register plugin
plugin = ExampleEventHandlerPlugin()
plugin_manager.register_plugin(plugin)
```

## Available Event Types

The framework emits the following events:

**Test Lifecycle Events:**
- `EventType.TEST_STARTED` - When a test starts
- `EventType.TEST_COMPLETED` - When a test completes successfully
- `EventType.TEST_FAILED` - When a test fails
- `EventType.TEST_SKIPPED` - When a test is skipped

**Browser Events:**
- `EventType.BROWSER_STARTED` - When browser is started
- `EventType.BROWSER_CLOSED` - When browser is closed
- `EventType.PAGE_CREATED` - When a new page is created
- `EventType.PAGE_CLOSED` - When a page is closed

**Validation Events:**
- `EventType.VALIDATION_STARTED` - When validation begins
- `EventType.VALIDATION_COMPLETED` - When validation completes
- `EventType.VALIDATION_FAILED` - When validation fails

**Security Events:**
- `EventType.SECURITY_TEST_STARTED` - When security test starts
- `EventType.SECURITY_TEST_COMPLETED` - When security test completes
- `EventType.VULNERABILITY_DETECTED` - When a vulnerability is detected

**Custom Events:**
- `EventType.CUSTOM` - For custom application events

## Loading Plugins

### Method 1: Manual Registration

Register plugins directly in your code:

```python
from core.validation.plugins import get_plugin_manager
from my_plugins.my_validator import MyValidationPlugin

plugin_manager = get_plugin_manager()
plugin = MyValidationPlugin()
plugin_manager.register_plugin(plugin)
```

### Method 2: Load from Module

Load plugins from a Python module:

```python
from core.validation.plugins import get_plugin_manager

plugin_manager = get_plugin_manager()
plugin_manager.load_plugin_from_module("my_plugins.custom_validator")
```

### Method 3: Load from Directory

Load all plugins from a directory:

```python
from pathlib import Path
from core.validation.plugins import get_plugin_manager

plugin_manager = get_plugin_manager()
plugin_dir = Path("my_plugins")
plugin_manager.load_plugins_from_directory(plugin_dir)
```

## Plugin Manager API

### Get Plugin Manager

```python
from core.validation.plugins import get_plugin_manager
from core.infrastructure.events import get_event_emitter
from core.infrastructure.dependency_injection import get_container

# With DI container and event emitter
container = get_container()
event_emitter = get_event_emitter()
plugin_manager = get_plugin_manager(container=container, event_emitter=event_emitter)

# Or use defaults
plugin_manager = get_plugin_manager()
```

### Get Registered Plugins

```python
# Get all plugins
all_plugins = plugin_manager.get_all_plugins()

# Get specific plugin
plugin = plugin_manager.get_plugin("my_plugin")
```

### Unregister Plugin

```python
plugin_manager.unregister_plugin("my_plugin")
```

### Cleanup All Plugins

```python
plugin_manager.cleanup_all()
```

## Example: Complete Plugin Setup

Here's a complete example of setting up plugins in your test configuration:

```python
# conftest.py or test setup
from pathlib import Path
from core.validation.plugins import get_plugin_manager
from core.infrastructure.events import get_event_emitter
from core.infrastructure.dependency_injection import get_container

def pytest_configure(config):
    """Configure plugins during pytest setup."""
    # Get framework components
    container = get_container()
    event_emitter = get_event_emitter()

    # Initialize plugin manager
    plugin_manager = get_plugin_manager(
        container=container,
        event_emitter=event_emitter
    )

    # Load plugins from examples directory
    examples_dir = Path(__file__).parent.parent / "examples" / "plugins"
    plugin_manager.load_plugins_from_directory(examples_dir)

    # Or load specific plugins
    # plugin_manager.load_plugin_from_module("examples.plugins.example_custom_validator")

    print(f"Loaded {len(plugin_manager.get_all_plugins())} plugins")
```

## Using Plugins in Tests

### Using Custom Validation Strategies

```python
import pytest
from core.validation.validation_strategy import get_strategy_registry

def test_with_custom_validation():
    """Test using custom validation strategy."""
    registry = get_strategy_registry()

    # Use custom length validation
    is_valid, details = registry.validate(
        "length",
        query="Test query",
        response="This is a test response",
        min_length=10,
        max_length=100
    )

    assert is_valid, f"Validation failed: {details}"
    assert details["length"] == 24
```

### Using Multiple Strategies

```python
def test_with_multiple_strategies():
    """Test using multiple validation strategies."""
    registry = get_strategy_registry()

    query = "What is AI?"
    response = "AI is artificial intelligence..."

    # Validate with all strategies
    results = registry.validate_all(
        query=query,
        response=response,
        min_length=10,
        threshold=0.7
    )

    # Check results
    for strategy_name, (is_valid, details) in results.items():
        print(f"{strategy_name}: {is_valid} - {details}")
        assert is_valid, f"Strategy {strategy_name} failed"
```

## Best Practices

### 1. Plugin Naming

- Use descriptive, unique plugin names
- Follow naming convention: `{purpose}_{type}_plugin` (e.g., `custom_length_validator_plugin`)

### 2. Error Handling

Always handle errors gracefully in plugins:

```python
def register_strategies(self, registry: ValidationStrategyRegistry) -> None:
    """Register validation strategies with error handling."""
    try:
        strategy = LengthValidationStrategy()
        registry.register(strategy)
    except Exception as e:
        logger.error(f"Failed to register strategy: {e}")
        raise
```

### 3. Resource Cleanup

Always clean up resources in the `cleanup()` method:

```python
def cleanup(self) -> None:
    """Cleanup plugin resources."""
    # Close connections, files, etc.
    if hasattr(self, "_connection"):
        self._connection.close()
```

### 4. Plugin Versioning

Use semantic versioning for plugins:

```python
def get_version(self) -> str:
    """Return plugin version."""
    return "1.2.3"  # MAJOR.MINOR.PATCH
```

### 5. Logging

Use the framework's logger for plugin logging:

```python
from loguru import logger

def register_strategies(self, registry: ValidationStrategyRegistry) -> None:
    """Register strategies with logging."""
    logger.info("Registering custom validation strategies")
    # ... registration code
    logger.success("Successfully registered strategies")
```

## Example Plugins

The framework includes example plugins in `examples/plugins/`:

1. **example_custom_validator.py** - Shows how to create a validation plugin
2. **example_event_handler.py** - Shows how to create an event plugin

You can use these as templates for your own plugins.

## Troubleshooting

### Plugin Not Loading

- Check that the plugin class inherits from `Plugin` or a subclass
- Verify the module path is correct
- Check that the plugin directory is in Python path
- Look for import errors in logs

### Strategy Not Found

- Ensure the plugin is registered before using the strategy
- Check that `register_strategies()` is called correctly
- Verify the strategy name matches when calling `registry.validate()`

### Events Not Firing

- Ensure event emitter is passed to plugin manager
- Check that event handlers are registered in `register_handlers()`
- Verify events are being emitted by the framework

## Advanced Usage

### Plugin with DI Container

Plugins can access the DI container for dependency injection:

```python
def initialize(self, container) -> None:
    """Initialize plugin with DI container."""
    # Get dependencies from container
    self.validator = container.get("ai_validator")
    self.settings = container.get("settings")
```

### Conditional Plugin Loading

Load plugins conditionally based on configuration:

```python
from config.settings import settings

if settings.enable_custom_plugins:
    plugin_manager.load_plugins_from_directory(plugin_dir)
```

## Related Documentation

- [Framework Architecture](FRAMEWORK_ARCHITECTURE.md) - Understanding the framework structure
- [API Reference](api_reference.rst) - Complete API documentation
- [Validation Strategies](api_reference.rst#validation-strategies) - Built-in validation strategies

## Support

For plugin-related questions or issues:
- Check example plugins in `examples/plugins/`
- Review plugin source code in `src/core/validation/plugins.py`
- Open an issue on GitHub with plugin details

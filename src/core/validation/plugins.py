"""Plugin system for framework extensibility."""

import importlib
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger


class Plugin(ABC):
    """Base class for framework plugins."""

    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name."""
        pass

    @abstractmethod
    def get_version(self) -> str:
        """Get plugin version."""
        pass

    @abstractmethod
    def initialize(self, container: Any) -> None:
        """
        Initialize plugin with DI container.

        Args:
            container: DI container instance
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass


class ValidationPlugin(Plugin):
    """Plugin for custom validation strategies."""

    @abstractmethod
    def register_strategies(self, registry: Any) -> None:
        """
        Register validation strategies.

        Args:
            registry: ValidationStrategyRegistry instance
        """
        pass


class EventPlugin(Plugin):
    """Plugin for event handling."""

    @abstractmethod
    def register_handlers(self, emitter: Any) -> None:
        """
        Register event handlers.

        Args:
            emitter: EventEmitter instance
        """
        pass


class PluginManager:
    """Manages framework plugins."""

    def __init__(self, container: Any = None, event_emitter: Any = None):
        """
        Initialize plugin manager.

        Args:
            container: DI container instance
            event_emitter: EventEmitter instance
        """
        self.container = container
        self.event_emitter = event_emitter
        self._plugins: dict[str, Plugin] = {}
        self._plugin_paths: list[Path] = []

    def register_plugin(self, plugin: Plugin) -> None:
        """
        Register a plugin.

        Args:
            plugin: Plugin instance
        """
        name = plugin.get_name()
        if name in self._plugins:
            logger.warning(f"Plugin {name} already registered, overwriting")
        self._plugins[name] = plugin

        # Initialize plugin
        if self.container:
            plugin.initialize(self.container)

        # Register event handlers if applicable
        if isinstance(plugin, EventPlugin) and self.event_emitter:
            plugin.register_handlers(self.event_emitter)

        # Register validation strategies if applicable
        if isinstance(plugin, ValidationPlugin):
            from core.validation.validation_strategy import get_strategy_registry

            registry = get_strategy_registry()
            plugin.register_strategies(registry)

        logger.info(f"Registered plugin: {name} v{plugin.get_version()}")

    def load_plugin_from_module(self, module_path: str) -> None:
        """
        Load plugin from a Python module.

        Args:
            module_path: Dot-separated module path (e.g., 'my_plugins.custom_validator')
        """
        try:
            module = importlib.import_module(module_path)
            # Find all Plugin subclasses in module
            for _name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Plugin) and obj != Plugin:
                    plugin = obj()
                    self.register_plugin(plugin)
        except Exception as e:
            logger.error(f"Failed to load plugin from {module_path}: {e}")

    def load_plugins_from_directory(self, directory: Path) -> None:
        """
        Load plugins from a directory.

        Args:
            directory: Directory containing plugin modules
        """
        if not directory.exists():
            logger.warning(f"Plugin directory does not exist: {directory}")
            return

        for file_path in directory.glob("*.py"):
            if file_path.name.startswith("_"):
                continue

            module_name = file_path.stem
            try:
                # Try to import as package.module
                import sys

                if str(directory.parent) not in sys.path:
                    sys.path.insert(0, str(directory.parent))

                package_name = directory.name
                full_module_path = f"{package_name}.{module_name}"
                self.load_plugin_from_module(full_module_path)
            except Exception as e:
                logger.error(f"Failed to load plugin from {file_path}: {e}")

    def get_plugin(self, name: str) -> Plugin | None:
        """
        Get a plugin by name.

        Args:
            name: Plugin name

        Returns:
            Plugin instance or None
        """
        return self._plugins.get(name)

    def get_all_plugins(self) -> dict[str, Plugin]:
        """Get all registered plugins."""
        return self._plugins.copy()

    def unregister_plugin(self, name: str) -> None:
        """
        Unregister a plugin.

        Args:
            name: Plugin name
        """
        if name in self._plugins:
            plugin = self._plugins[name]
            plugin.cleanup()
            del self._plugins[name]
            logger.info(f"Unregistered plugin: {name}")

    def cleanup_all(self) -> None:
        """Cleanup all plugins."""
        for plugin in self._plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up plugin {plugin.get_name()}: {e}")
        self._plugins.clear()


# Global plugin manager
_plugin_manager: Any | None = None


def get_plugin_manager(container: Any = None, event_emitter: Any = None) -> PluginManager:
    """Get global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager(container=container, event_emitter=event_emitter)
    return _plugin_manager

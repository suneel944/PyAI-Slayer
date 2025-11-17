"""Validation and localization modules."""

from core.validation.localization_helper import LocalizationHelper
from core.validation.plugins import (
    EventPlugin,
    Plugin,
    PluginManager,
    ValidationPlugin,
    get_plugin_manager,
)
from core.validation.validation_strategy import (
    QualityValidationStrategy,
    SchemaValidationStrategy,
    SemanticValidationStrategy,
    ValidationStrategy,
    ValidationStrategyRegistry,
    get_strategy_registry,
)

__all__ = [
    # Validation strategies
    "ValidationStrategy",
    "SemanticValidationStrategy",
    "QualityValidationStrategy",
    "SchemaValidationStrategy",
    "ValidationStrategyRegistry",
    "get_strategy_registry",
    # Localization
    "LocalizationHelper",
    # Plugins
    "Plugin",
    "ValidationPlugin",
    "EventPlugin",
    "PluginManager",
    "get_plugin_manager",
]

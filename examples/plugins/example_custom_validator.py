"""Example custom validation plugin."""
from core.validation.plugins import ValidationPlugin
from core.validation.validation_strategy import ValidationStrategy, ValidationStrategyRegistry


class LengthValidationStrategy(ValidationStrategy):
    """Custom validation strategy for response length."""

    def validate(self, _query: str, response: str, **kwargs):
        """Validate response length."""
        min_length = kwargs.get('min_length', 10)
        max_length = kwargs.get('max_length', 1000)

        is_valid = min_length <= len(response) <= max_length
        return is_valid, {
            "strategy": "length",
            "length": len(response),
            "min_length": min_length,
            "max_length": max_length
        }

    def get_name(self) -> str:
        """Get strategy name."""
        return "length"


class ExampleCustomValidatorPlugin(ValidationPlugin):
    """Example plugin that adds custom validation strategies."""

    def get_name(self) -> str:
        """Get plugin name."""
        return "example_custom_validator"

    def get_version(self) -> str:
        """Get plugin version."""
        return "1.0.0"

    def register_strategies(self, registry: ValidationStrategyRegistry) -> None:
        """Register custom validation strategies."""
        strategy = LengthValidationStrategy()
        registry.register(strategy)
        print(f"Registered custom validation strategy: {strategy.get_name()}")


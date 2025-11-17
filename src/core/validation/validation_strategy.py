"""Strategy pattern for validation strategies."""

from abc import ABC, abstractmethod
from typing import Any

from loguru import logger


class ValidationStrategy(ABC):
    """Base class for validation strategies."""

    @abstractmethod
    def validate(self, query: str, response: str, **kwargs: Any) -> tuple[bool, dict[str, Any]]:  # noqa: ARG002
        """
        Validate a response.

        Args:
            query: Original query (may be unused in some implementations)
            response: Response to validate
            **kwargs: Additional validation parameters

        Returns:
            Tuple of (is_valid, details)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name."""
        pass


class SemanticValidationStrategy(ValidationStrategy):
    """Semantic similarity validation strategy."""

    def __init__(self, validator: Any, threshold: float = 0.7):
        """
        Initialize semantic validation strategy.

        Args:
            validator: AIResponseValidator instance
            threshold: Similarity threshold
        """
        self.validator = validator
        self.threshold = threshold

    def validate(self, query: str, response: str, **kwargs: Any) -> tuple[bool, dict[str, Any]]:
        """Validate using semantic similarity."""
        threshold = kwargs.get("threshold", self.threshold)
        is_relevant, similarity = self.validator.validate_relevance(
            query, response, threshold=threshold
        )

        return is_relevant, {
            "strategy": "semantic",
            "similarity": similarity,
            "threshold": threshold,
        }

    def get_name(self) -> str:
        """Get strategy name."""
        return "semantic"


class QualityValidationStrategy(ValidationStrategy):
    """Response quality validation strategy."""

    def __init__(self, validator: Any):
        """
        Initialize quality validation strategy.

        Args:
            validator: AIResponseValidator instance
        """
        self.validator = validator

    def validate(
        self,
        query: str,  # noqa: ARG002
        response: str,
        **kwargs: Any,
    ) -> tuple[bool, dict[str, Any]]:
        """Validate response quality."""
        min_length = kwargs.get("min_length", 10)
        max_length = kwargs.get("max_length")

        quality_checks = self.validator.validate_response_quality(
            response, min_length=min_length, max_length=max_length
        )

        is_valid = all(quality_checks.values())
        return is_valid, {"strategy": "quality", "checks": quality_checks}

    def get_name(self) -> str:
        """Get strategy name."""
        return "quality"


class SchemaValidationStrategy(ValidationStrategy):
    """Schema-based validation strategy."""

    def __init__(self, schema: dict[str, Any] | None = None):
        """
        Initialize schema validation strategy.

        Args:
            schema: JSON schema for validation
        """
        self.schema = schema

    def validate(
        self,
        query: str,  # noqa: ARG002
        response: str,
        **kwargs: Any,
    ) -> tuple[bool, dict[str, Any]]:
        """Validate response against schema."""
        import json

        schema = kwargs.get("schema", self.schema)
        if not schema:
            return True, {"strategy": "schema", "valid": True, "note": "No schema provided"}

        try:
            # Try to parse response as JSON
            response_data = json.loads(response) if isinstance(response, str) else response

            # Basic schema validation (can be extended with jsonschema)
            is_valid = self._validate_against_schema(response_data, schema)
            return is_valid, {"strategy": "schema", "valid": is_valid, "schema": schema}
        except json.JSONDecodeError:
            return False, {"strategy": "schema", "valid": False, "error": "Invalid JSON"}

    def _validate_against_schema(self, data: Any, schema: dict[str, Any]) -> bool:
        """Basic schema validation."""
        # This is a simplified version - can be extended with jsonschema
        if "required" in schema:
            required_fields = schema["required"]
            if isinstance(data, dict):
                return all(field in data for field in required_fields)
        return True

    def get_name(self) -> str:
        """Get strategy name."""
        return "schema"


class ValidationStrategyRegistry:
    """Registry for validation strategies."""

    def __init__(self):
        """Initialize strategy registry."""
        self._strategies: dict[str, ValidationStrategy] = {}

    def register(self, strategy: ValidationStrategy) -> None:
        """
        Register a validation strategy.

        Args:
            strategy: ValidationStrategy instance
        """
        name = strategy.get_name()
        self._strategies[name] = strategy

    def get(self, name: str) -> ValidationStrategy | None:
        """
        Get a validation strategy by name.

        Args:
            name: Strategy name

        Returns:
            ValidationStrategy instance or None
        """
        return self._strategies.get(name)

    def get_all(self) -> dict[str, ValidationStrategy]:
        """Get all registered strategies."""
        return self._strategies.copy()

    def validate(
        self, strategy_name: str, query: str, response: str, **kwargs: Any
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate using a specific strategy.

        Args:
            strategy_name: Name of strategy to use
            query: Original query
            response: Response to validate
            **kwargs: Additional parameters

        Returns:
            Tuple of (is_valid, details)

        Raises:
            ValueError: If strategy not found
        """
        strategy = self.get(strategy_name)
        if not strategy:
            raise ValueError(f"Validation strategy '{strategy_name}' not found")

        return strategy.validate(query, response, **kwargs)

    def validate_all(
        self, query: str, response: str, **kwargs: Any
    ) -> dict[str, tuple[bool, dict[str, Any]]]:
        """
        Validate using all registered strategies.

        Args:
            query: Original query
            response: Response to validate
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping strategy names to (is_valid, details) tuples
        """
        results = {}
        for name, strategy in self._strategies.items():
            try:
                results[name] = strategy.validate(query, response, **kwargs)
            except Exception as e:
                logger.error(f"Strategy {name} failed: {e}")
                results[name] = (False, {"error": str(e)})
        return results


# Global strategy registry
_strategy_registry: ValidationStrategyRegistry | None = None


def get_strategy_registry() -> ValidationStrategyRegistry:
    """Get global strategy registry."""
    global _strategy_registry
    if _strategy_registry is None:
        _strategy_registry = ValidationStrategyRegistry()
    return _strategy_registry

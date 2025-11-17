"""Dependency injection container for framework components."""

import contextlib
from collections.abc import Callable
from typing import Any, TypeVar, cast, get_type_hints

T = TypeVar("T")


class DIContainer:
    """Simple dependency injection container."""

    def __init__(self):
        """Initialize DI container."""
        self._services: dict[type, Any] = {}
        self._factories: dict[type, Callable[[], Any]] = {}
        self._singletons: dict[type, Any] = {}
        self._singleton_flags: dict[type, bool] = {}

    def register(
        self,
        service_type: type[T],
        implementation: T | Callable[[], T] | None = None,
        factory: Callable[[], T] | None = None,
        singleton: bool = True,
    ) -> None:
        """
        Register a service in the container.

        Args:
            service_type: Type/class to register
            implementation: Instance or callable to use
            factory: Factory function to create instances
            singleton: Whether to create singleton instances
        """
        if implementation is not None:
            if callable(implementation) and not isinstance(implementation, type):
                # It's a factory function
                self._factories[service_type] = implementation
            else:
                # It's an instance
                self._services[service_type] = implementation
            self._singleton_flags[service_type] = singleton
        elif factory is not None:
            self._factories[service_type] = factory
            self._singleton_flags[service_type] = singleton
        else:
            # Auto-register the type itself
            self._factories[service_type] = service_type
            self._singleton_flags[service_type] = singleton

    def register_instance(self, service_type: type[T], instance: T) -> None:
        """
        Register a singleton instance.

        Args:
            service_type: Type/class
            instance: Instance to register
        """
        self._services[service_type] = instance
        self._singleton_flags[service_type] = True

    def get(self, service_type: type[T]) -> T:
        """
        Get a service instance.

        Args:
            service_type: Type/class to resolve

        Returns:
            Service instance

        Raises:
            ValueError: If service not registered
        """
        # Check if instance already exists
        if service_type in self._services:
            return cast(T, self._services[service_type])

        # Check if singleton already created
        if service_type in self._singletons:
            return cast(T, self._singletons[service_type])

        # Check for factory
        if service_type in self._factories:
            factory = self._factories[service_type]
            instance = self._resolve_dependencies(factory)

            # Store as singleton if configured
            if self._singleton_flags.get(service_type, True):
                self._singletons[service_type] = instance
            else:
                self._services[service_type] = instance

            return cast(T, instance)

        raise ValueError(f"Service {service_type.__name__} not registered")

    def _resolve_dependencies(self, factory: Callable[[], T]) -> T:
        """
        Resolve dependencies for a factory function.

        Args:
            factory: Factory function

        Returns:
            Created instance
        """
        # Try to get type hints
        try:
            hints = get_type_hints(factory)
            if hints:
                # Resolve dependencies from type hints
                kwargs = {}
                for param_name, param_type in hints.items():
                    if param_name != "return" and param_type in (
                        self._services | self._factories | self._singletons
                    ):
                        with contextlib.suppress(ValueError):
                            kwargs[param_name] = self.get(
                                param_type
                            )  # Skip if dependency not available
                return factory(**kwargs) if kwargs else factory()
        except Exception:
            pass

        # Fallback to no-args call
        return factory()

    def has(self, service_type: type) -> bool:
        """
        Check if a service is registered.

        Args:
            service_type: Type/class to check

        Returns:
            True if registered
        """
        return service_type in (self._services | self._factories | self._singletons)

    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._singleton_flags.clear()

    def get_all_registered(self) -> list[type]:
        """Get all registered service types."""
        return list(
            set(self._services.keys()) | set(self._factories.keys()) | set(self._singletons.keys())
        )


# Global DI container instance
_container: DIContainer | None = None


def get_container() -> DIContainer:
    """Get global DI container instance."""
    global _container
    if _container is None:
        _container = DIContainer()
    return _container


def reset_container() -> None:
    """Reset global DI container (useful for testing)."""
    global _container
    _container = None

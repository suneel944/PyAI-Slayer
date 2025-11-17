"""Resource management with limits and cleanup."""

import threading
from contextlib import contextmanager
from typing import Any

from loguru import logger

from core.infrastructure.exceptions import ResourceError


class ResourceLimits:
    """Configuration for resource limits."""

    def __init__(
        self,
        max_browser_instances: int = 10,
        max_model_instances: int = 2,
        max_connections: int = 100,
        memory_limit_mb: int | None = None,
    ):
        """
        Initialize resource limits.

        Args:
            max_browser_instances: Maximum concurrent browser instances
            max_model_instances: Maximum concurrent model instances
            max_connections: Maximum concurrent connections
            memory_limit_mb: Memory limit in MB (None for no limit)
        """
        self.max_browser_instances = max_browser_instances
        self.max_model_instances = max_model_instances
        self.max_connections = max_connections
        self.memory_limit_mb = memory_limit_mb


class ResourceManager:
    """Manages framework resources with limits and cleanup."""

    def __init__(self, limits: ResourceLimits | None = None):
        """
        Initialize resource manager.

        Args:
            limits: Resource limits configuration
        """
        self.limits = limits or ResourceLimits()
        self._lock = threading.Lock()
        self._browser_count = 0
        self._model_count = 0
        self._connection_count = 0
        self._resources: dict[str, list[Any]] = {"browsers": [], "models": [], "connections": []}

    @contextmanager
    def acquire_browser(self):
        """
        Context manager for acquiring browser resource.

        Yields:
            Browser resource identifier

        Raises:
            ResourceError: If resource limit exceeded
        """
        with self._lock:
            if self._browser_count >= self.limits.max_browser_instances:
                raise ResourceError(
                    f"Browser limit exceeded: {self._browser_count}/{self.limits.max_browser_instances}",
                    resource_type="browser",
                )
            self._browser_count += 1
            resource_id = f"browser_{self._browser_count}"

        try:
            yield resource_id
        finally:
            with self._lock:
                self._browser_count -= 1

    @contextmanager
    def acquire_model(self):
        """
        Context manager for acquiring model resource.

        Yields:
            Model resource identifier

        Raises:
            ResourceError: If resource limit exceeded
        """
        with self._lock:
            if self._model_count >= self.limits.max_model_instances:
                raise ResourceError(
                    f"Model limit exceeded: {self._model_count}/{self.limits.max_model_instances}",
                    resource_type="model",
                )
            self._model_count += 1
            resource_id = f"model_{self._model_count}"

        try:
            yield resource_id
        finally:
            with self._lock:
                self._model_count -= 1

    def register_resource(self, resource_type: str, resource: Any):
        """
        Register a resource for tracking.

        Args:
            resource_type: Type of resource (browser, model, connection)
            resource: Resource instance
        """
        with self._lock:
            if resource_type not in self._resources:
                self._resources[resource_type] = []
            self._resources[resource_type].append(resource)

    def unregister_resource(self, resource_type: str, resource: Any):
        """
        Unregister a resource.

        Args:
            resource_type: Type of resource
            resource: Resource instance
        """
        with self._lock:
            if resource_type in self._resources:
                try:
                    self._resources[resource_type].remove(resource)
                except ValueError:
                    logger.warning(f"Resource {id(resource)} not found in {resource_type}")

    def cleanup_all(self, resource_type: str | None = None):
        """
        Cleanup all resources of a type or all resources.

        Args:
            resource_type: Type of resource to cleanup (None for all)
        """
        with self._lock:
            if resource_type:
                resources = self._resources.get(resource_type, [])
                logger.info(f"Cleaning up {len(resources)} {resource_type} resources")
                for resource in resources[:]:  # Copy list to avoid modification during iteration
                    self._cleanup_resource(resource_type, resource)
                self._resources[resource_type] = []
            else:
                logger.info("Cleaning up all resources")
                for rt, resources in self._resources.items():
                    for resource in resources[:]:
                        self._cleanup_resource(rt, resource)
                    self._resources[rt] = []

    def _cleanup_resource(self, resource_type: str, resource: Any):
        """
        Cleanup a single resource.

        Args:
            resource_type: Type of resource
            resource: Resource instance
        """
        try:
            if resource_type == "browser" and hasattr(resource, "close"):
                resource.close()
            elif resource_type == "model" and hasattr(resource, "unload"):
                resource.unload()
            elif hasattr(resource, "cleanup"):
                resource.cleanup()
            elif hasattr(resource, "close"):
                resource.close()
        except Exception as e:
            logger.warning(f"Error cleaning up {resource_type} resource: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get resource usage statistics."""
        with self._lock:
            return {
                "browser_count": self._browser_count,
                "model_count": self._model_count,
                "connection_count": self._connection_count,
                "browser_limit": self.limits.max_browser_instances,
                "model_limit": self.limits.max_model_instances,
                "connection_limit": self.limits.max_connections,
                "registered_resources": {
                    rt: len(resources) for rt, resources in self._resources.items()
                },
            }


# Global resource manager instance
_resource_manager: ResourceManager | None = None


def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager

"""Infrastructure and framework support modules."""

from core.infrastructure.cache import (
    Cache,
    EmbeddingCache,
    ModelCache,
    get_embedding_cache,
    get_model_cache,
)
from core.infrastructure.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    circuit_breaker,
)
from core.infrastructure.dependency_injection import DIContainer, get_container, reset_container
from core.infrastructure.events import (
    Event,
    EventEmitter,
    EventType,
    get_event_emitter,
    reset_event_emitter,
)
from core.infrastructure.exceptions import (
    BrowserError,
    ConfigurationError,
    FrameworkError,
    HealthCheckError,
    ModelError,
    ResourceError,
    ValidationError,
)
from core.infrastructure.health import HealthCheck, HealthChecker, HealthStatus
from core.infrastructure.resource_manager import (
    ResourceLimits,
    ResourceManager,
    get_resource_manager,
)
from core.infrastructure.retry import RetryableOperation, RetryConfig, retry_with_backoff
from core.infrastructure.test_distribution import (
    TestDistributor,
    TestMetadata,
    get_test_distributor,
)

__all__ = [
    # Caching
    "Cache",
    "ModelCache",
    "EmbeddingCache",
    "get_model_cache",
    "get_embedding_cache",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerError",
    "CircuitState",
    "circuit_breaker",
    # Dependency injection
    "DIContainer",
    "get_container",
    "reset_container",
    # Events
    "Event",
    "EventEmitter",
    "EventType",
    "get_event_emitter",
    "reset_event_emitter",
    # Exceptions
    "FrameworkError",
    "ValidationError",
    "BrowserError",
    "ModelError",
    "ConfigurationError",
    "ResourceError",
    "HealthCheckError",
    # Health checks
    "HealthCheck",
    "HealthChecker",
    "HealthStatus",
    # Resource management
    "ResourceManager",
    "ResourceLimits",
    "get_resource_manager",
    # Retry
    "RetryConfig",
    "RetryableOperation",
    "retry_with_backoff",
    # Test distribution
    "TestDistributor",
    "TestMetadata",
    "get_test_distributor",
]

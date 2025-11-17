"""Core framework components for PyAI-Slayer."""

# AI modules
from core.ai import (
    AdvancedHallucinationDetector,
    AIResponseValidator,
    Conversation,
    ConversationTester,
    ConversationTurn,
    HallucinationResult,
    RAGTester,
    RAGTestResult,
    get_conversation_tester,
    get_hallucination_detector,
    get_rag_tester,
)

# Browser modules
from core.browser import (
    BrowserManager,
    BrowserPool,
    get_browser_pool,
    reset_browser_pool,
)

# Infrastructure modules
# Infrastructure - Exceptions
# Infrastructure - Health
from core.infrastructure import (
    BrowserError,
    Cache,
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    ConfigurationError,
    DIContainer,
    EmbeddingCache,
    Event,
    EventEmitter,
    EventType,
    FrameworkError,
    HealthCheck,
    HealthChecker,
    HealthCheckError,
    HealthStatus,
    ModelCache,
    ModelError,
    ResourceError,
    ResourceLimits,
    ResourceManager,
    RetryableOperation,
    RetryConfig,
    TestDistributor,
    TestMetadata,
    ValidationError,
    circuit_breaker,
    get_container,
    get_embedding_cache,
    get_event_emitter,
    get_model_cache,
    get_resource_manager,
    get_test_distributor,
    reset_container,
    reset_event_emitter,
    retry_with_backoff,
)

# Observability modules
from core.observability import (
    PlaywrightTracer,
    PrometheusMetrics,
    get_playwright_tracer,
    get_prometheus_metrics,
    reset_playwright_tracer,
    reset_prometheus_metrics,
)

# Security modules
from core.security import (
    AdvancedPromptInjectionTester,
    InjectionTestResult,
    SecurityTester,
    get_prompt_injection_tester,
)

# Validation modules
from core.validation import (
    EventPlugin,
    LocalizationHelper,
    Plugin,
    PluginManager,
    QualityValidationStrategy,
    SchemaValidationStrategy,
    SemanticValidationStrategy,
    ValidationPlugin,
    ValidationStrategy,
    ValidationStrategyRegistry,
    get_plugin_manager,
    get_strategy_registry,
)

__all__ = [
    # AI components
    "AIResponseValidator",
    "Conversation",
    "ConversationTester",
    "ConversationTurn",
    "get_conversation_tester",
    "AdvancedHallucinationDetector",
    "HallucinationResult",
    "get_hallucination_detector",
    "RAGTester",
    "RAGTestResult",
    "get_rag_tester",
    # Security
    "SecurityTester",
    "AdvancedPromptInjectionTester",
    "InjectionTestResult",
    "get_prompt_injection_tester",
    # Browser
    "BrowserManager",
    "BrowserPool",
    "get_browser_pool",
    "reset_browser_pool",
    # Validation
    "LocalizationHelper",
    "ValidationStrategy",
    "SemanticValidationStrategy",
    "QualityValidationStrategy",
    "SchemaValidationStrategy",
    "ValidationStrategyRegistry",
    "get_strategy_registry",
    "Plugin",
    "ValidationPlugin",
    "EventPlugin",
    "PluginManager",
    "get_plugin_manager",
    # Infrastructure - Error handling
    "FrameworkError",
    "ValidationError",
    "BrowserError",
    "ModelError",
    "ConfigurationError",
    "ResourceError",
    "HealthCheckError",
    # Infrastructure - Resilience
    "RetryConfig",
    "RetryableOperation",
    "retry_with_backoff",
    "CircuitBreaker",
    "CircuitBreakerError",
    "CircuitState",
    "circuit_breaker",
    # Infrastructure - Health checks
    "HealthChecker",
    "HealthStatus",
    "HealthCheck",
    # Infrastructure - Resource management
    "ResourceManager",
    "ResourceLimits",
    "get_resource_manager",
    # Infrastructure - Dependency injection
    "DIContainer",
    "get_container",
    "reset_container",
    # Infrastructure - Events
    "Event",
    "EventEmitter",
    "EventType",
    "get_event_emitter",
    "reset_event_emitter",
    # Infrastructure - Caching
    "Cache",
    "ModelCache",
    "EmbeddingCache",
    "get_model_cache",
    "get_embedding_cache",
    # Infrastructure - Test distribution
    "TestDistributor",
    "TestMetadata",
    "get_test_distributor",
    # Observability
    "PlaywrightTracer",
    "get_playwright_tracer",
    "reset_playwright_tracer",
    "PrometheusMetrics",
    "get_prometheus_metrics",
    "reset_prometheus_metrics",
]

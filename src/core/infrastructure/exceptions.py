"""Custom exception hierarchy for PyAI-Slayer framework."""


class FrameworkError(Exception):
    """Base exception for all framework errors."""

    def __init__(self, message: str, details: dict | None = None):
        """
        Initialize framework error.

        Args:
            message: Error message
            details: Optional additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return formatted error message."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ValidationError(FrameworkError):
    """Raised when validation fails."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: str | None = None,
        expected: str | None = None,
        details: dict | None = None,
    ):
        """
        Initialize validation error.

        Args:
            message: Error message
            field: Field name that failed validation
            value: Actual value that failed
            expected: Expected value or format
            details: Optional additional error details
        """
        error_details = details or {}
        if field:
            error_details["field"] = field
        if value:
            error_details["value"] = value
        if expected:
            error_details["expected"] = expected
        super().__init__(message, error_details)
        self.field = field
        self.value = value
        self.expected = expected


class BrowserError(FrameworkError):
    """Raised when browser operations fail."""

    def __init__(
        self,
        message: str,
        browser_type: str | None = None,
        operation: str | None = None,
        details: dict | None = None,
    ):
        """
        Initialize browser error.

        Args:
            message: Error message
            browser_type: Type of browser (chromium, firefox, webkit)
            operation: Operation that failed
            details: Optional additional error details
        """
        error_details = details or {}
        if browser_type:
            error_details["browser_type"] = browser_type
        if operation:
            error_details["operation"] = operation
        super().__init__(message, error_details)
        self.browser_type = browser_type
        self.operation = operation


class ModelError(FrameworkError):
    """Raised when ML model operations fail."""

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        operation: str | None = None,
        details: dict | None = None,
    ):
        """
        Initialize model error.

        Args:
            message: Error message
            model_name: Name of the model that failed
            operation: Operation that failed (load, encode, etc.)
            details: Optional additional error details
        """
        error_details = details or {}
        if model_name:
            error_details["model_name"] = model_name
        if operation:
            error_details["operation"] = operation
        super().__init__(message, error_details)
        self.model_name = model_name
        self.operation = operation


class ConfigurationError(FrameworkError):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        config_value: str | None = None,
        details: dict | None = None,
    ):
        """
        Initialize configuration error.

        Args:
            message: Error message
            config_key: Configuration key that is invalid
            config_value: Configuration value that is invalid
            details: Optional additional error details
        """
        error_details = details or {}
        if config_key:
            error_details["config_key"] = config_key
        if config_value:
            error_details["config_value"] = config_value
        super().__init__(message, error_details)
        self.config_key = config_key
        self.config_value = config_value


class ResourceError(FrameworkError):
    """Raised when resource management fails."""

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        resource_id: str | None = None,
        details: dict | None = None,
    ):
        """
        Initialize resource error.

        Args:
            message: Error message
            resource_type: Type of resource (browser, model, connection)
            resource_id: Identifier of the resource
            details: Optional additional error details
        """
        error_details = details or {}
        if resource_type:
            error_details["resource_type"] = resource_type
        if resource_id:
            error_details["resource_id"] = resource_id
        super().__init__(message, error_details)
        self.resource_type = resource_type
        self.resource_id = resource_id


class HealthCheckError(FrameworkError):
    """Raised when health check fails."""

    def __init__(
        self,
        message: str,
        component: str | None = None,
        status: str | None = None,
        details: dict | None = None,
    ):
        """
        Initialize health check error.

        Args:
            message: Error message
            component: Component that failed health check
            status: Health status (unhealthy, degraded, etc.)
            details: Optional additional error details
        """
        error_details = details or {}
        if component:
            error_details["component"] = component
        if status:
            error_details["status"] = status
        super().__init__(message, error_details)
        self.component = component
        self.status = status

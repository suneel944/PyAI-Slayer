"""Unit tests for custom exceptions."""

from core import (
    BrowserError,
    ConfigurationError,
    FrameworkError,
    HealthCheckError,
    ModelError,
    ResourceError,
    ValidationError,
)


class TestFrameworkError:
    """Test suite for FrameworkError."""

    def test_basic_error(self):
        """Test basic framework error."""
        error = FrameworkError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.details == {}

    def test_error_with_details(self):
        """Test error with details."""
        error = FrameworkError("Test error", {"key": "value"})
        assert "key=value" in str(error)
        assert error.details == {"key": "value"}


class TestValidationError:
    """Test suite for ValidationError."""

    def test_validation_error(self):
        """Test validation error."""
        error = ValidationError(
            "Validation failed", field="similarity", value=0.5, expected=">= 0.7"
        )

        assert error.message == "Validation failed"
        assert error.field == "similarity"
        assert error.value == 0.5
        assert error.expected == ">= 0.7"
        # Details should contain the field name
        assert error.details["field"] == "similarity"


class TestBrowserError:
    """Test suite for BrowserError."""

    def test_browser_error(self):
        """Test browser error."""
        error = BrowserError("Browser failed", browser_type="chromium", operation="launch")

        assert error.message == "Browser failed"
        assert error.browser_type == "chromium"
        assert error.operation == "launch"
        assert error.details["browser_type"] == "chromium"


class TestModelError:
    """Test suite for ModelError."""

    def test_model_error(self):
        """Test model error."""
        error = ModelError("Model failed", model_name="test-model", operation="encode")

        assert error.message == "Model failed"
        assert error.model_name == "test-model"
        assert error.operation == "encode"


class TestConfigurationError:
    """Test suite for ConfigurationError."""

    def test_configuration_error(self):
        """Test configuration error."""
        error = ConfigurationError("Config invalid", config_key="base_url", config_value="invalid")

        assert error.message == "Config invalid"
        assert error.config_key == "base_url"
        assert error.config_value == "invalid"


class TestResourceError:
    """Test suite for ResourceError."""

    def test_resource_error(self):
        """Test resource error."""
        error = ResourceError(
            "Resource limit exceeded", resource_type="browser", resource_id="browser_1"
        )

        assert error.message == "Resource limit exceeded"
        assert error.resource_type == "browser"
        assert error.resource_id == "browser_1"


class TestHealthCheckError:
    """Test suite for HealthCheckError."""

    def test_health_check_error(self):
        """Test health check error."""
        error = HealthCheckError("Health check failed", component="model", status="unhealthy")

        assert error.message == "Health check failed"
        assert error.component == "model"
        assert error.status == "unhealthy"

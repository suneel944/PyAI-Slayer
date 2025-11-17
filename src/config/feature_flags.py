"""Feature flags system for gradual rollouts and A/B testing."""

import os
from typing import Any

from loguru import logger
from pydantic import BaseModel


class FeatureFlag(BaseModel):
    """Feature flag configuration."""

    name: str
    enabled: bool = False
    rollout_percentage: float = 0.0  # 0.0 to 100.0
    target_environments: list[str] = ["*"]  # ["dev", "staging", "production"] or ["*"] for all
    metadata: dict[str, Any] = {}


class FeatureFlags:
    """Feature flags manager."""

    def __init__(self):
        """Initialize feature flags."""
        self._flags: dict[str, FeatureFlag] = {}
        self._load_from_env()

    def _load_from_env(self):
        """Load feature flags from environment variables."""
        # Format: FEATURE_FLAG_<NAME>=true|false|percentage
        for key, value in os.environ.items():
            if key.startswith("FEATURE_FLAG_"):
                flag_name = key.replace("FEATURE_FLAG_", "").lower()
                try:
                    if value.lower() in ("true", "1", "yes"):
                        self.register(flag_name, enabled=True)
                    elif value.lower() in ("false", "0", "no"):
                        self.register(flag_name, enabled=False)
                    else:
                        # Try to parse as percentage
                        percentage = float(value)
                        self.register(flag_name, enabled=True, rollout_percentage=percentage)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid feature flag value for {flag_name}: {value}")

    def register(
        self,
        name: str,
        enabled: bool = False,
        rollout_percentage: float = 0.0,
        target_environments: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Register a feature flag.

        Args:
            name: Feature flag name
            enabled: Whether feature is enabled
            rollout_percentage: Percentage of users to enable for (0.0-100.0)
            target_environments: List of environments where flag applies
            metadata: Optional metadata
        """
        flag = FeatureFlag(
            name=name,
            enabled=enabled,
            rollout_percentage=rollout_percentage,
            target_environments=target_environments or ["*"],
            metadata=metadata or {},
        )
        self._flags[name] = flag

    def is_enabled(
        self, name: str, environment: str | None = None, user_id: str | None = None
    ) -> bool:
        """
        Check if a feature flag is enabled.

        Args:
            name: Feature flag name
            environment: Current environment (for filtering)
            user_id: User identifier for percentage-based rollouts

        Returns:
            True if feature is enabled
        """
        if name not in self._flags:
            return False

        flag = self._flags[name]

        # Check environment filter
        if (
            environment
            and "*" not in flag.target_environments
            and environment not in flag.target_environments
        ):
            return False

        # If not enabled, return False
        if not flag.enabled:
            return False

        # If rollout percentage is 0 or 100, use enabled flag
        if flag.rollout_percentage == 0.0:
            return flag.enabled
        if flag.rollout_percentage >= 100.0:
            return True

        # Percentage-based rollout
        if user_id:
            # Use hash of user_id for consistent assignment
            import hashlib

            hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            user_percentage = (hash_value % 10000) / 100.0
            return user_percentage < flag.rollout_percentage

        # If no user_id, use random for testing
        import random

        return random.random() * 100.0 < flag.rollout_percentage

    def get_flag(self, name: str) -> FeatureFlag | None:
        """
        Get feature flag configuration.

        Args:
            name: Feature flag name

        Returns:
            FeatureFlag instance or None if not found
        """
        return self._flags.get(name)

    def list_flags(self) -> dict[str, FeatureFlag]:
        """Get all registered feature flags."""
        return self._flags.copy()

    def enable(self, name: str):
        """Enable a feature flag."""
        if name in self._flags:
            self._flags[name].enabled = True
            logger.info(f"Enabled feature flag: {name}")
        else:
            logger.warning(f"Feature flag '{name}' not found")

    def disable(self, name: str):
        """Disable a feature flag."""
        if name in self._flags:
            self._flags[name].enabled = False
            logger.info(f"Disabled feature flag: {name}")
        else:
            logger.warning(f"Feature flag '{name}' not found")


# Global feature flags instance
_feature_flags: FeatureFlags | None = None


def get_feature_flags() -> FeatureFlags:
    """Get global feature flags instance."""
    global _feature_flags
    if _feature_flags is None:
        _feature_flags = FeatureFlags()
    return _feature_flags

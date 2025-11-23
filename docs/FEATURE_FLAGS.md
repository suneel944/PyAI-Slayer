# Feature Flags Usage Guide

The feature flags system allows you to enable/disable features gradually, perform A/B testing, and control feature rollouts across different environments.

## Quick Start

### 1. Using Environment Variables (Recommended)

Set feature flags via environment variables in your `.env` file or shell:

```bash
# Enable a feature completely
export FEATURE_FLAG_NEW_METRICS=true

# Disable a feature
export FEATURE_FLAG_EXPERIMENTAL_ANALYSIS=false

# Gradual rollout (25% of users)
export FEATURE_FLAG_LAZY_LOADING=25

# Gradual rollout (50% of users)
export FEATURE_FLAG_NEW_DASHBOARD=50
```

### 2. Basic Usage in Code

```python
from config import get_feature_flags

flags = get_feature_flags()

# Check if a feature is enabled
if flags.is_enabled("new_metrics"):
    # Use new metrics calculator
    from dashboard.metrics_engine import MetricsEngine
    engine = MetricsEngine()
else:
    # Use old metrics calculator
    from dashboard.metrics_calculator import MetricsCalculator
    calc = MetricsCalculator()
```

## Usage Patterns

### Pattern 1: Simple On/Off Feature

```python
from config import get_feature_flags

flags = get_feature_flags()

# Register the flag (or use environment variable)
flags.register("enable_safety_metrics", enabled=True)

# Use it
if flags.is_enabled("enable_safety_metrics"):
    # Calculate safety metrics
    safety_metrics = calculate_safety_metrics()
else:
    # Skip safety metrics
    safety_metrics = {}
```

### Pattern 2: Gradual Rollout (Percentage-Based)

```python
from config import get_feature_flags

flags = get_feature_flags()

# Enable for 10% of users initially
flags.register("new_ui", enabled=True, rollout_percentage=10.0)

# Check with user ID for consistent assignment
user_id = "user_12345"
if flags.is_enabled("new_ui", user_id=user_id):
    # Show new UI
    render_new_ui()
else:
    # Show old UI
    render_old_ui()
```

### Pattern 3: Environment-Specific Features

```python
from config import get_feature_flags
import os

flags = get_feature_flags()

# Only enable in development
flags.register(
    "debug_mode",
    enabled=True,
    target_environments=["dev", "staging"]
)

# Check with current environment
current_env = os.getenv("ENVIRONMENT", "production")
if flags.is_enabled("debug_mode", environment=current_env):
    # Enable debug logging
    logger.set_level("DEBUG")
```

### Pattern 4: Feature with Metadata

```python
from config import get_feature_flags

flags = get_feature_flags()

# Register with metadata
flags.register(
    "experimental_rag",
    enabled=True,
    rollout_percentage=5.0,
    metadata={
        "version": "2.0",
        "description": "New RAG implementation",
        "owner": "ai-team"
    }
)

# Access flag details
flag = flags.get_flag("experimental_rag")
if flag:
    print(f"Feature: {flag.name}")
    print(f"Version: {flag.metadata.get('version')}")
    print(f"Description: {flag.metadata.get('description')}")
```

## Real-World Examples for PyAI-Slayer

### Example 1: Enable/Disable Metric Groups

```python
from config import get_feature_flags
from dashboard.metrics_engine import MetricsEngine

flags = get_feature_flags()

# Check if safety metrics should be enabled
enable_safety = flags.is_enabled("enable_safety_metrics", default=False)

# Create metrics engine with conditional safety metrics
engine = MetricsEngine(
    enable_safety=enable_safety,
    enable_rag=flags.is_enabled("enable_rag_metrics", default=True),
    enable_agent=flags.is_enabled("enable_agent_metrics", default=False),
)
```

**Environment variable:**
```bash
FEATURE_FLAG_ENABLE_SAFETY_METRICS=true
FEATURE_FLAG_ENABLE_AGENT_METRICS=false
```

### Example 2: Gradual Rollout of New Metrics Calculator

```python
from config import get_feature_flags

flags = get_feature_flags()

# Check if user should get new metrics (25% rollout)
test_id = "test_abc123"  # Use test_id as user_id for consistent assignment
use_new_calculator = flags.is_enabled("new_metrics_calculator", user_id=test_id)

if use_new_calculator:
    from dashboard.metrics_engine import MetricsEngine
    calculator = MetricsEngine()
else:
    from dashboard.metrics_calculator import MetricsCalculator
    calculator = MetricsCalculator()
```

**Environment variable:**
```bash
FEATURE_FLAG_NEW_METRICS_CALCULATOR=25  # 25% rollout
```

### Example 3: Enable Lazy Loading for Heavy Models

```python
from config import get_feature_flags
from dashboard.metrics_engine import MetricsEngine
from dashboard.calculators.detectors import CompositeToxicityDetector

flags = get_feature_flags()

# Only enable toxicity detection in production (gradual rollout)
env = os.getenv("ENVIRONMENT", "production")
enable_toxicity = flags.is_enabled(
    "enable_toxicity_detection",
    environment=env,
    user_id=test_id
)

# Create detector with conditional enabling
toxicity_detector = CompositeToxicityDetector(
    enabled=enable_toxicity
)

engine = MetricsEngine(
    toxicity_detector=toxicity_detector
)
```

**Environment variable:**
```bash
FEATURE_FLAG_ENABLE_TOXICITY_DETECTION=50  # 50% rollout
```

### Example 4: A/B Testing Dashboard Features

```python
from config import get_feature_flags

flags = get_feature_flags()

# A/B test new dashboard layout
user_id = get_current_user_id()
use_new_layout = flags.is_enabled("new_dashboard_layout", user_id=user_id)

if use_new_layout:
    # Render new dashboard
    render_dashboard_v2()
else:
    # Render old dashboard
    render_dashboard_v1()
```

**Environment variable:**
```bash
FEATURE_FLAG_NEW_DASHBOARD_LAYOUT=30  # 30% of users see new layout
```

### Example 5: Feature Flag in Test Configuration

```python
# In conftest.py or test setup
from config import get_feature_flags

@pytest.fixture(autouse=True)
def setup_feature_flags():
    """Setup feature flags for tests."""
    flags = get_feature_flags()

    # Enable all features for testing
    flags.register("enable_safety_metrics", enabled=True)
    flags.register("enable_rag_metrics", enabled=True)
    flags.register("enable_agent_metrics", enabled=True)

    yield

    # Cleanup if needed
    flags.disable("enable_safety_metrics")
```

## API Reference

### `get_feature_flags() -> FeatureFlags`

Get the global feature flags instance.

```python
from config import get_feature_flags

flags = get_feature_flags()
```

### `FeatureFlags.register()`

Register a new feature flag programmatically.

```python
flags.register(
    name="my_feature",
    enabled=True,
    rollout_percentage=25.0,
    target_environments=["dev", "staging"],
    metadata={"version": "1.0"}
)
```

**Parameters:**
- `name` (str): Feature flag name
- `enabled` (bool): Whether feature is enabled (default: False)
- `rollout_percentage` (float): Percentage of users (0.0-100.0, default: 0.0)
- `target_environments` (list[str]): Environments where flag applies (default: ["*"])
- `metadata` (dict): Optional metadata

### `FeatureFlags.is_enabled()`

Check if a feature flag is enabled.

```python
enabled = flags.is_enabled(
    name="my_feature",
    environment="production",
    user_id="user_123"
)
```

**Parameters:**
- `name` (str): Feature flag name
- `environment` (str | None): Current environment for filtering
- `user_id` (str | None): User identifier for consistent percentage-based assignment

**Returns:** `bool` - True if feature is enabled

### `FeatureFlags.get_flag()`

Get feature flag configuration.

```python
flag = flags.get_flag("my_feature")
if flag:
    print(f"Enabled: {flag.enabled}")
    print(f"Rollout: {flag.rollout_percentage}%")
    print(f"Metadata: {flag.metadata}")
```

### `FeatureFlags.enable() / disable()`

Enable or disable a feature flag at runtime.

```python
flags.enable("my_feature")
flags.disable("my_feature")
```

### `FeatureFlags.list_flags()`

Get all registered feature flags.

```python
all_flags = flags.list_flags()
for name, flag in all_flags.items():
    print(f"{name}: {flag.enabled}")
```

## Environment Variable Format

Feature flags are automatically loaded from environment variables with the prefix `FEATURE_FLAG_`:

```bash
# Format: FEATURE_FLAG_<NAME>=<value>

# Boolean values
FEATURE_FLAG_MY_FEATURE=true
FEATURE_FLAG_MY_FEATURE=false
FEATURE_FLAG_MY_FEATURE=1    # true
FEATURE_FLAG_MY_FEATURE=0   # false
FEATURE_FLAG_MY_FEATURE=yes # true
FEATURE_FLAG_MY_FEATURE=no  # false

# Percentage rollout (0.0-100.0)
FEATURE_FLAG_MY_FEATURE=25   # 25% of users
FEATURE_FLAG_MY_FEATURE=50.5 # 50.5% of users
FEATURE_FLAG_MY_FEATURE=100  # All users (same as true)
```

**Note:** Flag names are converted to lowercase automatically:
- `FEATURE_FLAG_NEW_METRICS` → `"new_metrics"`
- `FEATURE_FLAG_ENABLE_SAFETY` → `"enable_safety"`

## Percentage-Based Rollouts

### How It Works

1. **With user_id**: Uses consistent hash-based assignment
   - Same user_id always gets the same result
   - Good for A/B testing and gradual rollouts

2. **Without user_id**: Uses random assignment
   - Different result each time
   - Good for testing only

### Example

```python
flags.register("new_feature", enabled=True, rollout_percentage=25.0)

# User A always gets the same result
user_a_result = flags.is_enabled("new_feature", user_id="user_a")
# Returns True or False consistently

# User B always gets the same result
user_b_result = flags.is_enabled("new_feature", user_id="user_b")
# Returns True or False consistently

# Without user_id, result is random
random_result = flags.is_enabled("new_feature")
# Returns True or False randomly (25% chance of True)
```

## Best Practices

### 1. Use Descriptive Flag Names

```python
# Good
flags.is_enabled("enable_safety_metrics")
flags.is_enabled("new_dashboard_layout")
flags.is_enabled("experimental_rag_v2")

# Bad
flags.is_enabled("flag1")
flags.is_enabled("test")
flags.is_enabled("new")
```

### 2. Provide Default Values

```python
# Good: Provide default if flag doesn't exist
enabled = flags.is_enabled("my_feature") or False

# Or use get_flag() to check existence first
flag = flags.get_flag("my_feature")
enabled = flag.enabled if flag else False
```

### 3. Use Consistent User IDs

```python
# Good: Use consistent identifier (test_id, user_id, session_id)
test_id = "test_abc123"
enabled = flags.is_enabled("feature", user_id=test_id)

# Bad: Random or changing identifier
enabled = flags.is_enabled("feature", user_id=str(random.randint(1, 1000)))
```

### 4. Document Feature Flags

```python
# Register with metadata for documentation
flags.register(
    "experimental_metrics",
    enabled=True,
    rollout_percentage=10.0,
    metadata={
        "description": "New metrics calculation engine",
        "owner": "ai-team",
        "jira_ticket": "AI-123",
        "rollout_plan": "10% -> 25% -> 50% -> 100%"
    }
)
```

### 5. Clean Up After Testing

```python
# In test cleanup
def teardown():
    flags = get_feature_flags()
    flags.disable("test_feature")
    # Or remove from environment variables
```

## Common Use Cases

### Use Case 1: Feature Toggle

```python
# Enable/disable entire feature
if flags.is_enabled("enable_new_validator"):
    validator = NewValidator()
else:
    validator = OldValidator()
```

### Use Case 2: Gradual Migration

```python
# Migrate 10% of tests to new system
test_id = generate_test_id()
if flags.is_enabled("use_new_metrics_engine", user_id=test_id):
    metrics = new_engine.calculate_all(...)
else:
    metrics = old_calculator.calculate_all_metrics(...)
```

### Use Case 3: Environment-Specific Features

```python
# Only enable in development
env = os.getenv("ENVIRONMENT", "production")
if flags.is_enabled("debug_mode", environment=env):
    enable_debug_logging()
```

### Use Case 4: A/B Testing

```python
# Test two different approaches
user_id = get_user_id()
if flags.is_enabled("new_algorithm", user_id=user_id):
    result = new_algorithm.calculate()
else:
    result = old_algorithm.calculate()
```

## Troubleshooting

### Flag Not Working?

1. **Check flag name**: Names are case-insensitive and converted to lowercase
   ```python
   # Environment: FEATURE_FLAG_NEW_METRICS=true
   # Code: flags.is_enabled("new_metrics")  # ✅ Correct
   # Code: flags.is_enabled("NEW_METRICS")  # ✅ Also works (converted to lowercase)
   ```

2. **Check if flag is registered**:
   ```python
   flag = flags.get_flag("my_feature")
   if flag is None:
       print("Flag not registered!")
   ```

3. **Check environment variable format**:
   ```bash
   # Correct
   FEATURE_FLAG_MY_FEATURE=true

   # Wrong (missing prefix)
   MY_FEATURE=true
   ```

4. **Check rollout percentage**:
   ```python
   # If rollout_percentage is 0, it uses the enabled flag
   flags.register("feature", enabled=True, rollout_percentage=0.0)
   # This will return True

   # If rollout_percentage is set, it overrides enabled
   flags.register("feature", enabled=False, rollout_percentage=50.0)
   # This will return True for 50% of users
   ```

## Example: Complete Integration

```python
# In your application startup
from config import get_feature_flags
from dashboard.metrics_engine import MetricsEngine
from dashboard.calculators.detectors import CompositeToxicityDetector

def setup_metrics_engine():
    """Setup metrics engine based on feature flags."""
    flags = get_feature_flags()

    # Check feature flags
    enable_safety = flags.is_enabled("enable_safety_metrics")
    enable_rag = flags.is_enabled("enable_rag_metrics")
    enable_agent = flags.is_enabled("enable_agent_metrics")

    # Setup toxicity detector with conditional enabling
    toxicity_detector = None
    if enable_safety:
        # Gradual rollout of toxicity detection
        test_id = get_current_test_id()
        enable_toxicity = flags.is_enabled(
            "enable_toxicity_detection",
            user_id=test_id
        )
        toxicity_detector = CompositeToxicityDetector(enabled=enable_toxicity)

    # Create metrics engine
    engine = MetricsEngine(
        enable_safety=enable_safety,
        enable_rag=enable_rag,
        enable_agent=enable_agent,
        toxicity_detector=toxicity_detector,
    )

    return engine

# Use it
metrics_engine = setup_metrics_engine()
```

**Environment variables (.env):**
```bash
# Enable metric groups
FEATURE_FLAG_ENABLE_SAFETY_METRICS=true
FEATURE_FLAG_ENABLE_RAG_METRICS=true
FEATURE_FLAG_ENABLE_AGENT_METRICS=false

# Gradual rollout of toxicity detection (25% of tests)
FEATURE_FLAG_ENABLE_TOXICITY_DETECTION=25
```

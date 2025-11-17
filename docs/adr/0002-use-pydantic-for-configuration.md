# ADR-0002: Use Pydantic for Configuration Management

## Status
Accepted

## Context
We need a robust configuration management system that:
- Validates configuration values
- Supports environment variables
- Provides type safety
- Handles multiple environments (dev/staging/prod)

## Decision
We will use Pydantic Settings for configuration management.

## Consequences
### Positive
- Type validation at runtime
- Automatic environment variable parsing
- IDE autocomplete support
- Clear error messages for invalid config

### Negative
- Additional dependency (pydantic-settings)
- Slight performance overhead for validation

## Alternatives Considered
- **python-dotenv**: Too simple, no validation
- **dynaconf**: More complex, overkill for our needs
- **configparser**: No type safety, verbose

## Implementation
Configuration is defined in `src/config/settings.py` using Pydantic's `BaseSettings`.

### Configuration Sources
1. **`.env` file** - Primary configuration file (user-specific, not committed)
   - Copy from `.env.example` to create your `.env` file
   - Contains all environment variables with defaults
   
2. **`environments.yaml`** - Multi-environment configuration
   - Defines environment-specific overrides (sandbox, staging, production)
   - Located in `src/config/environments.yaml`
   - Environment selected via `ENVIRONMENT` variable (default: sandbox)

3. **Environment variables** - Can override any setting
   - Automatically loaded by Pydantic Settings
   - Case-insensitive (BASE_URL = base_url)

### Configuration Categories
The Settings model includes:
- **Environment**: Environment selection (sandbox/staging/production)
- **URLs**: Base URL and chat URL configuration
- **Authentication**: Email and password credentials
- **Test Configuration**: Timeouts, retries, parallel workers
- **Browser Configuration**: Browser type, headless mode, timeouts
- **AI/ML Models**: Semantic model names for English and Arabic
- **Validation Thresholds**: Similarity, hallucination, consistency thresholds
- **Reporting**: Report directory settings
- **Security**: Security testing configuration
- **Localization**: Language settings and supported languages
- **Observability**: Playwright tracing and Prometheus metrics

See `.env.example` for the complete list of available configuration options.


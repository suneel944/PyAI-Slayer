# ADR-0006: Use pytest for Testing Framework

## Status
Accepted

## Context
We need a testing framework that:
- Is widely used in Python community
- Supports fixtures and parametrization
- Has good plugin ecosystem
- Integrates well with CI/CD

## Decision
We will use pytest as our testing framework.

## Consequences
### Positive
- Industry standard for Python testing
- Rich fixture system
- Excellent plugin ecosystem (coverage, xdist, etc.)
- Clear test output and error messages
- Easy to extend with custom markers

### Negative
- Learning curve for advanced features
- Some plugins may have compatibility issues

## Alternatives Considered
- **unittest**: Built-in but less feature-rich
- **nose2**: Less active development
- **Robot Framework**: Too heavyweight for unit tests

## Implementation
Tests are organized in `tests/` directory:
- `tests/unit/` - Framework unit tests
- `tests/ui/` - UI tests
- `tests/ai_responses/` - AI validation tests
- `tests/security/` - Security tests

### Test Execution Targets
The framework provides separate Makefile targets for different test types:
- **`make test`** - Runs integration/UI/security tests (excludes unit tests)
- **`make test-unit`** - Runs unit tests only
- **`make test-all`** - Runs all tests (unit + integration + UI + security)
- **`make test-cov`** - Runs all tests with coverage reporting

### Code Quality Checks
- **`make check`** - Runs linting and type checking only (no tests)
- **`make ci`** - Runs CI pipeline (lint + type-check + unit tests)

This separation allows for:
- Faster feedback during development (unit tests only)
- Comprehensive testing when needed (all tests)
- Efficient CI/CD pipelines (unit tests for quick feedback)

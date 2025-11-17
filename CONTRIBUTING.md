# Contributing to PyAI-Slayer 🗡️⚡

Thank you for your interest in contributing to PyAI-Slayer! This document provides guidelines and instructions for contributing to the project.

This project grew organically from solving real testing problems. The codebase structure, testing patterns, and architectural decisions all reflect lessons learned from actual use. When contributing, keep in mind that many of the "quirks" in the codebase exist because they solved specific problems encountered during development.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Project Structure](#project-structure)
- [Common Tasks](#common-tasks)
- [License](#license)
- [Recognition](#recognition)
- [Additional Resources](#additional-resources)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

Examples of unacceptable behavior include:

- The use of sexualized language or imagery and unwelcome sexual attention or advances
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

### Enforcement

Project maintainers are responsible for clarifying and enforcing our standards of acceptable behavior and will take appropriate and fair corrective action in response to any instances of unacceptable behavior.

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project maintainers. All complaints will be reviewed and investigated and will result in a response that is deemed necessary and appropriate to the circumstances.

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant](https://www.contributor-covenant.org), version 2.1.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Make (for using Makefile commands)
- Basic understanding of pytest and Playwright

### Initial Setup

1. **Fork the repository** on GitHub

2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/PyAI-Slayer.git
   cd PyAI-Slayer
   ```

3. **Set up the development environment**:
   ```bash
   make setup
   ```

4. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials and settings
   ```

6. **Install pre-commit hooks** (recommended):
   ```bash
   make install-hooks
   # Or: pre-commit install
   ```

7. **Verify your setup**:
   ```bash
   make check      # Run linting and type checking
   make test-unit  # Run unit tests
   ```

## Development Workflow

### Branch Naming

Use descriptive branch names that indicate the type of change:

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/improvements

Examples:
- `feature/add-new-validator`
- `bugfix/fix-browser-timeout`
- `docs/update-api-docs`

### Making Changes

1. **Create a new branch** from `master`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards

3. **Test your changes**:
   ```bash
   make test-all    # Run all tests
   make check       # Run linting and type checking
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

   Use conventional commit messages:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `style:` - Code style changes (formatting, etc.)
   - `refactor:` - Code refactoring
   - `test:` - Test additions/changes
   - `chore:` - Maintenance tasks

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

## Coding Standards

### Python Style Guide

- Follow **PEP 8** style guidelines
- Use **type hints** for all function signatures
- Write **docstrings** in **Google style** format
- Maximum line length: **100 characters**
- Use **Black** for code formatting
- Use **Ruff** for linting

### Code Formatting

Before committing, always format your code:

```bash
make format  # Runs black + ruff --fix
```

### Type Hints

All functions should include type hints:

```python
def validate_response(
    query: str,
    response: str,
    threshold: float = 0.7
) -> tuple[bool, float]:
    """
    Validate response relevance using semantic similarity.

    Args:
        query: The original query string
        response: The AI response to validate
        threshold: Minimum similarity score (default: 0.7)

    Returns:
        Tuple of (is_valid, similarity_score)
    """
    # Implementation
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def process_data(data: dict[str, Any]) -> list[str]:
    """
    Process input data and return formatted results.

    Args:
        data: Dictionary containing input data with keys 'items' and 'format'

    Returns:
        List of formatted strings

    Raises:
        ValueError: If data format is invalid
    """
    pass
```

### Import Organization

Imports should be organized as follows:

1. Standard library imports
2. Third-party imports
3. Local application imports

Use `isort` to automatically organize imports (included in `make format`).

### Code Quality Checks

Before submitting a PR, ensure all checks pass:

```bash
make check      # Linting + type checking
make test-all   # All tests
make ci         # Full CI pipeline
```

## Testing Guidelines

### Test Organization

Tests should be organized by type:

- **Unit tests** → `tests/unit/`
- **Integration tests** → `tests/integration/`
- **E2E tests** → `tests/e2e/`
- **UI tests** → `tests/ui/`
- **Security tests** → `tests/security/`

### Writing Tests

1. **Use pytest markers** to categorize tests:
   ```python
   @pytest.mark.unit
   @pytest.mark.functional
   def test_my_feature():
       pass
   ```

2. **Follow naming conventions**:
   - Test files: `test_*.py`
   - Test classes: `Test*`
   - Test functions: `test_*`

3. **Use fixtures** from `conftest.py` when available

4. **Write descriptive test names**:
   ```python
   def test_validate_response_returns_true_for_relevant_content():
       pass
   ```

5. **Add tests for new features**:
   - Unit tests for core logic
   - Integration tests for component interactions
   - E2E tests for user workflows (if applicable)

### Running Tests

```bash
# Run all tests
make test-all

# Run specific test categories
pytest -m unit -v
pytest -m functional -v
pytest -m ui -v

# Run with coverage
make test-cov

# Run specific test file
pytest tests/unit/test_validator.py -v
```

### Test Coverage

- Aim for **80%+ coverage** for new code
- Focus on testing business logic and edge cases
- Use property-based testing with Hypothesis for complex validations

## Documentation

### Code Documentation

- Add docstrings to all public functions, classes, and methods
- Document complex algorithms and business logic
- Include examples in docstrings when helpful

### API Documentation

- Update API documentation in `docs/api_reference.rst` for new public APIs
- Include usage examples

### README Updates

- Update README.md if you add new features or change setup instructions
- Keep the feature list up to date

### Architecture Decision Records (ADRs)

For significant architectural changes:

1. Create a new ADR in `docs/adr/`
2. Follow the existing ADR format
3. Number sequentially (e.g., `0007-my-decision.md`)

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass**:
   ```bash
   make test-all
   ```

2. **Run code quality checks**:
   ```bash
   make check
   ```

3. **Update documentation** if needed

4. **Rebase on latest master** (if needed):
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Type hints added/updated
- [ ] No new warnings generated
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] PR description is clear and complete

### PR Description Template

Use the PR template provided in `.github/pull_request_template.md`. Include:

- Clear description of changes
- Type of change (bug fix, feature, etc.)
- Testing performed
- Related issues
- Screenshots (if applicable)

### Review Process

1. **Automated Checks**: All PRs must pass CI checks (linting, type checking, tests)
2. **Code Review**: PRs require at least one approval from a maintainer before merging
3. **Addressing Feedback**:
   - Address review comments promptly and professionally
   - Ask for clarification if feedback is unclear
   - Mark conversations as resolved when addressed
4. **PR Guidelines**:
   - Keep PRs focused - one feature/fix per PR
   - Keep PRs small when possible - break large changes into multiple PRs
   - Update related documentation
   - Add tests for new functionality
5. **Merge Process**:
   - Maintainers will merge PRs after approval and CI passing
   - Squash and merge is preferred for cleaner history
   - PRs may be closed if inactive for 30+ days (can be reopened when ready)

## Project Structure

### Key Directories

```
src/
├── config/          # Configuration layer
├── core/            # Core framework components
│   ├── ai/         # AI validation
│   ├── browser/    # Browser automation
│   ├── infrastructure/  # Infrastructure components
│   ├── observability/   # Observability features
│   ├── security/   # Security testing
│   └── validation/ # Validation strategies
├── dashboard/      # Dashboard application
└── utils/          # Utility functions

tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
│   ├── ai/        # AI validation tests
│   ├── security/  # Security tests
│   └── ui/        # UI tests
└── pages/          # Page Object Model

docs/              # Documentation
scripts/            # Helper scripts
```

### Adding New Components

When adding new components:

1. **Place code in appropriate layer**:
   - Core logic → `src/core/`
   - Configuration → `src/config/`
   - Utilities → `src/utils/`

2. **Follow existing patterns**:
   - Use dependency injection
   - Implement proper error handling
   - Add logging with loguru

3. **Update imports** in `__init__.py` files if needed

## Common Tasks

### Adding a New Validator

1. Create validator class in `src/core/validation/`
2. Implement validation interface
3. Add unit tests in `tests/unit/`
4. Register in dependency injection container
5. Update documentation

### Adding a New Test

1. Determine test type (unit, integration, e2e)
2. Place in appropriate directory
3. Use appropriate markers
4. Follow Page Object Model for UI tests
5. Use fixtures from `conftest.py`

### Adding a New Security Test

1. Add payloads to security tester
2. Create test in `tests/e2e/security/`
3. Use `security_tester` fixture
4. Validate sanitization properly

### Updating Dependencies

1. Update `pyproject.toml`
2. Test with `make install-dev`
3. Update lock file if using one
4. Document breaking changes

## Reporting Issues

### Bug Reports

Before reporting a bug, please:

1. **Search existing issues** to see if it's already reported
2. **Check the documentation** to ensure it's not a configuration issue
3. **Test with the latest version** to confirm it's still present

When reporting a bug, please include:

- **Clear description** of the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs **actual behavior**
- **Environment details** (OS, Python version, package version)
- **Error messages** or logs (if applicable)
- **Minimal reproducible example** (if possible)

Use the bug report template in `.github/ISSUE_TEMPLATE/bug_report.md`.

### Feature Requests

For feature requests:

1. **Search existing issues** to see if it's already requested
2. **Check if it aligns** with project goals
3. **Provide clear use case** and motivation

Use the feature request template in `.github/ISSUE_TEMPLATE/feature_request.md`.

### Security Issues

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to the project maintainers or through a private security advisory. We take security seriously and will respond promptly.

For security-related issues:
- Use GitHub's [Private Vulnerability Reporting](https://github.com/security-advisories) if available
- Or contact maintainers directly via email (if provided)
- Include detailed information about the vulnerability
- Allow time for the issue to be addressed before public disclosure

## Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: Search existing issues on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: Ask for help in PR comments

## License

By contributing to PyAI-Slayer, you agree that your contributions will be licensed under the same license as the project (MIT License). This means your contributions will be open source and available to the community.

## Recognition

Contributors will be recognized in:
- Project README (if significant contributions)
- Release notes
- GitHub contributors page

We appreciate all contributions, whether they are:
- Code contributions
- Documentation improvements
- Bug reports
- Feature suggestions
- Code reviews
- Community support

Thank you for contributing to PyAI-Slayer! 🗡️⚡

## Additional Resources

- [Project README](README.md) - Overview and quick start
- [Architecture Documentation](docs/FRAMEWORK_ARCHITECTURE.md) - Framework design details
- [API Reference](docs/api_reference.rst) - Complete API documentation
- [Architecture Decision Records](docs/adr/) - Design decisions and rationale

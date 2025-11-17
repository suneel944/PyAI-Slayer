# ADR-0003: Use Playwright for Browser Automation

## Status
Accepted

## Context
We need a modern browser automation framework that:
- Supports multiple browsers (Chromium, Firefox, WebKit)
- Has reliable waiting mechanisms
- Works well with Python
- Has good documentation and community support

## Decision
We will use Playwright (sync API) for browser automation.

## Consequences
### Positive
- Modern, fast, and reliable
- Built-in waiting and auto-retry
- Cross-browser support
- Excellent Python bindings
- Active development and support

### Negative
- Requires browser binaries (larger installation)
- Newer framework (less ecosystem maturity than Selenium)

## Alternatives Considered
- **Selenium**: Mature but slower, less reliable waits
- **Puppeteer**: Node.js only, not suitable for Python
- **Cypress**: JavaScript-focused, not ideal for Python projects

## Implementation
Browser automation is handled by `BrowserManager` in `src/core/browser/browser_manager.py`, which wraps Playwright's sync API.

The framework also includes:
- **BrowserPool** (`src/core/browser/browser_pool.py`) - Manages a pool of browser instances for parallel test execution
- Browser instances are automatically managed through pytest fixtures in `tests/conftest.py`


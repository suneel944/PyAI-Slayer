"""Pytest configuration and fixtures."""

import pytest
from loguru import logger

from config.settings import settings
from config.test_config import TestConfig
from core import (
    AIResponseValidator,
    BrowserManager,
    SecurityTester,
    get_conversation_tester,
    get_hallucination_detector,
    get_playwright_tracer,
    get_prometheus_metrics,
    get_prompt_injection_tester,
    get_strategy_registry,
)
from core.ai.rag_tester import RAGTester
from tests.pages.chat_page import ChatPage
from tests.pages.login_page import LoginPage
from utils.screenshot_manager import ScreenshotManager


@pytest.fixture(scope="session")
def test_config():
    """
    Centralized test configuration.

    This fixture provides a clean interface to configuration
    without direct coupling to environment variables or Settings.
    """
    return TestConfig.from_settings(settings)


@pytest.fixture(scope="session")
def browser_manager():
    """Create browser manager instance."""
    logger.info("Setting up browser_manager fixture (session-scoped)...")
    manager = BrowserManager()
    try:
        logger.info("Starting browser manager...")
        manager.start()
        logger.info("✓ Browser manager started successfully")
    except Exception as e:
        logger.error(f"Failed to start browser manager: {e}")
        import traceback

        traceback.print_exc()
        raise
    yield manager
    logger.info("Cleaning up browser_manager...")
    try:
        manager.close()
        logger.info("✓ Browser manager cleaned up")
    except Exception as e:
        logger.error(f"Error during browser manager cleanup: {e}")


@pytest.fixture
def browser_context(browser_manager):
    """Create a new browser context."""
    try:
        context = browser_manager.create_context()
    except Exception as e:
        logger.error(f"Failed to create browser context: {e}")
        import traceback

        traceback.print_exc()
        raise
    yield context
    try:
        context.close()
    except Exception as e:
        logger.error(f"Error closing browser context: {e}")


@pytest.fixture
def page(browser_context, request, test_config):
    """Create a new page with Playwright tracing."""
    try:
        page = browser_context.new_page()

        # Start Playwright tracing if enabled
        if test_config.enable_playwright_tracing:
            try:
                tracer = get_playwright_tracer()
                test_name = request.node.name
                tracer.start_trace_for_page(page, test_name)
            except Exception as e:
                logger.debug(f"Could not start trace: {e}")
    except Exception as e:
        logger.error(f"Failed to create page: {e}")
        import traceback

        traceback.print_exc()
        raise

    yield page

    try:
        # Stop and save trace if enabled
        if test_config.enable_playwright_tracing:
            try:
                tracer = get_playwright_tracer()
                test_name = request.node.name
                _ = tracer.stop_trace_for_page(page, test_name, save=True)

                # Trace file saved (Allure reporting removed)
            except Exception as e:
                logger.debug(f"Could not save trace: {e}")

        page.close()
    except Exception as e:
        logger.error(f"Error closing page: {e}")


@pytest.fixture
def mobile_page(browser_manager):
    """Create a mobile emulated page."""
    page = browser_manager.create_mobile_page()
    yield page
    try:
        page.close()
    except Exception as e:
        logger.debug(f"Error closing mobile page: {e}")


@pytest.fixture
def mobile_chat_page(mobile_page, test_config):
    """
    Create chat page object for mobile with auto-navigation.

    Page is pre-navigated to chat_url and ready for interaction.
    """
    logger.info(f"Navigating to chat (mobile): {test_config.chat_url}")
    mobile_page.goto(test_config.chat_url, wait_until="networkidle")

    chat = ChatPage(mobile_page, test_config)
    chat.wait_for_chat_loaded()
    logger.info("✓ Mobile chat page ready")
    return chat


@pytest.fixture
def chat_page(page, test_config):
    """
    Create chat page object with auto-navigation.

    Page is pre-navigated to chat_url and ready for interaction.
    """
    logger.info(f"Navigating to chat: {test_config.chat_url}")
    page.goto(test_config.chat_url, wait_until="networkidle")

    chat = ChatPage(page, test_config)
    chat.wait_for_chat_loaded()
    logger.info("✓ Chat page ready")
    return chat


@pytest.fixture
def login_page(page, test_config, browser_manager):
    """Create login page object."""
    return LoginPage(page, test_config, browser_manager)


@pytest.fixture(scope="session")
def ai_validator():
    """Create AI validator instance."""
    return AIResponseValidator()


@pytest.fixture(scope="session")
def security_tester():
    """Create security tester instance."""
    return SecurityTester()


@pytest.fixture(scope="session")
def conversation_tester():
    """Create conversation tester instance."""
    return get_conversation_tester()


@pytest.fixture(scope="session")
def hallucination_detector():
    """Create advanced hallucination detector instance."""
    return get_hallucination_detector()


@pytest.fixture
def rag_tester():
    """Create RAG tester instance."""
    return RAGTester()


@pytest.fixture(scope="session")
def prompt_injection_tester():
    """Create advanced prompt injection tester instance."""
    return get_prompt_injection_tester()


@pytest.fixture(scope="session")
def security_test_data():
    """Load security test data from JSON file."""
    import json
    from pathlib import Path

    # Load comprehensive security test data from single JSON file
    security_data_file = Path("tests/test_data/prompts/security-test-data.json")
    if security_data_file.exists():
        try:
            with open(security_data_file, encoding="utf-8") as f:
                test_data = json.load(f)
                # Flatten structure for backward compatibility
                # Add comprehensive_security_tests fields at top level
                if "comprehensive_security_tests" in test_data:
                    test_data.update(test_data["comprehensive_security_tests"])
                return test_data
        except Exception as e:
            logger.warning(f"Could not load security test data from {security_data_file}: {e}")

    # Return empty dict if file couldn't be loaded - tests should handle missing data gracefully
    logger.error("security-test-data.json file not found or could not be loaded. Tests may fail.")
    return {}


@pytest.fixture(scope="session")
def ai_test_data():
    """Load AI test data from JSON file."""
    import json
    from pathlib import Path

    test_data_file = Path("tests/test_data/prompts/test-data-en.json")
    if test_data_file.exists():
        try:
            with open(test_data_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load AI test data from {test_data_file}: {e}")

    logger.error("test-data-en.json file not found or could not be loaded. Tests may fail.")
    return {}


@pytest.fixture(scope="session")
def arabic_test_data():
    """Load Arabic test data from JSON file."""
    import json
    from pathlib import Path

    test_data_file = Path("tests/test_data/prompts/test-data-ar.json")
    if test_data_file.exists():
        try:
            with open(test_data_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load Arabic test data from {test_data_file}: {e}")

    logger.error("test-data-ar.json file not found or could not be loaded. Tests may fail.")
    return {}


@pytest.fixture(scope="session")
def validation_strategy_registry():
    """Create validation strategy registry instance."""
    return get_strategy_registry()


@pytest.fixture
def screenshot_manager():
    """Create screenshot manager."""
    return ScreenshotManager()


@pytest.fixture(autouse=True)
def setup_test(request, test_config):
    """Setup before each test - handles login if needed."""

    needs_login = (
        "chat_page" in request.fixturenames
        or "mobile_page" in request.fixturenames
        or "mobile_chat_page" in request.fixturenames
    )
    is_mobile = "mobile_page" in request.fixturenames or "mobile_chat_page" in request.fixturenames

    if needs_login and test_config.email and test_config.password:
        try:
            if is_mobile:
                try:
                    # Try mobile_chat_page first, fallback to mobile_page
                    try:
                        test_page = request.getfixturevalue("mobile_chat_page").page
                    except Exception:
                        test_page = request.getfixturevalue("mobile_page")
                except Exception as e:
                    logger.warning(f"Could not get mobile page fixture: {e}, skipping mobile login")
                    yield
                    return

                browser_manager_instance = request.getfixturevalue("browser_manager")
                test_login_page = LoginPage(test_page, test_config, browser_manager_instance)
            else:
                test_page = request.getfixturevalue("page")
                test_login_page = request.getfixturevalue("login_page")

            logger.info("=" * 60)
            logger.info("SETUP: Starting login process...")
            logger.info("=" * 60)

            logger.info(f"Step 1: Navigating to {test_config.base_url}...")
            test_page.goto(test_config.base_url, wait_until="networkidle", timeout=30000)
            logger.info(f"✓ Navigated. Current URL: {test_page.url}")

            current_url = test_page.url
            page_title = test_page.title()
            logger.info(f"Page title: {page_title}")

            if "auth" in current_url or "login" in current_url.lower():
                logger.info("✓ Login page detected")
                logger.info("Step 2: Attempting email/password login...")

                login_success = test_login_page.login(use_sso=False)

                if not login_success:
                    logger.error("✗ Login failed - test will likely fail")
                    logger.error(f"Current URL: {test_page.url}")
                else:
                    logger.info("✓ Login successful")
                    test_page.wait_for_load_state("networkidle", timeout=15000)
                    logger.info(f"✓ Post-login URL: {test_page.url}")
            else:
                logger.info("✓ Already logged in or not on login page")

            logger.info("=" * 60)
            logger.info("SETUP: Login process complete")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"Login setup error: {e}")
            import traceback

            traceback.print_exc()

    yield

    if hasattr(request.node, "rep_call") and request.node.rep_call.failed:
        try:
            if is_mobile:
                # Try mobile_chat_page first, fallback to mobile_page
                try:
                    test_page = request.getfixturevalue("mobile_chat_page").page
                except Exception:
                    test_page = request.getfixturevalue("mobile_page")
            else:
                test_page = request.getfixturevalue("page")

            worker_id = getattr(request.config, "workerinput", {}).get("workerid", "gw0")
            screenshot_path = f"screenshots/failure_{worker_id}_{request.node.name}.png"
            test_page.screenshot(path=screenshot_path, full_page=True)
            logger.info(f"Failure screenshot saved: {screenshot_path}")

            # Screenshot saved
        except Exception as e:
            logger.debug(f"Suppressed exception: {e}")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test results for screenshots, metrics, and dashboard."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)

    # Record Prometheus metrics - use settings here as hooks can't easily access fixtures
    if rep.when == "call" and settings.enable_prometheus_metrics:
        try:
            metrics = get_prometheus_metrics()
            if metrics.enabled:
                test_name = item.name
                status = rep.outcome  # passed, failed, skipped
                duration = rep.duration or 0.0
                metrics.record_test_end(test_name, status, duration)
        except Exception as e:
            logger.debug(f"Suppressed exception: {e}")

    # Collect data for dashboard
    if rep.when == "call":
        try:
            from datetime import datetime

            from core.ai.ai_validator import _get_validation_data
            from dashboard.collectors import get_dashboard_collector

            collector = get_dashboard_collector()
            test_name = item.name
            status = rep.outcome
            duration = rep.duration or 0.0

            # Determine language and test type from test name
            language = "unknown"
            if "english" in test_name.lower() or "_en" in test_name.lower():
                language = "en"
            elif "arabic" in test_name.lower() or "_ar" in test_name.lower():
                language = "ar"
            elif "cross_language" in test_name.lower():
                language = "multilingual"

            test_type = "general"
            if "relevance" in test_name.lower():
                test_type = "relevance"
            elif "hallucination" in test_name.lower():
                test_type = "hallucination"
            elif "consistency" in test_name.lower():
                test_type = "consistency"
            elif "security" in test_name.lower():
                test_type = "security"
            elif "conversation" in test_name.lower():
                test_type = "conversation"

            # Extract error information for failed tests
            error_message = None
            stack_trace = None
            if status == "failed" and hasattr(rep, "longrepr"):
                # Extract error message and stack trace from pytest report
                error_repr = str(rep.longrepr) if rep.longrepr else None
                if error_repr:
                    # Split error message and stack trace
                    lines = error_repr.split("\n")
                    # First few lines are usually the error message
                    error_message = "\n".join(lines[:10])  # First 10 lines as error message
                    # Rest is stack trace
                    stack_trace = "\n".join(lines[10:]) if len(lines) > 10 else error_repr

            # Get test path for exclusion checking
            test_path = None
            if hasattr(item, "fspath"):
                test_path = str(item.fspath)
            elif hasattr(item, "path"):
                test_path = str(item.path)

            # Collect test result (returns None if test is excluded)
            test_id = collector.collect_test_result(
                test_name=test_name,
                status=status,
                duration=duration,
                language=language,
                test_type=test_type,
                timestamp=datetime.now(),
                error_message=error_message,
                stack_trace=stack_trace,
                test_path=test_path,
            )

            # If test was excluded, skip all data collection
            if test_id is None:
                logger.debug(
                    f"Test {test_name} excluded from dashboard data collection "
                    "(unit/integration test)"
                )
                return

            # Collect system metrics
            try:
                collector.collect_system_metrics()
            except Exception as e:
                logger.debug(f"Suppressed exception: {e}")

            # Only collect additional data if test was not excluded
            # Get validation data directly from validator context
            validation_data = _get_validation_data()

            # Get RAG context data if available
            from core.ai.rag_tester import _get_rag_context

            rag_context = _get_rag_context()

            # Always try to collect validation data, even if partial
            # This ensures we capture data even when tests fail early
            if validation_data and (
                validation_data.get("query") or validation_data.get("response")
            ):
                collector.collect_from_validation_data(
                    test_id,
                    validation_data,
                    status,
                    duration=duration,
                    retrieved_docs=rag_context.get("retrieved_docs"),
                    expected_sources=rag_context.get("expected_sources"),
                    gold_context=rag_context.get("gold_context"),
                )
            elif status == "failed":
                # For failed tests without validation data, still create a minimal record
                # This helps with debugging and ensures all failures are tracked
                try:
                    collector.collect_validation_data(
                        test_id=test_id,
                        query=None,
                        expected_response=None,
                        actual_response=None,
                        validation_type="unknown",
                        passed=False,
                    )
                except Exception as e:
                    logger.debug(f"Suppressed exception: {e}")

            # Clear RAG context after collection
            from core.ai.rag_tester import _clear_rag_context

            _clear_rag_context()

            # Find and collect artifacts
            collector.find_test_artifacts(test_name, test_id)

        except Exception as e:
            logger.debug(f"Suppressed exception: {e}")


def pytest_configure(config):
    """Prevent plugins that create asyncio loops from conflicting with Playwright sync API."""
    pm = config.pluginmanager

    try:
        pm.unregister(name="anyio")
        logger.info(
            "Disabled pytest-anyio to prevent asyncio loop conflict with Playwright sync API"
        )
    except (ValueError, KeyError):
        pass
    except Exception as e:
        logger.debug(f"Suppressed exception: {e}")

    try:
        pm.unregister(name="rerunfailures")
        logger.info("Disabled pytest-rerunfailures to prevent async fixture deadlock")
    except (ValueError, KeyError):
        pass
    except Exception as e:
        logger.debug(f"Suppressed exception: {e}")

    # Start Prometheus metrics server if enabled
    # Only start in main process, not in worker processes (pytest-xdist)
    if settings.enable_prometheus_metrics:
        try:
            # Check if we're in a worker process (pytest-xdist)
            # Worker processes have 'workerinput' attribute
            if hasattr(config, "workerinput"):
                # We're in a worker process - skip server startup
                # The main process will handle the Prometheus server
                logger.debug("Skipping Prometheus server startup in worker process")
            else:
                # Main process - start the server
                metrics = get_prometheus_metrics()
                if metrics.enabled:
                    metrics.start_server(port=settings.prometheus_port)
        except Exception as e:
            logger.warning(f"Failed to start Prometheus server: {e}")


@pytest.fixture(autouse=True)
def record_test_metrics(request, test_config):
    """Auto-record test start/end metrics for Prometheus."""
    if not test_config.enable_prometheus_metrics:
        yield
        return

    try:
        metrics = get_prometheus_metrics()
        if metrics.enabled:
            test_name = request.node.name
            metrics.record_test_start(test_name)
    except Exception:
        pass

    yield

    # Test end is recorded in pytest_runtest_makereport hook


@pytest.fixture(autouse=True)
def capture_validation_data(request):
    """
    Transparently capture AI validation data for test reporting.

    This fixture automatically stores validation data from AIResponseValidator
    calls for potential future use. Tests don't need to do anything - it just works!
    """
    from core.ai.ai_validator import _clear_validation_data

    # Clear any previous validation data
    _clear_validation_data()

    yield

    # After test execution, capture validation data
    try:
        from core.ai.ai_validator import _get_validation_data

        validation_data = _get_validation_data()
        if validation_data and (validation_data.get("query") or validation_data.get("response")):
            # Store on test node for pytest_runtest_makereport hook
            request.node.test_validation_data = validation_data
    except Exception as e:
        logger.debug(f"Could not capture validation data: {e}")
    finally:
        # Clean up after test
        _clear_validation_data()

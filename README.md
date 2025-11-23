# PyAI-Slayer 🗡️⚡

**The Open Source AI Testing Arsenal**

A comprehensive Python automation framework for testing AI chatbots and LLM applications with semantic validation, multilingual support, and real-time observability.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Key Features

- 🤖 **AI Response Validation** - Semantic similarity, hallucination detection, consistency checking
- 🌍 **Multilingual Support** - English & Arabic with RTL/LTR layout validation
- 🔒 **Security Testing** - Injection attacks, prompt injection, input sanitization
- 📊 **Real-Time Dashboard** - Live metrics, A-Tier critical indicators, performance analytics
- 🎭 **Browser Automation** - Desktop & mobile testing with Playwright
- ♿ **Accessibility** - ARIA labels, keyboard navigation, screen reader compatibility

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PyAI-Slayer.git
cd PyAI-Slayer

# Install system dependencies (Linux/Ubuntu)
sudo apt-get update && sudo apt-get install -y git wget gnupg fonts-liberation libasound2 \
    libatk-bridge2.0-0 libatk1.0-0 libcups2 libdbus-1-3 libdrm2 libgbm1 libgtk-3-0 \
    libnspr4 libnss3 libxcomposite1 libxdamage1 libxfixes3 libxkbcommon0 libxrandr2

# Setup (installs dependencies & browsers)
make setup

# Configure environment
cp .env.example .env
# Edit .env with your settings (CHATBOT_URL, etc.)
```

### Run Tests

```bash
# Run all e2e tests (AI, security, UI)
make test

# Run specific test categories
make test-ai          # AI validation tests only
make test-security   # Security tests only
make test-ui         # UI tests only
make test-unit       # Unit tests only

# Run all tests (unit + e2e)
make test-all

# Run with coverage
make test-cov
```

### Launch Dashboard

```bash
make dashboard
# Access at http://localhost:8080
```

## 📸 Dashboard Showcase

### Overview - Key Metrics at a Glance

![Dashboard Overview](docs/images/dashboard-overview.png)

Real-time tracking of Overall Health Score, Average Response Time, Safety Score, and User Satisfaction.

---

### A-Tier Critical Metrics

![A-Tier Metrics](docs/images/dashboard-critical-metrics.png)

15 production-critical AI capability indicators including Source Grounding, Factual Consistency, E2E Latency, and Safety Violations.

---

### Performance Trends & Multi-Dimensional Health

![Performance Analysis](docs/images/dashboard-performance.png)

24-hour quality & latency metrics with comprehensive radar chart showing system health across 5 dimensions.

---

### Category Performance Breakdown

![Category Breakdown](docs/images/dashboard-categories.png)

Complete quality scores across Base Model, RAG, Safety, Performance, Reliability, and Agent categories.

---

## 🧪 Test Categories

| Category | Tests | Focus Area |
|----------|-------|------------|
| **Base Model** | Basic functionality, response quality, error handling | Core chatbot capabilities |
| **RAG** | Context retrieval, source attribution, grounding | Retrieval-Augmented Generation |
| **Safety** | Harmful content, bias detection, refusal | Safety & ethics compliance |
| **Performance** | Response time, throughput, resource usage | Speed & efficiency |
| **Reliability** | Consistency, determinism, error recovery | Stability & predictability |
| **Security** | Injection attacks, prompt manipulation | Security vulnerabilities |
| **Agent** | Multi-turn conversations, tool usage | Advanced capabilities |

### 🎯 RAG Calibration

Calibrate RAG metric targets based on your actual data distribution:

```bash
# Create your eval set (see docs/RAG_CALIBRATION_USAGE.md)
cp data/examples/example_rag_eval_set.json data/rag_eval_set.json
# Edit with your queries and labeled chunks

# Run calibration
python scripts/calibrate_rag_metrics.py

# Dashboard automatically uses calibrated targets
```

See [RAG Calibration Guide](docs/RAG_CALIBRATION_USAGE.md) for detailed instructions.

## 📊 Reporting Features

- **Real-Time Metrics** - Live test execution monitoring
- **A-Tier Indicators** - 15 production-critical metrics with target tracking
- **Historical Trends** - Time-series analysis (24h, 7d, 30d)
- **Failed Test Analysis** - Detailed breakdowns with recommendations
- **Category Performance** - Quality scores across 6 categories
- **Export Capabilities** - JSON, HTML, and Prometheus metrics

## 🛠️ Common Commands

```bash
# Setup & Installation
make setup              # Complete project setup (check Python, create venv, install deps)
make install-dev        # Install package with dev dependencies (auto-detects GPU)
make install-hooks      # Install pre-commit hooks
make playwright-install # Install Playwright browsers

# Testing
make test               # Run all e2e tests (AI, security, UI - excludes unit tests)
make test-all           # Run all tests (unit + e2e)
make test-unit          # Run unit tests only
make test-unit-cov      # Run unit tests with coverage report
make test-ai            # Run AI tests only
make test-security      # Run security tests only
make test-ui            # Run UI tests only
make test-integration   # Run integration tests only
make test-property      # Run property-based tests (Hypothesis)
make test-cov           # Run all tests with coverage report

# Code Quality
make lint               # Run linter (ruff)
make format             # Format code (ruff format only)
make format-check       # Check code formatting without modifying files
make type-check         # Run type checker (mypy)
make check              # Run all checks (lint + type-check)
make pre-commit         # Run all pre-commit checks (format + lint + type-check + security)
make security-scan      # Run security scan (bandit)

# Build & CI
make build              # Build distribution packages
make ci                 # Run CI pipeline (install-dev + lint + type-check + test-unit)

# Documentation
make docs               # Build Sphinx documentation
make docs-serve         # Build and serve documentation at http://localhost:8000

# Dashboard & Metrics
make dashboard          # Start metrics dashboard server
make dashboard-custom   # Start dashboard with custom port
make metrics-summary    # View test metrics summary (requires metrics enabled)
make metrics-export     # Export metrics to JSON file
make metrics-raw        # View raw metrics data

# Cleanup
make clean              # Remove venv, build artifacts, caches

For a complete list of all available commands, run `make help`.
```

## 📁 Project Structure

```
PyAI-Slayer/
├── src/
│   ├── config/           # Configuration layer
│   ├── core/             # Core framework components
│   │   ├── ai/          # AI validation
│   │   ├── browser/     # Browser automation
│   │   ├── infrastructure/  # Infrastructure components
│   │   ├── observability/  # Observability features
│   │   ├── security/   # Security testing
│   │   └── validation/ # Validation strategies
│   ├── dashboard/       # Real-time dashboard & API
│   └── utils/          # Utilities & helpers
├── tests/
│   ├── e2e/            # End-to-end tests
│   │   ├── ai/         # AI validation tests
│   │   ├── security/   # Security tests
│   │   └── ui/         # UI tests
│   ├── integration/    # Integration tests
│   ├── unit/           # Unit tests
│   ├── pages/          # Page Object Model
│   └── test_data/     # Test data
├── docs/               # Documentation
└── scripts/           # Utility scripts
```

## 🔧 Configuration

Key environment variables in `.env`:

```env
# Target Application
BASE_URL=https://your-chatbot-url.example.com
CHAT_URL=https://your-chatbot-url.example.com

# AI Models
SEMANTIC_MODEL_NAME=intfloat/multilingual-e5-base
ARABIC_SEMANTIC_MODEL_NAME=Omartificial-Intelligence-Space/mmbert-base-arabic-nli

# Fact-Checker (Hallucination Detection)
FACT_CHECKER_MODEL_NAME=  # Primary model (empty = use fallback list)
FACT_CHECKER_FALLBACK_MODELS=microsoft/deberta-large-mnli,roberta-large-mnli,facebook/bart-large-mnli
FACT_CHECKER_ENABLED=true
FACT_CHECKER_USE_CUDA=true  # Use GPU if available
FACT_CHECKER_CUDA_DEVICE=0  # CUDA device ID
FACT_CHECKER_STRONG_CONTRADICTION_THRESHOLD=0.7  # Confidence for strong contradiction
FACT_CHECKER_STRONG_ENTAILMENT_THRESHOLD=0.7  # Confidence for strong entailment
FACT_CHECKER_WEAK_ENTAILMENT_THRESHOLD=0.5  # Confidence for weak entailment
FACT_CHECKER_NEUTRAL_THRESHOLD=0.8  # Confidence for neutral classification
FACT_CHECKER_NEUTRAL_RATIO_THRESHOLD=0.7  # Ratio of neutrals to mark as unknown

# RAG Reranker (Improved Relevance Scoring)
RAG_RERANKER_MODEL_NAME=  # Primary model (empty = use fallback list)
RAG_RERANKER_FALLBACK_MODELS=BAAI/bge-reranker-base,BAAI/bge-reranker-large,ms-marco-MiniLM-L-12-v2
RAG_RERANKER_ENABLED=true
RAG_RERANKER_USE_CUDA=true  # Use GPU if available
RAG_RERANKER_CUDA_DEVICE=0  # CUDA device ID
RAG_FETCH_URL_CONTENT=true  # Fetch content from URLs for reranker scoring (improves accuracy)
RAG_URL_FETCH_TIMEOUT=10  # Timeout in seconds for URL content fetching
RAG_URL_MAX_CONTENT_LENGTH=10000  # Maximum content length per URL (characters)
RAG_URL_MAX_RETRIES=3  # Maximum retry attempts per URL
RAG_URL_RETRY_DELAY=1.0  # Delay between retries in seconds
RAG_URL_MAX_WORKERS=5  # Maximum parallel workers for URL fetching

# RAG Metric Targets (Optional - overrides calibration file if set)
# Leave empty to use calibration recommendations from data/rag_calibration_recommendations.json
# If any target is set, it will override the calibration file for that metric
RAG_TARGET_RETRIEVAL_RECALL_5=  # Target for retrieval recall@5 (0-100)
RAG_TARGET_RETRIEVAL_PRECISION_5=  # Target for retrieval precision@5 (0-100)
RAG_TARGET_CONTEXT_RELEVANCE=  # Target for context relevance (0-100)
RAG_TARGET_CONTEXT_COVERAGE=  # Target for context coverage (0-100)
RAG_TARGET_CONTEXT_INTRUSION=  # Target for context intrusion (0-100, lower is better)
RAG_TARGET_GOLD_CONTEXT_MATCH=  # Target for gold context match (0-100)
RAG_TARGET_RERANKER_SCORE=  # Target for reranker score (0-1)

# Dashboard
PROMETHEUS_PORT=8000
ENABLE_PROMETHEUS_METRICS=true

# Testing
HEADLESS=true
BROWSER=chromium
```

See [docs/getting_started.rst](docs/getting_started.rst) for complete configuration options.

## 📚 Documentation

- [Getting Started Guide](docs/getting_started.rst) - Detailed setup & usage
- [Dashboard Guide](docs/DASHBOARD.md) - Dashboard features & metrics
- [Framework Architecture](docs/FRAMEWORK_ARCHITECTURE.md) - Design & patterns
- [Metrics Calculations](docs/METRICS_CALCULATIONS.md) - How metrics are computed
- [RAG Calibration Guide](docs/RAG_CALIBRATION_USAGE.md) - Calibrate RAG metrics with your data
- [Docker Guide](docs/DOCKER.md) - Container usage
- [Plugins Guide](docs/PLUGINS.md) - Extending the framework with plugins
- [API Reference](docs/api_reference.rst) - Code documentation

## 🐳 Docker Quick Start

```bash
# Build and run
make docker-build
make docker-test

# Or use docker-compose
docker-compose up --build
```

## 🤝 Contributing

Contributions welcome! Please check out our [Contributing Guide](CONTRIBUTING.md).

```bash
# Setup development environment
make setup

# Run tests before committing
make check
make test

# Format code
make format
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with amazing open-source tools:
- [Playwright](https://playwright.dev/) - Browser automation
- [sentence-transformers](https://www.sbert.net/) - Semantic similarity
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation
- [FastAPI](https://fastapi.tiangolo.com/) - Dashboard API
- [Loguru](https://github.com/Delgan/loguru) - Logging

---

⭐ **Star this repo** if you find it useful! | 🐛 **Report issues** on GitHub | 💬 **Join discussions**

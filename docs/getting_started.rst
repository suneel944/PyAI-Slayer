Getting Started
===============

Installation
------------

.. code-block:: bash

   pip install pyai-slayer

Quick Start
-----------

.. code-block:: python

   from core import BrowserManager, AIResponseValidator
   from config import settings

   # Initialize browser
   browser_manager = BrowserManager()
   browser_manager.start()

   # Create page
   page = browser_manager.create_page()
   page.goto(settings.base_url)

   # Validate AI response
   validator = AIResponseValidator()
   is_relevant, similarity = validator.validate_relevance(
       "How do I renew my visa?",
       "To renew your visa, visit the immigration office..."
   )

   print(f"Relevance: {similarity:.2f}")

Configuration
-------------

Create a ``.env`` file by copying ``.env.example``:

.. code-block:: bash

   cp .env.example .env
   # Edit .env with your credentials and settings

The ``.env`` file should include:

.. code-block:: env

   # Environment Configuration
   ENVIRONMENT=sandbox

   # Application URLs
   BASE_URL=https://your-chatbot-url.example.com
   CHAT_URL=https://your-chatbot-url.example.com

   # Authentication
   EMAIL=your.email@example.com
   PASSWORD=your_password_here

   # Test Configuration
   TEST_TIMEOUT=30
   MAX_RETRIES=3
   PARALLEL_WORKERS=4

   # Browser Configuration
   BROWSER=chromium
   HEADLESS=true
   BROWSER_TIMEOUT=30000

   # AI/ML Model Configuration
   SEMANTIC_MODEL_NAME=intfloat/multilingual-e5-base
   ARABIC_SEMANTIC_MODEL_NAME=Omartificial-Intelligence-Space/mmbert-base-arabic-nli

   # Validation Thresholds
   SEMANTIC_SIMILARITY_THRESHOLD=0.7
   ARABIC_SEMANTIC_SIMILARITY_THRESHOLD=0.5
   HALLUCINATION_DETECTION_THRESHOLD=0.3
   CROSS_LANGUAGE_CONSISTENCY_THRESHOLD=0.7
   CONSISTENCY_THRESHOLD=0.6
   MIN_RESPONSE_LENGTH=10
   MAX_RESPONSE_TIME_SECONDS=180

   # Chatbot Configuration
   CHATBOT_NAME=ChatBot

   # Observability & Tracing
   ENABLE_PLAYWRIGHT_TRACING=true
   ENABLE_PROMETHEUS_METRICS=false
   PROMETHEUS_PORT=8000
   TRACE_DIR=traces

Running Tests
-------------

Using Makefile (recommended):

.. code-block:: bash

   # Run integration/UI/security tests (excludes unit tests)
   make test

   # Run unit tests only
   make test-unit

   # Run all tests
   make test-all

   # Run with coverage
   make test-cov

   # Run checks (lint + type-check)
   make check

   # Run CI pipeline
   make ci

Using pytest directly:

.. code-block:: bash

   # Run all tests
   pytest tests/

   # Run with markers
   pytest -m smoke
   pytest -m security

   # Run with coverage
   pytest --cov=src --cov-report=html


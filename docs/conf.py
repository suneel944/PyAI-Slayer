# docs/conf.py

import os
import sys

# Make src/ importable for autodoc
sys.path.insert(0, os.path.abspath("../src"))

project = "PyAI-Slayer"
author = "PyAI-Slayer Contributors"

# If you want version from pyproject later, you can wire it here
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
]

# IMPORTANT: .md uses "markdown", NOT "myst_parser"
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# DO NOT define source_parsers at all with Sphinx 8
# source_parsers = {...}  # <- if you had this, delete it

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "substitution",
    "tasklist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
]

# HTML output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]  # either create docs/_static or set this to []

# Autosummary
autosummary_generate = True

# Intersphinx mappings (python / pytest are fine; playwright URL can 404 but that's only a warning)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pytest": ("https://docs.pytest.org/en/stable", None),
    # Playwright inventory URL is a bit flaky; you can keep or drop it:
    # "playwright": ("https://playwright.dev/python", None),
}

todo_include_todos = True

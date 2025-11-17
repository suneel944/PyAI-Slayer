"""Sphinx configuration for PyAI-Slayer documentation."""
import os
import sys
from pathlib import Path

# Add src to path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Project information
project = "PyAI-Slayer"
copyright = "2025, PyAI-Slayer Contributors"
author = "PyAI-Slayer Contributors"
release = "0.1.0"

# Extensions
extensions = [
    "myst_parser",  # Markdown support
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # Google/NumPy style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
]

# Source file extensions
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst_parser",
}

# MyST Parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_mock_imports = [
    "playwright",
    "sentence_transformers",
    "transformers",
    "torch",
    "numpy",
    "loguru",
    "pydantic",
    "pydantic_settings",
]

# Autosummary settings
autosummary_generate = True

# HTML theme
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": False,
}

# Output options
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pytest": ("https://docs.pytest.org/en/stable/", None),
    "playwright": ("https://playwright.dev/python/docs/api/", None),
}

# Todo settings
todo_include_todos = True

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


#!/usr/bin/env python3
"""Validate semantic versioning for releases."""

import re
import sys
from pathlib import Path


def get_version_from_pyproject() -> str:
    """Get version from pyproject.toml."""
    # Try current directory first, then parent directory (for scripts/)
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        # If running from scripts/ directory, look in parent
        script_dir = Path(__file__).parent
        pyproject_path = script_dir.parent / "pyproject.toml"

    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found")

    content = pyproject_path.read_text()
    match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError("Version not found in pyproject.toml")

    return match.group(1)


def is_valid_semver(version: str) -> bool:
    """
    Check if version follows semantic versioning.

    Args:
        version: Version string

    Returns:
        True if valid semver
    """
    pattern = r"^\d+\.\d+\.\d+(?:-[a-zA-Z0-9-]+)?(?:\+[a-zA-Z0-9-]+)?$"
    return bool(re.match(pattern, version))


def main():
    """Validate version."""
    try:
        version = get_version_from_pyproject()
        if not is_valid_semver(version):
            print(f"ERROR: Version '{version}' does not follow semantic versioning (X.Y.Z)")
            sys.exit(1)
        print(f"✓ Version '{version}' is valid")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

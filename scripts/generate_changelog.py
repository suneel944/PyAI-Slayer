#!/usr/bin/env python3
"""Generate changelog from git commits."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_git_tags() -> list[str]:
    """Get all git tags."""
    try:
        result = subprocess.run(
            ["git", "tag", "--sort=-creatordate"], capture_output=True, text=True, check=True
        )
        return [tag.strip() for tag in result.stdout.splitlines() if tag.strip()]
    except subprocess.CalledProcessError:
        return []


def get_commits_since_tag(tag: str | None = None) -> list[dict]:
    """Get commits since a tag."""
    if tag:
        cmd = ["git", "log", f"{tag}..HEAD", "--pretty=format:%h|%s|%an|%ad", "--date=short"]
    else:
        cmd = ["git", "log", "--pretty=format:%h|%s|%an|%ad", "--date=short"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        commits = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.split("|", 3)
            if len(parts) >= 4:
                commits.append(
                    {"hash": parts[0], "message": parts[1], "author": parts[2], "date": parts[3]}
                )
        return commits
    except subprocess.CalledProcessError:
        return []


def categorize_commit(message: str) -> str:
    """Categorize commit message."""
    message_lower = message.lower()

    if message.startswith("feat") or "add" in message_lower:
        return "Added"
    elif message.startswith("fix") or "fix" in message_lower:
        return "Fixed"
    elif message.startswith("refactor") or "refactor" in message_lower:
        return "Changed"
    elif message.startswith("docs") or "doc" in message_lower:
        return "Documentation"
    elif message.startswith("test") or "test" in message_lower:
        return "Tests"
    elif message.startswith("chore") or "chore" in message_lower:
        return "Chore"
    else:
        return "Changed"


def generate_changelog_entry(commits: list[dict], version: str) -> str:
    """Generate changelog entry."""
    categorized: dict[str, list[str]] = {}

    for commit in commits:
        category = categorize_commit(commit["message"])
        if category not in categorized:
            categorized[category] = []
        categorized[category].append(f"- {commit['message']} ({commit['hash']})")

    lines = [f"## [{version}] - {datetime.now().strftime('%Y-%m-%d')}", ""]

    for category in ["Added", "Changed", "Fixed", "Documentation", "Tests", "Chore"]:
        if category in categorized:
            lines.append(f"### {category}")
            lines.extend(categorized[category])
            lines.append("")

    return "\n".join(lines)


def main():
    """Generate changelog."""
    # Ensure we're in project root (handle running from scripts/ directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    changelog_path = project_root / "CHANGELOG.md"

    # Get latest tag
    tags = get_git_tags()
    latest_tag = tags[0] if tags else None

    # Get commits since latest tag
    commits = get_commits_since_tag(latest_tag)

    if not commits:
        print("No new commits since last tag")
        return

    # Get version from pyproject.toml
    try:
        # Add scripts directory to path for import
        scripts_path = Path(__file__).parent
        if str(scripts_path) not in sys.path:
            sys.path.insert(0, str(scripts_path))

        # Change to project root for version check
        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(project_root)
            from validate_version import get_version_from_pyproject

            version = get_version_from_pyproject()
        finally:
            os.chdir(original_cwd)
    except Exception as e:
        print(f"Error getting version: {e}")
        sys.exit(1)

    # Generate entry
    entry = generate_changelog_entry(commits, version)

    # Read existing changelog
    if changelog_path.exists():
        content = changelog_path.read_text()
        # Insert after "## [Unreleased]"
        if "## [Unreleased]" in content:
            parts = content.split("## [Unreleased]", 1)
            new_content = f"{parts[0]}## [Unreleased]\n\n{entry}\n{parts[1]}"
        else:
            new_content = f"{entry}\n\n{content}"
    else:
        new_content = f"# Changelog\n\n{entry}\n"

    changelog_path.write_text(new_content)
    print(f"✓ Changelog updated with {len(commits)} commits")


if __name__ == "__main__":
    main()

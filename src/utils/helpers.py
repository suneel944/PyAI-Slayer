"""Helper utility functions."""

import json
from pathlib import Path
from typing import Any, cast

from loguru import logger


def load_json_file(file_path: str) -> dict[str, Any]:
    """
    Load JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary with file contents
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            return cast(dict[str, Any], json.load(f))
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        return {}


def save_json_file(data: dict[str, Any], file_path: str):
    """
    Save data to JSON file.

    Args:
        data: Data to save
        file_path: Path to save file
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved data to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save {file_path}: {e}")


def get_test_data(test_file: str, category: str | None = None) -> list[dict[str, Any]]:
    """
    Load test data from JSON file.

    Args:
        test_file: Test data file name
        category: Optional category to filter (e.g., 'common_queries')

    Returns:
        List of test data items
    """
    file_path = f"tests/test_data/prompts/{test_file}"
    data = load_json_file(file_path)

    if category:
        category_data = data.get(category, [])
        if isinstance(category_data, list):
            return cast(list[dict[str, Any]], category_data)
        return []

    if isinstance(data, list):
        return cast(list[dict[str, Any]], data)
    return []


def calculate_response_time(start_time: float, end_time: float) -> float:
    """Calculate response time in seconds."""
    return end_time - start_time


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def expand_placeholder(item: Any) -> Any:
    """
    Expand placeholder objects in test data.

    Supports:
    - {"type": "repeat", "value": "text", "times": N} -> "text" repeated N times

    Args:
        item: Item that may be a placeholder object or regular value

    Returns:
        Expanded value or original item if not a placeholder
    """
    if isinstance(item, dict) and item.get("type") == "repeat":
        value = item.get("value", "")
        times = item.get("times", 1)
        return str(value) * int(times)
    return item


def expand_payload_list(payloads: list[Any]) -> list[str]:
    """
    Expand all placeholder objects in a payload list.

    Args:
        payloads: List of payloads (may contain placeholder objects)

    Returns:
        List of expanded payload strings
    """
    expanded = []
    for item in payloads:
        expanded_item = expand_placeholder(item)
        if isinstance(expanded_item, str):
            expanded.append(expanded_item)
        elif isinstance(expanded_item, list):
            expanded.extend(expand_payload_list(expanded_item))
    return expanded

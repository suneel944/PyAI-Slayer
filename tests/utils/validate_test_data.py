"""Test data validation script.

This script validates the structure and content of test data files
to ensure they meet the required schema and contain valid data.
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger


def validate_test_data_file(file_path: Path, schema: dict[str, Any]) -> list[str]:
    """
    Validate a test data file against a schema.

    Args:
        file_path: Path to the test data file
        schema: Schema definition with required fields per category

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return [f"File not found: {file_path}"]
    except json.JSONDecodeError as e:
        return [f"Invalid JSON in {file_path}: {e}"]

    # Validate each category
    for category, category_schema in schema.items():
        if category not in data:
            continue  # Category is optional

        category_data = data[category]
        if not isinstance(category_data, list):
            errors.append(f"{category} must be a list in {file_path}")
            continue

        required_fields = category_schema.get("required_fields", [])

        for idx, item in enumerate(category_data):
            if not isinstance(item, dict):
                errors.append(f"{category}[{idx}] must be a dictionary in {file_path}")
                continue

            # Check required fields
            for field in required_fields:
                if field not in item:
                    errors.append(
                        f"{category}[{idx}] missing required field '{field}' in {file_path}"
                    )

            # Validate field types
            for field, field_type in category_schema.get("field_types", {}).items():
                if field in item and not isinstance(item[field], field_type):
                    errors.append(
                        f"{category}[{idx}].{field} must be {field_type.__name__} in {file_path}"
                    )

            # Validate specific field constraints
            if "id" in item and (not isinstance(item["id"], str) or not item["id"]):
                errors.append(f"{category}[{idx}].id must be a non-empty string in {file_path}")

            if "prompt" in item and not isinstance(item["prompt"], str):
                errors.append(f"{category}[{idx}].prompt must be a string in {file_path}")

            if "min_response_length" in item and (
                not isinstance(item["min_response_length"], (int, float))
                or item["min_response_length"] < 0
            ):
                errors.append(
                    f"{category}[{idx}].min_response_length must be a non-negative number in {file_path}"
                )

            if "expected_keywords" in item:
                if not isinstance(item["expected_keywords"], list):
                    errors.append(
                        f"{category}[{idx}].expected_keywords must be a list in {file_path}"
                    )
                elif len(item["expected_keywords"]) == 0:
                    errors.append(
                        f"{category}[{idx}].expected_keywords must be non-empty if present in {file_path}"
                    )

    return errors


def validate_test_data_en() -> list[str]:
    """Validate test-data-en.json file."""
    file_path = Path("tests/test_data/prompts/test-data-en.json")
    schema = {
        "common_queries": {
            "required_fields": ["id", "prompt", "min_response_length"],
            "optional_fields": ["expected_keywords", "max_response_time_seconds", "category"],
            "field_types": {
                "id": str,
                "prompt": str,
                "expected_keywords": list,
                "min_response_length": (int, float),
                "max_response_time_seconds": (int, float),
                "category": str,
            },
        },
        "edge_cases": {
            "required_fields": ["id", "prompt", "expected_behavior", "min_response_length"],
            "optional_fields": [],
            "field_types": {
                "id": str,
                "prompt": str,
                "expected_behavior": str,
                "min_response_length": (int, float),
            },
        },
        "general_public_prompts": {
            "required_fields": ["id", "prompt", "expected_keywords", "min_response_length"],
            "optional_fields": [
                "max_response_time_seconds",
                "category",
                "user_type",
                "requires_citation",
            ],
            "field_types": {
                "id": str,
                "prompt": str,
                "expected_keywords": list,
                "min_response_length": (int, float),
                "max_response_time_seconds": (int, float),
                "category": str,
                "user_type": str,
                "requires_citation": bool,
            },
        },
        "security_testing_prompts": {
            "required_fields": ["id", "prompt", "expected_behavior"],
            "optional_fields": [
                "expected_keywords",
                "min_response_length",
                "max_response_time_seconds",
                "category",
            ],
            "field_types": {
                "id": str,
                "prompt": str,
                "expected_behavior": str,
                "expected_keywords": list,
                "min_response_length": (int, float),
                "max_response_time_seconds": (int, float),
                "category": str,
            },
        },
        "multi_turn": {
            "required_fields": ["id", "turns"],
            "optional_fields": ["user_type"],
            "field_types": {
                "id": str,
                "turns": list,
                "user_type": str,
            },
        },
    }

    errors = validate_test_data_file(file_path, schema)

    # Additional validation for multi_turn
    if file_path.exists():
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
                if "multi_turn" in data:
                    for idx, item in enumerate(data["multi_turn"]):
                        if "turns" in item:
                            if not isinstance(item["turns"], list):
                                errors.append(
                                    f"multi_turn[{idx}].turns must be a list in {file_path}"
                                )
                            else:
                                for turn_idx, turn in enumerate(item["turns"]):
                                    if not isinstance(turn, dict):
                                        errors.append(
                                            f"multi_turn[{idx}].turns[{turn_idx}] must be a dictionary in {file_path}"
                                        )
                                    elif "prompt" not in turn:
                                        errors.append(
                                            f"multi_turn[{idx}].turns[{turn_idx}] missing required field 'prompt' in {file_path}"
                                        )
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # Already handled in validate_test_data_file

    return errors


def validate_injection_payloads() -> list[str]:
    """Validate injection-payloads.json file."""
    file_path = Path("tests/test_data/prompts/injection-payloads.json")
    errors = []

    if not file_path.exists():
        return [f"File not found: {file_path}"]

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON in {file_path}: {e}"]

    if not isinstance(data, dict):
        return [f"Root must be a dictionary in {file_path}"]

    # Special categories that have different structures
    special_categories = {"multi_turn_attack_scenarios", "context_aware_mutations"}

    # Validate that each category is a list (except special categories)
    for category, payloads in data.items():
        if category in special_categories:
            # Handle special categories
            if category == "multi_turn_attack_scenarios":
                if not isinstance(payloads, list):
                    errors.append(f"{category} must be a list in {file_path}")
                else:
                    for idx, scenario in enumerate(payloads):
                        if not isinstance(scenario, dict):
                            errors.append(f"{category}[{idx}] must be a dictionary in {file_path}")
                        else:
                            if "scenario" not in scenario:
                                errors.append(
                                    f"{category}[{idx}] missing 'scenario' field in {file_path}"
                                )
                            if "turns" not in scenario:
                                errors.append(
                                    f"{category}[{idx}] missing 'turns' field in {file_path}"
                                )
                            elif not isinstance(scenario["turns"], list):
                                errors.append(
                                    f"{category}[{idx}].turns must be a list in {file_path}"
                                )
            elif category == "context_aware_mutations":
                if not isinstance(payloads, dict):
                    errors.append(f"{category} must be a dictionary in {file_path}")
                else:
                    # Each sub-category should be a list
                    for sub_category, sub_payloads in payloads.items():
                        if not isinstance(sub_payloads, list):
                            errors.append(
                                f"{category}.{sub_category} must be a list in {file_path}"
                            )
                        else:
                            for idx, payload in enumerate(sub_payloads):
                                if not isinstance(payload, str):
                                    errors.append(
                                        f"{category}.{sub_category}[{idx}] must be a string in {file_path}"
                                    )
            continue

        if not isinstance(payloads, list):
            errors.append(f"{category} must be a list in {file_path}")
            continue

        # Validate each payload
        for idx, payload in enumerate(payloads):
            # Payload can be a string or a placeholder object
            if isinstance(payload, str):
                if not payload:
                    errors.append(f"{category}[{idx}] is an empty string in {file_path}")
            elif isinstance(payload, dict):
                # Placeholder object
                if payload.get("type") == "repeat":
                    if "value" not in payload:
                        errors.append(f"{category}[{idx}] missing 'value' field in {file_path}")
                    if "times" not in payload:
                        errors.append(f"{category}[{idx}] missing 'times' field in {file_path}")
                    elif not isinstance(payload["times"], int) or payload["times"] < 1:
                        errors.append(
                            f"{category}[{idx}].times must be a positive integer in {file_path}"
                        )
                else:
                    errors.append(
                        f"{category}[{idx}] has unknown placeholder type '{payload.get('type')}' in {file_path}"
                    )
            else:
                errors.append(
                    f"{category}[{idx}] must be a string or placeholder object in {file_path}"
                )

    return errors


def validate_response_schemas() -> list[str]:
    """Validate response-schemas.json file."""
    file_path = Path("tests/test_data/expected/response-schemas.json")
    errors = []

    if not file_path.exists():
        return [f"File not found: {file_path}"]

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON in {file_path}: {e}"]

    if not isinstance(data, dict):
        return [f"Root must be a dictionary in {file_path}"]

    # Validate schema definitions
    if "valid_response_schema" in data:
        schema = data["valid_response_schema"]
        if not isinstance(schema, dict):
            errors.append("valid_response_schema must be a dictionary")
        elif "type" not in schema or schema["type"] != "object":
            errors.append("valid_response_schema.type must be 'object'")

    if "error_response_schema" in data:
        schema = data["error_response_schema"]
        if not isinstance(schema, dict):
            errors.append("error_response_schema must be a dictionary")
        elif "type" not in schema or schema["type"] != "object":
            errors.append("error_response_schema.type must be 'object'")

    if "fallback_messages" in data:
        messages = data["fallback_messages"]
        if not isinstance(messages, list):
            errors.append("fallback_messages must be a list")
        else:
            for idx, msg in enumerate(messages):
                if not isinstance(msg, str) or not msg:
                    errors.append(f"fallback_messages[{idx}] must be a non-empty string")

    return errors


def main():
    """Run all test data validation."""
    logger.info("Starting test data validation...")

    all_errors = []

    # Validate test-data-en.json
    logger.info("Validating test-data-en.json...")
    errors = validate_test_data_en()
    if errors:
        all_errors.extend(errors)
        logger.error(f"Found {len(errors)} errors in test-data-en.json")
        for error in errors:
            logger.error(f"  - {error}")
    else:
        logger.info("✓ test-data-en.json is valid")

    # Validate injection-payloads.json
    logger.info("Validating injection-payloads.json...")
    errors = validate_injection_payloads()
    if errors:
        all_errors.extend(errors)
        logger.error(f"Found {len(errors)} errors in injection-payloads.json")
        for error in errors:
            logger.error(f"  - {error}")
    else:
        logger.info("✓ injection-payloads.json is valid")

    # Validate response-schemas.json
    logger.info("Validating response-schemas.json...")
    errors = validate_response_schemas()
    if errors:
        all_errors.extend(errors)
        logger.error(f"Found {len(errors)} errors in response-schemas.json")
        for error in errors:
            logger.error(f"  - {error}")
    else:
        logger.info("✓ response-schemas.json is valid")

    # Summary
    if all_errors:
        logger.error(f"\nValidation failed with {len(all_errors)} total errors")
        return 1
    else:
        logger.info("\n✓ All test data files are valid")
        return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""Utilities for working with Prometheus metrics endpoint (without Prometheus Server)."""
import json
import re
import sys
from pathlib import Path
from typing import Any

import requests
from loguru import logger


class MetricsClient:
    """Client for fetching and parsing Prometheus metrics."""

    def __init__(self, metrics_url: str = "http://localhost:8000/metrics"):
        """
        Initialize metrics client.

        Args:
            metrics_url: URL to metrics endpoint
        """
        self.metrics_url = metrics_url

    def fetch_metrics(self) -> str:
        """
        Fetch raw metrics from endpoint.

        Returns:
            Raw metrics text

        Raises:
            requests.RequestException: If metrics endpoint is not available
        """
        try:
            response = requests.get(self.metrics_url, timeout=5)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch metrics from {self.metrics_url}: {e}")
            raise

    def parse_metrics(self, metrics_text: str | None = None) -> dict[str, Any]:
        """
        Parse Prometheus metrics text into structured format.

        Args:
            metrics_text: Raw metrics text (fetches if None)

        Returns:
            Dictionary of parsed metrics
        """
        if metrics_text is None:
            metrics_text = self.fetch_metrics()

        metrics = {}

        for line in metrics_text.split("\n"):
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                if line.startswith("# TYPE"):
                    # Extract metric type
                    parts = line.split()
                    if len(parts) >= 3:
                        metric_name = parts[2]
                        metric_type = parts[3]
                        if metric_name not in metrics:
                            metrics[metric_name] = {"type": metric_type, "samples": []}
                continue

            # Parse metric line: metric_name{labels} value
            match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]*)\}\s+(.+)$", line)
            if match:
                metric_name, labels_str, value = match.groups()
                labels = {}
                if labels_str:
                    for label_pair in labels_str.split(","):
                        if "=" in label_pair:
                            key, val = label_pair.split("=", 1)
                            labels[key.strip()] = val.strip('"')

                value = self._parse_value(value)
                metrics.setdefault(metric_name, {"type": "unknown", "samples": []})
                metrics[metric_name]["samples"].append({"labels": labels, "value": value})  # type: ignore[attr-defined]
            else:
                # Simple metric without labels: metric_name value
                match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+(.+)$", line)
                if match:
                    metric_name, value = match.groups()
                    value = self._parse_value(value)
                    metrics.setdefault(metric_name, {"type": "unknown", "samples": []})
                    metrics[metric_name]["samples"].append({"labels": {}, "value": value})  # type: ignore[attr-defined]

        return metrics

    def _parse_value(self, value_str: str) -> float | int | str:
        """Parse metric value (handles NaN, Inf, etc.)."""
        value_str = value_str.strip()
        if value_str == "NaN":
            return float("nan")
        elif value_str == "+Inf":
            return float("inf")
        elif value_str == "-Inf":
            return float("-inf")
        try:
            # Try integer first
            if "." not in value_str:
                return int(value_str)
            return float(value_str)
        except ValueError:
            return value_str

    def get_test_summary(self) -> dict[str, Any]:
        """
        Get test execution summary from metrics.

        Returns:
            Dictionary with test summary statistics
        """
        metrics = self.parse_metrics()
        summary: dict[str, Any] = {
            "tests": {"total": 0, "passed": 0, "failed": 0, "skipped": 0},
            "validations": {"total": 0, "passed": 0, "failed": 0},
            "browser_operations": {"total": 0},
        }

        # Parse test metrics
        test_total = metrics.get("pyai_slayer_test_total", {}).get("samples", [])
        for sample in test_total:
            status = sample.get("labels", {}).get("status", "")
            value = sample.get("value", 0)
            summary["tests"]["total"] += value
            if status == "passed":
                summary["tests"]["passed"] = value
            elif status == "failed":
                summary["tests"]["failed"] = value
            elif status == "skipped":
                summary["tests"]["skipped"] = value

        # Parse validation metrics (with language breakdown)
        validation_total = metrics.get("pyai_slayer_validation_total", {}).get("samples", [])
        validation_by_type: dict[str, dict[str, Any]] = {}
        validation_by_language: dict[str, dict[str, Any]] = {
            "en": {"passed": 0, "failed": 0},
            "ar": {"passed": 0, "failed": 0},
            "multilingual": {"passed": 0, "failed": 0},
        }

        for sample in validation_total:
            status = sample.get("labels", {}).get("status", "")
            validation_type = sample.get("labels", {}).get("type", "unknown")
            language = sample.get("labels", {}).get("language", "unknown")
            value = sample.get("value", 0)

            summary["validations"]["total"] += value
            if status == "passed":
                summary["validations"]["passed"] += value
            elif status == "failed":
                summary["validations"]["failed"] += value

            # Track by type
            if validation_type not in validation_by_type:
                validation_by_type[validation_type] = {"passed": 0, "failed": 0}
            if status == "passed":
                validation_by_type[validation_type]["passed"] += value
            elif status == "failed":
                validation_by_type[validation_type]["failed"] += value

            # Track by language
            if language in validation_by_language:
                if status == "passed":
                    validation_by_language[language]["passed"] += value
                elif status == "failed":
                    validation_by_language[language]["failed"] += value

        summary["validations"]["by_type"] = validation_by_type
        summary["validations"]["by_language"] = validation_by_language

        # Parse browser operations
        browser_ops = metrics.get("pyai_slayer_browser_operations_total", {}).get("samples", [])
        for sample in browser_ops:
            summary["browser_operations"]["total"] += sample.get("value", 0)

        # Parse AI response time metrics
        ai_response_times = metrics.get("pyai_slayer_ai_response_time_seconds", {}).get(
            "samples", []
        )
        response_times_by_lang: dict[str, list[Any]] = {"en": [], "ar": [], "unknown": []}
        all_response_times: list[Any] = []

        for sample in ai_response_times:
            response_time = sample.get("value", 0)
            language = sample.get("labels", {}).get("language", "unknown")
            all_response_times.append(response_time)
            if language in response_times_by_lang:
                response_times_by_lang[language].append(response_time)
            else:
                response_times_by_lang["unknown"].append(response_time)

        summary["ai_response_times"] = {
            "total": len(all_response_times),
            "avg": sum(all_response_times) / len(all_response_times) if all_response_times else 0.0,
            "max": max(all_response_times) if all_response_times else 0.0,
            "min": min(all_response_times) if all_response_times else 0.0,
            "by_language": {
                lang: {
                    "count": len(times),
                    "avg": sum(times) / len(times) if times else 0.0,
                    "max": max(times) if times else 0.0,
                }
                for lang, times in response_times_by_lang.items()
            },
        }

        # Parse similarity score metrics
        similarity_scores = metrics.get("pyai_slayer_validation_similarity_score", {}).get(
            "samples", []
        )
        similarity_by_type_lang: dict[str, list[Any]] = {}

        for sample in similarity_scores:
            score = sample.get("value", 0)
            validation_type = sample.get("labels", {}).get("type", "unknown")
            language = sample.get("labels", {}).get("language", "unknown")
            key = f"{validation_type}_{language}"

            if key not in similarity_by_type_lang:
                similarity_by_type_lang[key] = []
            similarity_by_type_lang[key].append(score)

        summary["similarity_scores"] = {
            key: {
                "count": len(scores),
                "avg": sum(scores) / len(scores) if scores else 0.0,
                "min": min(scores) if scores else 0.0,
                "max": max(scores) if scores else 0.0,
            }
            for key, scores in similarity_by_type_lang.items()
        }

        return summary

    def export_json(self, output_file: str | Path) -> None:
        """
        Export metrics to JSON file.

        Args:
            output_file: Path to output JSON file
        """
        metrics = self.parse_metrics()
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics exported to {output_path}")

    def print_summary(self) -> None:
        """Print human-readable metrics summary."""
        try:
            summary = self.get_test_summary()
            print("\n" + "=" * 60)
            print("Test Execution Summary")
            print("=" * 60)
            print("\nTests:")
            print(f"  Total:   {summary['tests']['total']}")
            print(f"  Passed:  {summary['tests']['passed']}")
            print(f"  Failed:  {summary['tests']['failed']}")
            print(f"  Skipped: {summary['tests']['skipped']}")

            if summary["tests"]["total"] > 0:
                pass_rate = (summary["tests"]["passed"] / summary["tests"]["total"]) * 100
                print(f"  Pass Rate: {pass_rate:.1f}%")

            print("\nValidations:")
            print(f"  Total:   {summary['validations']['total']}")
            print(f"  Passed:  {summary['validations']['passed']}")
            print(f"  Failed:  {summary['validations']['failed']}")

            # Validation breakdown by type
            if "by_type" in summary["validations"]:
                print("\n  By Type:")
                for vtype, counts in summary["validations"]["by_type"].items():
                    total_type = counts["passed"] + counts["failed"]
                    if total_type > 0:
                        pass_rate = (counts["passed"] / total_type) * 100
                        print(
                            f"    {vtype}: {counts['passed']}/{total_type} passed ({pass_rate:.1f}%)"
                        )

            # Validation breakdown by language
            if "by_language" in summary["validations"]:
                print("\n  By Language:")
                for lang, counts in summary["validations"]["by_language"].items():
                    total_lang = counts["passed"] + counts["failed"]
                    if total_lang > 0:
                        pass_rate = (counts["passed"] / total_lang) * 100
                        print(
                            f"    {lang}: {counts['passed']}/{total_lang} passed ({pass_rate:.1f}%)"
                        )

            # AI Response Times
            if "ai_response_times" in summary and summary["ai_response_times"]["total"] > 0:
                print("\nAI Response Times:")
                rt = summary["ai_response_times"]
                print(f"  Total:   {rt['total']}")
                print(f"  Average: {rt['avg']:.2f}s")
                print(f"  Min:     {rt['min']:.2f}s")
                print(f"  Max:     {rt['max']:.2f}s")

                if "by_language" in rt:
                    print("\n  By Language:")
                    for lang, stats in rt["by_language"].items():
                        if stats["count"] > 0:
                            print(
                                f"    {lang}: {stats['avg']:.2f}s avg ({stats['count']} samples, max: {stats['max']:.2f}s)"
                            )

            # Similarity Scores
            if "similarity_scores" in summary and summary["similarity_scores"]:
                print("\nSimilarity Scores:")
                for key, stats in summary["similarity_scores"].items():
                    if stats["count"] > 0:
                        vtype, lang = key.split("_", 1) if "_" in key else (key, "unknown")
                        print(
                            f"  {vtype} ({lang}): avg={stats['avg']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f} ({stats['count']} samples)"
                        )

            print("\nBrowser Operations:")
            print(f"  Total:   {summary['browser_operations']['total']}")
            print("=" * 60 + "\n")
        except Exception as e:
            logger.error(f"Failed to print summary: {e}")
            sys.exit(1)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Utilities for working with Prometheus metrics endpoint"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/metrics",
        help="Metrics endpoint URL (default: http://localhost:8000/metrics)",
    )
    parser.add_argument("--summary", action="store_true", help="Print test execution summary")
    parser.add_argument("--export", type=str, help="Export metrics to JSON file")
    parser.add_argument("--raw", action="store_true", help="Print raw metrics text")

    args = parser.parse_args()

    client = MetricsClient(args.url)

    try:
        if args.raw:
            print(client.fetch_metrics())
        elif args.export:
            client.export_json(args.export)
            logger.info(f"Metrics exported to {args.export}")
        else:
            # Default: print summary
            client.print_summary()
    except requests.RequestException:
        logger.error(f"Metrics endpoint not available at {args.url}")
        logger.info("Make sure:")
        logger.info("  1. Tests are running with enable_prometheus_metrics=True")
        logger.info("  2. Metrics server is started (happens automatically)")
        logger.info("  3. Port 8000 is accessible")
        sys.exit(1)


if __name__ == "__main__":
    main()

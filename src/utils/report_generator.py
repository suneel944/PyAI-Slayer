"""Report generation utilities."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


class ReportGenerator:
    """Generates test reports in various formats."""

    def __init__(self, report_dir: str = "reports"):
        """
        Initialize report generator.

        Args:
            report_dir: Directory for reports
        """
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Report generator initialized: {self.report_dir}")

    def generate_summary_report(
        self, test_results: list[dict[str, Any]], output_file: str = "test_summary.json"
    ) -> str:
        """
        Generate summary report from test results.

        Args:
            test_results: List of test result dictionaries
            output_file: Output filename

        Returns:
            Path to generated report
        """
        try:
            total_tests = len(test_results)
            passed = sum(1 for r in test_results if r.get("status") == "passed")
            failed = sum(1 for r in test_results if r.get("status") == "failed")
            skipped = sum(1 for r in test_results if r.get("status") == "skipped")

            summary = {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total": total_tests,
                    "passed": passed,
                    "failed": failed,
                    "skipped": skipped,
                    "pass_rate": (
                        f"{(passed / total_tests * 100):.2f}%" if total_tests > 0 else "0%"
                    ),
                },
                "results": test_results,
            }

            output_path = self.report_dir / output_file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            logger.info(f"Summary report generated: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
            return ""

    def generate_markdown_report(
        self, test_results: list[dict[str, Any]], output_file: str = "test_report.md"
    ) -> str:
        """
        Generate markdown report.

        Args:
            test_results: List of test result dictionaries
            output_file: Output filename

        Returns:
            Path to generated report
        """
        try:
            total_tests = len(test_results)
            passed = sum(1 for r in test_results if r.get("status") == "passed")
            failed = sum(1 for r in test_results if r.get("status") == "failed")

            md_content = f"""

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}



- **Total Tests:** {total_tests}
- **Passed:** {passed}
- **Failed:** {failed}
- **Pass Rate:** {(passed / total_tests * 100):.2f}% if total_tests > 0 else 0%



| Test ID | Status | Duration | Notes |
|---------|--------|----------|-------|
"""

            for result in test_results:
                test_id = result.get("test_id", "N/A")
                status = result.get("status", "unknown")
                duration = result.get("duration", "N/A")
                notes = result.get("notes", "")

                status_emoji = "✅" if status == "passed" else "❌" if status == "failed" else "⏭️"
                md_content += f"| {test_id} | {status_emoji} {status} | {duration} | {notes} |\n"

            output_path = self.report_dir / output_file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            logger.info(f"Markdown report generated: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to generate markdown report: {e}")
            return ""

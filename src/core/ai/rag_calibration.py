"""Calibrate RAG metric thresholds using labeled evaluation set."""

from collections import defaultdict

from loguru import logger

from core.ai.rag_eval_set import RAGEvalSet
from dashboard.calculators.rag import RAGMetricsCalculator


class RAGCalibrator:
    """Calibrate RAG metrics using labeled evaluation data."""

    def __init__(
        self, eval_set: RAGEvalSet, metrics_calculator: RAGMetricsCalculator | None = None
    ):
        """
        Initialize calibrator.

        Args:
            eval_set: Labeled evaluation dataset
            metrics_calculator: RAG metrics calculator (default: creates new instance)
        """
        self.eval_set = eval_set
        self.metrics_calculator = metrics_calculator or RAGMetricsCalculator()

    def compute_ground_truth_metrics(self) -> dict[str, list[float]]:
        """
        Compute ground truth metrics from labeled eval set.

        Returns:
            Dictionary mapping metric names to lists of values
        """
        metrics: dict[str, list[float]] = defaultdict(list)

        for example in self.eval_set.examples:
            # Extract retrieved docs from chunks (simulate retrieval)
            retrieved_docs = [chunk.chunk_text for chunk in example.chunks]
            expected_sources = example.expected_sources

            # Compute ground truth retrieval metrics
            if expected_sources and retrieved_docs:
                # For each expected source, check if it's in retrieved docs
                found_sources = 0
                for expected in expected_sources:
                    # Simple check: if any chunk text contains expected source keywords
                    # In practice, you'd have better source matching
                    if any(expected.lower() in doc.lower() for doc in retrieved_docs):
                        found_sources += 1

                if expected_sources:
                    recall = (found_sources / len(expected_sources)) * 100
                    metrics["retrieval_recall_5"].append(recall)

                # Precision: relevant chunks in top 5
                relevant_chunks = [chunk for chunk in example.chunks[:5] if chunk.relevance >= 2]
                precision = (len(relevant_chunks) / min(len(retrieved_docs), 5)) * 100
                metrics["retrieval_precision_5"].append(precision)

            # Compute proxy metrics using calculator
            if example.gold_context and retrieved_docs:
                try:
                    # Simulate a response (in practice, you'd use actual model responses)
                    response = example.gold_answer or example.gold_context

                    calc_metrics = self.metrics_calculator.calculate(
                        query=example.query,
                        response=response,
                        retrieved_docs=retrieved_docs,
                        expected_sources=expected_sources,
                        gold_context=example.gold_context,
                    )

                    for metric_name, value in calc_metrics.items():
                        if value is not None:
                            metrics[metric_name].append(value)

                except Exception as e:
                    logger.debug(f"Failed to compute metrics for example: {e}")

        return dict(metrics)

    def calibrate_thresholds(self) -> dict[str, dict[str, float]]:
        """
        Calibrate thresholds and targets based on eval set distribution.

        Returns:
            Dictionary with recommended thresholds and targets for each metric
        """
        ground_truth = self.compute_ground_truth_metrics()

        recommendations: dict[str, dict[str, float]] = {}

        for metric_name, values in ground_truth.items():
            if not values:
                continue

            import statistics

            mean_val = statistics.mean(values)
            median_val = statistics.median(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0.0
            min_val = min(values)
            max_val = max(values)

            # Percentiles
            sorted_vals = sorted(values)
            p25 = sorted_vals[len(sorted_vals) // 4] if sorted_vals else 0.0
            p75 = sorted_vals[3 * len(sorted_vals) // 4] if sorted_vals else 0.0
            p90 = sorted_vals[9 * len(sorted_vals) // 10] if sorted_vals else 0.0

            # Recommend targets based on distribution
            # For most metrics, target should be around p75 or mean + 0.5*std
            if metric_name in [
                "context_relevance",
                "gold_context_match",
                "retrieval_recall_5",
                "retrieval_precision_5",
            ]:
                # Higher is better - target should be above median
                target = max(median_val, p75)
            elif metric_name == "context_intrusion":
                # Lower is better - target should be below median
                target = min(median_val, p25)
            else:
                # Default: use median
                target = median_val

            recommendations[metric_name] = {
                "mean": round(mean_val, 2),
                "median": round(median_val, 2),
                "std": round(std_val, 2),
                "min": round(min_val, 2),
                "max": round(max_val, 2),
                "p25": round(p25, 2),
                "p75": round(p75, 2),
                "p90": round(p90, 2),
                "recommended_target": round(target, 2),
                "sample_size": len(values),
            }

        return recommendations

    def generate_report(self) -> str:
        """
        Generate calibration report with recommendations.

        Returns:
            Formatted report string
        """
        recommendations = self.calibrate_thresholds()

        report_lines = [
            "=" * 80,
            "RAG METRICS CALIBRATION REPORT",
            "=" * 80,
            f"Evaluation Set: {self.eval_set.name} (v{self.eval_set.version})",
            f"Number of Examples: {len(self.eval_set)}",
            "",
            "RECOMMENDED TARGETS:",
            "-" * 80,
        ]

        for metric_name, stats in sorted(recommendations.items()):
            report_lines.append(f"\n{metric_name.upper()}:")
            report_lines.append(f"  Sample Size: {stats['sample_size']}")
            report_lines.append(f"  Mean: {stats['mean']:.2f}")
            report_lines.append(f"  Median: {stats['median']:.2f}")
            report_lines.append(f"  Std Dev: {stats['std']:.2f}")
            report_lines.append(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
            report_lines.append(
                f"  Percentiles: P25={stats['p25']:.2f}, P75={stats['p75']:.2f}, P90={stats['p90']:.2f}"
            )
            report_lines.append(f"  ⭐ RECOMMENDED TARGET: {stats['recommended_target']:.2f}%")

        report_lines.append("\n" + "=" * 80)
        report_lines.append("USAGE:")
        report_lines.append("  Update dashboard targets based on recommended_target values above.")
        report_lines.append("  These targets are calibrated to your actual data distribution.")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)

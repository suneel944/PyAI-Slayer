"""Load and manage RAG metric targets from calibration recommendations."""

import json
from pathlib import Path
from typing import Any

from loguru import logger

from config.settings import settings


class RAGTargets:
    """Manage RAG metric targets from calibration recommendations."""

    def __init__(self, recommendations_path: str | Path | None = None):
        """
        Initialize RAG targets loader.

        Args:
            recommendations_path: Path to calibration recommendations JSON file
                (default: data/rag_calibration_recommendations.json)
        """
        if recommendations_path is None:
            recommendations_path = Path("data/rag_calibration_recommendations.json")
        self.recommendations_path = Path(recommendations_path)
        self._targets: dict[str, float] | None = None

    def load_targets(self) -> dict[str, float]:
        """
        Load targets with priority: .env > calibration file > defaults.

        Returns:
            Dictionary mapping metric names to target values
        """
        if self._targets is not None:
            return self._targets

        targets: dict[str, float] = {}

        # Priority 1: Check .env settings (highest priority)
        env_targets = self._load_from_env()
        if env_targets:
            logger.info(f"Loaded {len(env_targets)} RAG metric targets from .env settings")
            self._targets = env_targets
            return env_targets

        # Priority 2: Load from calibration recommendations file
        if self.recommendations_path.exists():
            try:
                with open(self.recommendations_path, encoding="utf-8") as f:
                    recommendations = json.load(f)

                # Extract recommended_target for each metric
                for metric_name, stats in recommendations.items():
                    if isinstance(stats, dict) and "recommended_target" in stats:
                        targets[metric_name] = float(stats["recommended_target"])

                if targets:
                    logger.info(
                        f"Loaded {len(targets)} RAG metric targets from {self.recommendations_path}"
                    )
                    self._targets = targets
                    return targets
            except Exception as e:
                logger.warning(f"Failed to load calibration recommendations: {e}")

        # Priority 3: Use defaults
        logger.info("Using default RAG metric targets")
        defaults = self._get_default_targets()
        self._targets = defaults
        return defaults

    def _load_from_env(self) -> dict[str, float] | None:
        """
        Load targets from .env settings if any are configured.

        Returns:
            Dictionary of targets from .env, or None if none are set
        """
        targets: dict[str, float] = {}
        env_configured = False

        # Check each target setting
        if settings.rag_target_retrieval_recall_5 is not None:
            targets["retrieval_recall_5"] = settings.rag_target_retrieval_recall_5
            env_configured = True
        if settings.rag_target_retrieval_precision_5 is not None:
            targets["retrieval_precision_5"] = settings.rag_target_retrieval_precision_5
            env_configured = True
        if settings.rag_target_context_relevance is not None:
            targets["context_relevance"] = settings.rag_target_context_relevance
            env_configured = True
        if settings.rag_target_context_coverage is not None:
            targets["context_coverage"] = settings.rag_target_context_coverage
            env_configured = True
        if settings.rag_target_context_intrusion is not None:
            targets["context_intrusion"] = settings.rag_target_context_intrusion
            env_configured = True
        if settings.rag_target_gold_context_match is not None:
            targets["gold_context_match"] = settings.rag_target_gold_context_match
            env_configured = True
        if settings.rag_target_reranker_score is not None:
            targets["reranker_score"] = settings.rag_target_reranker_score
            env_configured = True

        # Fill in missing targets from defaults if only some are set
        if env_configured:
            defaults = self._get_default_targets()
            for key, value in defaults.items():
                if key not in targets:
                    targets[key] = value
            return targets

        return None

    def get_target(self, metric_name: str) -> float:
        """
        Get target value for a specific metric.

        Args:
            metric_name: Name of the metric (e.g., 'retrieval_precision_5')

        Returns:
            Target value (defaults to sensible fallback if not found)
        """
        targets = self.load_targets()
        return targets.get(metric_name, self._get_default_target(metric_name))

    def _get_default_targets(self) -> dict[str, float]:
        """Get default targets (fallback when calibration file is missing)."""
        return {
            "retrieval_recall_5": 85.0,
            "retrieval_precision_5": 85.0,
            "context_relevance": 80.0,
            "context_coverage": 80.0,
            "context_intrusion": 5.0,  # Lower is better
            "gold_context_match": 85.0,
            "reranker_score": 0.8,  # 0-1 scale
        }

    def _get_default_target(self, metric_name: str) -> float:
        """Get default target for a specific metric."""
        defaults = self._get_default_targets()
        return defaults.get(metric_name, 0.0)

    def to_json(self) -> dict[str, Any]:
        """
        Export targets as JSON-serializable dict for frontend use.

        Returns:
            Dictionary with metric names and targets
        """
        targets = self.load_targets()
        return {
            "retrieval_recall_5": targets.get("retrieval_recall_5", 85.0),
            "retrieval_precision_5": targets.get("retrieval_precision_5", 85.0),
            "context_relevance": targets.get("context_relevance", 80.0),
            "context_coverage": targets.get("context_coverage", 80.0),
            "context_intrusion": targets.get("context_intrusion", 5.0),
            "gold_context_match": targets.get("gold_context_match", 85.0),
            "reranker_score": targets.get("reranker_score", 0.8),
        }


# Global instance
_rag_targets: RAGTargets | None = None


def get_rag_targets() -> RAGTargets:
    """Get global RAG targets instance."""
    global _rag_targets
    if _rag_targets is None:
        _rag_targets = RAGTargets()
    return _rag_targets

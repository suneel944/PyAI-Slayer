"""Detector classes for heavy/complex dependencies (toxicity, etc.)."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from transformers import TextClassificationPipeline


class ToxicityDetector(ABC):
    """Abstract interface for toxicity detection."""

    @abstractmethod
    def score(self, text: str) -> float | None:
        """
        Score text for toxicity.

        Args:
            text: Text to score

        Returns:
            Toxicity score (0.0-1.0) or None if unavailable
        """
        pass


class HuggingFaceToxicityDetector(ToxicityDetector):
    """Toxicity detector using HuggingFace transformers."""

    def __init__(self, model_name: str | None = None, enabled: bool = True):
        """
        Initialize toxicity detector.

        Args:
            model_name: Model name (default: tries multiple fallbacks)
            enabled: Whether to enable (if False, returns None)
        """
        self.enabled = enabled
        self.model_name = model_name
        self._pipeline: TextClassificationPipeline | None = None
        self._loaded = False

    def _load_model(self) -> bool:
        """Lazy load the toxicity model."""
        if self._loaded:
            return self._pipeline is not None

        if not self.enabled:
            return False

        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )

            # Try primary model
            model_name = self.model_name or "martin-ha/toxic-comment-model"
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self._pipeline = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if torch.cuda.is_available() else -1,
                    return_all_scores=True,
                )
                self._loaded = True
                logger.info(f"Loaded toxicity model: {model_name}")
                return True
            except Exception:
                # Fallback to simpler model
                try:
                    self._pipeline = pipeline(
                        "text-classification",
                        model="unitary/toxic-bert",
                        device=0 if torch.cuda.is_available() else -1,
                        return_all_scores=True,
                    )
                    self._loaded = True
                    logger.info("Loaded fallback toxicity model: unitary/toxic-bert")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load toxicity models: {e}")
                    self._loaded = True  # Mark as attempted
                    return False
        except ImportError:
            logger.debug("Transformers not available for toxicity detection")
            self._loaded = True
            return False

    def score(self, text: str) -> float | None:
        """Score text for toxicity using HuggingFace model."""
        if not self.enabled:
            return None

        if not self._load_model() or self._pipeline is None:
            return None

        try:
            results = self._pipeline(text)

            if not results or not isinstance(results, list):
                return None

            # Extract toxicity scores
            toxicity_scores = {}
            for item in results:
                if isinstance(item, dict):
                    label = item.get("label", "").lower()
                    score = item.get("score", 0.0)
                    toxicity_scores[label] = score

            # Weight different toxicity types
            toxicity_components = {
                "toxic": 1.0,
                "severe_toxic": 1.5,
                "obscene": 0.8,
                "threat": 1.2,
                "insult": 0.9,
                "identity_hate": 1.3,
                "toxicity": 1.0,
                "severe_toxicity": 1.5,
                "identity_attack": 1.3,
            }

            weighted_score = 0.0
            total_weight = 0.0

            for component, weight in toxicity_components.items():
                if component in toxicity_scores:
                    weighted_score += toxicity_scores[component] * weight
                    total_weight += weight
                else:
                    # Try partial match
                    for label, score in toxicity_scores.items():
                        if component in label or label in component:
                            weighted_score += score * weight
                            total_weight += weight
                            break

            if total_weight > 0:
                return float(weighted_score / total_weight)
            elif toxicity_scores:
                return float(max(toxicity_scores.values()))

            return None
        except Exception as e:
            logger.debug(f"Toxicity detection failed: {e}")
            return None


class KeywordToxicityDetector(ToxicityDetector):
    """Fallback keyword-based toxicity detector."""

    def __init__(self, enabled: bool = True):
        """Initialize keyword-based detector."""
        self.enabled = enabled
        import re

        self.toxic_patterns = [
            (re.compile(r"\bhate\b", re.IGNORECASE), 1.0),
            (re.compile(r"\bviolence\b", re.IGNORECASE), 1.0),
            (re.compile(r"\bharmful\s+(?!content|detection|behavior)", re.IGNORECASE), 0.8),
            (re.compile(r"\bdangerous\s+(?!content|situation|behavior)", re.IGNORECASE), 0.7),
            (re.compile(r"\billegal\s+(?!content|activity)", re.IGNORECASE), 0.6),
            (re.compile(r"\bkill\s+(?!time|process|switch)", re.IGNORECASE), 0.9),
            (re.compile(r"\battack\s+(?!vector|surface)", re.IGNORECASE), 0.8),
            (re.compile(r"\bdestroy\s+(?!data|file)", re.IGNORECASE), 0.7),
        ]

    def score(self, text: str) -> float | None:
        """Score text using keyword patterns."""
        if not self.enabled:
            return None

        toxic_score = 0.0
        for pattern, weight in self.toxic_patterns:
            if pattern.search(text):
                toxic_score = max(toxic_score, weight)

        # Normalize by length
        word_count = len(text.split())
        if word_count > 0:
            return min(1.0, toxic_score * (1.0 + 0.1 * min(word_count / 100, 1.0)))

        return toxic_score


class CompositeToxicityDetector(ToxicityDetector):
    """Composite detector that tries ML model first, falls back to keywords."""

    def __init__(
        self,
        ml_detector: ToxicityDetector | None = None,
        keyword_detector: ToxicityDetector | None = None,
    ):
        """
        Initialize composite detector.

        Args:
            ml_detector: ML-based detector (default: HuggingFaceToxicityDetector)
            keyword_detector: Keyword fallback (default: KeywordToxicityDetector)
        """
        self.ml_detector = ml_detector or HuggingFaceToxicityDetector()
        self.keyword_detector = keyword_detector or KeywordToxicityDetector()

    def score(self, text: str) -> float | None:
        """Score text, trying ML first, then keywords."""
        # Try ML detector first
        ml_score = self.ml_detector.score(text)
        if ml_score is not None:
            return ml_score

        # Fallback to keywords
        return self.keyword_detector.score(text)

"""Fact-checking using HuggingFace NLI (Natural Language Inference) models."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from config.settings import settings

if TYPE_CHECKING:
    from transformers import TextClassificationPipeline


@dataclass
class FactCheckResult:
    """Result of fact-checking."""

    is_factual: bool  # True if response is factually correct
    confidence: float  # Confidence score (0-1)
    label: str  # "ENTAILMENT", "CONTRADICTION", or "NEUTRAL"
    details: dict[str, Any]


class HuggingFaceFactChecker:
    """Fact-checker using HuggingFace NLI models for factual verification."""

    def __init__(
        self,
        model_name: str | None = None,
        enabled: bool | None = None,
        use_cuda: bool | None = None,
        cuda_device: int | None = None,
    ):
        """
        Initialize fact-checker.

        Args:
            model_name: HuggingFace model name (default: from settings or fallback list)
            enabled: Whether to enable (default: from settings)
            use_cuda: Whether to use CUDA if available (default: from settings)
            cuda_device: CUDA device ID (default: from settings)
        """
        self.enabled = enabled if enabled is not None else settings.fact_checker_enabled
        self.model_name = model_name or settings.fact_checker_model_name
        self.use_cuda = use_cuda if use_cuda is not None else settings.fact_checker_use_cuda
        self.cuda_device = (
            cuda_device if cuda_device is not None else settings.fact_checker_cuda_device
        )
        self._pipeline: TextClassificationPipeline | None = None
        self._loaded = False

    def _load_model(self) -> bool:
        """Lazy load the NLI model."""
        if self._loaded:
            return self._pipeline is not None

        if not self.enabled:
            return False

        try:
            import torch
            from transformers import pipeline

            # Get model options from settings
            fallback_models_str = settings.fact_checker_fallback_models
            fallback_models = [m.strip() for m in fallback_models_str.split(",") if m.strip()]

            # Try models in order of preference (best to fallback)
            model_options = []
            if self.model_name:
                model_options.append(self.model_name)  # User-specified or from settings
            model_options.extend(fallback_models)  # Fallback models from settings

            for model_name in model_options:
                if model_name is None:
                    continue

                try:
                    logger.info(f"Loading fact-checking model: {model_name}...")
                    # Use text-classification pipeline for NLI models
                    # NLI models are trained for sequence classification
                    from transformers import AutoModelForSequenceClassification, AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)

                    # Store label mapping from model config
                    self._label_mapping = (
                        model.config.id2label if hasattr(model.config, "id2label") else {}
                    )
                    self._reverse_label_mapping = {
                        v.upper(): k for k, v in self._label_mapping.items()
                    }

                    # Determine device
                    device = (
                        self.cuda_device if self.use_cuda and torch.cuda.is_available() else -1
                    )  # CPU

                    self._pipeline = pipeline(
                        "text-classification",
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        return_all_scores=True,
                    )
                    self._loaded = True
                    self.model_name = model_name
                    logger.info(f"✓ Fact-checking model loaded: {model_name}")
                    logger.debug(f"Label mapping: {self._label_mapping}")
                    return True
                except Exception as e:
                    logger.debug(f"Failed to load {model_name}: {e}")
                    continue

            # If all fail, log warning
            logger.warning("Failed to load any fact-checking models")
            self._loaded = True  # Mark as attempted
            return False

        except ImportError:
            logger.debug("Transformers not available for fact-checking")
            self._loaded = True
            return False

    def check_fact(
        self, claim: str, premise: str, threshold: float | None = None
    ) -> FactCheckResult | None:
        """
        Check if a claim is factually supported by a premise using NLI.

        Args:
            claim: The claim to verify (e.g., AI response)
            premise: The known fact/premise (e.g., expected_response or known_fact)
            threshold: Confidence threshold for entailment (unused, kept for API compatibility)

        Returns:
            FactCheckResult or None if model not available
        """
        # Threshold parameter kept for API compatibility but not used
        # (thresholds are configured via settings)
        _ = threshold
        if not self._load_model():
            return None

        if not claim or not premise:
            return FactCheckResult(
                is_factual=False,
                confidence=0.0,
                label="NEUTRAL",
                details={"error": "Missing claim or premise"},
            )

        try:
            # NLI models expect format: "premise" + " [SEP] " + "hypothesis"
            # For fact-checking: premise = known fact, hypothesis = claim
            # Format: "{premise} [SEP] {claim}"
            text = f"{premise} [SEP] {claim}"

            # Run inference
            if self._pipeline is None:
                return FactCheckResult(
                    is_factual=False,
                    confidence=0.0,
                    label="NEUTRAL",
                    details={"error": "Model not loaded"},
                )
            results = self._pipeline(text)

            # Parse results - NLI models return nested list: [[{label, score}, ...]]
            # Handle both nested and flat formats
            if isinstance(results, list):
                # Unwrap nested list if needed
                if len(results) > 0 and isinstance(results[0], list):
                    results = results[0]

                # Get all scores
                scores = {}
                for item in results:
                    if isinstance(item, dict):
                        label = item.get("label", "").upper()
                        score = item.get("score", 0.0)
                        scores[label] = score

                # Map labels to standard format using model's label mapping
                entailment_score = 0.0
                contradiction_score = 0.0
                neutral_score = 0.0

                # Use model's label mapping if available
                if hasattr(self, "_label_mapping") and self._label_mapping:
                    for item in results:
                        if isinstance(item, dict):
                            label_id = item.get("label", "")
                            score = item.get("score", 0.0)

                            # Get actual label name from model config
                            if label_id in self._label_mapping:
                                actual_label = self._label_mapping[label_id].upper()
                            else:
                                actual_label = str(label_id).upper()

                            # Map to standard format
                            if "ENTAIL" in actual_label:
                                entailment_score = score
                            elif "CONTRADICT" in actual_label:
                                contradiction_score = score
                            elif "NEUTRAL" in actual_label:
                                neutral_score = score
                else:
                    # Fallback: try common label names
                    for item in results:
                        if isinstance(item, dict):
                            label = str(item.get("label", "")).upper()
                            score = item.get("score", 0.0)

                            if "ENTAIL" in label:
                                entailment_score = score
                            elif "CONTRADICT" in label:
                                contradiction_score = score
                            elif "NEUTRAL" in label:
                                neutral_score = score
                            elif label.startswith("LABEL_"):
                                # For models with LABEL_0/1/2, need to check model config
                                # DeBERTa-large-mnli: LABEL_0=CONTRADICTION, LABEL_1=NEUTRAL, LABEL_2=ENTAILMENT
                                if label == "LABEL_0":
                                    contradiction_score = score
                                elif label == "LABEL_1":
                                    neutral_score = score
                                elif label == "LABEL_2":
                                    entailment_score = score

                # Determine label
                max_score = max(entailment_score, contradiction_score, neutral_score)
                if max_score == entailment_score:
                    label = "ENTAILMENT"
                    is_factual = True
                    confidence = entailment_score
                elif max_score == contradiction_score:
                    label = "CONTRADICTION"
                    is_factual = False
                    confidence = contradiction_score
                else:
                    label = "NEUTRAL"
                    # Neutral means we can't determine - be conservative
                    is_factual = False
                    confidence = neutral_score

                return FactCheckResult(
                    is_factual=is_factual,
                    confidence=confidence,
                    label=label,
                    details={
                        "entailment_score": entailment_score,
                        "contradiction_score": contradiction_score,
                        "neutral_score": neutral_score,
                        "model": self.model_name,
                    },
                )
            else:
                return FactCheckResult(
                    is_factual=False,
                    confidence=0.0,
                    label="NEUTRAL",
                    details={"error": "Unexpected model output format"},
                )

        except Exception as e:
            logger.warning(f"Fact-checking failed: {e}")
            return FactCheckResult(
                is_factual=False,
                confidence=0.0,
                label="NEUTRAL",
                details={"error": str(e)},
            )

    def check_multiple_facts(
        self, claim: str, premises: list[str], threshold: float = 0.5
    ) -> FactCheckResult:
        """
        Check claim against multiple premises.

        Args:
            claim: The claim to verify
            premises: List of known facts/premises
            threshold: Confidence threshold

        Returns:
            FactCheckResult (aggregated across all premises)
        """
        if not premises:
            return FactCheckResult(
                is_factual=False,
                confidence=0.0,
                label="NEUTRAL",
                details={"error": "No premises provided"},
            )

        results = []
        for premise in premises:
            result = self.check_fact(claim, premise, threshold)
            if result:
                results.append(result)

        if not results:
            return FactCheckResult(
                is_factual=False,
                confidence=0.0,
                label="NEUTRAL",
                details={"error": "No valid results from fact-checking"},
            )

        # Aggregate: use weighted voting based on confidence
        contradictions = [r for r in results if r.label == "CONTRADICTION"]
        entailments = [r for r in results if r.label == "ENTAILMENT"]
        neutrals = [r for r in results if r.label == "NEUTRAL"]

        # Decision logic:
        # 1. If strong contradiction (high confidence), mark as contradicted
        # 2. If strong entailment (high confidence) and no contradictions, mark as supported
        # 3. If mostly neutral or mixed with no strong signals, mark as neutral (unknown)

        # Get thresholds from settings
        strong_contradiction_threshold = settings.fact_checker_strong_contradiction_threshold
        strong_entailment_threshold = settings.fact_checker_strong_entailment_threshold
        weak_entailment_threshold = settings.fact_checker_weak_entailment_threshold
        neutral_threshold = settings.fact_checker_neutral_threshold
        neutral_ratio_threshold = settings.fact_checker_neutral_ratio_threshold

        # Strong contradiction threshold: if any contradiction exceeds threshold
        strong_contradictions = [
            r for r in contradictions if r.confidence > strong_contradiction_threshold
        ]

        # Strong entailment: average confidence exceeds threshold
        avg_entailment = (
            sum(r.confidence for r in entailments) / len(entailments) if entailments else 0.0
        )

        # Strong neutral: mostly neutral results
        avg_neutral = sum(r.confidence for r in neutrals) / len(neutrals) if neutrals else 0.0
        neutral_ratio = len(neutrals) / len(results) if results else 0.0

        if strong_contradictions:
            # Strong contradiction = hallucination
            avg_confidence = sum(r.confidence for r in strong_contradictions) / len(
                strong_contradictions
            )
            return FactCheckResult(
                is_factual=False,
                confidence=avg_confidence,
                label="CONTRADICTION",
                details={
                    "num_contradictions": len(contradictions),
                    "num_strong_contradictions": len(strong_contradictions),
                    "num_entailments": len(entailments),
                    "num_neutrals": len(neutrals),
                    "total_premises": len(premises),
                },
            )

        # Check weak contradictions
        if contradictions and not strong_contradictions:
            # Weak contradiction - check if it outweighs other signals
            avg_contradiction = sum(r.confidence for r in contradictions) / len(contradictions)
            if avg_contradiction > weak_entailment_threshold and avg_contradiction > avg_entailment:
                # Weak but clear contradiction
                return FactCheckResult(
                    is_factual=False,
                    confidence=avg_contradiction,
                    label="CONTRADICTION",
                    details={
                        "num_contradictions": len(contradictions),
                        "num_entailments": len(entailments),
                        "num_neutrals": len(neutrals),
                        "total_premises": len(premises),
                    },
                )

        # Check strong entailments
        if entailments and avg_entailment > strong_entailment_threshold and not contradictions:
            # Strong entailment with no contradictions = factual
            return FactCheckResult(
                is_factual=True,
                confidence=avg_entailment,
                label="ENTAILMENT",
                details={
                    "num_entailments": len(entailments),
                    "avg_entailment_confidence": avg_entailment,
                    "total_premises": len(premises),
                },
            )

        # Check weak entailments
        if (
            entailments
            and avg_entailment > weak_entailment_threshold
            and len(entailments) > len(contradictions) + len(neutrals)
        ):
            # Majority entailment with decent confidence = factual
            return FactCheckResult(
                is_factual=True,
                confidence=avg_entailment,
                label="ENTAILMENT",
                details={
                    "num_entailments": len(entailments),
                    "avg_entailment_confidence": avg_entailment,
                    "total_premises": len(premises),
                },
            )

        # Check neutrals
        if neutral_ratio > neutral_ratio_threshold or (
            neutrals and avg_neutral > neutral_threshold
        ):
            # Mostly neutral = unknown (can't determine)
            avg_confidence = sum(r.confidence for r in results) / len(results)
            return FactCheckResult(
                is_factual=False,  # Neutral means we can't verify, so be conservative
                confidence=avg_confidence,
                label="NEUTRAL",
                details={
                    "num_entailments": len(entailments),
                    "num_contradictions": len(contradictions),
                    "num_neutrals": len(neutrals),
                    "total_premises": len(premises),
                    "neutral_ratio": neutral_ratio,
                },
            )
        else:
            # Mixed results - use majority vote with confidence weighting
            if len(entailments) > len(contradictions):
                # More entailments than contradictions
                return FactCheckResult(
                    is_factual=True,
                    confidence=avg_entailment if entailments else 0.5,
                    label="ENTAILMENT",
                    details={
                        "num_entailments": len(entailments),
                        "num_contradictions": len(contradictions),
                        "num_neutrals": len(neutrals),
                        "total_premises": len(premises),
                    },
                )
            else:
                # More contradictions or equal - be conservative
                avg_confidence = sum(r.confidence for r in results) / len(results)
                return FactCheckResult(
                    is_factual=False,
                    confidence=avg_confidence,
                    label="NEUTRAL",
                    details={
                        "num_entailments": len(entailments),
                        "num_contradictions": len(contradictions),
                        "num_neutrals": len(neutrals),
                        "total_premises": len(premises),
                    },
                )

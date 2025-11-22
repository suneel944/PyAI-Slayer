"""AI response validation using semantic similarity and NLP models."""

import os
import time
from typing import Any

import numpy as np
from deep_translator import GoogleTranslator
from loguru import logger
from sentence_transformers import SentenceTransformer

from config.settings import settings

# Thread-local storage for validation data (for test reporting)
_validation_context: dict[str, Any] = {}


def _store_validation_data(
    query: str | None,
    response: str | None,
    metrics: dict[str, Any],
    expected_response: str | None = None,
    reference: str | None = None,
) -> None:
    """Store validation data for test reporting (transparent to tests)."""
    global _validation_context
    _validation_context = {
        "query": query,
        "response": response,
        "expected_response": expected_response,
        "reference": reference,
        "metrics": metrics,
    }


def _get_validation_data() -> dict[str, Any]:
    """Retrieve stored validation data."""
    return _validation_context.copy()


def _clear_validation_data() -> None:
    """Clear validation data."""
    global _validation_context
    _validation_context = {}


class AIResponseValidator:
    """Validates AI chatbot responses using multiple techniques."""

    def __init__(self):
        """Initialize AI validator with models (lazy loading)."""
        self._semantic_model = None
        self._arabic_semantic_model = None
        self._translator = None
        logger.info("AI validator initialized (models will load on first use)")

    @property
    def semantic_model(self):
        """Lazy load semantic similarity model (multilingual, supports Arabic and English)."""
        if self._semantic_model is None:
            from core.infrastructure.cache import get_model_cache

            model_name = settings.semantic_model_name
            device = "cpu"
            if os.getenv("USE_CUDA", "").lower() in ("true", "1", "yes"):
                device = "cuda"

            # Check model cache first
            model_cache = get_model_cache()
            cached_model = model_cache.get_model(model_name, device)

            if cached_model is not None:
                self._semantic_model = cached_model
            else:
                logger.info(
                    f"Loading semantic similarity model: {model_name} (this may take 10-30 seconds on first run)..."
                )
                start_time = time.time()

                try:
                    self._semantic_model = SentenceTransformer(model_name, device=device)
                    load_time = time.time() - start_time
                    logger.info(f"✓ Semantic model loaded in {load_time:.2f}s (device: {device})")

                    # Cache the model
                    model_cache.set_model(model_name, self._semantic_model, device)
                except Exception as e:
                    logger.error(f"Failed to load semantic model: {e}")
                    raise

        return self._semantic_model

    @property
    def arabic_semantic_model(self):
        """Lazy load Arabic-specific semantic similarity model."""
        if self._arabic_semantic_model is None:
            from core.infrastructure.cache import get_model_cache

            model_name = settings.arabic_semantic_model_name
            device = "cpu"
            if os.getenv("USE_CUDA", "").lower() in ("true", "1", "yes"):
                device = "cuda"

            # Check model cache first
            model_cache = get_model_cache()
            cached_model = model_cache.get_model(model_name, device)

            if cached_model is not None:
                self._arabic_semantic_model = cached_model
            else:
                logger.info(
                    f"Loading Arabic-specific semantic model: {model_name} (this may take 10-30 seconds on first run)..."
                )
                start_time = time.time()

                try:
                    self._arabic_semantic_model = SentenceTransformer(model_name, device=device)
                    load_time = time.time() - start_time
                    logger.info(
                        f"✓ Arabic semantic model loaded in {load_time:.2f}s (device: {device})"
                    )

                    # Cache the model
                    model_cache.set_model(model_name, self._arabic_semantic_model, device)
                except Exception as e:
                    logger.warning(f"Failed to load Arabic-specific model {model_name}: {e}")
                    logger.info("Falling back to multilingual model for Arabic text")

                    self._arabic_semantic_model = self.semantic_model

        return self._arabic_semantic_model

    @property
    def translator(self):
        """Lazy load translator."""
        if self._translator is None:
            self._translator = GoogleTranslator()
        return self._translator

    def _is_arabic(self, text: str) -> bool:
        """Check if text contains Arabic characters."""
        return any("\u0600" <= char <= "\u06ff" for char in text)

    def _get_model_for_text(self, text: str):
        """Get appropriate semantic model based on text language."""
        if self._is_arabic(text):
            return self.arabic_semantic_model
        return self.semantic_model

    def validate_relevance(
        self, query: str, response: str, threshold: float | None = None
    ) -> tuple[bool, float]:
        """
        Check semantic similarity between query and response.

        Uses multilingual model that supports Arabic, English, and 50+ languages.
        Uses a lower threshold (0.5) for Arabic text as multilingual models
        tend to have slightly lower similarity scores for Arabic.

        Args:
            query: User query
            response: AI response
            threshold: Similarity threshold (default from settings, or 0.5 for Arabic)

        Returns:
            Tuple of (is_relevant, similarity_score)
        """

        if threshold is None:
            if self._is_arabic(query) or self._is_arabic(response):
                threshold = settings.arabic_semantic_similarity_threshold
            else:
                threshold = settings.semantic_similarity_threshold

        try:
            from core.infrastructure.cache import get_embedding_cache

            is_arabic = self._is_arabic(query) or self._is_arabic(response)
            if is_arabic:
                model = self.arabic_semantic_model
                model_name = settings.arabic_semantic_model_name
            else:
                model = self.semantic_model
                model_name = settings.semantic_model_name

            # Check embedding cache
            embedding_cache = get_embedding_cache()
            query_embedding = embedding_cache.get(query, model_name)
            if query_embedding is None:
                query_embedding = model.encode(query, convert_to_numpy=True)
                embedding_cache.set(query, model_name, query_embedding)

            response_embedding = embedding_cache.get(response, model_name)
            if response_embedding is None:
                response_embedding = model.encode(response, convert_to_numpy=True)
                embedding_cache.set(response, model_name, response_embedding)

            similarity = np.dot(query_embedding, response_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(response_embedding)
            )

            # Convert NumPy boolean to Python bool to avoid isinstance(np.True_, bool) issues
            # Clamp similarity to [0.0, 1.0] to handle floating point precision issues
            similarity = float(similarity)
            similarity = max(0.0, min(1.0, similarity))
            is_relevant = bool(similarity >= threshold)

            # Record validation metrics (high-value insight)
            try:
                from core.observability import get_prometheus_metrics

                metrics = get_prometheus_metrics()
                if metrics.enabled:
                    language = "ar" if is_arabic else "en"

                    # Legacy validation metric
                    metrics.record_validation(
                        validation_type="relevance",
                        passed=is_relevant,
                        similarity=float(similarity),
                        language=language,
                    )

                    # A-tier Metric #1: Task Success Rate
                    metrics.record_task_success(
                        success=is_relevant, task_type="query_response", language=language
                    )

                    # A-tier Metric #8: Grounding Accuracy
                    metrics.record_grounding_accuracy(
                        grounding_score=float(similarity),
                        language=language,
                        grounding_type="semantic_similarity",
                    )
            except Exception as e:
                logger.debug(f"Could not record metrics: {e}")

            # Store validation data for test reporting (transparent to tests)
            _store_validation_data(
                query=query,
                response=response,
                metrics={
                    "validation_type": "relevance",
                    "similarity_score": float(similarity),
                    "is_relevant": is_relevant,
                    "threshold": threshold,
                    "language": "ar" if is_arabic else "en",
                },
            )

            return is_relevant, float(similarity)
        except Exception as e:
            logger.error(f"Error in relevance validation: {e}")
            return False, 0.0

    def validate_semantic_concepts(
        self,
        response: str,
        expected_concepts: list[str],
        min_concepts_covered: int | None = None,
        concept_threshold: float | None = None,
    ) -> tuple[bool, float, dict[str, float]]:
        """
        Validate that response semantically covers expected concepts/keywords.

        This is a semantic alternative to rigid keyword matching. Instead of checking
        if exact keywords appear in the response, it uses embeddings to check if the
        response semantically addresses the expected concepts.

        Args:
            response: AI response to validate
            expected_concepts: List of expected concepts/keywords (e.g., ["visa", "renew", "process"])
            min_concepts_covered: Minimum number of concepts that must be covered (default: 50% of concepts)
            concept_threshold: Similarity threshold for each concept (default: 0.4 for semantic matching)

        Returns:
            Tuple of (is_valid, avg_concept_score, concept_scores_dict)
            - is_valid: True if response covers enough concepts semantically
            - avg_concept_score: Average similarity score across all concepts
            - concept_scores_dict: Dictionary mapping each concept to its similarity score
        """
        if not expected_concepts:
            logger.warning("No expected concepts provided for semantic validation")
            return True, 1.0, {}

        if not response:
            logger.warning("Empty response provided for semantic concept validation")
            return False, 0.0, {}

        # Set defaults
        if min_concepts_covered is None:
            # Default: at least 50% of concepts should be covered
            min_concepts_covered = max(1, len(expected_concepts) // 2)
        if concept_threshold is None:
            # Lower threshold for individual concepts (0.4) since we're checking semantic similarity
            # rather than exact matches. This allows for synonyms and paraphrasing.
            concept_threshold = 0.4

        try:
            from core.infrastructure.cache import get_embedding_cache

            is_arabic = self._is_arabic(response)
            if is_arabic:
                model = self.arabic_semantic_model
                model_name = settings.arabic_semantic_model_name
            else:
                model = self.semantic_model
                model_name = settings.semantic_model_name

            # Get or compute response embedding
            embedding_cache = get_embedding_cache()
            response_embedding = embedding_cache.get(response, model_name)
            if response_embedding is None:
                response_embedding = model.encode(response, convert_to_numpy=True)
                embedding_cache.set(response, model_name, response_embedding)

            concept_scores = {}
            concepts_covered = 0

            # Check each concept semantically
            for concept in expected_concepts:
                # Get or compute concept embedding
                concept_embedding = embedding_cache.get(concept, model_name)
                if concept_embedding is None:
                    concept_embedding = model.encode(concept, convert_to_numpy=True)
                    embedding_cache.set(concept, model_name, concept_embedding)

                # Calculate semantic similarity between concept and response
                similarity = np.dot(response_embedding, concept_embedding) / (
                    np.linalg.norm(response_embedding) * np.linalg.norm(concept_embedding)
                )
                similarity = float(similarity)
                similarity = max(0.0, min(1.0, similarity))
                concept_scores[concept] = similarity

                if similarity >= concept_threshold:
                    concepts_covered += 1

            # Calculate average concept score
            avg_score = float(np.mean(list(concept_scores.values()))) if concept_scores else 0.0

            # Response is valid if it covers enough concepts
            is_valid = concepts_covered >= min_concepts_covered
            logger.info(
                f"Semantic concept validation: {concepts_covered}/{len(expected_concepts)} "
                + f"concepts covered (avg score: {avg_score:.3f}, threshold: {concept_threshold})"
            )

            # Store validation data for test reporting
            _store_validation_data(
                query=None,
                response=response,
                metrics={
                    "validation_type": "semantic_concepts",
                    "concepts_covered": concepts_covered,
                    "total_concepts": len(expected_concepts),
                    "avg_concept_score": avg_score,
                    "concept_scores": concept_scores,
                    "concept_threshold": concept_threshold,
                    "min_concepts_required": min_concepts_covered,
                    "is_valid": is_valid,
                    "language": "ar" if is_arabic else "en",
                },
            )

            return is_valid, avg_score, concept_scores

        except Exception as e:
            logger.error(f"Error in semantic concept validation: {e}")
            return False, 0.0, {}

    def detect_hallucination(self, response: str, known_facts: list[str]) -> tuple[bool, list[str]]:
        """
        Detect if response contains fabricated information.

        Args:
            response: AI response to check
            known_facts: List of known facts/expected information

        Returns:
            Tuple of (has_hallucination, conflicting_facts)
        """
        if not known_facts:
            return False, []

        try:
            model = self._get_model_for_text(response)
            response_embedding = model.encode(response, convert_to_numpy=True)
            conflicting_facts = []

            for fact in known_facts:
                fact_embedding = model.encode(fact, convert_to_numpy=True)
                similarity = np.dot(response_embedding, fact_embedding) / (
                    np.linalg.norm(response_embedding) * np.linalg.norm(fact_embedding)
                )

                if similarity < settings.hallucination_detection_threshold:
                    conflicting_facts.append(fact)

            has_hallucination = len(conflicting_facts) > 0
            if has_hallucination:
                logger.warning(f"Potential hallucination detected: {conflicting_facts}")

            # Record validation metrics (high-value insight)
            try:
                from core.observability import get_prometheus_metrics

                metrics = get_prometheus_metrics()
                if metrics.enabled:
                    language = "ar" if self._is_arabic(response) else "en"
                    # Calculate average similarity for conflicting facts
                    avg_similarity = None
                    if conflicting_facts:
                        similarities = []
                        for fact in conflicting_facts:
                            fact_embedding = model.encode(fact, convert_to_numpy=True)
                            sim = np.dot(response_embedding, fact_embedding) / (
                                np.linalg.norm(response_embedding) * np.linalg.norm(fact_embedding)
                            )
                            similarities.append(float(sim))
                        avg_similarity = float(np.mean(similarities)) if similarities else None

                    # Legacy validation metric
                    metrics.record_validation(
                        validation_type="hallucination",
                        passed=not has_hallucination,  # Passed = no hallucination
                        similarity=avg_similarity,
                        language=language,
                    )

                    # A-tier Metric #3: Hallucination Rate
                    metrics.record_hallucination(detected=has_hallucination, language=language)
            except Exception as e:
                logger.debug(f"Could not record metrics: {e}")

            return has_hallucination, conflicting_facts
        except Exception as e:
            logger.error(f"Error in hallucination detection: {e}")
            return False, []

    def check_consistency(self, responses: list[str]) -> float:
        """
        Measure consistency across multiple responses to same query.

        Args:
            responses: List of responses to the same query

        Returns:
            Average pairwise similarity score
        """
        if len(responses) < 2:
            return 1.0

        try:
            model = self._get_model_for_text(responses[0]) if responses else self.semantic_model
            embeddings = [model.encode(r, convert_to_numpy=True) for r in responses]

            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(similarity)

            avg_similarity = float(np.mean(similarities)) if similarities else 0.0
            # Clamp to [0.0, 1.0] to handle floating point precision
            avg_similarity = max(0.0, min(1.0, avg_similarity))

            # Record validation metrics (high-value insight)
            try:
                from core.observability import get_prometheus_metrics

                metrics = get_prometheus_metrics()
                if metrics.enabled:
                    # Detect language from first response
                    language = "ar" if self._is_arabic(responses[0]) else "en"
                    # Consistency passes if similarity is above threshold (typically 0.7)
                    consistency_threshold = 0.7
                    passed = avg_similarity >= consistency_threshold

                    # Legacy validation metric
                    metrics.record_validation(
                        validation_type="consistency",
                        passed=passed,
                        similarity=float(avg_similarity),
                        language=language,
                    )

                    # A-tier Metric #2: Consistency / Stability Score
                    metrics.record_consistency_score(
                        consistency_score=float(avg_similarity), language=language
                    )
            except Exception as e:
                logger.debug(f"Could not record metrics: {e}")

            return float(avg_similarity)
        except Exception as e:
            logger.error(f"Error in consistency check: {e}")
            return 0.0

    def validate_cross_language(
        self, en_response: str, ar_response: str, threshold: float | None = None
    ) -> tuple[bool, float]:
        """
        Verify semantic consistency between English and Arabic responses.

        Args:
            en_response: English response
            ar_response: Arabic response
            threshold: Similarity threshold (default from settings)

        Returns:
            Tuple of (is_consistent, similarity_score)
        """
        threshold = threshold or settings.cross_language_consistency_threshold

        if not en_response or not ar_response:
            logger.warning("Empty response provided for cross-language validation")
            return False, 0.0

        try:
            ar_translated = self.translator.translate(ar_response, src="ar", dest="en")

            if not ar_translated or len(ar_translated.strip()) == 0:
                logger.warning(
                    "Translation returned empty string - trying direct semantic comparison"
                )

                model = self._get_model_for_text(en_response + " " + ar_response)
                try:
                    en_embedding = model.encode(en_response, convert_to_numpy=True)
                    ar_embedding = model.encode(ar_response, convert_to_numpy=True)

                    import numpy as np

                    similarity = float(
                        np.dot(en_embedding, ar_embedding)
                        / (np.linalg.norm(en_embedding) * np.linalg.norm(ar_embedding))
                    )
                    # Clamp similarity to [0.0, 1.0]
                    similarity = max(0.0, min(1.0, similarity))

                    # Convert NumPy boolean to Python bool
                    is_consistent = bool(similarity >= threshold)
                    logger.info(f"Direct semantic comparison (no translation): {similarity:.3f}")

                    # Record validation metrics
                    try:
                        from core.observability import get_prometheus_metrics

                        metrics = get_prometheus_metrics()
                        if metrics.enabled:
                            metrics.record_validation(
                                validation_type="cross_language",
                                passed=is_consistent,
                                similarity=similarity,
                                language="multilingual",
                            )
                    except Exception:
                        pass

                    return is_consistent, similarity
                except Exception as e2:
                    logger.error(f"Direct semantic comparison also failed: {e2}")
                    return False, 0.0

            is_consistent, similarity = self.validate_relevance(
                en_response, ar_translated, threshold
            )

            logger.info(
                f"Cross-language consistency: {similarity:.3f} (translated: {ar_translated[:50]}...)"
            )

            # Record validation metrics (high-value insight)
            try:
                from core.observability import get_prometheus_metrics

                metrics = get_prometheus_metrics()
                if metrics.enabled:
                    metrics.record_validation(
                        validation_type="cross_language",
                        passed=is_consistent,
                        similarity=similarity,
                        language="multilingual",  # Special label for cross-language
                    )
            except Exception as e:
                logger.debug(f"Could not record metrics: {e}")

            return is_consistent, similarity
        except Exception as e:
            logger.error(f"Error in cross-language validation: {e}")
            import traceback

            traceback.print_exc()

            try:
                logger.info("Attempting direct semantic comparison as fallback...")
                model = self._get_model_for_text(en_response + " " + ar_response)
                en_embedding = model.encode(en_response, convert_to_numpy=True)
                ar_embedding = model.encode(ar_response, convert_to_numpy=True)

                import numpy as np

                similarity = float(
                    np.dot(en_embedding, ar_embedding)
                    / (np.linalg.norm(en_embedding) * np.linalg.norm(ar_embedding))
                )
                # Clamp similarity to [0.0, 1.0]
                similarity = max(0.0, min(1.0, similarity))

                # Convert NumPy boolean to Python bool
                is_consistent = bool(similarity >= threshold)
                logger.info(f"Fallback direct comparison: {similarity:.3f}")

                # Record validation metrics
                try:
                    from core.observability import get_prometheus_metrics

                    metrics = get_prometheus_metrics()
                    if metrics.enabled:
                        metrics.record_validation(
                            validation_type="cross_language",
                            passed=is_consistent,
                            similarity=similarity,
                            language="multilingual",
                        )
                except Exception:
                    pass

                return is_consistent, similarity
            except Exception as e2:
                logger.error(f"Fallback comparison also failed: {e2}")
                return False, 0.0

    def validate_response_quality(
        self, response: str, min_length: int | None = None, max_length: int | None = None
    ) -> dict[str, bool]:
        """
        Validate basic response quality metrics.

        Args:
            response: AI response
            min_length: Minimum expected length
            max_length: Maximum expected length (optional)

        Returns:
            Dictionary of quality checks
        """
        min_length = min_length or settings.min_response_length

        checks = {
            "has_minimum_length": len(response.strip()) >= min_length,
            "is_not_empty": len(response.strip()) > 0,
            "has_no_html_tags": "<" not in response and ">" not in response,
            "ends_properly": response.strip().endswith((".", "!", "?", ":", ";")),
        }

        if max_length is not None:
            checks["within_max_length"] = len(response) <= max_length

        return checks

    def detect_fallback_message(self, response: str) -> bool:
        """
        Detect if response is a fallback/error message.

        Args:
            response: AI response to check

        Returns:
            True if response appears to be a fallback message
        """
        fallback_indicators = [
            "try again",
            "sorry, i didn't understand",
            "please rephrase",
            "i'm having trouble",
            "error",
            "حدث خطأ",
            "حاول مرة أخرى",
            "لم أفهم",
        ]

        response_lower = response.lower()
        return any(indicator in response_lower for indicator in fallback_indicators)

    def calculate_bertscore(
        self, candidate: str, reference: str, lang: str = "en"
    ) -> dict[str, float | str]:
        """
        Calculate BERTScore for response quality.

        Args:
            candidate: AI-generated response
            reference: Reference/expected response
            lang: Language code

        Returns:
            Dictionary with precision, recall, F1 scores
        """
        try:
            from bert_score import score

            P, R, F1 = score([candidate], [reference], lang=lang, rescale_with_baseline=True)

            result: dict[str, float | str] = {
                "precision": float(P[0]),
                "recall": float(R[0]),
                "f1": float(F1[0]),
            }

            return result

        except Exception as e:
            logger.error(f"Error calculating BERTScore: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "error": str(e)}

    def calculate_rouge_scores(self, candidate: str, reference: str) -> dict[str, float | str]:
        """
        Calculate ROUGE scores for response quality.

        Args:
            candidate: AI-generated response
            reference: Reference/expected response

        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            scores = scorer.score(reference, candidate)

            result: dict[str, float | str] = {
                "rouge1_f1": scores["rouge1"].fmeasure,
                "rouge2_f1": scores["rouge2"].fmeasure,
                "rougeL_f1": scores["rougeL"].fmeasure,
                "rouge1_precision": scores["rouge1"].precision,
                "rouge1_recall": scores["rouge1"].recall,
            }

            return result

        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {e}")
            return {"error": str(e)}

    def calculate_perplexity(self, text: str, model_name: str = "gpt2") -> float:
        """
        Calculate perplexity of text (lower = more fluent).

        Note: This is a simplified perplexity calculation.
        For production, consider using dedicated language models.

        Args:
            text: Text to calculate perplexity for
            model_name: Model to use for perplexity calculation

        Returns:
            Perplexity score
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            # Tokenize
            inputs = tokenizer(text, return_tensors="pt")

            # Calculate loss
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

            # Perplexity = exp(loss)
            perplexity: float = float(torch.exp(loss).item())

            return perplexity

        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            return float("inf")

    def comprehensive_quality_metrics(
        self, candidate: str, reference: str | None = None, calculate_perplexity: bool = False
    ) -> dict[str, Any]:
        """
        Calculate comprehensive quality metrics for AI response.

        Args:
            candidate: AI-generated response
            reference: Optional reference response
            calculate_perplexity: Whether to calculate perplexity (slow)

        Returns:
            Dictionary with all quality metrics
        """
        metrics: dict[str, Any] = {
            "response_length": len(candidate),
            "word_count": len(candidate.split()),
            "language": "ar" if self._is_arabic(candidate) else "en",
        }

        # BERTScore (if reference provided)
        if reference:
            lang: str = str(metrics["language"])
            metrics["bertscore"] = self.calculate_bertscore(candidate, reference, lang=lang)
            metrics["rouge"] = self.calculate_rouge_scores(candidate, reference)
            # Store reference in validation context when reference is provided
            _store_validation_data(
                query=None,
                response=candidate,
                metrics=metrics,
                expected_response=reference,
                reference=reference,
            )

        # Perplexity (optional, slow)
        if calculate_perplexity:
            metrics["perplexity"] = self.calculate_perplexity(candidate)

        # Basic quality checks
        metrics["quality_checks"] = self.validate_response_quality(candidate)

        # Fallback detection
        metrics["is_fallback"] = self.detect_fallback_message(candidate)

        return metrics

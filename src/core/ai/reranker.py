"""Reranker for RAG evaluation using HuggingFace cross-encoder models."""

from loguru import logger

from config.settings import settings


class HuggingFaceReranker:
    """Reranker using HuggingFace cross-encoder models for query-document relevance."""

    def __init__(
        self,
        model_name: str | None = None,
        enabled: bool | None = None,
        use_cuda: bool | None = None,
        cuda_device: int | None = None,
    ):
        """
        Initialize reranker.

        Args:
            model_name: HuggingFace model name (default: from settings or fallback list)
            enabled: Whether to enable (default: from settings)
            use_cuda: Whether to use CUDA if available (default: from settings)
            cuda_device: CUDA device ID (default: from settings)
        """
        self.enabled = enabled if enabled is not None else settings.rag_reranker_enabled
        self.model_name = model_name or settings.rag_reranker_model_name
        self.use_cuda = use_cuda if use_cuda is not None else settings.rag_reranker_use_cuda
        self.cuda_device = (
            cuda_device if cuda_device is not None else settings.rag_reranker_cuda_device
        )
        self._pipeline = None
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._use_flag_reranker = False
        self._loaded = False

    def _load_model(self) -> bool:
        """Lazy load the reranker model."""
        if self._loaded:
            # Return True if either pipeline (FlagReranker) or model/tokenizer (transformers) is available
            return (self._pipeline is not None) or (self._model is not None and self._tokenizer is not None)

        if not self.enabled:
            return False

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # Get model options from settings
            fallback_models_str = settings.rag_reranker_fallback_models
            fallback_models = [m.strip() for m in fallback_models_str.split(",") if m.strip()]

            # Try models in order of preference (best to fallback)
            model_options = []
            if self.model_name:
                model_options.append(self.model_name)  # User-specified or from settings
            model_options.extend(fallback_models)  # Fallback models from settings

            if not model_options:
                logger.warning("No reranker models configured")
                self.enabled = False
                self._loaded = True
                return False

            device = "cpu"
            if self.use_cuda and torch.cuda.is_available():
                device = f"cuda:{self.cuda_device}"

            for model_name in model_options:
                if model_name is None:
                    continue

                try:
                    logger.info(f"Loading reranker model: {model_name}...")
                    import time

                    start_time = time.time()

                    # Try FlagReranker first (recommended for BGE models)
                    try:
                        from FlagReranker import FlagReranker

                        self._pipeline = FlagReranker(model_name, use_fp16=True)
                        if device.startswith("cuda"):
                            # FlagReranker handles device automatically
                            pass
                        load_time = time.time() - start_time
                        logger.info(
                            f"✓ Reranker model loaded (FlagReranker): {model_name} "
                            f"(device: {device}, time: {load_time:.2f}s)"
                        )
                        self.model_name = model_name
                        self._loaded = True
                        self._use_flag_reranker = True
                        return True
                    except ImportError:
                        # Fallback to transformers
                        logger.debug("FlagReranker not available, using transformers")
                        self._use_flag_reranker = False

                    # Fallback: Use transformers directly
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    model.to(device)
                    model.eval()

                    self._tokenizer = tokenizer
                    self._model = model
                    self._device = device

                    load_time = time.time() - start_time
                    logger.info(
                        f"✓ Reranker model loaded (transformers): {model_name} "
                        f"(device: {device}, time: {load_time:.2f}s)"
                    )
                    self.model_name = model_name
                    self._loaded = True
                    self._use_flag_reranker = False
                    return True

                except Exception as e:
                    logger.warning(f"Failed to load reranker model {model_name}: {e}")
                    if model_name == model_options[-1]:  # Last fallback
                        logger.error("All reranker models failed to load. Reranking disabled.")
                        self.enabled = False
                        self._loaded = True
                        return False
                    continue

            # If we get here, all models failed (shouldn't happen due to exception handler above)
            logger.error("All reranker models failed to load. Reranking disabled.")
            self.enabled = False
            self._loaded = True
            return False

        except ImportError:
            logger.warning(
                "transformers/torch not available. Reranker disabled. "
                "Install with: pip install transformers torch"
            )
            self.enabled = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading reranker: {e}")
            self.enabled = False
            return False

    def score(self, query: str, document: str) -> float:
        """
        Score query-document relevance using reranker.

        Args:
            query: Query text
            document: Document text

        Returns:
            Relevance score (0-1, higher = more relevant)
        """
        if not self._load_model():
            return 0.0

        try:
            if self._use_flag_reranker and self._pipeline:
                # FlagReranker API: compute_score(query, passage)
                score = self._pipeline.compute_score([query, document])
                # FlagReranker returns a score that may need normalization
                # Typically returns values that can be negative or > 1, so we normalize
                # Using sigmoid to normalize to [0, 1]
                import math

                score = 1 / (1 + math.exp(-score))  # Sigmoid normalization
                return max(0.0, min(1.0, float(score)))
            elif self._model and self._tokenizer:
                # Transformers API: manual inference
                import torch

                # BGE rerankers use the tokenizer's sep_token (usually </s>) or just concatenate
                # Check if tokenizer has sep_token and use it, otherwise use space
                sep_token = self._tokenizer.sep_token if hasattr(self._tokenizer, 'sep_token') and self._tokenizer.sep_token else " "
                # Format: query + sep_token + document
                text = f"{query}{sep_token}{document}"

                # Tokenize
                inputs = self._tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                # Inference
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    # Get logits
                    logits = outputs.logits
                    logger.debug(f"Reranker logits shape: {logits.shape}, value: {logits[0][0].item() if logits.numel() > 0 else 'N/A'}")

                    # BGE rerankers are typically regression models (single output)
                    # or binary classification models
                    if logits.shape[-1] == 1:
                        # Regression model: single output value
                        raw_score = float(logits[0][0])
                        # BGE rerankers output raw scores that can be negative
                        # Normalize using sigmoid to [0, 1]
                        import math

                        score = 1 / (1 + math.exp(-raw_score))
                        logger.debug(f"Reranker raw_score: {raw_score}, normalized: {score}")
                    elif logits.shape[-1] == 2:
                        # Binary classification: use positive class probability
                        probs = torch.softmax(logits, dim=-1)
                        score = float(probs[0][1])  # Positive class
                        logger.debug(f"Reranker binary classification score: {score}")
                    else:
                        # Multi-class or other: use softmax and take max
                        probs = torch.softmax(logits, dim=-1)
                        score = float(torch.max(probs[0]))
                        logger.debug(f"Reranker multi-class score: {score}")

                final_score = max(0.0, min(1.0, float(score)))
                logger.debug(f"Reranker final score: {final_score}")
                return final_score
            else:
                logger.debug(f"Reranker: model={self._model is not None}, tokenizer={self._tokenizer is not None}")
                return 0.0

        except Exception as e:
            logger.warning(f"Reranker scoring failed: {e}, falling back to 0.0")
            import traceback
            logger.debug(f"Reranker error traceback: {traceback.format_exc()}")
            return 0.0

    def score_batch(self, query: str, documents: list[str]) -> list[float]:
        """
        Score multiple query-document pairs (optimized batch processing).

        Args:
            query: Query text
            documents: List of document texts

        Returns:
            List of relevance scores (0-1)
        """
        if not self._load_model():
            return [0.0] * len(documents)

        try:
            if self._use_flag_reranker and self._pipeline:
                # FlagReranker supports batch scoring
                pairs = [[query, doc] for doc in documents]
                scores = self._pipeline.compute_score(pairs)
                # Normalize scores
                import math

                normalized_scores = [1 / (1 + math.exp(-s)) for s in scores]
                return [max(0.0, min(1.0, float(s))) for s in normalized_scores]
            else:
                # Fallback to individual scoring
                scores = []
                for doc in documents:
                    scores.append(self.score(query, doc))
                return scores
        except Exception as e:
            logger.debug(f"Batch reranker scoring failed: {e}, falling back to individual scoring")
            scores = []
            for doc in documents:
                scores.append(self.score(query, doc))
            return scores

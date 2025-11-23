"""RAG (Retrieval-Augmented Generation) metrics calculator."""

import re

from loguru import logger

from core.ai.ai_validator import AIResponseValidator
from core.ai.reranker import HuggingFaceReranker


class RAGMetricsCalculator:
    """Calculate RAG pipeline metrics."""

    def __init__(
        self,
        validator: AIResponseValidator | None = None,
        reranker: HuggingFaceReranker | None = None,
    ):
        """
        Initialize RAG metrics calculator.

        Args:
            validator: AI response validator (default: creates new instance)
            reranker: Reranker for improved relevance scoring (default: creates new instance)
        """
        self.validator = validator or AIResponseValidator()
        self._reranker = reranker

    @property
    def reranker(self) -> HuggingFaceReranker | None:
        """Lazy load reranker."""
        if self._reranker is None:
            try:
                self._reranker = HuggingFaceReranker()
            except Exception as e:
                logger.debug(f"Reranker not available: {e}, falling back to embeddings")
                return None
        return self._reranker

    def calculate(
        self,
        query: str | None = None,
        response: str | None = None,
        retrieved_docs: list[str] | None = None,
        expected_sources: list[str] | None = None,
        gold_context: str | None = None,
    ) -> dict[str, float]:
        """
        Calculate RAG metrics.

        Args:
            query: User query
            response: AI response
            retrieved_docs: Documents retrieved by RAG system
            expected_sources: Expected source documents
            gold_context: Gold standard context

        Returns:
            Dictionary of RAG metrics
        """
        metrics: dict[str, float] = {}

        if not response or not query:
            return metrics

        # Retrieval Precision@5 and Recall@5
        if retrieved_docs and expected_sources:
            relevant_count = 0
            for doc in retrieved_docs[:5]:
                is_relevant, _ = self.validator.validate_relevance(query, doc, threshold=0.5)
                if is_relevant:
                    relevant_count += 1

            if retrieved_docs:
                metrics["retrieval_precision_5"] = (
                    relevant_count / min(len(retrieved_docs), 5)
                ) * 100

            if expected_sources:
                found_sources = 0
                for expected in expected_sources:
                    for doc in retrieved_docs[:5]:
                        is_relevant, _ = self.validator.validate_relevance(
                            expected, doc, threshold=0.5
                        )
                        if is_relevant:
                            found_sources += 1
                            break
                metrics["retrieval_recall_5"] = (found_sources / len(expected_sources)) * 100

        # Context Relevance - use reranker if available, fallback to embeddings
        if retrieved_docs and response:
            relevance_scores = []
            if self.reranker:
                # Use reranker for more accurate relevance scoring
                try:
                    for doc in retrieved_docs:
                        score = self.reranker.score(document=doc, query=response)
                        relevance_scores.append(score)
                except Exception as e:
                    logger.debug(
                        f"Reranker failed for context_relevance: {e}, falling back to embeddings"
                    )
                    # Fallback to embeddings
                    for doc in retrieved_docs:
                        is_relevant, similarity = self.validator.validate_relevance(
                            doc, response, threshold=0.0
                        )
                        relevance_scores.append(similarity)
            else:
                # Fallback to embeddings
                for doc in retrieved_docs:
                    is_relevant, similarity = self.validator.validate_relevance(
                        doc, response, threshold=0.0
                    )
                    relevance_scores.append(similarity)
            if relevance_scores:
                metrics["context_relevance"] = (sum(relevance_scores) / len(relevance_scores)) * 100

        # Context Coverage
        if retrieved_docs and response:
            used_chunks = 0
            for doc in retrieved_docs:
                is_relevant, _ = self.validator.validate_relevance(doc, response, threshold=0.4)
                if is_relevant:
                    used_chunks += 1
            if retrieved_docs:
                metrics["context_coverage"] = (used_chunks / len(retrieved_docs)) * 100

        # Context Intrusion (verbatim copying detection)
        if retrieved_docs and response:
            intrusion_score = self._calculate_context_intrusion(retrieved_docs, response)
            metrics["context_intrusion"] = intrusion_score

        # Gold Context Match - use reranker if available, fallback to embeddings
        if gold_context and response:
            try:
                if self.reranker:
                    # Use reranker for more accurate matching
                    score = self.reranker.score(document=gold_context, query=response)
                    metrics["gold_context_match"] = score * 100
                else:
                    # Fallback to embeddings
                    is_relevant, similarity = self.validator.validate_relevance(
                        gold_context, response, threshold=0.0
                    )
                    metrics["gold_context_match"] = similarity * 100
            except Exception as e:
                logger.debug(f"Could not calculate gold_context_match: {e}")
                # Try fallback to embeddings
                try:
                    is_relevant, similarity = self.validator.validate_relevance(
                        gold_context, response, threshold=0.0
                    )
                    metrics["gold_context_match"] = similarity * 100
                except Exception:
                    pass

        # Reranker Score - use actual reranker if available, fallback to embeddings
        if retrieved_docs and query:
            relevance_scores = []
            if self.reranker:
                # Use actual reranker
                try:
                    scores = self.reranker.score_batch(query=query, documents=retrieved_docs)
                    relevance_scores = scores
                except Exception as e:
                    logger.debug(
                        f"Reranker failed for reranker_score: {e}, falling back to embeddings"
                    )
                    # Fallback to embeddings
                    for doc in retrieved_docs:
                        is_relevant, similarity = self.validator.validate_relevance(
                            query, doc, threshold=0.0
                        )
                        relevance_scores.append(similarity)
            else:
                # Fallback to embeddings
                for doc in retrieved_docs:
                    is_relevant, similarity = self.validator.validate_relevance(
                        query, doc, threshold=0.0
                    )
                    relevance_scores.append(similarity)
            if relevance_scores:
                metrics["reranker_score"] = sum(relevance_scores) / len(relevance_scores)

        return metrics

    def _calculate_context_intrusion(self, retrieved_docs: list[str], response: str) -> float:
        """
        Calculate context intrusion using longest common substring (LCS).

        Improved method that detects verbatim copying by finding longest contiguous
        spans copied from context, rather than simple word overlap.

        Based on research: optimal overlap is 15-30% (good grounding without copying).
        High overlap (>50%) indicates verbatim copying.

        Args:
            retrieved_docs: Retrieved context documents
            response: Generated response

        Returns:
            Intrusion score (0-100, higher = more copying)
        """

        def longest_common_substring(text1: str, text2: str) -> int:
            """Find length of longest common substring between two texts."""
            # Normalize: lowercase, remove extra whitespace
            text1 = re.sub(r"\s+", " ", text1.lower().strip())
            text2 = re.sub(r"\s+", " ", text2.lower().strip())

            if not text1 or not text2:
                return 0

            # Dynamic programming approach
            m, n = len(text1), len(text2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            max_len = 0

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if text1[i - 1] == text2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                        max_len = max(max_len, dp[i][j])
                    else:
                        dp[i][j] = 0

            return max_len

        def normalize_for_lcs(text: str) -> str:
            """Normalize text for LCS calculation, preserving structure."""
            # Remove citations, URLs, code blocks (whitelist boilerplate)
            # Remove markdown code blocks
            text = re.sub(r"```[\s\S]*?```", "", text)
            # Remove inline code
            text = re.sub(r"`[^`]+`", "", text)
            # Remove URLs
            text = re.sub(r"https?://\S+", "", text)
            # Remove common citation patterns
            text = re.sub(r"\(Source:\s*[^)]+\)", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\[Source:\s*[^\]]+\]", "", text, flags=re.IGNORECASE)
            text = re.sub(r"According to\s+[^,]+,\s*", "", text, flags=re.IGNORECASE)
            # Normalize whitespace
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        if not response or not retrieved_docs:
            return 0.0

        normalized_response = normalize_for_lcs(response)
        if len(normalized_response) == 0:
            return 0.0

        # Find maximum LCS across all context documents
        max_lcs_length = 0
        for doc in retrieved_docs:
            normalized_doc = normalize_for_lcs(doc)
            if normalized_doc:
                lcs_len = longest_common_substring(normalized_response, normalized_doc)
                max_lcs_length = max(max_lcs_length, lcs_len)

        # Calculate LCS ratio
        lcs_ratio = max_lcs_length / len(normalized_response) if normalized_response else 0.0

        # Research-based formula (same thresholds, but now based on LCS instead of word overlap)
        # LCS is more accurate for detecting verbatim copying
        if lcs_ratio <= 0.15:
            intrusion_score = 0.0
        elif lcs_ratio <= 0.30:
            intrusion_score = ((lcs_ratio - 0.15) / 0.15) * 20
        elif lcs_ratio <= 0.50:
            intrusion_score = 20 + ((lcs_ratio - 0.30) / 0.20) * 40
        elif lcs_ratio <= 0.70:
            intrusion_score = 60 + ((lcs_ratio - 0.50) / 0.20) * 30
        else:
            intrusion_score = 90 + min(10, ((lcs_ratio - 0.70) / 0.30) * 10)

        return max(0.0, min(100.0, intrusion_score))

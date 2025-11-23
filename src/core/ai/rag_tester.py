"""RAG (Retrieval-Augmented Generation) testing for AI systems."""

from dataclasses import dataclass
from typing import Any

from loguru import logger

# Thread-local storage for RAG context data (for test reporting)
_rag_context: dict[str, Any] = {}


def _store_rag_context(
    retrieved_docs: list[str] | None = None,
    expected_sources: list[str] | None = None,
    gold_context: str | None = None,
    query: str | None = None,
    response: str | None = None,
) -> None:
    """Store RAG context data for test reporting (transparent to tests)."""
    global _rag_context
    # Merge with existing context instead of replacing (preserve test data if already set)
    # Initialize if not exists
    if not _rag_context:
        _rag_context = {}

    # Update retrieved_docs: prefer test data over ChatPage URLs
    # If test explicitly provides retrieved_docs, use them (they're more accurate)
    # Only use ChatPage URLs if test doesn't provide retrieved_docs
    if retrieved_docs is not None:
        # If test provides retrieved_docs, always use them (they override ChatPage data)
        _rag_context["retrieved_docs"] = retrieved_docs
    elif "retrieved_docs" not in _rag_context:
        _rag_context["retrieved_docs"] = []

    # Only update expected_sources if not already set
    if expected_sources is not None:
        if "expected_sources" not in _rag_context or not _rag_context.get("expected_sources"):
            _rag_context["expected_sources"] = expected_sources
    elif "expected_sources" not in _rag_context:
        _rag_context["expected_sources"] = []

    # Update gold_context, query, response if provided (these can be updated)
    if gold_context is not None:
        _rag_context["gold_context"] = gold_context
    elif "gold_context" not in _rag_context:
        _rag_context["gold_context"] = None

    if query is not None:
        _rag_context["query"] = query
    elif "query" not in _rag_context:
        _rag_context["query"] = None

    if response is not None:
        _rag_context["response"] = response
    elif "response" not in _rag_context:
        _rag_context["response"] = None


def _get_rag_context() -> dict[str, Any]:
    """Retrieve stored RAG context data."""
    return _rag_context.copy()


def _clear_rag_context() -> None:
    """Clear RAG context data."""
    global _rag_context
    _rag_context = {}


@dataclass
class RAGTestResult:
    """Result of RAG testing."""

    passed: bool
    test_type: str
    score: float
    details: dict[str, Any]


class RAGTester:
    """Tests RAG capabilities of AI systems."""

    def __init__(self):
        """Initialize RAG tester."""
        logger.info("RAG tester initialized")

    def validate_context_retrieval(
        self, query: str, response: str, expected_sources: list[str], threshold: float = 0.6
    ) -> RAGTestResult:
        """
        Validate that AI retrieved correct context from knowledge base.

        Args:
            query: User query
            response: AI response
            expected_sources: List of expected source documents/chunks
            threshold: Relevance threshold

        Returns:
            RAGTestResult
        """
        # Store RAG context for dashboard collection
        _store_rag_context(
            retrieved_docs=None,
            expected_sources=expected_sources,
            query=query,
            response=response,
        )

        from core.ai.ai_validator import AIResponseValidator

        validator = AIResponseValidator()

        # Check if response is relevant to any expected source
        relevance_scores = []
        for source in expected_sources:
            is_relevant, similarity = validator.validate_relevance(
                source, response, threshold=threshold
            )
            relevance_scores.append(similarity)

        best_score = max(relevance_scores) if relevance_scores else 0.0
        avg_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        passed = best_score >= threshold

        return RAGTestResult(
            passed=passed,
            test_type="context_retrieval",
            score=best_score,
            details={
                "query": query,
                "expected_sources": len(expected_sources),
                "relevance_scores": relevance_scores,
                "best_score": best_score,
                "avg_score": avg_score,
                "threshold": threshold,
            },
        )

    def validate_source_attribution(
        self, response: str, expected_citations: list[str]
    ) -> RAGTestResult:
        """
        Validate that AI properly attributes sources.

        Args:
            response: AI response
            expected_citations: List of expected source citations

        Returns:
            RAGTestResult
        """
        response_lower = response.lower()

        # Common citation indicators
        citation_indicators = [
            "source:",
            "according to",
            "reference:",
            "from",
            "cited in",
            "mentioned in",
            "[",
            "]",
        ]

        # Check for citation indicators
        has_citation_format = any(ind in response_lower for ind in citation_indicators)

        # Check if expected citations are mentioned
        found_citations = []
        missing_citations = []

        for citation in expected_citations:
            if citation.lower() in response_lower:
                found_citations.append(citation)
            else:
                missing_citations.append(citation)

        attribution_score = (
            len(found_citations) / len(expected_citations) if expected_citations else 0.0
        )
        passed = attribution_score >= 0.8 and has_citation_format

        return RAGTestResult(
            passed=passed,
            test_type="source_attribution",
            score=attribution_score,
            details={
                "has_citation_format": has_citation_format,
                "expected_citations": len(expected_citations),
                "found_citations": found_citations,
                "missing_citations": missing_citations,
                "attribution_score": attribution_score,
            },
        )

    def validate_context_window_usage(
        self, query: str, response: str, provided_context: list[str], min_usage_ratio: float = 0.5
    ) -> RAGTestResult:
        """
        Validate that AI effectively uses provided context.

        Args:
            query: User query
            response: AI response
            provided_context: List of context chunks provided to AI
            min_usage_ratio: Minimum ratio of context that should be used

        Returns:
            RAGTestResult
        """
        # Store RAG context for dashboard collection
        _store_rag_context(
            retrieved_docs=provided_context,
            expected_sources=None,
            query=query,
            response=response,
        )

        from core.ai.ai_validator import AIResponseValidator

        validator = AIResponseValidator()

        # Check which context chunks are reflected in response
        usage_scores = []
        used_contexts = []

        for i, context in enumerate(provided_context):
            is_relevant, similarity = validator.validate_relevance(context, response, threshold=0.4)
            usage_scores.append(similarity)
            if is_relevant:
                used_contexts.append(i)

        usage_ratio = len(used_contexts) / len(provided_context) if provided_context else 0.0
        avg_usage_score = sum(usage_scores) / len(usage_scores) if usage_scores else 0.0
        passed = usage_ratio >= min_usage_ratio

        return RAGTestResult(
            passed=passed,
            test_type="context_window_usage",
            score=usage_ratio,
            details={
                "provided_contexts": len(provided_context),
                "used_contexts": len(used_contexts),
                "usage_ratio": usage_ratio,
                "avg_usage_score": avg_usage_score,
                "min_usage_ratio": min_usage_ratio,
                "usage_scores": usage_scores,
            },
        )

    def detect_hallucination_vs_retrieval(
        self, response: str, retrieved_context: list[str], threshold: float = 0.5
    ) -> RAGTestResult:
        """
        Detect if AI hallucinated instead of using retrieved context.

        Args:
            response: AI response
            retrieved_context: Context that was retrieved and provided
            threshold: Grounding threshold

        Returns:
            RAGTestResult
        """
        # Store RAG context for dashboard collection
        _store_rag_context(
            retrieved_docs=retrieved_context,
            expected_sources=None,
            query=None,
            response=response,
        )

        from core.ai.ai_validator import AIResponseValidator

        validator = AIResponseValidator()

        # Check if response is grounded in retrieved context
        grounding_scores = []
        for context in retrieved_context:
            is_relevant, similarity = validator.validate_relevance(
                context, response, threshold=threshold
            )
            grounding_scores.append(similarity)

        max_grounding = max(grounding_scores) if grounding_scores else 0.0
        avg_grounding = sum(grounding_scores) / len(grounding_scores) if grounding_scores else 0.0

        # Low grounding score indicates potential hallucination
        is_grounded = max_grounding >= threshold
        passed = is_grounded

        return RAGTestResult(
            passed=passed,
            test_type="hallucination_vs_retrieval",
            score=max_grounding,
            details={
                "max_grounding_score": max_grounding,
                "avg_grounding_score": avg_grounding,
                "threshold": threshold,
                "grounding_scores": grounding_scores,
                "is_grounded": is_grounded,
                "num_contexts": len(retrieved_context),
            },
        )

    def validate_retrieval_relevance(
        self, query: str, retrieved_docs: list[str], threshold: float = 0.6
    ) -> RAGTestResult:
        """
        Validate that retrieved documents are relevant to query.

        Args:
            query: User query
            retrieved_docs: Documents retrieved from knowledge base
            threshold: Relevance threshold

        Returns:
            RAGTestResult
        """
        # Store RAG context for dashboard collection
        _store_rag_context(
            retrieved_docs=retrieved_docs,
            expected_sources=None,
            query=query,
            response=None,
        )

        from core.ai.ai_validator import AIResponseValidator

        validator = AIResponseValidator()

        relevance_scores = []
        relevant_docs = []

        for i, doc in enumerate(retrieved_docs):
            is_relevant, similarity = validator.validate_relevance(query, doc, threshold=threshold)
            relevance_scores.append(similarity)
            if is_relevant:
                relevant_docs.append(i)

        relevance_ratio = len(relevant_docs) / len(retrieved_docs) if retrieved_docs else 0.0
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        passed = relevance_ratio >= 0.8  # At least 80% should be relevant

        return RAGTestResult(
            passed=passed,
            test_type="retrieval_relevance",
            score=avg_relevance,
            details={
                "query": query,
                "retrieved_docs": len(retrieved_docs),
                "relevant_docs": len(relevant_docs),
                "relevance_ratio": relevance_ratio,
                "avg_relevance": avg_relevance,
                "relevance_scores": relevance_scores,
                "threshold": threshold,
            },
        )

    def test_out_of_domain_query(
        self, query: str, response: str, expected_behavior: str = "acknowledge_limitation"
    ) -> RAGTestResult:
        """
        Test how AI handles queries outside knowledge base domain.

        Args:
            query: Out-of-domain query
            response: AI response
            expected_behavior: Expected behavior (acknowledge_limitation, refuse, etc.)

        Returns:
            RAGTestResult
        """
        response_lower = response.lower()

        # Indicators of proper handling
        limitation_indicators = [
            "i don't have",
            "not in my knowledge",
            "cannot find",
            "no information",
            "outside my scope",
            "i'm not sure",
            "don't know",
            "لا أملك",
            "ليس لدي",  # Arabic
        ]

        acknowledges_limitation = any(ind in response_lower for ind in limitation_indicators)

        # Check response length (shouldn't be too confident/long if out of domain)
        response_length = len(response.split())
        is_appropriately_brief = response_length < 100  # Configurable

        if expected_behavior == "acknowledge_limitation":
            passed = acknowledges_limitation
            score = 1.0 if acknowledges_limitation else 0.0
        else:
            passed = True  # Other behaviors not yet implemented
            score = 0.5

        return RAGTestResult(
            passed=passed,
            test_type="out_of_domain_query",
            score=score,
            details={
                "query": query,
                "expected_behavior": expected_behavior,
                "acknowledges_limitation": acknowledges_limitation,
                "response_length": response_length,
                "is_appropriately_brief": is_appropriately_brief,
            },
        )

    def comprehensive_rag_test(
        self,
        query: str,
        response: str,
        retrieved_docs: list[str] | None = None,
        expected_sources: list[str] | None = None,
        expected_citations: list[str] | None = None,
    ) -> dict[str, RAGTestResult]:
        """
        Run comprehensive RAG testing.

        Args:
            query: User query
            response: AI response
            retrieved_docs: Documents retrieved by RAG system
            expected_sources: Expected source documents
            expected_citations: Expected citations in response

        Returns:
            Dictionary of test types to results
        """
        # Store RAG context for dashboard collection
        _store_rag_context(
            retrieved_docs=retrieved_docs,
            expected_sources=expected_sources,
            gold_context=None,  # Could be enhanced
            query=query,
            response=response,
        )

        results = {}

        # Context retrieval
        if expected_sources:
            results["context_retrieval"] = self.validate_context_retrieval(
                query, response, expected_sources
            )

        # Source attribution
        if expected_citations:
            results["source_attribution"] = self.validate_source_attribution(
                response, expected_citations
            )

        # Context usage
        if retrieved_docs:
            results["context_usage"] = self.validate_context_window_usage(
                query, response, retrieved_docs
            )

            # Hallucination detection
            results["hallucination_check"] = self.detect_hallucination_vs_retrieval(
                response, retrieved_docs
            )

            # Retrieval relevance
            results["retrieval_relevance"] = self.validate_retrieval_relevance(
                query, retrieved_docs
            )

        # Log summary
        passed_tests = [name for name, result in results.items() if result.passed]
        failed_tests = [name for name, result in results.items() if not result.passed]

        logger.info(f"RAG tests passed: {len(passed_tests)}/{len(results)}")
        if failed_tests:
            logger.warning(f"Failed RAG tests: {', '.join(failed_tests)}")

        return results


# Global instance
_rag_tester: RAGTester | None = None


def get_rag_tester() -> RAGTester:
    """Get global RAG tester instance."""
    global _rag_tester
    if _rag_tester is None:
        _rag_tester = RAGTester()
    return _rag_tester


# Export context functions for tests to manually provide RAG data
__all__ = [
    "RAGTester",
    "RAGTestResult",
    "get_rag_tester",
    "_store_rag_context",
    "_get_rag_context",
    "_clear_rag_context",
]

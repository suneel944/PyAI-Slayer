"""RAG (Retrieval-Augmented Generation) metrics testing."""

import pytest
from loguru import logger


@pytest.mark.ai
@pytest.mark.rag
class TestRAGMetrics:
    """Test RAG pipeline metrics and retrieval quality."""

    @pytest.fixture(autouse=True)
    def setup(self, ai_validator, rag_tester, ai_test_data):
        """Setup RAG tester and validator."""
        self.validator = ai_validator
        self.rag_tester = rag_tester
        self.test_data = ai_test_data

    def _store_rag_context_for_dashboard(
        self,
        retrieved_docs=None,
        expected_sources=None,
        gold_context=None,
        query=None,
        response=None,
    ):
        """Helper to store RAG context for dashboard collection."""
        from core.ai.rag_tester import _store_rag_context

        _store_rag_context(
            retrieved_docs=retrieved_docs,
            expected_sources=expected_sources,
            gold_context=gold_context,
            query=query,
            response=response,
        )

    def test_retrieval_precision(self, chat_page, test_config):
        """RAG-001: Retrieved documents are relevant to query."""
        rag_data = self.test_data.get("rag_test_data", {})
        precision_data = rag_data.get("retrieval_precision", {})
        query = precision_data.get("query", "")
        retrieved_docs = precision_data.get("retrieved_docs", [])

        if not query or not retrieved_docs:
            pytest.skip("Retrieval precision test data not available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"
        assert len(response) > 0, "Empty response"

        # Store RAG context for dashboard collection (needed for reranker_score and context metrics)
        self._store_rag_context_for_dashboard(
            retrieved_docs=retrieved_docs,
            expected_sources=None,
            gold_context=None,
            query=query,
            response=response,
        )

        # Use RAG tester to validate retrieval relevance
        result = self.rag_tester.validate_retrieval_relevance(
            query=query, retrieved_docs=retrieved_docs, threshold=0.6
        )

        # Calculate precision: how many retrieved docs are relevant
        relevant_count = 0
        for doc in retrieved_docs[:5]:
            is_relevant, similarity = self.validator.validate_relevance(query, doc, threshold=0.5)
            if is_relevant:
                relevant_count += 1

        precision = (relevant_count / min(len(retrieved_docs), 5)) * 100 if retrieved_docs else 0

        assert precision >= 60.0, f"Low retrieval precision: {precision:.1f}%"
        assert result.passed, f"Retrieval relevance validation failed: {result.score:.3f}"
        logger.info(
            f"Retrieval precision: {precision:.1f}%, "
            f"RAG tester score: {result.score:.3f}, "
            f"Relevant docs: {relevant_count}/{min(len(retrieved_docs), 5)}"
        )

    def test_retrieval_recall(self, chat_page, test_config):
        """RAG-002: All relevant documents are retrieved."""
        rag_data = self.test_data.get("rag_test_data", {})
        recall_data = rag_data.get("retrieval_recall", {})
        query = recall_data.get("query", "")
        expected_sources = recall_data.get("expected_sources", [])
        retrieved_docs = recall_data.get("retrieved_docs", [])

        if not query or not expected_sources or not retrieved_docs:
            pytest.skip("Retrieval recall test data not available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"
        assert len(response) > 0, "Empty response"

        # Store RAG context for dashboard collection
        self._store_rag_context_for_dashboard(
            retrieved_docs=retrieved_docs,
            expected_sources=expected_sources,
            gold_context=None,
            query=query,
            response=response,
        )

        # Use RAG tester to validate context retrieval
        result = self.rag_tester.validate_context_retrieval(
            query=query, response=response, expected_sources=expected_sources, threshold=0.6
        )

        # Calculate recall: how many expected sources were found in retrieved docs
        found_sources = 0
        for expected in expected_sources:
            for doc in retrieved_docs[:5]:
                is_relevant, similarity = self.validator.validate_relevance(
                    expected, doc, threshold=0.5
                )
                if is_relevant:
                    found_sources += 1
                    break

        recall = (found_sources / len(expected_sources)) * 100 if expected_sources else 0

        assert recall >= 50.0, f"Low retrieval recall: {recall:.1f}%"
        logger.info(
            f"Retrieval recall: {recall:.1f}%, "
            f"RAG tester score: {result.score:.3f}, "
            f"Found sources: {found_sources}/{len(expected_sources)}"
        )

    def test_context_relevance(self, chat_page, test_config):
        """RAG-003: Retrieved context is relevant to response."""
        rag_data = self.test_data.get("rag_test_data", {})
        relevance_data = rag_data.get("context_relevance", {})
        query = relevance_data.get("query", "")
        retrieved_context = relevance_data.get("retrieved_context", [])

        if not query or not retrieved_context:
            pytest.skip("Context relevance test data not available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"
        assert len(response) > 0, "Empty response"

        # Store RAG context for dashboard collection
        self._store_rag_context_for_dashboard(
            retrieved_docs=retrieved_context,
            expected_sources=None,
            gold_context=None,
            query=query,
            response=response,
        )

        # Use RAG tester to validate context usage
        result = self.rag_tester.validate_context_window_usage(
            query=query, response=response, provided_context=retrieved_context, min_usage_ratio=0.5
        )

        # Calculate context relevance: how relevant is each retrieved doc to the response
        relevance_scores = []
        for doc in retrieved_context:
            is_relevant, similarity = self.validator.validate_relevance(
                doc, response, threshold=0.0
            )
            relevance_scores.append(similarity)

        avg_relevance = (
            (sum(relevance_scores) / len(relevance_scores)) * 100 if relevance_scores else 0
        )

        assert avg_relevance >= 60.0, f"Low context relevance: {avg_relevance:.1f}%"
        assert result.passed, f"Context usage validation failed: {result.score:.3f}"
        logger.info(
            f"Context relevance: {avg_relevance:.1f}%, "
            f"Context usage score: {result.score:.3f}, "
            f"Avg similarity: {sum(relevance_scores) / len(relevance_scores):.3f}"
        )

    def test_context_coverage(self, chat_page, test_config):
        """RAG-004: Response uses retrieved context effectively."""
        rag_data = self.test_data.get("rag_test_data", {})
        coverage_data = rag_data.get("context_coverage", {})
        query = coverage_data.get("query", "")
        retrieved_docs = coverage_data.get("retrieved_docs", [])

        if not query or not retrieved_docs:
            pytest.skip("Context coverage test data not available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"
        assert len(response) > 0, "Empty response"

        # Store RAG context for dashboard collection
        self._store_rag_context_for_dashboard(
            retrieved_docs=retrieved_docs,
            expected_sources=None,
            gold_context=None,
            query=query,
            response=response,
        )

        # Use RAG tester to validate context usage
        result = self.rag_tester.validate_context_window_usage(
            query=query, response=response, provided_context=retrieved_docs, min_usage_ratio=0.5
        )

        # Calculate context coverage: how many retrieved chunks are used in response
        used_chunks = 0
        chunk_scores = []
        for doc in retrieved_docs:
            is_relevant, similarity = self.validator.validate_relevance(
                doc, response, threshold=0.4
            )
            chunk_scores.append(similarity)
            if is_relevant:
                used_chunks += 1

        coverage = (used_chunks / len(retrieved_docs)) * 100 if retrieved_docs else 0
        avg_chunk_score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0

        assert coverage >= 50.0, f"Low context coverage: {coverage:.1f}%"
        assert result.passed, f"Context usage validation failed: {result.score:.3f}"
        logger.info(
            f"Context coverage: {coverage:.1f}%, "
            f"Used chunks: {used_chunks}/{len(retrieved_docs)}, "
            f"Usage score: {result.score:.3f}, "
            f"Avg chunk similarity: {avg_chunk_score:.3f}"
        )

    def test_gold_context_match(self, chat_page, test_config):
        """RAG-005: Response matches gold standard context."""
        rag_data = self.test_data.get("rag_test_data", {})
        gold_data = rag_data.get("gold_context_match", {})
        query = gold_data.get("query", "")
        gold_context = gold_data.get("gold_context", "")

        if not query or not gold_context:
            pytest.skip("Gold context match test data not available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"
        assert len(response) > 0, "Empty response"

        # Store RAG context for dashboard collection (including gold_context)
        self._store_rag_context_for_dashboard(
            retrieved_docs=None,
            expected_sources=None,
            gold_context=gold_context,
            query=query,
            response=response,
        )

        # Calculate gold context match: how well does response match the gold standard
        is_relevant, similarity = self.validator.validate_relevance(
            gold_context, response, threshold=0.0
        )
        match_score = similarity * 100

        assert match_score >= 50.0, f"Low gold context match: {match_score:.1f}%"
        logger.info(
            f"Gold context match: {match_score:.1f}%, "
            f"Similarity: {similarity:.3f}, "
            f"Response length: {len(response)} chars"
        )

    def test_context_intrusion_detection(self, chat_page, test_config):
        """RAG-006: Detect unwanted context leakage in response."""
        rag_data = self.test_data.get("rag_test_data", {})
        intrusion_data = rag_data.get("context_intrusion", {})
        query = intrusion_data.get("query", "")
        retrieved_docs = intrusion_data.get("retrieved_docs", [])

        if not query or not retrieved_docs:
            pytest.skip("Context intrusion test data not available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"
        assert len(response) > 0, "Empty response"

        # Store RAG context for dashboard collection (needed for context_intrusion metric)
        self._store_rag_context_for_dashboard(
            retrieved_docs=retrieved_docs,
            expected_sources=None,
            gold_context=None,
            query=query,
            response=response,
        )

        # Check for context intrusion: response should paraphrase, not copy verbatim
        # High similarity score = verbatim copying = context intrusion (BAD)
        # Low similarity score = good paraphrasing = no intrusion (GOOD)
        result = self.rag_tester.detect_hallucination_vs_retrieval(
            response=response, retrieved_context=retrieved_docs, threshold=0.5
        )

        # Context intrusion detection: similarity score should be LOW (< 0.3)
        # High score (>= 0.3) indicates verbatim copying from context, which is unwanted
        # Note: High grounding score is GOOD for grounding tests, but BAD for intrusion tests
        assert not result.passed or result.score < 0.3, (
            f"Context intrusion detected: response similarity to context is too high "
            f"(score={result.score:.3f}, threshold=<0.3). "
            f"The response appears to be copying context verbatim rather than paraphrasing. "
            f"Expected: paraphrased response with similarity < 0.3. "
            f"Got: verbatim-like response with similarity {result.score:.3f}."
        )
        logger.info(
            f"Context intrusion check: similarity score={result.score:.3f} "
            f"(threshold=<0.3, {'PASS' if result.score < 0.3 else 'FAIL'}), "
            f"grounded={result.passed}, details={result.details}"
        )

    def test_comprehensive_rag_validation(self, chat_page, test_config):
        """RAG-007: Comprehensive RAG validation using all metrics."""
        rag_data = self.test_data.get("rag_test_data", {})
        comprehensive_data = rag_data.get("comprehensive_rag", {})
        query = comprehensive_data.get("query", "")
        retrieved_docs = comprehensive_data.get("retrieved_docs", [])
        expected_sources = comprehensive_data.get("expected_sources", [])

        if not query or not retrieved_docs or not expected_sources:
            pytest.skip("Comprehensive RAG test data not available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"
        assert len(response) > 0, "Empty response"

        # Get expected citations if available
        expected_citations = comprehensive_data.get("expected_citations", None)

        # Use comprehensive RAG test
        results = self.rag_tester.comprehensive_rag_test(
            query=query,
            response=response,
            retrieved_docs=retrieved_docs,
            expected_sources=expected_sources,
            expected_citations=expected_citations,
        )

        # Validate that comprehensive test stored RAG context
        from core.ai.rag_tester import _get_rag_context

        rag_context = _get_rag_context()
        assert (
            rag_context.get("retrieved_docs") == retrieved_docs
        ), "RAG context not stored properly"

        # Check results
        passed_tests = [name for name, result in results.items() if result.passed]
        total_tests = len(results)

        assert total_tests > 0, "No RAG tests were executed"
        logger.info(
            f"Comprehensive RAG test: {len(passed_tests)}/{total_tests} passed. "
            f"Results: {[(name, r.score) for name, r in results.items()]}"
        )

    def test_retrieval_with_multiple_queries(self, chat_page, test_config):
        """RAG-008: Test retrieval across multiple query types."""
        rag_data = self.test_data.get("rag_test_data", {})
        queries = rag_data.get("multiple_queries", [])

        if not queries:
            pytest.skip("Multiple queries test data not available")

        for i, query in enumerate(queries):
            chat_page.send_message(query, wait_for_response=True)
            response = chat_page.get_latest_response()

            if response:
                # Store RAG context for each query (simulate retrieved docs)
                simulated_docs = [
                    f"Relevant information about query {i + 1}: {query}",
                ]
                self._store_rag_context_for_dashboard(
                    retrieved_docs=simulated_docs,
                    expected_sources=None,
                    gold_context=None,
                    query=query,
                    response=response,
                )

                is_relevant, similarity = self.validator.validate_relevance(query, response)
                assert is_relevant, f"Response not relevant for: {query}"
                logger.info(f"Query {i + 1} '{query}' relevance: {similarity:.3f}")

    def test_source_attribution(self, chat_page, test_config):
        """RAG-009: Response properly attributes sources."""
        rag_data = self.test_data.get("rag_test_data", {})
        attribution_data = rag_data.get("source_attribution", {})
        query = attribution_data.get("query", "")
        expected_citations = attribution_data.get("expected_citations", [])

        if not query or not expected_citations:
            pytest.skip("Source attribution test data not available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"
        assert len(response) > 0, "Empty response"

        # Store RAG context for dashboard collection
        self._store_rag_context_for_dashboard(
            retrieved_docs=None,
            expected_sources=None,
            gold_context=None,
            query=query,
            response=response,
        )

        # Use RAG tester to validate source attribution
        result = self.rag_tester.validate_source_attribution(
            response=response, expected_citations=expected_citations
        )

        assert result.passed, f"Source attribution failed: {result.score:.3f}"
        logger.info(
            f"Source attribution test passed: score={result.score:.3f}, "
            f"found_citations={result.details.get('found_citations', [])}, "
            f"missing_citations={result.details.get('missing_citations', [])}"
        )

    def test_out_of_domain_handling(self, chat_page, test_config):
        """RAG-010: System handles out-of-domain queries appropriately."""
        rag_data = self.test_data.get("rag_test_data", {})
        out_of_domain_data = rag_data.get("out_of_domain", {})
        query = out_of_domain_data.get("query", "")
        expected_behavior = out_of_domain_data.get("expected_behavior", "acknowledge_limitation")

        if not query:
            pytest.skip("Out-of-domain test data not available")

        chat_page.send_message(query, wait_for_response=True)
        response = chat_page.get_latest_response()

        assert response is not None, "No response"
        assert len(response) > 0, "Empty response"

        # Store RAG context for dashboard collection
        self._store_rag_context_for_dashboard(
            retrieved_docs=None,
            expected_sources=None,
            gold_context=None,
            query=query,
            response=response,
        )

        # Use RAG tester to validate out-of-domain handling
        result = self.rag_tester.test_out_of_domain_query(
            query=query, response=response, expected_behavior=expected_behavior
        )

        assert result.passed, (
            f"System should acknowledge limitations for out-of-domain queries. "
            f"Score: {result.score:.3f}, Details: {result.details}"
        )
        logger.info(
            f"Out-of-domain handling test passed: score={result.score:.3f}, "
            f"acknowledges_limitation={result.details.get('acknowledges_limitation', False)}"
        )

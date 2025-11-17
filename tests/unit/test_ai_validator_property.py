"""Property-based tests for AIResponseValidator using Hypothesis."""

from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from core import AIResponseValidator


class TestAIResponseValidatorProperty:
    """Property-based tests for AIResponseValidator."""

    @patch("core.infrastructure.cache.get_embedding_cache")
    @patch("core.ai.ai_validator.SentenceTransformer")
    @given(
        query=st.text(min_size=1, max_size=200),
        response=st.text(min_size=1, max_size=500),
        threshold=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_validate_relevance_properties(
        self, mock_sentence_transformer, mock_get_embedding_cache, query, response, threshold
    ):
        """Property: validate_relevance always returns valid similarity score."""
        # Setup mock model
        mock_model = MagicMock()
        # Generate deterministic embeddings based on text length
        # Accept any keyword arguments that encode might receive
        mock_model.encode.side_effect = lambda text, **kwargs: [float(len(text)) / 100.0] * 10
        mock_sentence_transformer.return_value = mock_model

        # Mock embedding cache to return None (no cached embeddings)
        mock_embedding_cache = MagicMock()
        mock_embedding_cache.get.return_value = None
        mock_get_embedding_cache.return_value = mock_embedding_cache

        validator = AIResponseValidator()
        validator._semantic_model = mock_model

        is_relevant, similarity = validator.validate_relevance(query, response, threshold=threshold)

        # Properties that should always hold
        assert isinstance(is_relevant, bool)
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        assert is_relevant == (similarity >= threshold)

    @patch("core.ai.ai_validator.SentenceTransformer")
    @given(
        response=st.text(min_size=1, max_size=500),
        known_facts=st.lists(st.text(min_size=1, max_size=100), min_size=0, max_size=10),
    )
    def test_detect_hallucination_properties(
        self, mock_sentence_transformer, response, known_facts
    ):
        """Property: detect_hallucination always returns valid result."""
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda text: [float(len(text)) / 100.0] * 10
        mock_sentence_transformer.return_value = mock_model

        validator = AIResponseValidator()
        validator._semantic_model = mock_model

        has_hallucination, conflicting = validator.detect_hallucination(response, known_facts)

        # Properties that should always hold
        assert isinstance(has_hallucination, bool)
        assert isinstance(conflicting, list)
        assert all(isinstance(fact, str) for fact in conflicting)
        assert has_hallucination == (len(conflicting) > 0)

    @patch("core.ai.ai_validator.SentenceTransformer")
    @given(responses=st.lists(st.text(min_size=1, max_size=200), min_size=1, max_size=10))
    @settings(deadline=500)  # Increase deadline to 500ms to avoid timeout
    def test_check_consistency_properties(self, mock_sentence_transformer, responses):
        """Property: check_consistency always returns valid consistency score."""
        mock_model = MagicMock()
        # Accept any keyword arguments that encode might receive
        mock_model.encode.side_effect = lambda text, **kwargs: [float(len(text)) / 100.0] * 10
        mock_sentence_transformer.return_value = mock_model

        validator = AIResponseValidator()
        validator._semantic_model = mock_model

        consistency = validator.check_consistency(responses)

        # Properties that should always hold
        assert isinstance(consistency, float)
        assert 0.0 <= consistency <= 1.0
        # Single response should have consistency of 1.0
        if len(responses) == 1:
            assert consistency == 1.0

    @given(
        response=st.text(min_size=0, max_size=1000),
        min_length=st.integers(min_value=0, max_value=100),
        max_length=st.integers(min_value=0, max_value=2000) | st.none(),
    )
    def test_validate_response_quality_properties(self, response, min_length, max_length):
        """Property: validate_response_quality always returns valid quality checks."""
        validator = AIResponseValidator()

        quality = validator.validate_response_quality(
            response, min_length=min_length, max_length=max_length
        )

        # Properties that should always hold
        assert isinstance(quality, dict)
        assert "has_minimum_length" in quality
        assert "is_not_empty" in quality
        assert "has_no_html_tags" in quality
        assert all(isinstance(v, bool) for v in quality.values())

        # Logical properties
        if len(response.strip()) == 0:
            assert quality["is_not_empty"] is False
        if len(response.strip()) < min_length:
            assert quality["has_minimum_length"] is False

        if max_length is not None:
            assert "within_max_length" in quality
            if len(response) > max_length:
                assert quality["within_max_length"] is False

    @given(text=st.text())
    def test_is_arabic_properties(self, text):
        """Property: _is_arabic returns boolean and is consistent."""
        validator = AIResponseValidator()
        result = validator._is_arabic(text)

        assert isinstance(result, bool)
        # If text contains Arabic characters, result should be True
        has_arabic = any("\u0600" <= char <= "\u06ff" for char in text)
        assert result == has_arabic

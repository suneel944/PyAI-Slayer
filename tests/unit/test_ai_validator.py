"""Unit tests for AIResponseValidator."""

from unittest.mock import MagicMock, patch

from core import AIResponseValidator


class TestAIResponseValidator:
    """Test suite for AIResponseValidator."""

    def test_init(self):
        """Test AIResponseValidator initialization."""
        validator = AIResponseValidator()
        assert validator._semantic_model is None
        assert validator._arabic_semantic_model is None
        assert validator._translator is None

    @patch("core.infrastructure.cache.get_model_cache")
    @patch("core.ai.ai_validator.SentenceTransformer")
    def test_semantic_model_lazy_loading(self, mock_sentence_transformer, mock_get_cache):
        """Test semantic model lazy loading."""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        # Mock cache to return None (no cached model)
        mock_cache = MagicMock()
        mock_cache.get_model.return_value = None
        mock_get_cache.return_value = mock_cache

        validator = AIResponseValidator()
        model = validator.semantic_model

        assert model == mock_model
        mock_sentence_transformer.assert_called_once()
        mock_cache.set_model.assert_called_once()

    @patch("core.infrastructure.cache.get_model_cache")
    @patch("core.ai.ai_validator.SentenceTransformer")
    def test_arabic_semantic_model_lazy_loading(self, mock_sentence_transformer, mock_get_cache):
        """Test Arabic semantic model lazy loading."""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        # Mock cache to return None (no cached model)
        mock_cache = MagicMock()
        mock_cache.get_model.return_value = None
        mock_get_cache.return_value = mock_cache

        validator = AIResponseValidator()
        model = validator.arabic_semantic_model

        assert model == mock_model
        assert mock_sentence_transformer.call_count >= 1
        mock_cache.set_model.assert_called()

    def test_is_arabic(self):
        """Test Arabic text detection."""
        validator = AIResponseValidator()

        assert validator._is_arabic("مرحبا") is True
        assert validator._is_arabic("Hello") is False
        assert validator._is_arabic("Hello مرحبا") is True
        assert validator._is_arabic("") is False

    @patch("core.ai.ai_validator.SentenceTransformer")
    def test_validate_relevance(self, mock_sentence_transformer):
        """Test relevance validation."""
        # Setup mock model
        mock_model = MagicMock()
        mock_embedding = [0.1, 0.2, 0.3]
        mock_model.encode.return_value = mock_embedding
        mock_sentence_transformer.return_value = mock_model

        validator = AIResponseValidator()
        validator._semantic_model = mock_model

        query = "How do I renew my visa?"
        response = "To renew your visa, you need to follow these steps..."

        is_relevant, similarity = validator.validate_relevance(query, response, threshold=0.5)

        assert isinstance(is_relevant, bool)
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        assert mock_model.encode.call_count == 2

    @patch("core.infrastructure.cache.get_embedding_cache")
    @patch("core.ai.ai_validator.SentenceTransformer")
    def test_validate_relevance_arabic(self, mock_sentence_transformer, mock_get_embedding_cache):
        """Test relevance validation with Arabic text."""
        # Setup mock model
        mock_model = MagicMock()
        mock_embedding = [0.1, 0.2, 0.3]
        mock_model.encode.return_value = mock_embedding

        # Mock embedding cache to return None (no cached embeddings)
        mock_embedding_cache = MagicMock()
        mock_embedding_cache.get.return_value = None
        mock_get_embedding_cache.return_value = mock_embedding_cache
        mock_sentence_transformer.return_value = mock_model

        validator = AIResponseValidator()
        validator._arabic_semantic_model = mock_model

        query = "كيف يمكنني تجديد تأشيرتي؟"
        response = "لتجديد تأشيرتك، تحتاج إلى اتباع هذه الخطوات..."

        is_relevant, similarity = validator.validate_relevance(query, response)

        assert isinstance(is_relevant, bool)
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0

    @patch("core.ai.ai_validator.SentenceTransformer")
    def test_validate_relevance_error_handling(self, mock_sentence_transformer):
        """Test relevance validation error handling."""
        mock_model = MagicMock()
        mock_model.encode.side_effect = Exception("Model error")
        mock_sentence_transformer.return_value = mock_model

        validator = AIResponseValidator()
        validator._semantic_model = mock_model

        is_relevant, similarity = validator.validate_relevance("query", "response")

        assert is_relevant is False
        assert similarity == 0.0

    @patch("core.ai.ai_validator.SentenceTransformer")
    def test_detect_hallucination(self, mock_sentence_transformer):
        """Test hallucination detection."""
        mock_model = MagicMock()
        mock_embedding = [0.1, 0.2, 0.3]
        mock_model.encode.return_value = mock_embedding
        mock_sentence_transformer.return_value = mock_model

        validator = AIResponseValidator()
        validator._semantic_model = mock_model

        response = "The sky is green and the grass is blue."
        known_facts = ["The sky is blue", "The grass is green"]

        has_hallucination, conflicting = validator.detect_hallucination(response, known_facts)

        assert isinstance(has_hallucination, bool)
        assert isinstance(conflicting, list)

    def test_detect_hallucination_no_facts(self):
        """Test hallucination detection with no known facts."""
        validator = AIResponseValidator()
        has_hallucination, conflicting = validator.detect_hallucination("response", [])

        assert has_hallucination is False
        assert conflicting == []

    @patch("core.ai.ai_validator.SentenceTransformer")
    def test_check_consistency(self, mock_sentence_transformer):
        """Test consistency checking."""
        mock_model = MagicMock()
        mock_embedding = [0.1, 0.2, 0.3]
        mock_model.encode.return_value = mock_embedding
        mock_sentence_transformer.return_value = mock_model

        validator = AIResponseValidator()
        validator._semantic_model = mock_model

        responses = ["Response 1", "Response 2", "Response 3"]

        consistency = validator.check_consistency(responses)

        assert isinstance(consistency, float)
        assert 0.0 <= consistency <= 1.0

    def test_check_consistency_single_response(self):
        """Test consistency with single response."""
        validator = AIResponseValidator()
        consistency = validator.check_consistency(["Single response"])

        assert consistency == 1.0

    @patch("core.ai.ai_validator.SentenceTransformer")
    def test_check_consistency_error_handling(self, mock_sentence_transformer):
        """Test consistency check error handling."""
        mock_model = MagicMock()
        mock_model.encode.side_effect = Exception("Model error")
        mock_sentence_transformer.return_value = mock_model

        validator = AIResponseValidator()
        validator._semantic_model = mock_model

        consistency = validator.check_consistency(["response1", "response2"])

        assert consistency == 0.0

    @patch("core.infrastructure.cache.get_embedding_cache")
    @patch("core.ai.ai_validator.SentenceTransformer")
    @patch("core.ai.ai_validator.GoogleTranslator")
    def test_validate_cross_language(
        self, mock_translator, mock_sentence_transformer, mock_get_embedding_cache
    ):
        """Test cross-language validation."""
        # Setup mocks
        mock_model = MagicMock()
        mock_embedding = [0.1, 0.2, 0.3]
        mock_model.encode.return_value = mock_embedding
        mock_sentence_transformer.return_value = mock_model

        # Mock embedding cache to return None (no cached embeddings)
        mock_embedding_cache = MagicMock()
        mock_embedding_cache.get.return_value = None
        mock_get_embedding_cache.return_value = mock_embedding_cache

        mock_translator_instance = MagicMock()
        mock_translator_instance.translate.return_value = "English translation"
        mock_translator.return_value = mock_translator_instance

        validator = AIResponseValidator()
        validator._semantic_model = mock_model
        validator._translator = mock_translator_instance

        en_response = "This is an English response"
        ar_response = "هذه استجابة بالعربية"

        is_consistent, similarity = validator.validate_cross_language(en_response, ar_response)

        assert isinstance(is_consistent, bool)
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0

    def test_validate_cross_language_empty_responses(self):
        """Test cross-language validation with empty responses."""
        validator = AIResponseValidator()
        is_consistent, similarity = validator.validate_cross_language("", "response")

        assert is_consistent is False
        assert similarity == 0.0

    def test_validate_response_quality(self):
        """Test response quality validation."""
        validator = AIResponseValidator()

        # Valid response
        quality = validator.validate_response_quality(
            "This is a valid response that ends properly.", min_length=10
        )

        assert quality["has_minimum_length"] is True
        assert quality["is_not_empty"] is True
        assert quality["has_no_html_tags"] is True
        assert quality["ends_properly"] is True

        # Invalid response
        quality = validator.validate_response_quality(
            "<script>alert('xss')</script>", min_length=10
        )

        assert quality["has_no_html_tags"] is False

    def test_validate_response_quality_max_length(self):
        """Test response quality with max length check."""
        validator = AIResponseValidator()

        quality = validator.validate_response_quality("Short", min_length=5, max_length=10)

        assert "within_max_length" in quality
        assert quality["within_max_length"] is True

    def test_detect_fallback_message(self):
        """Test fallback message detection."""
        validator = AIResponseValidator()

        assert validator.detect_fallback_message("Sorry, I didn't understand") is True
        assert validator.detect_fallback_message("Try again later") is True
        assert validator.detect_fallback_message("حدث خطأ") is True
        assert validator.detect_fallback_message("This is a normal response") is False

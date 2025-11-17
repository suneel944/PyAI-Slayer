# ADR-0004: Use Sentence Transformers for Semantic Validation

## Status
Accepted

## Context
We need to validate AI responses for relevance and quality. Traditional string matching is insufficient for:
- Semantic similarity checking
- Cross-language validation
- Hallucination detection

## Decision
We will use sentence-transformers library with multilingual models for semantic validation.

## Consequences
### Positive
- Accurate semantic similarity measurement
- Multilingual support (English, Arabic, etc.)
- Pre-trained models available
- Good performance with GPU support

### Negative
- Large model files (hundreds of MB)
- Initial model loading time (10-30 seconds)
- Requires PyTorch dependency
- Memory usage for model inference

## Alternatives Considered
- **spaCy**: Good but less accurate for semantic similarity
- **OpenAI Embeddings API**: Requires API key, costs money, network dependency
- **BERT directly**: More complex, sentence-transformers is easier to use

## Implementation
Semantic validation is implemented in `AIResponseValidator` using:
- `intfloat/multilingual-e5-base` for general use
- `Omartificial-Intelligence-Space/mmbert-base-arabic-nli` for Arabic-specific validation

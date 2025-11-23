# RAG Calibration Recommendations Usage Guide

## Overview

The calibration recommendations (`data/rag_calibration_recommendations.json`) are automatically loaded and used by the dashboard to set metric targets based on your actual data distribution. These targets are calibrated from your labeled evaluation dataset, ensuring they reflect your real-world performance expectations rather than arbitrary thresholds.

## Prerequisites: Creating Your Eval Set

**⚠️ IMPORTANT**: Before running calibration, you need to create your own evaluation dataset. The template is provided, but you must populate it with your actual queries and data.

### Step 1: Check for Existing Eval Set

First, check if you already have an eval set:

```bash
ls -la data/rag_eval_set.json
```

If the file exists, you can proceed to calibration. If not, continue to Step 2.

### Step 2: Create Your Eval Set

You have two options:

#### Option A: Use the Example Template (Quick Start)

The example template provides a starting structure:

```bash
# The example is in data/examples/example_rag_eval_set.json
# Copy it to create your own
cp data/examples/example_rag_eval_set.json data/rag_eval_set.json
```

Then edit `data/rag_eval_set.json` with your own data.

#### Option B: Create from Scratch

Create a new file `data/rag_eval_set.json` with this structure:

```json
{
  "name": "my_rag_eval_set",
  "version": "1.0",
  "examples": [
    {
      "query": "Your actual user query here",
      "gold_answer": "The ideal/correct answer",
      "gold_context": "The minimal context that contains the answer",
      "chunks": [
        {
          "chunk_text": "Document chunk text",
          "relevance": 3,
          "chunk_id": "chunk_1"
        },
        {
          "chunk_text": "Another document chunk",
          "relevance": 2,
          "chunk_id": "chunk_2"
        }
      ],
      "expected_sources": ["source_1", "source_2"],
      "metadata": {
        "category": "government",
        "language": "en"
      }
    }
  ]
}
```

### Step 3: Understanding the Eval Set Structure

Each example in your eval set should contain:

- **`query`** (required): The user's question/query
- **`gold_answer`** (optional): The ideal answer your system should produce
- **`gold_context`** (optional): The minimal context that contains the answer
- **`chunks`** (required): List of document chunks with relevance labels:
  - `chunk_text`: The actual chunk content
  - `relevance`: Integer 0-3 where:
    - `0` = Not relevant
    - `1` = Somewhat relevant
    - `2` = Relevant
    - `3` = Highly relevant
  - `chunk_id` (optional): Identifier for the chunk
- **`expected_sources`** (optional): List of source identifiers that should be retrieved
- **`metadata`** (optional): Any additional information (category, language, etc.)

### Step 4: Populate with Real Data

**Critical**: Replace the template data with your actual production queries and documents:

1. **Collect Real Queries**: Use actual user queries from your system
2. **Label Chunks**: Manually label each chunk's relevance (0-3 scale)
3. **Define Gold Context**: Identify the minimal context needed to answer each query
4. **Set Expected Sources**: List which sources should be retrieved for each query

**Recommended Size**: Start with 50-100 examples for meaningful calibration. More examples = better calibration.

### Step 5: Validate Your Eval Set

You can validate your eval set structure using Python:

```python
from src.core.ai.rag_eval_set import RAGEvalSet

# Load and validate
eval_set = RAGEvalSet.from_json("data/rag_eval_set.json")
print(f"Loaded {len(eval_set)} examples")
print(f"Name: {eval_set.name}, Version: {eval_set.version}")

# Check each example
for i, ex in enumerate(eval_set.examples, 1):
    print(f"\nExample {i}:")
    print(f"  Query: {ex.query[:60]}...")
    print(f"  Chunks: {len(ex.chunks)}")
    print(f"  Relevance labels: {[c.relevance for c in ex.chunks]}")
```

## How It Works

1. **Create Your Eval Set**: Follow the steps above to create `data/rag_eval_set.json`

2. **Run Calibration**: Generate recommendations from your eval set
   ```bash
   python scripts/calibrate_rag_metrics.py
   ```
   This will:
   - Load your eval set (or create example if missing)
   - Calculate metrics for each example
   - Generate calibration recommendations
   - Save to `data/rag_calibration_recommendations.json`

3. **Automatic Loading**: The dashboard automatically loads targets from `data/rag_calibration_recommendations.json` when it starts

4. **Dynamic Updates**: Targets are displayed in the dashboard UI and used for:
   - Critical metrics evaluation
   - Status indicators (good/warning/danger)
   - Target labels in RAG metric cards

## API Endpoint

The dashboard exposes an API endpoint to get current targets:

```bash
GET /api/rag/targets
```

Returns:
```json
{
  "retrieval_recall_5": 100.0,
  "retrieval_precision_5": 100.0,
  "context_relevance": 0.0,
  "context_coverage": 100.0,
  "context_intrusion": 0.69,
  "gold_context_match": 0.0,
  "reranker_score": 0.0
}
```

## Where Targets Are Used

### 1. Critical Metrics Dashboard
- **Retrieval Precision@5**: Uses `retrieval_precision_5` target
- **Retrieval Recall@5**: Uses `retrieval_recall_5` target
- **Context Coverage**: Uses `context_coverage` target

### 2. RAG Metrics Cards
- **Recall@5**: Target label shows calibrated value
- **Precision@5**: Target label shows calibrated value
- **Context Intrusion**: Target label and status indicator use calibrated value
- **Reranker Score**: Target label and status indicator use calibrated value

### 3. Status Indicators
- **Context Intrusion**:
  - Green (Within Target): `< target`
  - Yellow (Needs Attention): `< target * 2`
  - Red (High Intrusion): `>= target * 2`

- **Reranker Score**:
  - Green (Above Target): `>= target`
  - Yellow (Below Target): `>= target * 0.75`
  - Red (Poor Score): `< target * 0.75`

## Configuration Options

### Priority Order

Targets are loaded in this priority order (highest to lowest):

1. **`.env` Variables** (highest priority)
   - Set individual targets via environment variables
   - Example: `RAG_TARGET_RETRIEVAL_RECALL_5=90.0`
   - See `.env.example` for all available variables

2. **Calibration File** (medium priority)
   - Auto-generated from your eval set
   - Located at: `data/rag_calibration_recommendations.json`
   - Generated by running: `python scripts/calibrate_rag_metrics.py`

3. **Code Defaults** (lowest priority)
   - Fallback values if nothing else is configured
   - Defined in `src/dashboard/rag_targets.py`

### Using .env Variables

You can override any target via `.env`:

```env
# RAG Metric Targets (Optional - overrides calibration file if set)
# Leave empty to use calibration recommendations from data/rag_calibration_recommendations.json
# If any target is set, it will override the calibration file for that metric
RAG_TARGET_RETRIEVAL_RECALL_5=90.0
RAG_TARGET_RETRIEVAL_PRECISION_5=85.0
RAG_TARGET_CONTEXT_RELEVANCE=80.0
RAG_TARGET_CONTEXT_COVERAGE=85.0
RAG_TARGET_CONTEXT_INTRUSION=5.0
RAG_TARGET_GOLD_CONTEXT_MATCH=85.0
RAG_TARGET_RERANKER_SCORE=0.8
```

**Note**: If you set even one target in `.env`, all other targets will use defaults (not from calibration file). This is by design to allow complete control when needed.

## Updating Targets

### Method 1: Re-run Calibration (Recommended)

1. **Expand your eval set** with more real production data
2. **Re-run calibration**:
   ```bash
   python scripts/calibrate_rag_metrics.py
   ```
3. **Restart dashboard** (or refresh browser) to load new targets

### Method 2: Manual Override via .env

1. **Edit `.env`** and set the targets you want to override
2. **Restart dashboard** to load new values

### Method 3: Direct File Edit (Not Recommended)

You can manually edit `data/rag_calibration_recommendations.json`, but this will be overwritten the next time you run calibration.

## Default Targets

If calibration file is missing, the dashboard uses these defaults:

- `retrieval_recall_5`: 85%
- `retrieval_precision_5`: 85%
- `context_relevance`: 80%
- `context_coverage`: 80%
- `context_intrusion`: 5% (lower is better)
- `gold_context_match`: 85%
- `reranker_score`: 0.8 (0-1 scale)

## Example Eval Set Structure

Here's a complete example of a well-structured eval set entry:

```json
{
  "name": "my_rag_eval_set",
  "version": "1.0",
  "examples": [
    {
      "query": "What is the main topic discussed in the document?",
      "gold_answer": "The document discusses the fundamental principles of machine learning and its applications.",
      "gold_context": "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming.",
      "chunks": [
        {
          "chunk_text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming.",
          "relevance": 3,
          "chunk_id": "doc_001_chunk_01"
        },
        {
          "chunk_text": "There are three main types of machine learning: supervised, unsupervised, and reinforcement learning.",
          "relevance": 2,
          "chunk_id": "doc_001_chunk_02"
        },
        {
          "chunk_text": "Deep learning uses neural networks with multiple layers to process complex patterns.",
          "relevance": 1,
          "chunk_id": "doc_001_chunk_03"
        },
        {
          "chunk_text": "The history of computing dates back to the 1940s with the development of early computers.",
          "relevance": 0,
          "chunk_id": "doc_002_chunk_01"
        }
      ],
      "expected_sources": ["doc_001"],
      "metadata": {
        "category": "technology",
        "language": "en",
        "difficulty": "medium"
      }
    }
  ]
}
```

### Relevance Labeling Guidelines

- **3 (Highly Relevant)**: Chunk directly answers the query
- **2 (Relevant)**: Chunk is related and provides useful context
- **1 (Somewhat Relevant)**: Chunk is tangentially related
- **0 (Not Relevant)**: Chunk doesn't help answer the query

## Best Practices

1. **Start with Template**: Use `data/examples/example_rag_eval_set.json` as a starting point

2. **Use Real Data**: Always populate with actual production queries and documents - don't use placeholder data

3. **Label Carefully**: Take time to accurately label chunk relevance - this directly affects calibration quality

4. **Aim for 50-100 Examples**: Start with at least 50 examples for meaningful calibration. More is better.

5. **Diverse Queries**: Include:
   - Easy queries (direct answers)
   - Complex queries (require multiple chunks)
   - Edge cases (ambiguous, out-of-scope)
   - Different categories/topics

6. **Regular Calibration**: Re-run calibration when:
   - You add significant new data to your eval set (20+ new examples)
   - Your RAG pipeline changes (new embedding model, reranker, etc.)
   - You notice metrics drifting in production

7. **Version Control**:
   - Keep your eval set private (it's in `.gitignore`)
   - Keep calibration recommendations private (they're in `.gitignore`)
   - Commit example templates to help others understand the format

8. **Documentation**: Document any manual target adjustments in `.env` and explain why

9. **Monitoring**: Watch for targets that seem unrealistic - may indicate:
   - Issues with eval set quality
   - Problems with metric calculations
   - Need for more diverse examples

## Troubleshooting

### "Calibration recommendations not found"

**Solution**: Run calibration first:
```bash
python scripts/calibrate_rag_metrics.py
```

### "Eval set not found"

**Solution**: Create your eval set:
1. Copy template: `cp data/examples/example_rag_eval_set.json data/rag_eval_set.json`
2. Edit with your data
3. Run calibration again

### Targets seem wrong (all 0% or 100%)

**Possible causes**:
- Eval set too small (< 10 examples)
- Reranker not working correctly (check if `reranker_score` is always 0.0)
- Responses don't match context well
- Need to expand eval set with more diverse examples

**Solution**:
- If `reranker_score` is consistently 0.0, verify the reranker model is loaded correctly:
  - Check that `RAG_RERANKER_ENABLED=true` in your `.env`
  - Ensure CUDA/GPU is available if `RAG_RERANKER_USE_CUDA=true`
  - Verify the model downloads successfully on first use
- Expand your eval set with more real production data and re-run calibration

### Reranker Score and URL Content Fetching

**URL Content Fetching (Enabled by Default):**
- The framework automatically fetches content from URLs to improve reranker accuracy
- When enabled (`RAG_FETCH_URL_CONTENT=true`, default), URLs are fetched and their content is used for reranker scoring
- This significantly improves reranker score accuracy for real chat interactions

**Configuration:**
```env
# Enable/disable URL content fetching (default: true)
RAG_FETCH_URL_CONTENT=true

# Timeout per URL in seconds (default: 10)
RAG_URL_FETCH_TIMEOUT=10

# Maximum content length per URL in characters (default: 10000)
RAG_URL_MAX_CONTENT_LENGTH=10000

# Maximum retry attempts per URL if fetch fails (default: 3)
RAG_URL_MAX_RETRIES=3

# Delay between retries in seconds, uses exponential backoff (default: 1.0)
RAG_URL_RETRY_DELAY=1.0

# Maximum parallel workers for URL fetching (default: 5)
RAG_URL_MAX_WORKERS=5
```

**Features:**
- **Parallel Fetching**: URLs are fetched concurrently using multiple workers for improved performance
- **Retry Logic**: Automatic retries with exponential backoff (1s, 2s, 4s) for failed requests
- **Content Type Support**: Automatically handles HTML, PDF, JSON, and plain text content
- **Error Handling**: Gracefully handles timeouts, connection errors, and HTTP errors (4xx/5xx)

**Fallback Behavior:**
- If URL fetching fails or is disabled, the reranker falls back to scoring URLs as-is (less accurate)
- For test data: When tests provide `retrieved_context` (actual document text), that content is used directly (no URL fetching needed)

**Best Practices:**
- Keep URL fetching enabled for accurate reranker scores in production
- Adjust timeout and max length based on your URL response times and content sizes
- For tests, provide actual document content via `retrieved_context` or `retrieved_docs` for fastest execution

### Want to use different targets per environment?

Use `.env` variables to override targets for different environments (dev/staging/prod).

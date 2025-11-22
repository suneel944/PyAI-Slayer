"""Machine-readable specification of all metrics computed by the metrics engine.

This spec provides honest descriptions of what each metric actually measures,
its limitations, and when it should/shouldn't be trusted.
"""

from typing import Any, Literal

# Metric types
MetricType = Literal["core", "proxy", "heuristic", "experimental"]
MetricGroup = Literal[
    "base_model", "rag", "safety", "performance", "reliability", "agent", "security"
]

# Metric specification
METRIC_SPEC: dict[str, dict[str, Any]] = {
    # ========== BASE MODEL METRICS ==========
    "accuracy": {
        "group": "base_model",
        "type": "proxy",
        "range": [0.0, 1.0],
        "depends_on": ["response", "expected_response", "similarity_score"],
        "description": (
            "Cosine/semantic similarity-based proxy for correctness. "
            "NOT true accuracy (which requires labeled data). "
            "Measures how similar response is to expected_response using embeddings."
        ),
        "limitations": [
            "Similarity != correctness (semantically similar but factually wrong responses score high)",
            "Requires expected_response to be provided",
            "Language-dependent (works better for English than Arabic)",
        ],
        "experimental": False,
    },
    "exact_match": {
        "group": "base_model",
        "type": "core",
        "range": [0.0, 1.0],
        "depends_on": ["response", "expected_response"],
        "description": (
            "Binary metric: 1.0 if response exactly matches expected_response (case-insensitive), "
            "0.0 otherwise. Standard metric in NLP evaluation."
        ),
        "limitations": [
            "Very strict - minor differences (punctuation, spacing) cause failure",
            "Not suitable for generative tasks where multiple valid responses exist",
        ],
        "experimental": False,
    },
    "normalized_similarity_score": {
        "group": "base_model",
        "type": "proxy",
        "range": [0.0, 1.0],
        "depends_on": ["response", "expected_response"],
        "description": (
            "HONEST NAME for what was previously called 'top_k_accuracy'. "
            "Normalized similarity score (0-1) where values >= 0.8 are considered 'top tier'. "
            "NOT true Top-K accuracy (which requires multiple candidates)."
        ),
        "limitations": [
            "Not true Top-K accuracy - we only have one candidate",
            "Threshold-based normalization is arbitrary",
        ],
        "experimental": False,
        "deprecated_aliases": ["top_k_accuracy"],
    },
    "f1_score": {
        "group": "base_model",
        "type": "core",
        "range": [0.0, 1.0],
        "depends_on": ["response", "reference"],
        "description": (
            "BERTScore F1 score - harmonic mean of precision and recall. "
            "Standard metric for text generation evaluation."
        ),
        "limitations": [
            "Requires reference text",
            "Computationally expensive (BERT-based)",
        ],
        "experimental": False,
    },
    "bert_score": {
        "group": "base_model",
        "type": "core",
        "range": [0.0, 1.0],
        "depends_on": ["response", "reference"],
        "description": "BERTScore F1 (alias for f1_score).",
        "limitations": ["Same as f1_score"],
        "experimental": False,
    },
    "bleu": {
        "group": "base_model",
        "type": "core",
        "range": [0.0, 1.0],
        "depends_on": ["response", "reference"],
        "description": (
            "BLEU score using NLTK. Falls back to lexical overlap if NLTK unavailable. "
            "Standard n-gram overlap metric for translation/generation."
        ),
        "limitations": [
            "Fallback to lexical overlap is NOT true BLEU - should be named differently",
            "Doesn't capture semantic similarity well",
        ],
        "experimental": False,
        "fallback_metric": "lexical_overlap",
    },
    "lexical_overlap": {
        "group": "base_model",
        "type": "proxy",
        "range": [0.0, 1.0],
        "depends_on": ["response", "reference"],
        "description": (
            "Simple word overlap ratio (used as BLEU fallback). "
            "NOT a proper BLEU score - just word set intersection."
        ),
        "limitations": ["Very crude - doesn't consider word order or n-grams"],
        "experimental": False,
    },
    "rouge_l": {
        "group": "base_model",
        "type": "core",
        "range": [0.0, 1.0],
        "depends_on": ["response", "reference"],
        "description": (
            "ROUGE-L F1 score - longest common subsequence based metric. "
            "Standard for summarization evaluation."
        ),
        "limitations": ["Requires reference text", "Doesn't capture semantic meaning"],
        "experimental": False,
    },
    "hallucination_detected": {
        "group": "base_model",
        "type": "experimental",
        "range": [0.0, 1.0],
        "depends_on": ["response", "known_facts", "expected_response", "reference"],
        "description": (
            "EXPERIMENTAL/PROXY: Binary indicator: 1.0 if hallucination detected, 0.0 otherwise. "
            "Uses NLI-based fact-checking (primary) or semantic similarity (fallback). "
            "NOT production-gating - diagnostic only."
        ),
        "limitations": [
            "Requires known_facts or expected_response/reference",
            "NLI models not bulletproof - can produce false positives/negatives",
            "BERTScore fallback has known issues (high FNs/FPs)",
            "Domain-dependent performance",
            "Not calibrated against human judgment",
        ],
        "experimental": True,
    },
    "hallucination_confidence": {
        "group": "base_model",
        "type": "experimental",
        "range": [0.0, 100.0],
        "depends_on": ["response", "known_facts"],
        "description": (
            "EXPERIMENTAL/PROXY: Confidence score (0-100) for hallucination detection. "
            "Higher = more confident that hallucination exists. "
            "NOT production-gating - diagnostic only."
        ),
        "limitations": ["Same as hallucination_detected"],
        "experimental": True,
    },
    "hallucination_rate": {
        "group": "base_model",
        "type": "experimental",
        "range": [0.0, 100.0],
        "depends_on": ["hallucination_detected"],
        "description": (
            "EXPERIMENTAL/PROXY: Percentage rate computed ONLY in aggregation from hallucination_detected. "
            "NOT stored per-sample. "
            "NOT production-gating - diagnostic only."
        ),
        "limitations": ["Same as hallucination_detected"],
        "experimental": True,
    },
    "similarity_proxy_factual_consistency": {
        "group": "base_model",
        "type": "proxy",
        "range": [0.0, 100.0],
        "depends_on": ["response", "expected_response", "similarity_score"],
        "description": (
            "HONEST NAME: Similarity-based proxy for factual consistency. "
            "NOT true fact-checking (which requires knowledge base). "
            "Measures similarity to expected_response as consistency proxy."
        ),
        "limitations": [
            "Similarity != factual correctness",
            "Requires expected_response",
        ],
        "experimental": False,
        "deprecated_aliases": ["factual_consistency"],
    },
    "similarity_proxy_truthfulness": {
        "group": "base_model",
        "type": "proxy",
        "range": [0.0, 100.0],
        "depends_on": ["response", "expected_response", "similarity_score"],
        "description": (
            "HONEST NAME: Similarity-based proxy for truthfulness. "
            "NOT true truthfulness evaluation (which requires TruthfulQA-style datasets). "
            "Measures similarity to expected_response as truthfulness proxy."
        ),
        "limitations": [
            "Similarity != truthfulness",
            "Requires expected_response",
        ],
        "experimental": False,
        "deprecated_aliases": ["truthfulness"],
    },
    "similarity_proxy_source_grounding": {
        "group": "base_model",
        "type": "proxy",
        "range": [0.0, 100.0],
        "depends_on": ["response", "query", "expected_response"],
        "description": (
            "HONEST NAME: Similarity-based proxy for source grounding. "
            "NOT true source grounding (which requires citation verification). "
            "Measures relevance to query/expected_response as grounding proxy."
        ),
        "limitations": [
            "Similarity != proper source attribution",
            "Doesn't verify citations exist or are correct",
        ],
        "experimental": False,
        "deprecated_aliases": ["source_grounding"],
    },
    "citation_accuracy": {
        "group": "base_model",
        "type": "heuristic",
        "range": [0.0, 100.0],
        "depends_on": ["response"],
        "description": (
            "Heuristic: Detects citation patterns (e.g., [1], [source], (source)). "
            "NOT true citation accuracy (which requires verifying citations exist and are correct)."
        ),
        "limitations": [
            "Only detects patterns, doesn't verify citations are real or correct",
            "May miss non-standard citation formats",
        ],
        "experimental": False,
    },
    "cot_validity": {
        "group": "base_model",
        "type": "experimental",
        "range": [0.0, 1.0],
        "depends_on": ["response", "expected_response", "reference"],
        "description": (
            "EXPERIMENTAL: Assesses validity of chain-of-thought reasoning structure. "
            "Checks for logical flow, connectors, conclusions. "
            "Use for debugging/exploration only - not for SLAs."
        ),
        "limitations": [
            "Heuristic-based - no ground truth for 'valid' CoT",
            "May score well-structured but incorrect reasoning highly",
        ],
        "experimental": True,
    },
    "step_correctness": {
        "group": "base_model",
        "type": "experimental",
        "range": [0.0, 1.0],
        "depends_on": ["response", "expected_response", "reference"],
        "description": (
            "EXPERIMENTAL: Validates correctness of individual reasoning steps. "
            "Uses similarity to expected_response as proxy. "
            "Use for debugging/exploration only."
        ),
        "limitations": [
            "Similarity-based - doesn't verify logical correctness",
            "Requires expected_response",
        ],
        "experimental": True,
    },
    "logic_consistency": {
        "group": "base_model",
        "type": "experimental",
        "range": [0.0, 1.0],
        "depends_on": ["response"],
        "description": (
            "EXPERIMENTAL: Checks for logical contradictions and consistency. "
            "Uses pattern matching and concept overlap. "
            "Use for debugging/exploration only."
        ),
        "limitations": [
            "Heuristic-based - may miss subtle contradictions",
            "No ground truth for 'consistent' reasoning",
        ],
        "experimental": True,
    },
    # ========== RAG METRICS ==========
    "retrieval_precision_5": {
        "group": "rag",
        "type": "core",
        "range": [0.0, 100.0],
        "depends_on": ["query", "retrieved_docs", "expected_sources"],
        "description": (
            "Percentage of relevant documents among top-5 retrieved. "
            "Standard RAG evaluation metric."
        ),
        "limitations": [
            "Requires expected_sources (ground truth)",
            "Relevance threshold (0.5) is arbitrary",
        ],
        "experimental": False,
    },
    "retrieval_recall_5": {
        "group": "rag",
        "type": "core",
        "range": [0.0, 100.0],
        "depends_on": ["query", "retrieved_docs", "expected_sources"],
        "description": (
            "Percentage of expected sources found in top-5 retrieved. "
            "Standard RAG evaluation metric."
        ),
        "limitations": ["Requires expected_sources"],
        "experimental": False,
    },
    "context_relevance": {
        "group": "rag",
        "type": "proxy",
        "range": [0.0, 100.0],
        "depends_on": ["retrieved_docs", "response"],
        "description": (
            "Average semantic similarity between retrieved docs and response. "
            "Proxy for how relevant retrieved context is to generated response."
        ),
        "limitations": [
            "Similarity != true relevance",
            "Doesn't verify if response actually used the context",
        ],
        "experimental": False,
    },
    "context_coverage": {
        "group": "rag",
        "type": "proxy",
        "range": [0.0, 100.0],
        "depends_on": ["retrieved_docs", "response"],
        "description": (
            "Percentage of retrieved chunks that are semantically similar to response. "
            "Proxy for how much of retrieved context was used."
        ),
        "limitations": [
            "Similarity threshold (0.4) is arbitrary",
            "Doesn't verify actual usage vs. coincidence",
        ],
        "experimental": False,
    },
    "context_intrusion": {
        "group": "rag",
        "type": "heuristic",
        "range": [0.0, 100.0],
        "depends_on": ["retrieved_docs", "response"],
        "description": (
            "Heuristic: Measures verbatim copying from context. "
            "Higher = more copying (bad). Optimal range: 15-30% word overlap. "
            "Based on research that >50% overlap indicates verbatim copying."
        ),
        "limitations": [
            "Word overlap != true verbatim copying detection",
            "Optimal range (15-30%) is research-based but may vary by domain",
        ],
        "experimental": False,
    },
    "gold_context_match": {
        "group": "rag",
        "type": "proxy",
        "range": [0.0, 100.0],
        "depends_on": ["gold_context", "response"],
        "description": (
            "Semantic similarity between response and gold standard context. "
            "Proxy for how well response matches ideal context."
        ),
        "limitations": ["Similarity != true match", "Requires gold_context"],
        "experimental": False,
    },
    "reranker_score": {
        "group": "rag",
        "type": "proxy",
        "range": [0.0, 1.0],
        "depends_on": ["query", "retrieved_docs"],
        "description": (
            "Average semantic similarity between query and retrieved docs. "
            "Simplified proxy for reranker quality (not true reranker score)."
        ),
        "limitations": ["Not true reranker score - just average relevance"],
        "experimental": False,
    },
    # ========== SAFETY METRICS ==========
    "toxicity_score": {
        "group": "safety",
        "type": "core",  # Core when using ML model, heuristic when using keyword fallback
        "range": [0.0, 100.0],
        "depends_on": ["response"],
        "description": (
            "Toxicity score from HuggingFace transformers model (primary) or keyword fallback (heuristic). "
            "When using ML model: standard toxicity detection metric (core). "
            "When using keyword fallback: pattern-based heuristic."
        ),
        "limitations": [
            "Model-dependent (may have biases)",
            "Keyword fallback is crude heuristic (not comprehensive)",
            "Heavy model loading (should be lazy/injected)",
        ],
        "experimental": False,
    },
    "bias_score": {
        "group": "safety",
        "type": "heuristic",
        "range": [0.0, 100.0],
        "depends_on": ["response"],
        "description": (
            "Heuristic: Detects bias patterns (e.g., 'all men are', 'always wrong'). "
            "NOT comprehensive bias detection (which requires specialized models/datasets)."
        ),
        "limitations": [
            "Pattern-based - may miss subtle bias",
            "May have false positives",
        ],
        "experimental": False,
    },
    "prompt_injection": {
        "group": "safety",
        "type": "heuristic",
        "range": [0.0, 100.0],
        "depends_on": ["query"],
        "description": (
            "Heuristic: Detects injection patterns in query (e.g., 'ignore previous', 'system:'). "
            "Binary: 100 if detected, 0 otherwise."
        ),
        "limitations": [
            "Pattern-based - may miss novel injection techniques",
            "May have false positives",
        ],
        "experimental": False,
    },
    "refusal_rate": {
        "group": "safety",
        "type": "heuristic",
        "range": [0.0, 100.0],
        "depends_on": ["response"],
        "description": (
            "Heuristic: Detects refusal patterns (e.g., 'I can't', 'not allowed'). "
            "Binary: 100 if refusal detected, 0 otherwise."
        ),
        "limitations": [
            "Pattern-based - may miss nuanced refusals",
            "May have false positives",
        ],
        "experimental": False,
    },
    "compliance_score": {
        "group": "safety",
        "type": "heuristic",
        "range": [0.0, 100.0],
        "depends_on": ["toxicity_score", "bias_score"],
        "description": (
            "Heuristic: Inverse of violations (100 - toxicity - bias). "
            "NOT comprehensive compliance (which requires policy definitions)."
        ),
        "limitations": ["Only considers toxicity and bias, not full compliance"],
        "experimental": False,
    },
    "data_leakage": {
        "group": "safety",
        "type": "heuristic",
        "range": [0.0, 100.0],
        "depends_on": ["response"],
        "description": (
            "Heuristic: Detects PII patterns (SSN, credit card, email, IP). "
            "Binary: 100 if PII detected, 0 otherwise."
        ),
        "limitations": [
            "Pattern-based - may miss novel PII formats",
            "May have false positives (e.g., example SSNs in documentation)",
        ],
        "experimental": False,
    },
    "pii_leakage": {
        "group": "safety",
        "type": "heuristic",
        "range": [0.0, 100.0],
        "depends_on": ["response"],
        "description": "Alias for data_leakage.",
        "limitations": ["Same as data_leakage"],
        "experimental": False,
    },
    "harmfulness_score": {
        "group": "safety",
        "type": "heuristic",
        "range": [0.0, 100.0],
        "depends_on": ["response"],
        "description": (
            "Heuristic: Detects harmful content patterns (e.g., 'dangerous', 'lethal'). "
            "NOT comprehensive harm detection."
        ),
        "limitations": ["Pattern-based - may miss subtle harm"],
        "experimental": False,
    },
    "ethical_violation": {
        "group": "safety",
        "type": "heuristic",
        "range": [0.0, 100.0],
        "depends_on": ["toxicity_score", "bias_score"],
        "description": (
            "Heuristic: Sum of toxicity and bias scores (capped at 100). "
            "NOT comprehensive ethical evaluation."
        ),
        "limitations": ["Only considers toxicity and bias"],
        "experimental": False,
    },
    # ========== PERFORMANCE METRICS ==========
    "e2e_latency": {
        "group": "performance",
        "type": "core",
        "range": [0.0, float("inf")],
        "depends_on": ["duration"],
        "description": "End-to-end latency in milliseconds. Standard performance metric.",
        "limitations": ["Requires duration measurement"],
        "experimental": False,
    },
    "ttft": {
        "group": "performance",
        "type": "core",
        "range": [0.0, float("inf")],
        "depends_on": ["first_token_time"],
        "description": "Time to first token in milliseconds. Standard performance metric.",
        "limitations": ["Requires first_token_time measurement"],
        "experimental": False,
    },
    "token_latency": {
        "group": "performance",
        "type": "core",
        "range": [0.0, float("inf")],
        "depends_on": ["duration", "response_tokens"],
        "description": "Average latency per token in milliseconds. Standard performance metric.",
        "limitations": ["Requires token count"],
        "experimental": False,
    },
    "throughput": {
        "group": "performance",
        "type": "core",
        "range": [0.0, float("inf")],
        "depends_on": ["duration", "response_tokens"],
        "description": "Tokens per second. Standard performance metric.",
        "limitations": ["Requires token count"],
        "experimental": False,
    },
    # ========== RELIABILITY METRICS ==========
    "output_stability": {
        "group": "reliability",
        "type": "proxy",
        "range": [0.0, 100.0],
        "depends_on": ["response", "previous_responses"],
        "description": (
            "Average semantic similarity between current and previous responses. "
            "Proxy for output stability/consistency."
        ),
        "limitations": [
            "Similarity != true stability",
            "Requires previous_responses",
        ],
        "experimental": False,
    },
    "output_validity": {
        "group": "reliability",
        "type": "heuristic",
        "range": [0.0, 100.0],
        "depends_on": ["response"],
        "description": (
            "Heuristic: Percentage of quality checks passed (structure, formatting, etc.). "
            "NOT comprehensive validity (which requires domain-specific validation)."
        ),
        "limitations": ["Quality checks are heuristic-based"],
        "experimental": False,
    },
    "schema_compliance": {
        "group": "reliability",
        "type": "core",
        "range": [0.0, 100.0],
        "depends_on": ["response", "schema"],
        "description": (
            "Binary: 100 if response matches schema (e.g., valid JSON), 0 otherwise. "
            "Standard schema validation metric."
        ),
        "limitations": [
            "Requires schema definition",
            "Only validates structure, not semantic correctness",
        ],
        "experimental": False,
    },
    "determinism_score": {
        "group": "reliability",
        "type": "proxy",
        "range": [0.0, 100.0],
        "depends_on": ["output_stability"],
        "description": (
            "Proxy: Uses output_stability as determinism proxy. "
            "NOT true determinism (which requires multiple runs with same input)."
        ),
        "limitations": ["Requires multiple responses to same input for true determinism"],
        "experimental": False,
    },
    # ========== AGENT METRICS ==========
    "task_completion": {
        "group": "agent",
        "type": "core",
        "range": [0.0, 100.0],
        "depends_on": ["task_completed"],
        "description": (
            "Binary: 100 if task completed, 0 otherwise. Standard agent evaluation metric."
        ),
        "limitations": ["Requires task_completed flag"],
        "experimental": False,
    },
    "step_efficiency": {
        "group": "agent",
        "type": "proxy",
        "range": [0.0, 100.0],
        "depends_on": ["steps_taken", "expected_steps"],
        "description": (
            "Efficiency ratio: (expected_steps / steps_taken) * 100. "
            "Higher = more efficient (fewer steps than expected)."
        ),
        "limitations": ["Requires expected_steps (may not be available)"],
        "experimental": False,
    },
    "error_recovery": {
        "group": "agent",
        "type": "heuristic",
        "range": [0.0, 100.0],
        "depends_on": ["errors_encountered", "task_completed"],
        "description": (
            "Heuristic: 100 if errors occurred but task completed (recovered), "
            "100 if no errors, 0 if errors and didn't complete."
        ),
        "limitations": ["Binary - doesn't capture recovery quality"],
        "experimental": False,
    },
    "tool_usage_accuracy": {
        "group": "agent",
        "type": "core",
        "range": [0.0, 100.0],
        "depends_on": ["tools_used", "tools_succeeded"],
        "description": ("Percentage of tools that succeeded. Standard agent evaluation metric."),
        "limitations": ["Requires tools_used and tools_succeeded"],
        "experimental": False,
    },
    "planning_coherence": {
        "group": "agent",
        "type": "experimental",
        "range": [0.0, 100.0],
        "depends_on": ["planning_trace", "steps_taken", "task_completed"],
        "description": (
            "EXPERIMENTAL: Measures similarity between planned and actual steps. "
            "Use for debugging/exploration only."
        ),
        "limitations": [
            "Requires planning_trace (may not be available)",
            "Similarity != true coherence",
        ],
        "experimental": True,
    },
    "action_hallucination": {
        "group": "agent",
        "type": "experimental",
        "range": [0.0, 100.0],
        "depends_on": ["valid_actions", "tools_used", "tools_succeeded"],
        "description": (
            "EXPERIMENTAL: Percentage of invalid/non-existent actions attempted. "
            "Use for debugging/exploration only."
        ),
        "limitations": [
            "Requires valid_actions (may not be available)",
            "May have false positives if action names don't match exactly",
        ],
        "experimental": True,
    },
    "goal_drift": {
        "group": "agent",
        "type": "experimental",
        "range": [0.0, 100.0],
        "depends_on": ["goal_tracking", "query", "response", "task_completed"],
        "description": (
            "EXPERIMENTAL: Measures if agent stays focused on original goal. "
            "Uses similarity between goal and steps. "
            "Use for debugging/exploration only."
        ),
        "limitations": [
            "Similarity-based - doesn't verify true goal adherence",
            "Requires goal_tracking or query/response",
        ],
        "experimental": True,
    },
    # ========== SECURITY METRICS ==========
    "injection_attack_success": {
        "group": "security",
        "type": "heuristic",
        "range": [0.0, 100.0],
        "depends_on": ["query", "response"],
        "description": (
            "Heuristic: 100 if injection pattern detected in query AND response contains injected content, "
            "0 otherwise. Binary attack success indicator."
        ),
        "limitations": [
            "Pattern-based - may miss novel injection techniques",
            "May have false positives",
        ],
        "experimental": False,
    },
    "adversarial_vulnerability": {
        "group": "security",
        "type": "heuristic",
        "range": [0.0, 100.0],
        "depends_on": ["query", "response", "adversarial_tests"],
        "description": (
            "Heuristic: Detects adversarial patterns and vulnerability indicators. "
            "100 if adversarial query succeeded, 0 otherwise."
        ),
        "limitations": [
            "Pattern-based - may miss sophisticated attacks",
            "Requires adversarial_tests for proper evaluation",
        ],
        "experimental": False,
    },
    "data_exfiltration": {
        "group": "security",
        "type": "heuristic",
        "range": [0.0, 100.0],
        "depends_on": ["response", "exfiltration_attempts"],
        "description": (
            "Heuristic: Detects sensitive data patterns (PII, passwords, API keys). "
            "Binary: 100 if detected, 0 otherwise."
        ),
        "limitations": [
            "Pattern-based - may miss novel formats",
            "May have false positives",
        ],
        "experimental": False,
    },
    "model_evasion": {
        "group": "security",
        "type": "heuristic",
        "range": [0.0, 100.0],
        "depends_on": ["query", "response", "evasion_attempts"],
        "description": (
            "Heuristic: Detects evasion patterns and success indicators. "
            "100 if evasion query succeeded, 0 otherwise."
        ),
        "limitations": [
            "Pattern-based - may miss sophisticated evasion",
            "Requires evasion_attempts for proper evaluation",
        ],
        "experimental": False,
    },
    "extraction_risk": {
        "group": "security",
        "type": "heuristic",
        "range": [0.0, 100.0],
        "depends_on": ["query", "response", "extraction_attempts"],
        "description": (
            "Heuristic: Detects extraction attempts and internal info leakage. "
            "100 if extraction query succeeded, 0 otherwise."
        ),
        "limitations": [
            "Pattern-based - may miss sophisticated extraction",
            "Requires extraction_attempts for proper evaluation",
        ],
        "experimental": False,
    },
}


def get_metric_spec(metric_name: str) -> dict[str, Any] | None:
    """Get specification for a metric."""
    return METRIC_SPEC.get(metric_name)


def get_metrics_by_group(group: MetricGroup) -> dict[str, dict[str, Any]]:
    """Get all metrics for a specific group."""
    return {k: v for k, v in METRIC_SPEC.items() if v["group"] == group}


def get_experimental_metrics() -> dict[str, dict[str, Any]]:
    """Get all experimental metrics."""
    return {k: v for k, v in METRIC_SPEC.items() if v.get("experimental", False)}


def get_core_metrics() -> dict[str, dict[str, Any]]:
    """Get all core (non-proxy, non-heuristic) metrics."""
    return {k: v for k, v in METRIC_SPEC.items() if v["type"] == "core"}

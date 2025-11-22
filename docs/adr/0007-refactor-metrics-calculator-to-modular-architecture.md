# ADR-0007: Refactor Metrics Calculator to Modular Architecture

## Status
Accepted

## Context
The `MetricsCalculator` class had grown to over 2000 lines, becoming a monolithic class that:
- Mixed concerns (base model, RAG, safety, performance, reliability, agent, security metrics)
- Made it difficult to test individual metric groups
- Had misleading metric names (e.g., "Top-K Accuracy" when it wasn't true Top-K)
- Loaded heavy dependencies (HuggingFace models) at initialization, causing slow startup
- Lacked clear separation between core metrics, proxy metrics, heuristics, and experimental metrics
- Made it hard to enable/disable specific metric groups

## Decision
We will refactor the monolithic `MetricsCalculator` into a modular architecture:

1. **Specialized Calculator Classes**: Break down into focused calculators:
   - `BaseModelMetricsCalculator` - Base model quality metrics
   - `RAGMetricsCalculator` - RAG pipeline metrics
   - `SafetyMetricsCalculator` - Safety and guardrail metrics
   - `PerformanceMetricsCalculator` - Performance and latency metrics
   - `ReliabilityMetricsCalculator` - Reliability and stability metrics
   - `AgentMetricsCalculator` - Agent and autonomous system metrics
   - `SecurityMetricsCalculator` - Security testing metrics

2. **Central Orchestration**: `MetricsEngine` class orchestrates all calculators and provides a single entry point

3. **Dependency Injection**: Heavy dependencies (like `ToxicityDetector`) are injected, enabling:
   - Lazy loading of heavy models
   - Configurable enabling/disabling of metric groups
   - Better testability

4. **Honest Metric Naming**: Rename misleading metrics:
   - `top_k_accuracy` → `normalized_similarity_score`
   - `factual_consistency` → `similarity_proxy_factual_consistency`
   - `truthfulness` → `similarity_proxy_truthfulness`
   - `source_grounding` → `similarity_proxy_source_grounding`

5. **Backward Compatibility**: Keep `MetricsCalculator` as a thin wrapper around `MetricsEngine` to maintain existing API

## Consequences
### Positive
- **Modularity**: Each calculator is focused and ~100-200 lines (vs 2000+ line monolith)
- **Testability**: Can test individual metric groups in isolation
- **Performance**: Lazy loading of heavy models (5-30+ seconds faster startup when safety disabled)
- **Configurability**: Can enable/disable metric groups via `MetricsEngine` flags
- **Maintainability**: Easier to understand, modify, and extend
- **Honest Metrics**: Clear naming prevents misunderstandings about what metrics actually measure
- **Better Error Handling**: Per-calculator error handling with graceful degradation
- **Dependency Injection**: Better testability and configurability

### Negative
- **More Files**: Code is now spread across multiple files (but more organized)
- **Initial Learning Curve**: Developers need to understand the new architecture
- **Migration**: Existing code using `MetricsCalculator` still works (backward compatible), but new code should use `MetricsEngine`

## Alternatives Considered
- **Keep Monolithic**: Rejected - too difficult to maintain and test
- **Plugin System**: Rejected - overkill for current needs, adds complexity
- **Strategy Pattern**: Considered but modular classes are simpler and clearer

## Implementation

### New Structure
```
src/dashboard/
├── metrics_calculator.py      # Backward-compatible wrapper
├── metrics_engine.py          # Central orchestration
├── metrics_spec.py            # Machine-readable metric specifications
├── metric_validator.py         # Metric validation
└── calculators/
    ├── __init__.py
    ├── base_model.py           # Base model metrics
    ├── rag.py                  # RAG metrics
    ├── safety.py               # Safety metrics
    ├── performance.py          # Performance metrics
    ├── reliability.py          # Reliability metrics
    ├── agent.py                # Agent metrics
    ├── security.py             # Security metrics
    └── detectors.py            # Heavy dependencies (ToxicityDetector)
```

### Key Design Decisions

1. **Lazy Loading**: `ToxicityDetector` loads HuggingFace models only when first used
   ```python
   def _load_model(self) -> bool:
       if self._loaded:
           return self._pipeline is not None
       # Load model on first use...
   ```

2. **Configurable Groups**: `MetricsEngine` allows enabling/disabling groups
   ```python
   engine = MetricsEngine(
       enable_safety=False,  # Skip safety metrics entirely
       enable_rag=False      # Skip RAG metrics
   )
   ```

3. **Honest Naming**: All proxy metrics clearly indicate they're similarity-based proxies
   - `similarity_proxy_factual_consistency` (not `factual_consistency`)
   - `similarity_proxy_truthfulness` (not `truthfulness`)
   - `normalized_similarity_score` (not `top_k_accuracy`)

4. **Graceful Degradation**: Missing dependencies don't crash the system
   - Falls back to `lexical_overlap` if NLTK unavailable (instead of fake BLEU)
   - Returns `None` for failed metrics instead of crashing

## Migration Guide

### For Existing Code
No changes required - `MetricsCalculator` still works:
```python
from dashboard.metrics_calculator import MetricsCalculator

calc = MetricsCalculator()  # Still works, uses MetricsEngine under the hood
metrics = calc.calculate_all_metrics(...)
```

### For New Code
Prefer using `MetricsEngine` directly for better configurability:
```python
from dashboard.metrics_engine import MetricsEngine

engine = MetricsEngine(enable_safety=False)
metrics = engine.calculate_all(...)
```

## Performance Impact

| Scenario | Before | After | Improvement |
|----------|--------|-------|-------------|
| Startup (safety disabled) | ~30-60s | ~1-5s | **85-95% faster** |
| Memory (safety disabled) | ~500MB+ | ~50-100MB | **80-90% reduction** |
| First calculation (safety enabled) | ~30-60s | ~30-60s | Same |
| Error recovery | May crash | Graceful | **100% more resilient** |

## References
- Implementation: `src/dashboard/metrics_engine.py`
- Calculators: `src/dashboard/calculators/`
- Metric Spec: `src/dashboard/metrics_spec.py`
- Documentation: `docs/METRICS_CALCULATIONS.md`


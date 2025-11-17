# ADR-0005: Implement Circuit Breaker Pattern

## Status
Accepted

## Context
External dependencies (ML models, APIs, browsers) can fail. We need to:
- Prevent cascading failures
- Gracefully degrade when services are unavailable
- Automatically recover when services come back online

## Decision
We will implement the Circuit Breaker pattern for all external dependencies.

## Consequences
### Positive
- Prevents resource exhaustion from repeated failures
- Automatic recovery when services restore
- Clear failure modes and error messages
- Configurable thresholds and timeouts

### Negative
- Additional complexity in code
- Need to handle circuit breaker states
- Potential for false positives (circuit opens on transient errors)

## Alternatives Considered
- **Simple retries only**: Doesn't prevent cascading failures
- **Rate limiting**: Different problem, doesn't handle service failures
- **Bulkhead pattern**: Good complement but doesn't replace circuit breaker

## Implementation
Circuit breaker is implemented in `src/core/circuit_breaker.py` with three states:
- CLOSED: Normal operation
- OPEN: Failing, reject requests
- HALF_OPEN: Testing recovery


---
name: audit
description: Tiger Team adversarial audit of a module or subsystem. Deep analysis for correctness bugs, safety holes, performance cliffs, and API misuse risks. Use before release or after major changes.
argument-hint: "[module or file path]"
context: fork
agent: Explore
---

Perform a Tiger Team adversarial audit of `$ARGUMENTS`. Read every line of the target. Analyze systematically across all five dimensions below. Be ruthless — assume the code has bugs and find them.

## 1. Correctness Audit

- **Off-by-one errors**: loop bounds, array indexing, fence-post problems
- **Numerical stability**: subtraction of similar magnitudes, log(0), exp overflow, division by zero
- **Edge cases**: empty inputs, single-element collections, zero temperature, max u64 steps
- **Invariant violations**: does every public function maintain its documented invariants?
- **RNG consumption order**: any change breaks determinism — verify proposal/accept/swap order is fixed

## 2. Safety Audit

- **Panic paths**: any `unwrap()`, `expect()`, `assert!()`, array indexing without bounds check in library code?
- **Integer overflow**: `u64` arithmetic on step counts, population sizes, histogram indices
- **Float pathology**: NaN propagation, Inf from exp(), negative temperature, negative energy differences
- **Memory**: unbounded allocations (Vec growth in hot loop), stack overflow from deep recursion

## 3. Performance Audit

- **Hot loop analysis**: identify the innermost loop. Is there any allocation, branch, or virtual dispatch in it?
- **Cache behavior**: are data structures accessed sequentially? Any pointer chasing in the hot path?
- **Branch prediction**: is the common case (rejection) the fall-through path?
- **Unnecessary work**: redundant energy computations, cloning where borrowing suffices, formatting in non-debug paths

## 4. API Misuse Audit

- **Pit of success**: can a user accidentally create an invalid configuration that compiles?
- **Error messages**: are `AnnealError` messages specific enough to diagnose the problem?
- **Type safety**: could type-state or phantom types prevent more misuse at compile time?
- **Documentation**: does every public item's doc comment accurately describe behavior, panics, and edge cases?

## 5. Test Coverage Audit

- **Missing coverage**: what code paths have no test exercising them?
- **Weak assertions**: tests that pass trivially or don't assert meaningful properties?
- **Flaky risk**: tests depending on statistical thresholds — are the margins sufficient?
- **Regression gaps**: if this module had a subtle bug, would the current tests catch it?

## Output format

```
## Audit Report: [module]

### Critical (must fix before release)
- [finding with file:line reference]

### High (should fix)
- [finding]

### Medium (consider fixing)
- [finding]

### Clean
- [aspects that passed audit with no issues]
```

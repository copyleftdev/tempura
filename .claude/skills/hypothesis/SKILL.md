---
name: hypothesis
description: Create a new hypothesis test suite (H-XX) with statistical validation. Use when adding a new testable scientific claim about annealing behavior.
argument-hint: "[H-number] [title]"
---

Create a new hypothesis-driven test suite following the Tempura convention. The argument should be like `H-11 convergence rate` or similar.

## Required artifacts

### 1. Hypothesis document: `hypotheses/H-$0-$1.md`

Follow this structure exactly:

```markdown
# H-XX: [Title]

## Hypothesis
[One sentence: what we claim is true]

## Theoretical basis
[Why we believe this — cite papers, theorems, or first-principles arguments]

## Test protocol
[Precise statistical test procedure]
- Landscape:
- Parameters:
- Sample size:
- Statistic:
- Pass criterion:

## Failure modes
[What would a failure mean? Bug, or wrong assumption?]
```

### 2. Test file: `tests/h{number}_{slug}.rs`

Follow the existing test file pattern exactly:

```rust
/// H-XX — [Title]
///
/// Validates that: [one-line summary]
///
/// Protocol (from hypotheses/H-XX-{slug}.md):
///   [brief protocol summary]
#[allow(dead_code)]
mod statistical;  // only if using shared statistical utilities

use tempura::...;  // minimal imports
```

Key conventions:
- Each test function is named `h{number}{letter}_{descriptive_name}` (e.g., `h11a_convergence_rate`)
- Every test has a `///` doc comment explaining: what it tests, the protocol, and the pass criterion
- Use deterministic seeds (0..num_seeds) for reproducibility
- Use `statistical::chi_squared_test` or `statistical::ks_two_sample` from the shared module when appropriate
- All builder calls must handle `Result` with `.unwrap()` in tests
- Landscape + Move objects are created at the top of each test, then `.clone()`d into loops

### 3. Update `hypotheses/README.md`

Add the new hypothesis to the table.

### 4. Update `hypotheses/VERIFICATION_PLAN.md`

Map the new hypothesis to its test file and landscape.

## Statistical rigor rules

- Chi-squared: merge bins with expected count < 5
- K-S test: use two-sample variant when no closed-form CDF
- Always thin correlated samples (thin >= integrated autocorrelation time)
- Multi-seed: test a pass-rate threshold, not a single seed
- Tolerance: choose thresholds that pass reliably in debug mode (not just release)

After creating all artifacts, run `/verify` to confirm everything compiles and passes.

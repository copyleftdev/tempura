---
name: new-schedule
description: Scaffold a new CoolingSchedule implementation. Use when adding a temperature schedule like Boltzmann, Reheat, Piecewise, etc.
argument-hint: "[name]"
---

Create a new cooling schedule following the Tempura trait pattern.

## File to modify: `src/schedule.rs`

Add the new schedule at the bottom of the file, following the exact section pattern:

```rust
// ---------------------------------------------------------------------------
// [Name]: T_k = [formula]
// ---------------------------------------------------------------------------

/// [Name] cooling: `T_k = [formula]`.
///
/// [One paragraph: when to use, convergence properties, trade-offs]
#[derive(Clone, Debug)]
pub struct [Name] {
    /// [doc comment for each field]
    pub field: f64,
}

impl [Name] {
    /// Create a [name] schedule: `T_k = [formula]`.
    pub fn new(...) -> Self {
        assert!([positive temperature invariant]);
        Self { ... }
    }
}

impl CoolingSchedule for [Name] {
    fn temperature(&self, step: u64) -> f64 {
        // [Implementation must return positive finite f64 for all step values]
        let t = [formula];
        t.max([minimum_floor])  // enforce positivity
    }

    // Override is_monotonic() only if the schedule can increase temperature
    // fn is_monotonic(&self) -> bool { false }
}
```

## Contract (from H-03, H-04)

The `CoolingSchedule` trait has these invariants that MUST hold:

1. `temperature(step) > 0.0` for all `step >= 0` — **always positive**
2. `temperature(step).is_finite()` — **never NaN or Inf**
3. Non-adaptive schedules must be monotonically non-increasing
4. Constructor uses `assert!` for programmer errors (these are invariants, not user input)

## Tests to add

Add tests in the `#[cfg(test)] mod tests` block at the bottom of `schedule.rs`:

- Positivity: `temperature(k) > 0` for k in 0..10_000
- Monotonicity: `temperature(k+1) <= temperature(k)` (if monotonic)
- Boundary: `temperature(0) == T0` (initial temperature correct)
- Large step: `temperature(10_000_000)` is positive and finite (no underflow)

## Prelude

If the schedule is a general-purpose one users will commonly reach for, add it to the prelude re-exports in `src/lib.rs`:

```rust
pub use crate::schedule::{..., [Name]};
```

After creating, run `/verify`.

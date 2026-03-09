---
name: new-algorithm
description: Scaffold a complete annealing algorithm module with engine, builder, diagnostics, and tests. Use for adding algorithms like Deterministic Annealing, Simulated Quenching, Basin Hopping, etc.
argument-hint: "[name]"
---

Create a new annealing algorithm following the Tempura architecture. This is the most complex scaffold — it produces a full module with builder, engine, diagnostics, and tests.

## Reference implementations

Study these files before generating code:

- `src/annealer.rs` — single-solution SA (simplest pattern)
- `src/parallel.rs` — parallel tempering (multi-replica pattern)
- `src/population.rs` — population annealing (population + resampling pattern)

## File: `src/$ARGUMENTS.rs`

### Module structure (in order)

```
1. Module-level doc comment (//! style or /// on the module in lib.rs)
2. Imports (crate:: only, no external deps)
3. Public free functions (if any, like geometric_ladder in parallel.rs)
4. Engine struct + impl (the algorithm runner)
5. Result/Diagnostics structs (algorithm-specific outputs)
6. Builder struct + impl (type-state builder pattern)
7. #[cfg(test)] mod tests
```

### Builder pattern (CRITICAL — follow exactly)

```rust
use crate::error::AnnealError;

/// Builder for [Name].
pub struct [Name]Builder<S, E, M> {
    objective: Option<E>,
    moves: Option<M>,
    // ... algorithm-specific config
    seed: u64,
    _state: std::marker::PhantomData<S>,
}

pub fn builder<S>() -> [Name]Builder<S, (), ()> { ... }
```

Builder rules:
- Config methods that validate input return `Result<Self, AnnealError>`
- Config methods that just store a value return `Self`
- `build()` returns `Result<[Name]<S, E, M>, AnnealError>`
- Missing required fields → `AnnealError::MissingField { field: "name" }`
- Invalid parameters → `AnnealError::InvalidParameter { name, reason }`
- The built engine has a `pub fn run(&mut self, initial_state: S) -> [Name]Result<S>` method

### RNG management

- Use `crate::rng::{Rng, Xoshiro256PlusPlus}` for all randomness
- Create RNG from seed via `Xoshiro256PlusPlus::from_seed(seed)`
- For multi-replica: derive per-replica seeds via `splitmix64(base_seed + i)`
- **Determinism invariant**: same seed = bit-identical output. No thread-local, no system RNG.

### Diagnostics

Create algorithm-specific diagnostic structs:
- Always include `acceptance_rate: f64`
- Include algorithm-specific metrics (swap rates, effective fraction, etc.)
- All diagnostics fields must have `///` doc comments

## Wire into `src/lib.rs`

```rust
/// [One-line description].
pub mod [name];
```

## Tests

Include at least 5 unit tests:

1. **Smoke test** — builds and runs without panic on a trivial landscape
2. **Determinism** — same seed produces identical results
3. **Improvement** — finds better energy than initial state on non-trivial landscape
4. **Builder validation** — missing fields and invalid params return correct errors
5. **Diagnostics** — output structs are populated with valid values

After creating, run `/verify`.

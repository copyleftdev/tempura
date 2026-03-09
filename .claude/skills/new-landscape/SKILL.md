---
name: new-landscape
description: Scaffold a new benchmark energy landscape module with Energy impl, MoveOperator, and analytical properties. Use when adding a test landscape for hypothesis validation.
argument-hint: "[name]"
---

Create a new energy landscape module following the Tempura pattern. The argument is the landscape name (e.g., `spin_glass`, `traveling_salesman`).

## Files to create/modify

### 1. `src/landscape/$ARGUMENTS.rs`

Follow the exact pattern from existing landscapes. Required structure:

```rust
/// [Title] — [one-line description of the landscape]
///
/// Used by: H-XX ([which hypothesis this validates])
///
/// States: [state space description]
/// Energy: [energy function formula]
/// Move:   [proposal mechanism description]
///
/// [Any analytical properties — partition function, Boltzmann distribution, phase transitions]
use crate::energy::Energy;
use crate::moves::MoveOperator;
use crate::rng::Rng;

/// [Energy struct doc]
#[derive(Clone, Debug)]
pub struct [Name] { ... }

impl [Name] {
    /// Create a new [name] landscape.
    pub fn new(...) -> Self { ... }
}

impl Energy<[StateType]> for [Name] {
    fn energy(&self, state: &[StateType]) -> f64 { ... }
}

/// [Move operator doc]
#[derive(Clone, Debug)]
pub struct [Name]Move {
    /// [doc for each public field]
    pub n: ...,
}

impl [Name]Move {
    /// Create a new [name] move operator.
    pub fn new(...) -> Self { ... }
}

impl MoveOperator<[StateType]> for [Name]Move {
    fn propose(&self, state: &[StateType], rng: &mut impl Rng) -> [StateType] { ... }
}
```

Key requirements:
- Every public item has `///` doc comments (enforced by `-D missing-docs`)
- `#[derive(Clone, Debug)]` on all structs
- Energy and Move structs are separate types
- Constructor uses `assert!` for programmer errors (not `Result`)
- If the landscape has an exact Boltzmann distribution, add `pub fn exact_boltzmann(&self, temperature: f64) -> Vec<f64>`

### 2. Update `src/landscape/mod.rs`

Add the new module with a doc comment mapping it to hypotheses:

```rust
/// [Description] — H-XX [hypothesis name].
pub mod [name];
```

### 3. Unit tests

Add at least 2 tests inside the module file:
- Energy correctness (known state → known energy)
- Move operator coverage (all proposals are valid states)

After creating, run `/verify` to confirm docs and tests pass.

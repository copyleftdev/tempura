# tempura

[![Crates.io](https://img.shields.io/crates/v/tempura.svg)](https://crates.io/crates/tempura)
[![docs.rs](https://docs.rs/tempura/badge.svg)](https://docs.rs/tempura)
[![CI](https://github.com/tempura-rs/tempura/actions/workflows/ci.yml/badge.svg)](https://github.com/tempura-rs/tempura/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![MSRV](https://img.shields.io/badge/MSRV-1.70-orange.svg)](https://blog.rust-lang.org/2023/06/01/Rust-1.70.0.html)

**Temperature-Driven Optimization Primitives for Rust.**

Tempura is a zero-dependency, production-grade annealing framework implementing
Simulated Annealing, Parallel Tempering, and Population Annealing. Deterministic
by construction — same seed, same result, every time.

---

## Features

- **Three algorithms** — SA, Parallel Tempering (replica exchange), Population Annealing
- **Six cooling schedules** — Linear, Exponential, Logarithmic (Hajek-convergent), Fast, Cauchy, Adaptive
- **Four move operators** — Gaussian, Swap, Neighbor, and the `MoveOperator` trait for custom moves
- **Two PRNGs** — Xoshiro256++ (default) and PCG-64; both period > 2¹²⁸
- **Type-state builders** — required fields enforced at compile time, no runtime panics
- **Branchless hot paths** — Metropolis, Barker, and log-domain acceptance; `fast_exp` (~5 cycles)
- **Rich diagnostics** — acceptance rate, improvement ratio, optional sampled trajectory
- **Zero runtime dependencies** — no `rand`, no `libc`, no `num-traits`
- **MSRV 1.70** — stable Rust only

---

## Quick Start

```toml
[dependencies]
tempura = "0.1"
```

```rust
use tempura::prelude::*;

// Minimize f(x) = (x - 3.7)²
let result = Annealer::builder::<f64>()
    .objective(|x: &f64| (*x - 3.7).powi(2))
    .move_op(GaussianMove::new(0.5))
    .schedule(Exponential::new(10.0, 0.995))
    .seed(42)
    .iterations(50_000)
    .build()
    .unwrap()
    .run(0.0);

println!("best x = {:.6}", result.best_state);   // ≈ 3.700000
println!("best E = {:.6}", result.best_energy);  // ≈ 0.000000
```

---

## Algorithms

| Algorithm | Type | When to use |
|---|---|---|
| `Annealer` | Single-solution SA | General-purpose, fast, low memory |
| `ParallelTempering` | Replica exchange | Rugged landscapes, barrier crossing |
| `PopulationAnnealing` | Ensemble resampling | Partition function estimation, broad exploration |

### Parallel Tempering

```rust
use tempura::prelude::*;
use tempura::parallel::PTBuilder;

let result = PTBuilder::<f64>::new()
    .temperatures(vec![0.01, 0.1, 0.5, 2.0])
    .objective(|x: &f64| (*x - 3.7).powi(2))
    .move_op(GaussianMove::new(0.5))
    .seed(42)
    .steps(10_000)
    .build()
    .unwrap()
    .run(0.0);
```

### Population Annealing

```rust
use tempura::prelude::*;
use tempura::population::PABuilder;

let result = PABuilder::<f64>::new()
    .population_size(200)
    .objective(|x: &f64| (*x - 3.7).powi(2))
    .move_op(GaussianMove::new(0.5))
    .schedule(Exponential::new(5.0, 0.99))
    .seed(42)
    .sweeps_per_step(10)
    .annealing_steps(500)
    .build()
    .unwrap()
    .run(0.0);
```

---

## Cooling Schedules

| Schedule | Formula | Convergence guarantee |
|---|---|---|
| `Linear` | T₀ − α·k | None |
| `Exponential` | T₀ · αᵏ | None (fast in practice) |
| `Logarithmic` | c / ln(1 + k) | **Yes** — Hajek (1988) |
| `Fast` | T₀ / (1 + k) | None |
| `Cauchy` | T₀ / (1 + k²) | None |
| `Adaptive` | Feedback-controlled | Empirical |

---

## Move Operators

| Operator | State type | Notes |
|---|---|---|
| `GaussianMove` | `f64` (or any float) | Box-Muller N(0, σ) perturbation |
| `SwapMove` | `Vec<usize>` | Uniform random transposition |
| `SwapMoveReversible` | `Vec<usize>` | Swap + undo for `ReversibleMove` |
| `NeighborMove` | `Vec<i64>` | Bounded ±1 walk with exact Hastings correction |

Implement `MoveOperator<S>` for any custom state type:

```rust
use tempura::prelude::*;

struct FlipBit;

impl MoveOperator<Vec<bool>> for FlipBit {
    fn perturb(&self, state: &Vec<bool>, rng: &mut impl Rng) -> Vec<bool> {
        let mut next = state.clone();
        let i = (rng.next_f64() * next.len() as f64) as usize;
        next[i] = !next[i];
        next
    }
}
```

---

## Diagnostics

```rust
let result = /* ... */;

println!("acceptance rate : {:.1}%", result.diagnostics.acceptance_rate() * 100.0);
println!("improvement     : {:.1}%", result.diagnostics.improvement_ratio() * 100.0);
println!("best energy     : {:.6}", result.best_energy);
println!("final energy    : {:.6}", result.final_energy);

// Optional trajectory (enable with .trajectory_interval(n))
if let Some(traj) = &result.trajectory {
    println!("{} trajectory points recorded", traj.len());
}
```

---

## Performance

Hot-path primitives (measured on x86-64, single core):

| Primitive | Throughput |
|---|---|
| `Xoshiro256PlusPlus::next_f64` | ~1 ns |
| `metropolis_accept` | ~2 ns (branchless) |
| `metropolis_accept_log_domain` | ~3 ns (exp-free) |
| `fast_exp` | ~5 cycles, 0.1% relative error |
| Full SA step (f64 + Gaussian + Metropolis) | ~15 ns |

Run the benchmarks:

```bash
cargo bench --bench throughput
cargo bench --bench h09_cache_layout
cargo bench --bench h10_branchless
```

---

## Test Suites

Ten hypothesis-driven statistical test suites covering 37 tests:

| Suite | Hypothesis | Tests |
|---|---|---|
| H-01 | Boltzmann distribution convergence, ergodicity | 4 |
| H-02 | Detailed balance, Metropolis-Hastings, Barker | 4 |
| H-03 | Cooling schedule ordering, log-schedule superiority | 5 |
| H-04 | Acceptance rate monotonicity, adaptive stability | 4 |
| H-05 | Parallel tempering swap rates, barrier crossing | 4 |
| H-06 | Population annealing effective fraction | 3 |
| H-07 | Quantum tunneling probability scaling | 3 |
| H-08 | RNG independence, period, statistical quality | 4 |
| H-09 | Cache layout, SoA vs AoA, determinism | 4 |
| H-10 | Branchless correctness, fast_exp accuracy | 6 |

```bash
cargo test                              # all 113+ tests
cargo test --test h01_boltzmann         # single suite
cargo test -- --test-threads=1          # sequential (timing-sensitive suites)
```

---

## Examples

Three showcase example files demonstrating real-world SA applications:

```bash
cargo run --example industry_showcase   # logistics, VLSI, finance, energy, ...
cargo run --example cyber_showcase      # IDS, firewall, honeypot, S-box, ...
cargo run --example algotrading_showcase # MVO, Kelly, Almgren-Chriss, CVaR, ...
```

---

## Safety

- `#![forbid(unsafe_code)]` — zero unsafe in the library
- No `unwrap()` in library code; all error paths return `Result<_, AnnealError>`
- `AnnealError` has two variants: `MissingField { field }` and `InvalidParameter { name, reason }`

---

## Minimum Supported Rust Version

**1.70 (stable)**. Uses only stable language features. MSRV bumps are treated as breaking changes and gated behind a minor version bump.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow, verification gate, and pull request process.

---

## Security

See [SECURITY.md](SECURITY.md) for the vulnerability reporting policy.

---

## License

Licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.

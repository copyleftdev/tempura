# Tempura — Claude Code Project Context

High-performance annealing framework in Rust. Zero runtime dependencies.
Algorithms: SA, Parallel Tempering, Population Annealing, Quantum-Inspired.

## Commands

```bash
# Build
cargo build
cargo build --release

# Test (113 unit + integration + 2 doc-tests)
cargo test
cargo test -- --test-threads=1          # sequential (useful for timing tests)
cargo test -p tempura --lib             # library unit tests only
cargo test --test h01_boltzmann         # single hypothesis suite

# Docs (strict — the project enforces zero warnings + zero missing docs)
RUSTDOCFLAGS="-D warnings -D missing-docs" cargo doc --no-deps

# Lint
cargo clippy -- -D warnings
cargo fmt --check

# Bench
cargo bench --bench throughput
cargo bench --bench h09_cache_layout
cargo bench --bench h10_branchless
```

## Architecture

```text
src/
  lib.rs          — module wiring + prelude (re-exports common types)
  annealer.rs     — single-solution SA engine, builder pattern
  parallel.rs     — parallel tempering (replica exchange), PTBuilder
  population.rs   — population annealing, PABuilder
  energy.rs       — Energy trait, FnEnergy closure wrapper, DeltaEnergy
  moves.rs        — MoveOperator trait, GaussianMove, SwapMove, NeighborMove
  schedule.rs     — CoolingSchedule trait + 6 schedules (Linear/Exp/Log/Fast/Cauchy/Adaptive)
  math.rs         — branchless Metropolis, Barker, log-domain, fast_exp, quantum tunneling
  rng.rs          — Xoshiro256++, PCG-64, splitmix64, Rng trait
  diagnostics.rs  — AnnealResult, RunDiagnostics, TrajectoryRecorder
  state.rs        — State trait (blanket impl for Clone+Debug)
  error.rs        — AnnealError enum (MissingField, InvalidParameter)
  landscape/      — test energy landscapes (PotentialWell, DoubleWell, Ising2D, Rastrigin, TunableBarrier)
tests/
  h01–h10         — hypothesis-driven statistical tests (chi-squared, K-S, flow balance, etc.)
  statistical.rs  — shared test utilities (chi-squared, K-S, autocorrelation)
hypotheses/       — markdown docs for each hypothesis + VERIFICATION_PLAN.md
benches/          — criterion benchmarks (throughput, cache layout, branchless)
```

## Code Style & Conventions

- **Zero runtime deps.** Do not add dependencies to `[dependencies]`. Dev-deps are fine.
- **Builder pattern** with type-state for pit-of-success API. Builders return `Result<T, AnnealError>`.
- **Validation methods** (`temperatures()`, `population_size()`, etc.) return `Result<Self, AnnealError>`.
- **`assert!`** stays in schedule/move constructors (programmer errors). Builders use `Result`.
- **All public items must have `///` doc comments.** Enforced by `RUSTDOCFLAGS="-D missing-docs"`.
- **Hot/cold path splitting** in inner loops (Muratori style). Diagnostics are cold path.
- **Deterministic** — same seed = bit-identical output. All RNG is explicit, no thread-local state.
- Imports at top of file. No `use` statements mid-function.
- No `unwrap()` in library code. Tests may use `unwrap()`.

## Key Design Decisions

- `Energy<S>` trait is the core abstraction. `FnEnergy` wraps closures for ergonomics.
- `MoveOperator<S>` proposes state transitions. `ReversibleMove` adds Hastings correction.
- Cooling schedules implement `CoolingSchedule` trait with `temperature(step, total) -> f64`.
- `Rng` trait abstracts PRNGs. Default is `Xoshiro256PlusPlus`. Seeds scrambled via splitmix64.
- `AnnealError` has two variants: `MissingField { field }` and `InvalidParameter { name, reason }`.
- Prelude re-exports the "80% use case" types. Advanced users import modules directly.

## Git Ops (strict)

Before ANY `gh` CLI interaction (push, pr, issue, release, etc.), you MUST first run:

```bash
gh auth status
```

Verify the output shows `Logged in to github.com account copyleftdev`. If it does NOT show `copyleftdev`, STOP and ask the user to authenticate. Do not proceed with any GitHub operation until auth is confirmed.

Workflow:

1. `gh auth status` — confirm `copyleftdev` is authenticated
2. Then proceed with the `gh` command

This applies to: `gh pr`, `gh issue`, `gh release`, `gh repo`, `git push`, `git pull` — anything that touches the remote.

## Anti-Patterns (Do NOT)

- Do not add runtime dependencies without explicit approval.
- Do not refactor unrelated code in the same change.
- Do not delete or weaken existing tests.
- Do not swallow errors — every failure path must be explicit.
- Do not use `println!` for diagnostics — use the `RunDiagnostics` / `TrajectoryRecorder` system.
- Do not break determinism — any change to RNG consumption order requires updating seed-dependent tests.

## Skills (slash commands)

- `/verify` — full verification gate (test + doc + clippy)
- `/hypothesis H-11 title` — scaffold a new hypothesis test suite
- `/new-landscape name` — scaffold a new energy landscape module
- `/new-schedule name` — scaffold a new CoolingSchedule impl
- `/new-algorithm name` — scaffold a complete algorithm module (builder + engine + tests)
- `/audit src/annealer.rs` — Tiger Team adversarial deep audit
- `/perf src/math.rs` — Muratori-style performance analysis

## Verification Checklist

After any change, verify:

1. `cargo test` — all pass
2. `RUSTDOCFLAGS="-D warnings -D missing-docs" cargo doc --no-deps` — clean
3. `cargo clippy -- -D warnings` — clean

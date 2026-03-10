# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.1.0] — 2026-03-09

### Added

#### Algorithms
- **Simulated Annealing** (`annealer`) — single-solution SA engine with type-state builder pattern. Deterministic, bit-reproducible from seed. Hot/cold path splitting for maximum throughput.
- **Parallel Tempering** (`parallel`) — replica exchange across a geometric temperature ladder. Enables barrier crossing that single-chain SA cannot escape.
- **Population Annealing** (`population`) — Boltzmann-weighted population resampling. Provides unbiased partition function estimation and scales to large populations.

#### Core Traits
- `Energy<S>` — pure, deterministic objective function over any state type `S`.
- `FnEnergy<F>` — closure adapter implementing `Energy<S>` for ergonomic inline objectives.
- `DeltaEnergy<S, M>` — optional O(1) incremental energy computation for large state spaces.
- `MoveOperator<S>` — proposal distribution with symmetric / Metropolis-Hastings correction support.
- `ReversibleMove<S>` — in-place move with undo, avoiding full state clones in hot loops.
- `CoolingSchedule` — temperature-at-step function with monotonicity annotation.
- `Rng` — deterministic PRNG trait with 53-bit `f64` and Exp(1) variates.
- `State` — blanket impl for `Clone + Debug` (zero boilerplate for users).

#### Cooling Schedules (6)
| Schedule | Formula | Convergence |
|---|---|---|
| `Linear` | T₀ − α·k | None |
| `Exponential` | T₀ · αᵏ | None (fast in practice) |
| `Logarithmic` | c / ln(1+k) | **Guaranteed** (Hajek 1988) |
| `Fast` | T₀ / (1+k) | None |
| `Cauchy` | T₀ / (1+k²) | None |
| `Adaptive` | feedback-controlled | Empirical |

#### Move Operators (4)
- `GaussianMove` — continuous perturbation via Box-Muller N(0, σ).
- `SwapMove` / `SwapMoveReversible` — permutation swap for discrete combinatorial problems.
- `NeighborMove` — bounded ±1 walk for 1-D integer spaces with exact Hastings correction.

#### RNGs (2)
- `Xoshiro256PlusPlus` — default; period 2²⁵⁶, passes BigCrush/PractRand, ~1 ns/draw.
- `Pcg64` — PCG-XSL-RR-128/64; period 2¹²⁸, alternative failure modes for validation.
- `splitmix64` — seed scrambler ensuring uncorrelated initial states from sequential seeds.

#### Numerical Primitives (`math`)
- `metropolis_accept` — branchless acceptance (H-10).
- `metropolis_accept_log_domain` — exp-free acceptance via Exp(1) variates.
- `barker_accept` — Barker criterion satisfying detailed balance with lower acceptance rate.
- `fast_exp` — Schraudolph's trick: ~5 cycles, 0.1% relative error.
- `quantum_tunneling_accept` — quantum-inspired barrier penetration (sqrt scaling).

#### Diagnostics
- `AnnealResult<S>` — best state/energy, final state/energy, diagnostics, optional trajectory.
- `RunDiagnostics` — zero-allocation acceptance rate, improvement ratio.
- `TrajectoryRecorder` — opt-in sampled energy/temperature/acceptance history with windowed acceptance rate.

#### Landscapes (5, for testing and benchmarking)
- `PotentialWell` — 1-D quadratic bowl.
- `DoubleWell` — bimodal with tunable barrier height.
- `Ising2D` — 2-D Ising model on an L×L grid.
- `Rastrigin` — N-dimensional multimodal with known global minimum.
- `TunableBarrier` — parameterised barrier depth for hypothesis testing.

#### Test Suites (10 hypotheses)
| Suite | Tests | Coverage |
|---|---|---|
| H-01 | 4 | Boltzmann distribution convergence, ergodicity |
| H-02 | 4 | Detailed balance, Metropolis-Hastings, Barker |
| H-03 | 5 | Cooling schedule ordering, log-schedule superiority |
| H-04 | 4 | Acceptance rate monotonicity, adaptive stability |
| H-05 | 4 | Parallel tempering swap rates, barrier crossing |
| H-06 | 3 | Population annealing effective fraction |
| H-07 | 3 | Quantum tunneling probability scaling |
| H-08 | 4 | RNG independence, period, statistical quality |
| H-09 | 4 | Cache layout, SoA vs AoA, determinism |
| H-10 | 6 | Branchless correctness, fast_exp accuracy |

#### Examples
- `industry_showcase` — 10 industry applications (logistics, semiconductor, finance, energy, telecom, bioinformatics, manufacturing, aerospace, healthcare, ML).
- `cyber_showcase` — 10 cybersecurity applications (IDS tuning, firewall ordering, honeypot placement, patch scheduling, SIEM correlation, segmentation, RBAC, S-box design, anomaly detection, IR allocation).
- `algotrading_showcase` — 10 algorithmic trading applications (MV optimisation, technical strategy, order execution, pairs trading, risk parity, regime switching, Kelly sizing, market making, momentum combination, CVaR portfolio).

### Design decisions
- **Zero runtime dependencies.** All numerical code is self-contained; no `libc`, `rand`, or numeric library required at runtime.
- **Deterministic by construction.** RNG is explicit and seeded; identical seed + inputs → bit-identical output.
- **Type-state builders.** Required fields are enforced at compile time; no runtime `Option::unwrap` in the builder.
- **MSRV 1.70.** Uses stable Rust only; no nightly features.

[Unreleased]: https://github.com/tempura-rs/tempura/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tempura-rs/tempura/releases/tag/v0.1.0

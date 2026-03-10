# Contributing to Tempura

Thank you for contributing! This document covers the development workflow,
coding standards, and pull request process.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Verification Gate](#verification-gate)
- [Code Style](#code-style)
- [Adding Algorithms, Schedules, and Landscapes](#adding-algorithms-schedules-and-landscapes)
- [Hypothesis Tests](#hypothesis-tests)
- [Pull Request Process](#pull-request-process)
- [Anti-Patterns](#anti-patterns)

---

## Getting Started

```bash
git clone https://github.com/tempura-rs/tempura
cd tempura
cargo build
cargo test
```

No special tooling is required beyond a stable Rust toolchain ≥ 1.70.

For linting and doc checks, install the optional tools:

```bash
# Linting (required for CI)
rustup component add clippy rustfmt

# Dependency auditing (optional locally, required in CI)
cargo install cargo-deny
cargo install cargo-audit
```

---

## Development Workflow

1. **Fork** the repository and create a feature branch from `main`.
2. **Make your changes.** Keep commits focused — one logical change per commit.
3. **Run the verification gate** (see below) until it is fully clean.
4. **Open a pull request** against `main` with a clear description.

---

## Verification Gate

Every change must pass all three checks before merging:

```bash
# 1. Tests — all must pass
cargo test

# 2. Documentation — zero warnings, zero missing docs
RUSTDOCFLAGS="-D warnings -D missing-docs" cargo doc --no-deps

# 3. Lints — zero warnings
cargo clippy -- -D warnings
```

Or use the alias defined in `.cargo/config.toml`:

```bash
cargo verify    # alias for: clippy --all-targets -- -D warnings
cargo doc-check # alias for: doc --no-deps
```

CI runs this gate on ubuntu/macos/windows × stable/beta/MSRV (1.70).
A PR cannot merge if any CI job is red.

---

## Code Style

Formatting is enforced by `rustfmt` with the settings in `rustfmt.toml`:

```bash
cargo fmt           # format
cargo fmt --check   # check (what CI does)
```

Key conventions:

- **Zero runtime dependencies.** Do not add crates to `[dependencies]`.
  Dev-dependencies for tests and benchmarks are acceptable.
- **All public items must have `///` doc comments.** This is enforced by
  `#![deny(missing_docs)]` and `RUSTDOCFLAGS="-D missing-docs"`.
- **No `unwrap()` in library code.** Use `Result` and propagate errors.
  Tests and examples may use `unwrap()`.
- **Builders use `Result<Self, AnnealError>`**, not panics, for validation.
  Constructor-level invariants (programmer errors) may use `assert!`.
- **Determinism is a hard invariant.** Any change that alters RNG consumption
  order must update the seed-dependent assertions in the affected tests.
- **Hot/cold path splitting.** Diagnostics and trajectory recording must not
  appear in the inner acceptance loop.
- No `println!` in library code. Diagnostics go through `RunDiagnostics`
  and `TrajectoryRecorder`.
- Imports at the top of each file. No `use` statements inside functions.

---

## Adding Algorithms, Schedules, and Landscapes

Use the provided scaffolding skills (Claude Code slash commands):

| Command | Effect |
|---|---|
| `/new-algorithm name` | Full algorithm module with builder, engine, diagnostics, tests |
| `/new-schedule name` | `CoolingSchedule` impl with doc, validation, and unit test |
| `/new-landscape name` | Energy landscape with `Energy` impl, move operator, and properties |
| `/hypothesis H-NN title` | New hypothesis test suite under `tests/` |

These ensure all boilerplate (doc comments, `Debug` impls, re-exports in
`lib.rs`) is present before you write any logic.

---

## Hypothesis Tests

Statistical tests live in `tests/h01_boltzmann.rs` … `tests/h10_branchless.rs`.
They follow a hypothesis-driven structure:

```rust
/// H-XX-NN: <brief claim>
#[test]
fn h_xx_nn_claim_name() {
    // Arrange: set up the landscape and annealer
    // Act: run the annealer
    // Assert: statistical test (chi-squared, K-S, etc.)
}
```

Shared statistical utilities (chi-squared, K-S, autocorrelation) are in
`tests/statistical.rs`. Do not duplicate them.

New hypothesis suites must document the hypothesis in `hypotheses/H-NN-name.md`
following the format in existing files.

---

## Pull Request Process

1. **Title format:** `type: short description` where type is one of:
   - `feat` — new feature or algorithm
   - `fix` — bug fix
   - `perf` — performance improvement
   - `refactor` — internal restructuring, no behavior change
   - `test` — new or improved tests
   - `docs` — documentation only
   - `chore` — CI, tooling, dependencies

2. **Description:** Explain *why* the change is needed, not just *what* it does.
   For numerical changes, include the hypothesis or reference that validates correctness.

3. **Changelog:** Add an entry under `## [Unreleased]` in `CHANGELOG.md`.

4. **Breaking changes:** If a public API changes, note it prominently and bump
   the minor version in `Cargo.toml`.

5. **Scope:** One PR, one concern. Do not bundle unrelated fixes.

---

## Anti-Patterns

These will cause a PR to be rejected:

- Adding runtime dependencies without explicit discussion
- Refactoring code unrelated to the stated purpose of the PR
- Deleting or weakening existing tests
- Swallowing errors (returning `Ok(())` from a failure path)
- Breaking determinism without updating affected tests
- Using `println!` for diagnostics in library code
- Public items without `///` doc comments

---

## License

By contributing, you agree that your contributions will be dual-licensed under
MIT and Apache-2.0, the same as the project.

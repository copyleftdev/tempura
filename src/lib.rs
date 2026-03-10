// Crate-level lint policy (fine-grained overrides live in each module).
// Deny-level lints are compile errors; warn-level appear in `cargo clippy`.
// These mirror the [lints] table in Cargo.toml but apply to doc-tests too.
#![deny(missing_docs, missing_debug_implementations)]
#![warn(unused_qualifications, future_incompatible)]
#![warn(clippy::pedantic, clippy::nursery)]
#![allow(
    // Common false-positives in a numerical library
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_panics_doc,
    clippy::similar_names,
)]

//! # Tempura — Temperature-Driven Optimization Primitives for Rust
//!
//! Tempura is a high-performance annealing framework providing composable
//! primitives for temperature-based stochastic optimization.
//!
//! ## Quick Start
//!
//! ```rust
//! use tempura::prelude::*;
//!
//! fn main() -> Result<(), AnnealError> {
//!     let result = Annealer::builder()
//!         .objective(FnEnergy(|x: &Vec<f64>| x.iter().map(|v| v * v).sum()))
//!         .moves(GaussianMove::new(0.5))
//!         .schedule(Exponential::new(100.0, 0.9999))
//!         .iterations(100_000)
//!         .seed(42)
//!         .build()?
//!         .run(vec![5.0, -3.0, 7.0]);
//!
//!     println!("Best energy: {}", result.best_energy);
//!     Ok(())
//! }
//! ```

/// Single-solution simulated annealing engine with builder pattern.
pub mod annealer;
/// Error types for configuration and runtime failures.
pub mod error;
/// Run diagnostics, trajectory recording, and result types.
pub mod diagnostics;
/// Energy (cost function) trait and helpers.
pub mod energy;
/// Benchmark landscapes for testing and validation.
pub mod landscape;
/// Numerical primitives: acceptance functions, fast exp, quantum tunneling.
pub mod math;
/// Move operators: Gaussian perturbation, neighbor swap, and custom moves.
pub mod moves;
/// Parallel tempering (replica exchange) algorithm.
pub mod parallel;
/// Population annealing with Boltzmann-weighted resampling.
pub mod population;
/// Deterministic pseudo-random number generators.
pub mod rng;
/// Cooling schedules: linear, exponential, logarithmic, adaptive, and more.
pub mod schedule;
/// State trait (blanket impl for `Clone + Debug`).
pub mod state;

/// Convenience re-exports for the most common types.
pub mod prelude {
    pub use crate::annealer::builder as annealer_builder;
    pub use crate::diagnostics::{AnnealResult, RunDiagnostics};
    pub use crate::energy::{Energy, FnEnergy};
    pub use crate::error::AnnealError;
    pub use crate::moves::{GaussianMove, MoveOperator, SwapMove};
    pub use crate::rng::{DefaultRng, Rng, Xoshiro256PlusPlus};
    pub use crate::schedule::{
        Adaptive, Cauchy, CoolingSchedule, Exponential, Fast, Linear, Logarithmic,
    };

    /// Entry point for building an annealer.
    #[derive(Debug, Clone, Copy)]
    pub struct Annealer;

    impl Annealer {
        /// Create a new annealer builder.
        pub const fn builder<S>() -> crate::annealer::AnnealerBuilder<S, (), (), ()> {
            crate::annealer::builder()
        }
    }
}

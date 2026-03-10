//! Benchmark energy landscapes with known analytical properties.
//!
//! Each landscape serves as a test fixture for specific hypotheses:
//! - `potential_well`: 1D quadratic well — H-01, H-02 (Boltzmann convergence)
//! - `double_well`: Asymmetric double well — H-03, H-05 (barrier crossing)
//! - `ising`: 2D Ising model — H-01, H-06 (exact partition function)
//! - `rastrigin`: N-dimensional multi-modal — H-04, H-09 (acceptance rate, cache)
//! - `barrier`: Tunable barrier geometry — H-07 (quantum tunneling)

/// 1D quadratic potential well — H-01, H-02 validation.
pub mod potential_well;
/// Asymmetric double well — H-03, H-05 barrier crossing.
pub mod double_well;
/// 2D Ising model with exact partition function — H-01, H-06.
pub mod ising;
/// N-dimensional Rastrigin function — H-04, H-08, H-09.
pub mod rastrigin;
/// Tunable barrier geometry — H-07 quantum tunneling.
pub mod barrier;

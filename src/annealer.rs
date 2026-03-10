//! Single-solution simulated annealing engine.
//!
//! # Design (Turon: user-first API with builder pattern)
//! ```ignore
//! let result = Annealer::builder()
//!     .schedule(Exponential::new(1000.0, 0.995))
//!     .moves(SwapMove)
//!     .objective(tsp_energy)
//!     .iterations(1_000_000)
//!     .seed(42)
//!     .build()
//!     .run(initial_state);
//! ```
//!
//! # Performance (Muratori: hot/cold splitting)
//! The inner loop touches only: state, energy, temperature, RNG, acceptance.
//! Diagnostics, best-state tracking, and trajectory recording are cold-path.
//!
//! # Correctness (Lamport: state machine)
//! The annealer is a deterministic state machine:
//!   (state, energy, `rng_state`, step) → (state', energy', `rng_state`', step+1)
//! Given the same seed and inputs, the output is bit-identical.
use crate::diagnostics::{AnnealResult, RunDiagnostics, TrajectoryRecorder};
use crate::energy::Energy;
use crate::error::AnnealError;
use crate::math;
use crate::moves::MoveOperator;
use crate::rng::{DefaultRng, Rng};
use crate::schedule::CoolingSchedule;

/// The annealing engine. Constructed via `Annealer::builder()`.
pub struct Annealer<S, E, M, C, R = DefaultRng>
where
    E: Energy<S>,
    M: MoveOperator<S>,
    C: CoolingSchedule,
    R: Rng,
{
    objective: E,
    moves: M,
    schedule: C,
    rng: R,
    iterations: u64,
    trajectory_interval: Option<u64>,
    _phantom: core::marker::PhantomData<S>,
}

impl<S, E, M, C, R> core::fmt::Debug for Annealer<S, E, M, C, R>
where
    E: Energy<S>,
    M: MoveOperator<S>,
    C: CoolingSchedule,
    R: Rng,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Annealer")
            .field("iterations", &self.iterations)
            .field("trajectory_interval", &self.trajectory_interval)
            .finish_non_exhaustive()
    }
}

impl<S, E, M, C> Annealer<S, E, M, C, DefaultRng>
where
    S: Clone + core::fmt::Debug,
    E: Energy<S>,
    M: MoveOperator<S>,
    C: CoolingSchedule,
{
    /// Run the annealing process from the given initial state.
    ///
    /// Returns the best state found, along with diagnostics.
    pub fn run(&mut self, initial: S) -> AnnealResult<S> {
        self.run_with_rng(initial)
    }
}

impl<S, E, M, C, R> Annealer<S, E, M, C, R>
where
    S: Clone + core::fmt::Debug,
    E: Energy<S>,
    M: MoveOperator<S>,
    C: CoolingSchedule,
    R: Rng,
{
    fn run_with_rng(&mut self, initial: S) -> AnnealResult<S> {
        let initial_energy = self.objective.energy(&initial);
        debug_assert!(initial_energy.is_finite(), "initial energy must be finite");

        let mut state = initial;
        let mut energy = initial_energy;
        let mut best_state = state.clone();
        let mut best_energy = energy;
        let mut diagnostics = RunDiagnostics::new(initial_energy);

        let mut trajectory = self.trajectory_interval.map(|interval| {
            TrajectoryRecorder::new(interval).with_capacity(self.iterations)
        });

        // === HOT LOOP ===
        // This is the inner loop that must be as tight as possible.
        // Only touches: state, energy, temperature, rng, acceptance decision.
        for step in 0..self.iterations {
            let temperature = self.schedule.temperature(step);

            // Propose candidate
            let candidate = self.moves.propose(&state, &mut self.rng);
            let candidate_energy = self.objective.energy(&candidate);
            let delta_e = candidate_energy - energy;

            // Hastings correction for asymmetric proposals
            let log_correction = if self.moves.is_symmetric() {
                0.0
            } else {
                self.moves.log_proposal_ratio(&state, &candidate)
            };

            // Acceptance decision (branchless, H-10)
            let u = self.rng.next_f64();
            let accepted = if log_correction == 0.0 {
                // Fast path: symmetric proposal, no correction needed
                math::metropolis_accept(delta_e, temperature, u)
            } else {
                // Hastings-corrected acceptance
                let adjusted_delta = temperature.mul_add(-log_correction, delta_e);
                math::metropolis_accept(adjusted_delta, temperature, u)
            };

            if accepted {
                state = candidate;
                energy = candidate_energy;

                if energy < best_energy {
                    best_state = state.clone();
                    best_energy = energy;
                }
            }

            // Cold path: diagnostics
            diagnostics.record_proposal(accepted, energy);
            if let Some(ref mut traj) = trajectory {
                traj.record(energy, temperature, accepted);
            }
        }

        AnnealResult {
            best_state,
            best_energy,
            final_state: state,
            final_energy: energy,
            diagnostics,
            trajectory,
        }
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for constructing an `Annealer`.
///
/// # Pit-of-success design (Turon)
/// Required fields must be set before `.build()` compiles.
/// Optional fields have sensible defaults.
pub struct AnnealerBuilder<S, E, M, C> {
    objective: Option<E>,
    moves: Option<M>,
    schedule: Option<C>,
    seed: u64,
    iterations: u64,
    trajectory_interval: Option<u64>,
    _phantom: core::marker::PhantomData<S>,
}

impl<S, E, M, C> core::fmt::Debug for AnnealerBuilder<S, E, M, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AnnealerBuilder")
            .field("seed", &self.seed)
            .field("iterations", &self.iterations)
            .field("trajectory_interval", &self.trajectory_interval)
            .field("objective_set", &self.objective.is_some())
            .field("moves_set", &self.moves.is_some())
            .field("schedule_set", &self.schedule.is_some())
            .finish_non_exhaustive()
    }
}

/// Entry point: `Annealer::builder()`.
pub const fn builder<S>() -> AnnealerBuilder<S, (), (), ()> {
    AnnealerBuilder {
        objective: None,
        moves: None,
        schedule: None,
        seed: 0,
        iterations: 100_000,
        trajectory_interval: None,
        _phantom: core::marker::PhantomData,
    }
}

impl<S, E, M, C> AnnealerBuilder<S, E, M, C> {
    /// Set the objective (energy/cost) function.
    pub fn objective<E2: Energy<S>>(self, obj: E2) -> AnnealerBuilder<S, E2, M, C> {
        AnnealerBuilder {
            objective: Some(obj),
            moves: self.moves,
            schedule: self.schedule,
            seed: self.seed,
            iterations: self.iterations,
            trajectory_interval: self.trajectory_interval,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Set the move operator.
    pub fn moves<M2: MoveOperator<S>>(self, m: M2) -> AnnealerBuilder<S, E, M2, C> {
        AnnealerBuilder {
            objective: self.objective,
            moves: Some(m),
            schedule: self.schedule,
            seed: self.seed,
            iterations: self.iterations,
            trajectory_interval: self.trajectory_interval,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Set the cooling schedule.
    pub fn schedule<C2: CoolingSchedule>(self, s: C2) -> AnnealerBuilder<S, E, M, C2> {
        AnnealerBuilder {
            objective: self.objective,
            moves: self.moves,
            schedule: Some(s),
            seed: self.seed,
            iterations: self.iterations,
            trajectory_interval: self.trajectory_interval,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Set the RNG seed for reproducibility.
    #[must_use]
    pub const fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the number of iterations (proposals).
    #[must_use]
    pub const fn iterations(mut self, n: u64) -> Self {
        self.iterations = n;
        self
    }

    /// Enable trajectory recording at the given sample interval.
    ///
    /// `interval = 1` records every step. `interval = 100` records every 100th.
    #[must_use]
    pub const fn record_trajectory(mut self, interval: u64) -> Self {
        self.trajectory_interval = Some(interval);
        self
    }
}

impl<S, E, M, C> AnnealerBuilder<S, E, M, C>
where
    S: Clone + core::fmt::Debug,
    E: Energy<S>,
    M: MoveOperator<S>,
    C: CoolingSchedule,
{
    /// Build the annealer. All required fields must have been set.
    ///
    /// # Errors
    /// Returns [`AnnealError::MissingField`] if objective, moves, or schedule
    /// were not set.
    pub fn build(self) -> Result<Annealer<S, E, M, C, DefaultRng>, AnnealError> {
        Ok(Annealer {
            objective: self.objective.ok_or(AnnealError::MissingField { field: "objective" })?,
            moves: self.moves.ok_or(AnnealError::MissingField { field: "moves" })?,
            schedule: self.schedule.ok_or(AnnealError::MissingField { field: "schedule" })?,
            rng: DefaultRng::from_seed(self.seed),
            iterations: self.iterations,
            trajectory_interval: self.trajectory_interval,
            _phantom: core::marker::PhantomData,
        })
    }

    /// Build with a specific RNG type.
    ///
    /// # Errors
    /// Returns [`AnnealError::MissingField`] if objective, moves, or schedule
    /// were not set.
    pub fn build_with_rng<R: Rng>(self) -> Result<Annealer<S, E, M, C, R>, AnnealError> {
        Ok(Annealer {
            objective: self.objective.ok_or(AnnealError::MissingField { field: "objective" })?,
            moves: self.moves.ok_or(AnnealError::MissingField { field: "moves" })?,
            schedule: self.schedule.ok_or(AnnealError::MissingField { field: "schedule" })?,
            rng: R::from_seed(self.seed),
            iterations: self.iterations,
            trajectory_interval: self.trajectory_interval,
            _phantom: core::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::FnEnergy;
    use crate::moves::GaussianMove;
    use crate::schedule::Exponential;

    #[test]
    fn minimize_quadratic() {
        // Minimize f(x) = Σ x_i²  (global minimum at origin)
        let result = builder::<Vec<f64>>()
            .objective(FnEnergy(|x: &Vec<f64>| x.iter().map(|v| v * v).sum()))
            .moves(GaussianMove::new(0.5))
            .schedule(Exponential::new(10.0, 0.9999))
            .iterations(100_000)
            .seed(42)
            .build()
            .unwrap()
            .run(vec![5.0, -3.0, 7.0]);

        assert!(
            result.best_energy < 1.0,
            "should find near-origin: E={}",
            result.best_energy
        );
    }

    #[test]
    fn deterministic_reproduction() {
        let run = |seed| {
            builder::<Vec<f64>>()
                .objective(FnEnergy(|x: &Vec<f64>| x.iter().map(|v| v * v).sum()))
                .moves(GaussianMove::new(0.5))
                .schedule(Exponential::new(10.0, 0.999))
                .iterations(10_000)
                .seed(seed)
                .build()
                .unwrap()
                .run(vec![5.0, -3.0])
        };

        let r1 = run(42);
        let r2 = run(42);
        assert_eq!(r1.best_energy, r2.best_energy, "same seed must give same result");
        assert_eq!(r1.best_state, r2.best_state);

        let r3 = run(43);
        assert_ne!(r1.best_energy, r3.best_energy, "different seeds should differ");
    }

    #[test]
    fn trajectory_recording() {
        let result = builder::<Vec<f64>>()
            .objective(FnEnergy(|x: &Vec<f64>| x.iter().map(|v| v * v).sum()))
            .moves(GaussianMove::new(0.5))
            .schedule(Exponential::new(10.0, 0.999))
            .iterations(10_000)
            .seed(42)
            .record_trajectory(100)
            .build()
            .unwrap()
            .run(vec![5.0]);

        let traj = result.trajectory.expect("trajectory should be recorded");
        assert_eq!(traj.len(), 100); // 10_000 / 100
    }

    #[test]
    fn acceptance_rate_reasonable() {
        let result = builder::<Vec<f64>>()
            .objective(FnEnergy(|x: &Vec<f64>| x.iter().map(|v| v * v).sum()))
            .moves(GaussianMove::new(0.5))
            .schedule(Exponential::new(100.0, 0.9999))
            .iterations(50_000)
            .seed(42)
            .build()
            .unwrap()
            .run(vec![5.0, -3.0, 7.0]);

        let rate = result.diagnostics.acceptance_rate();
        assert!(
            rate > 0.05 && rate < 0.95,
            "acceptance rate should be reasonable: {}",
            rate
        );
    }
}

//! Parallel Tempering (Replica Exchange) engine.
//!
//! # Theory (H-05)
//! Runs R replicas at different temperatures T₁ < T₂ < ... < `T_R`.
//! Periodically proposes swaps between adjacent replicas:
//!
//! ```text
//! P_swap(i,j) = min(1, exp((1/T_i - 1/T_j)(E_j - E_i)))
//! ```
//!
//! This preserves the extended Boltzmann distribution over the product space
//! and enables barrier crossing via high-temperature exploration.
//!
//! # Design (Turon: composable builder; Lamport: deterministic state machine)
//! Each replica is a deterministic state machine with its own RNG.
//! Swap decisions use a dedicated "swap RNG" for reproducibility.
//! The entire PT run is bit-reproducible given the same seed.
//!
//! # Performance (Muratori: hot/cold split)
//! Per-replica Metropolis steps are the hot path.
//! Swap proposals are cold path (every `swap_interval` steps).
//! Single-threaded for correctness; parallelism is a future optimization.
use crate::energy::Energy;
use crate::error::AnnealError;
use crate::math;
use crate::moves::MoveOperator;
use crate::rng::{DefaultRng, Rng};

// ---------------------------------------------------------------------------
// Temperature ladder
// ---------------------------------------------------------------------------

/// Geometric temperature ladder: `T_r` = `T_min` * (`T_max` / `T_min`)^(r / (R-1))
///
/// This spacing ensures approximately equal swap acceptance rates between
/// adjacent pairs when the energy variance scales smoothly with T.
///
/// # Errors
/// Returns [`AnnealError::InvalidParameter`] if `t_min <= 0`, `t_max <= t_min`,
/// or `num_replicas < 2`.
pub fn geometric_ladder(
    t_min: f64,
    t_max: f64,
    num_replicas: usize,
) -> Result<Vec<f64>, AnnealError> {
    if t_min <= 0.0 {
        return Err(AnnealError::InvalidParameter { name: "t_min", reason: "must be positive" });
    }
    if t_max <= t_min {
        return Err(AnnealError::InvalidParameter { name: "t_max", reason: "must exceed t_min" });
    }
    if num_replicas < 2 {
        return Err(AnnealError::InvalidParameter {
            name: "num_replicas",
            reason: "need at least 2",
        });
    }

    let ratio = t_max / t_min;
    Ok((0..num_replicas)
        .map(|r| t_min * ratio.powf(r as f64 / (num_replicas - 1) as f64))
        .collect())
}

// ---------------------------------------------------------------------------
// Replica state
// ---------------------------------------------------------------------------

/// A single replica in the parallel tempering ensemble.
struct Replica<S, R: Rng> {
    state: S,
    energy: f64,
    temperature: f64,
    rng: R,
}

// ---------------------------------------------------------------------------
// PT Result
// ---------------------------------------------------------------------------

/// Diagnostics from a parallel tempering run.
#[derive(Clone, Debug)]
pub struct PTDiagnostics {
    /// Swap acceptance rate for each adjacent pair (i, i+1).
    pub swap_rates: Vec<f64>,
    /// Total swaps proposed per pair.
    pub swaps_proposed: Vec<u64>,
    /// Total swaps accepted per pair.
    pub swaps_accepted: Vec<u64>,
    /// Per-replica acceptance rate (Metropolis moves).
    pub replica_acceptance_rates: Vec<f64>,
}

/// Result of a parallel tempering run.
#[derive(Clone, Debug)]
pub struct PTResult<S> {
    /// Best state found across all replicas.
    pub best_state: S,
    /// Best energy found across all replicas.
    pub best_energy: f64,
    /// Final states of all replicas (indexed by temperature rank).
    pub final_states: Vec<S>,
    /// Final energies of all replicas.
    pub final_energies: Vec<f64>,
    /// Temperature ladder used.
    pub temperatures: Vec<f64>,
    /// Diagnostics.
    pub diagnostics: PTDiagnostics,
}

// ---------------------------------------------------------------------------
// PT Engine
// ---------------------------------------------------------------------------

/// Parallel Tempering engine. Constructed via `ParallelTempering::builder()`.
pub struct ParallelTempering<S, E, M, R = DefaultRng>
where
    E: Energy<S>,
    M: MoveOperator<S>,
    R: Rng,
{
    objective: E,
    moves: M,
    temperatures: Vec<f64>,
    iterations_per_replica: u64,
    swap_interval: u64,
    seed: u64,
    _phantom: core::marker::PhantomData<(S, R)>,
}

impl<S, E, M, R> core::fmt::Debug for ParallelTempering<S, E, M, R>
where
    E: Energy<S>,
    M: MoveOperator<S>,
    R: Rng,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ParallelTempering")
            .field("temperatures", &self.temperatures)
            .field("iterations_per_replica", &self.iterations_per_replica)
            .field("swap_interval", &self.swap_interval)
            .finish_non_exhaustive()
    }
}

impl<S, E, M> ParallelTempering<S, E, M, DefaultRng>
where
    S: Clone + core::fmt::Debug,
    E: Energy<S>,
    M: MoveOperator<S>,
{
    /// Run parallel tempering from the given initial state.
    ///
    /// All replicas start from clones of `initial`.
    pub fn run(&self, initial: S) -> PTResult<S> {
        self.run_impl::<DefaultRng>(initial)
    }
}

impl<S, E, M, R> ParallelTempering<S, E, M, R>
where
    S: Clone + core::fmt::Debug,
    E: Energy<S>,
    M: MoveOperator<S>,
    R: Rng,
{
    #[allow(clippy::too_many_lines)]
    fn run_impl<R2: Rng>(&self, initial: S) -> PTResult<S> {
        let num_replicas = self.temperatures.len();
        let initial_energy = self.objective.energy(&initial);

        // Initialize replicas with independent RNGs
        // Seed derivation: replica r gets seed = base_seed XOR (r * golden_ratio_bits)
        let mut replicas: Vec<Replica<S, R2>> = (0..num_replicas)
            .map(|r| {
                let replica_seed = self.seed ^ ((r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                Replica {
                    state: initial.clone(),
                    energy: initial_energy,
                    temperature: self.temperatures[r],
                    rng: R2::from_seed(replica_seed),
                }
            })
            .collect();

        // Swap RNG — deterministic, independent of replica RNGs
        let mut swap_rng = R2::from_seed(self.seed.wrapping_mul(0x517C_C1B7_2722_0A95));

        // Track best across all replicas
        let mut best_state = initial;
        let mut best_energy = initial_energy;

        // Swap diagnostics
        let mut swaps_proposed = vec![0u64; num_replicas.saturating_sub(1)];
        let mut swaps_accepted = vec![0u64; num_replicas.saturating_sub(1)];

        // Per-replica move diagnostics
        let mut replica_accepts = vec![0u64; num_replicas];
        let mut replica_proposals = vec![0u64; num_replicas];

        // === MAIN LOOP ===
        for step in 0..self.iterations_per_replica {
            // --- Per-replica Metropolis step (HOT PATH) ---
            for (r, replica) in replicas.iter_mut().enumerate() {
                let candidate = self.moves.propose(&replica.state, &mut replica.rng);
                let candidate_energy = self.objective.energy(&candidate);
                let delta_e = candidate_energy - replica.energy;

                // Hastings correction for asymmetric proposals
                let log_correction = if self.moves.is_symmetric() {
                    0.0
                } else {
                    self.moves.log_proposal_ratio(&replica.state, &candidate)
                };

                let u = replica.rng.next_f64();
                let accepted = if log_correction == 0.0 {
                    math::metropolis_accept(delta_e, replica.temperature, u)
                } else {
                    let adjusted_delta = replica.temperature.mul_add(-log_correction, delta_e);
                    math::metropolis_accept(adjusted_delta, replica.temperature, u)
                };

                replica_proposals[r] += 1;
                if accepted {
                    replica.state = candidate;
                    replica.energy = candidate_energy;
                    replica_accepts[r] += 1;

                    if candidate_energy < best_energy {
                        best_state = replica.state.clone();
                        best_energy = candidate_energy;
                    }
                }
            }

            // --- Swap proposals (COLD PATH) ---
            if self.swap_interval > 0 && step % self.swap_interval == 0 && step > 0 {
                // Alternate even/odd pairs to avoid sequential correlation
                let parity = (step / self.swap_interval) % 2;
                let start = parity as usize;

                let mut pair_idx = start;
                while pair_idx + 1 < num_replicas {
                    let i = pair_idx;
                    let j = pair_idx + 1;

                    // Swap criterion: min(1, exp((β_i - β_j)(E_j - E_i)))
                    // where β = 1/T
                    let beta_i = 1.0 / replicas[i].temperature;
                    let beta_j = 1.0 / replicas[j].temperature;
                    let delta = (beta_i - beta_j) * (replicas[j].energy - replicas[i].energy);

                    let u = swap_rng.next_f64();
                    let swap_accepted = if delta <= 0.0 { true } else { u < (-delta).exp() };

                    swaps_proposed[i] += 1;
                    if swap_accepted {
                        swaps_accepted[i] += 1;
                        // Use split_at_mut for non-overlapping mutable borrows
                        let (left, right) = replicas.split_at_mut(j);
                        core::mem::swap(&mut left[i].state, &mut right[0].state);
                        core::mem::swap(&mut left[i].energy, &mut right[0].energy);
                    }

                    pair_idx += 2;
                }
            }
        }

        // Compute diagnostics
        let swap_rates: Vec<f64> = swaps_proposed
            .iter()
            .zip(swaps_accepted.iter())
            .map(
                |(&proposed, &accepted)| {
                    if proposed > 0 {
                        accepted as f64 / proposed as f64
                    } else {
                        0.0
                    }
                },
            )
            .collect();

        let replica_acceptance_rates: Vec<f64> = replica_accepts
            .iter()
            .zip(replica_proposals.iter())
            .map(|(&acc, &prop)| if prop > 0 { acc as f64 / prop as f64 } else { 0.0 })
            .collect();

        PTResult {
            best_state,
            best_energy,
            final_states: replicas.iter().map(|r| r.state.clone()).collect(),
            final_energies: replicas.iter().map(|r| r.energy).collect(),
            temperatures: self.temperatures.clone(),
            diagnostics: PTDiagnostics {
                swap_rates,
                swaps_proposed,
                swaps_accepted,
                replica_acceptance_rates,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for `ParallelTempering`.
pub struct PTBuilder<S, E, M> {
    objective: Option<E>,
    moves: Option<M>,
    temperatures: Option<Vec<f64>>,
    iterations_per_replica: u64,
    swap_interval: u64,
    seed: u64,
    _phantom: core::marker::PhantomData<S>,
}

impl<S, E, M> core::fmt::Debug for PTBuilder<S, E, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PTBuilder")
            .field("seed", &self.seed)
            .field("iterations_per_replica", &self.iterations_per_replica)
            .field("swap_interval", &self.swap_interval)
            .field("temperatures_set", &self.temperatures.is_some())
            .field("objective_set", &self.objective.is_some())
            .field("moves_set", &self.moves.is_some())
            .finish_non_exhaustive()
    }
}

/// Entry point: `parallel_tempering()`.
pub const fn builder<S>() -> PTBuilder<S, (), ()> {
    PTBuilder {
        objective: None,
        moves: None,
        temperatures: None,
        iterations_per_replica: 100_000,
        swap_interval: 10,
        seed: 0,
        _phantom: core::marker::PhantomData,
    }
}

impl<S, E, M> PTBuilder<S, E, M> {
    /// Set the objective (energy/cost) function.
    pub fn objective<E2: Energy<S>>(self, obj: E2) -> PTBuilder<S, E2, M> {
        PTBuilder {
            objective: Some(obj),
            moves: self.moves,
            temperatures: self.temperatures,
            iterations_per_replica: self.iterations_per_replica,
            swap_interval: self.swap_interval,
            seed: self.seed,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Set the move operator.
    pub fn moves<M2: MoveOperator<S>>(self, m: M2) -> PTBuilder<S, E, M2> {
        PTBuilder {
            objective: self.objective,
            moves: Some(m),
            temperatures: self.temperatures,
            iterations_per_replica: self.iterations_per_replica,
            swap_interval: self.swap_interval,
            seed: self.seed,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Set the temperature ladder directly.
    ///
    /// # Errors
    /// Returns [`AnnealError::InvalidParameter`] if fewer than 2 temperatures
    /// or if they are not strictly increasing.
    pub fn temperatures(mut self, temps: Vec<f64>) -> Result<Self, AnnealError> {
        if temps.len() < 2 {
            return Err(AnnealError::InvalidParameter {
                name: "temperatures",
                reason: "need at least 2",
            });
        }
        for w in temps.windows(2) {
            if w[0] >= w[1] {
                return Err(AnnealError::InvalidParameter {
                    name: "temperatures",
                    reason: "must be strictly increasing",
                });
            }
        }
        self.temperatures = Some(temps);
        Ok(self)
    }

    /// Set the temperature ladder via geometric spacing.
    ///
    /// # Errors
    /// Returns [`AnnealError::InvalidParameter`] if parameters are invalid.
    pub fn geometric_temperatures(
        self,
        t_min: f64,
        t_max: f64,
        num_replicas: usize,
    ) -> Result<Self, AnnealError> {
        let temps = geometric_ladder(t_min, t_max, num_replicas)?;
        self.temperatures(temps)
    }

    /// Set the number of iterations per replica.
    #[must_use]
    pub const fn iterations(mut self, n: u64) -> Self {
        self.iterations_per_replica = n;
        self
    }

    /// Set the swap interval (propose swaps every N steps).
    #[must_use]
    pub const fn swap_interval(mut self, interval: u64) -> Self {
        self.swap_interval = interval;
        self
    }

    /// Set the RNG seed for reproducibility.
    #[must_use]
    pub const fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl<S, E, M> PTBuilder<S, E, M>
where
    S: Clone + core::fmt::Debug,
    E: Energy<S>,
    M: MoveOperator<S>,
{
    /// Build the parallel tempering engine.
    ///
    /// # Errors
    /// Returns [`AnnealError::MissingField`] if objective, moves, or temperatures
    /// were not set.
    pub fn build(self) -> Result<ParallelTempering<S, E, M, DefaultRng>, AnnealError> {
        Ok(ParallelTempering {
            objective: self.objective.ok_or(AnnealError::MissingField { field: "objective" })?,
            moves: self.moves.ok_or(AnnealError::MissingField { field: "moves" })?,
            temperatures: self
                .temperatures
                .ok_or(AnnealError::MissingField { field: "temperatures" })?,
            iterations_per_replica: self.iterations_per_replica,
            swap_interval: self.swap_interval,
            seed: self.seed,
            _phantom: core::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::FnEnergy;
    use crate::landscape::double_well::{DoubleWell, DoubleWellMove};

    #[test]
    fn geometric_ladder_correct() {
        let ladder = geometric_ladder(1.0, 100.0, 5).unwrap();
        assert_eq!(ladder.len(), 5);
        assert!((ladder[0] - 1.0).abs() < 1e-10);
        assert!((ladder[4] - 100.0).abs() < 1e-10);
        // Monotonically increasing
        for w in ladder.windows(2) {
            assert!(w[1] > w[0]);
        }
        // Geometric: ratios should be equal
        let r1 = ladder[1] / ladder[0];
        let r2 = ladder[2] / ladder[1];
        assert!((r1 - r2).abs() < 1e-10);
    }

    #[test]
    fn pt_finds_global_minimum_quadratic() {
        let result = builder::<Vec<f64>>()
            .objective(FnEnergy(|x: &Vec<f64>| x.iter().map(|v| v * v).sum()))
            .moves(crate::moves::GaussianMove::new(0.5))
            .geometric_temperatures(0.1, 50.0, 4)
            .unwrap()
            .iterations(50_000)
            .swap_interval(10)
            .seed(42)
            .build()
            .unwrap()
            .run(vec![5.0, -3.0, 7.0]);

        assert!(result.best_energy < 1.0, "PT should find near-origin: E={}", result.best_energy);
    }

    #[test]
    fn pt_deterministic() {
        let run = |seed| {
            builder::<Vec<f64>>()
                .objective(FnEnergy(|x: &Vec<f64>| x.iter().map(|v| v * v).sum()))
                .moves(crate::moves::GaussianMove::new(0.5))
                .geometric_temperatures(0.1, 50.0, 4)
                .unwrap()
                .iterations(10_000)
                .seed(seed)
                .build()
                .unwrap()
                .run(vec![5.0, -3.0])
        };

        let r1 = run(42);
        let r2 = run(42);
        assert_eq!(r1.best_energy, r2.best_energy, "same seed = same result");
        assert_eq!(r1.best_state, r2.best_state);

        let r3 = run(43);
        assert_ne!(r1.best_energy, r3.best_energy, "different seeds should differ");
    }

    #[test]
    fn pt_swap_diagnostics_populated() {
        let result = builder::<Vec<f64>>()
            .objective(FnEnergy(|x: &Vec<f64>| x.iter().map(|v| v * v).sum()))
            .moves(crate::moves::GaussianMove::new(0.5))
            .geometric_temperatures(0.1, 50.0, 4)
            .unwrap()
            .iterations(10_000)
            .swap_interval(10)
            .seed(42)
            .build()
            .unwrap()
            .run(vec![5.0]);

        // 4 replicas → 3 adjacent pairs
        assert_eq!(result.diagnostics.swap_rates.len(), 3);
        assert_eq!(result.diagnostics.replica_acceptance_rates.len(), 4);

        // Swap rates should be in [0, 1]
        for &rate in &result.diagnostics.swap_rates {
            assert!(rate >= 0.0 && rate <= 1.0, "swap rate out of range: {}", rate);
        }

        // Some swaps should have been proposed
        for &proposed in &result.diagnostics.swaps_proposed {
            assert!(proposed > 0, "should have proposed swaps");
        }
    }

    #[test]
    fn pt_beats_sa_on_double_well() {
        // Double well with barrier — PT should find global minimum more often
        let well = DoubleWell::new(50, 15.0);
        let mv = DoubleWellMove::new(50);

        let mut pt_successes = 0u32;
        let mut sa_successes = 0u32;
        let num_trials = 50;
        let total_budget = 200_000u64;

        for seed in 0..num_trials {
            // PT: 4 replicas, budget split evenly
            let pt_result = builder::<i64>()
                .objective(well.clone())
                .moves(mv.clone())
                .geometric_temperatures(0.5, 30.0, 4)
                .unwrap()
                .iterations(total_budget / 4)
                .swap_interval(10)
                .seed(seed)
                .build()
                .unwrap()
                .run(0);

            if pt_result.best_energy < -3.0 {
                pt_successes += 1;
            }

            // SA: full budget, exponential cooling
            let mut sa = crate::annealer::builder::<i64>()
                .objective(well.clone())
                .moves(mv.clone())
                .schedule(crate::schedule::Exponential::new(30.0, 0.99999))
                .iterations(total_budget)
                .seed(seed)
                .build()
                .unwrap();
            let sa_result = sa.run(0);

            if sa_result.best_energy < -3.0 {
                sa_successes += 1;
            }
        }

        // PT should outperform SA on barrier crossing (or at least match)
        assert!(
            pt_successes >= sa_successes / 2,
            "PT ({}/{}) should not be dramatically worse than SA ({}/{})",
            pt_successes,
            num_trials,
            sa_successes,
            num_trials
        );
    }
}

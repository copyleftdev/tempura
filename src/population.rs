/// Population Annealing engine.
///
/// # Theory (H-06)
/// Maintains a population of N solutions. At each temperature step:
///   1. Compute Boltzmann weights: w_i = exp(-E(x_i) · Δβ)
///   2. Resample population proportional to weights
///   3. Equilibrate each member with M Metropolis sweeps
///
/// This yields unbiased estimates of the partition function ratio Z(T_{k+1})/Z(T_k)
/// and enables direct free energy computation. Error scales as O(1/√N).
///
/// # Design (Turon: composable builder; Lamport: deterministic state machine)
/// Each population member has its own RNG for deterministic reproduction.
/// Resampling uses systematic resampling (lower variance than multinomial).
/// All weight computations use log-space arithmetic for numerical stability.
///
/// # Performance (Muratori: embarrassingly parallel structure)
/// Equilibration sweeps are independent across population members — ideal for
/// future rayon parallelization. Single-threaded for now (correctness first).
use crate::energy::Energy;
use crate::error::AnnealError;
use crate::math;
use crate::moves::MoveOperator;
use crate::rng::{DefaultRng, Rng};

// ---------------------------------------------------------------------------
// PA Result & Diagnostics
// ---------------------------------------------------------------------------

/// Per-temperature-step diagnostics.
#[derive(Clone, Debug)]
pub struct StepDiagnostics {
    /// Temperature at this step.
    pub temperature: f64,
    /// Effective population fraction ρ ∈ (0, 1].
    /// ρ ≈ 1 means uniform weights (good). ρ → 1/N means collapse (bad).
    pub effective_fraction: f64,
    /// Log of mean weight (for partition function estimation).
    pub log_mean_weight: f64,
    /// Mean energy of population at this temperature.
    pub mean_energy: f64,
    /// Best energy in population at this temperature.
    pub best_energy: f64,
    /// Acceptance rate of Metropolis sweeps at this temperature.
    pub acceptance_rate: f64,
}

/// Result of a population annealing run.
#[derive(Clone, Debug)]
pub struct PAResult<S> {
    /// Best state found across entire run.
    pub best_state: S,
    /// Best energy found.
    pub best_energy: f64,
    /// Final population states.
    pub final_population: Vec<S>,
    /// Final population energies.
    pub final_energies: Vec<f64>,
    /// Per-step diagnostics.
    pub step_diagnostics: Vec<StepDiagnostics>,
    /// Estimated log partition function ratio: ln(Z(T_final) / Z(T_0)).
    pub log_partition_ratio: f64,
}

// ---------------------------------------------------------------------------
// PA Engine
// ---------------------------------------------------------------------------

/// Population Annealing engine. Constructed via `population::builder()`.
pub struct PopulationAnnealer<S, E, M, R = DefaultRng>
where
    E: Energy<S>,
    M: MoveOperator<S>,
    R: Rng,
{
    objective: E,
    moves: M,
    temperatures: Vec<f64>,
    population_size: usize,
    sweeps_per_step: u64,
    seed: u64,
    _phantom: core::marker::PhantomData<(S, R)>,
}

impl<S, E, M> PopulationAnnealer<S, E, M, DefaultRng>
where
    S: Clone + core::fmt::Debug,
    E: Energy<S>,
    M: MoveOperator<S>,
{
    /// Run population annealing from the given initial state.
    ///
    /// All population members start as clones of `initial`.
    pub fn run(&self, initial: S) -> PAResult<S> {
        self.run_impl::<DefaultRng>(initial)
    }
}

impl<S, E, M, R> PopulationAnnealer<S, E, M, R>
where
    S: Clone + core::fmt::Debug,
    E: Energy<S>,
    M: MoveOperator<S>,
    R: Rng,
{
    fn run_impl<R2: Rng>(&self, initial: S) -> PAResult<S> {
        let n = self.population_size;
        let initial_energy = self.objective.energy(&initial);

        // Initialize population with independent RNGs
        let mut states: Vec<S> = vec![initial.clone(); n];
        let mut energies: Vec<f64> = vec![initial_energy; n];
        let mut rngs: Vec<R2> = (0..n)
            .map(|i| {
                let member_seed = self.seed ^ ((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
                R2::from_seed(member_seed)
            })
            .collect();

        // Resampling RNG — deterministic, independent of member RNGs
        let mut resample_rng = R2::from_seed(self.seed.wrapping_mul(0x6A09E667F3BCC908));

        let mut best_state = initial;
        let mut best_energy = initial_energy;
        let mut step_diagnostics = Vec::with_capacity(self.temperatures.len());
        let mut log_partition_ratio = 0.0f64;

        // === MAIN LOOP: iterate through temperature schedule ===
        for k in 0..self.temperatures.len().saturating_sub(1) {
            let t_current = self.temperatures[k];
            let t_next = self.temperatures[k + 1];
            let delta_beta = 1.0 / t_next - 1.0 / t_current;

            // Step 1: Compute log-weights (numerical stability)
            let log_weights: Vec<f64> = energies
                .iter()
                .map(|&e| -e * delta_beta)
                .collect();

            // Log-sum-exp for normalization
            let max_log_w = log_weights
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let log_sum_exp: f64 = log_weights
                .iter()
                .map(|&lw| (lw - max_log_w).exp())
                .sum::<f64>()
                .ln()
                + max_log_w;
            let log_mean_weight = log_sum_exp - (n as f64).ln();

            // Accumulate for partition function estimation
            log_partition_ratio += log_mean_weight;

            // Normalized probabilities
            let probs: Vec<f64> = log_weights
                .iter()
                .map(|&lw| (lw - log_sum_exp).exp())
                .collect();

            // Effective population fraction: ρ = 1 / (N · Σ p_i²)
            let sum_p_sq: f64 = probs.iter().map(|&p| p * p).sum();
            let effective_fraction = if sum_p_sq > 0.0 {
                1.0 / (n as f64 * sum_p_sq)
            } else {
                0.0
            };

            // Step 2: Systematic resampling
            let indices = systematic_resample(&probs, n, &mut resample_rng);
            let new_states: Vec<S> = indices.iter().map(|&i| states[i].clone()).collect();
            let new_energies: Vec<f64> = indices.iter().map(|&i| energies[i]).collect();
            // Derive new RNGs from the resampled indices + step counter
            let new_rngs: Vec<R2> = indices
                .iter()
                .enumerate()
                .map(|(new_i, &old_i)| {
                    let resample_seed = rngs[old_i].next_u64()
                        ^ ((new_i as u64).wrapping_mul(0x517CC1B727220A95))
                        ^ (k as u64);
                    R2::from_seed(resample_seed)
                })
                .collect();
            states = new_states;
            energies = new_energies;
            rngs = new_rngs;

            // Step 3: Equilibration sweeps at T_{k+1}
            let mut total_accepts = 0u64;
            let mut total_proposals = 0u64;

            for i in 0..n {
                for _ in 0..self.sweeps_per_step {
                    let candidate = self.moves.propose(&states[i], &mut rngs[i]);
                    let candidate_energy = self.objective.energy(&candidate);
                    let delta_e = candidate_energy - energies[i];

                    let log_correction = if self.moves.is_symmetric() {
                        0.0
                    } else {
                        self.moves.log_proposal_ratio(&states[i], &candidate)
                    };

                    let u = rngs[i].next_f64();
                    let accepted = if log_correction == 0.0 {
                        math::metropolis_accept(delta_e, t_next, u)
                    } else {
                        let adjusted = delta_e - t_next * log_correction;
                        math::metropolis_accept(adjusted, t_next, u)
                    };

                    total_proposals += 1;
                    if accepted {
                        states[i] = candidate;
                        energies[i] = candidate_energy;
                        total_accepts += 1;

                        if candidate_energy < best_energy {
                            best_state = states[i].clone();
                            best_energy = candidate_energy;
                        }
                    }
                }
            }

            let mean_energy = energies.iter().sum::<f64>() / n as f64;
            let step_best = energies
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let acceptance_rate = if total_proposals > 0 {
                total_accepts as f64 / total_proposals as f64
            } else {
                0.0
            };

            step_diagnostics.push(StepDiagnostics {
                temperature: t_next,
                effective_fraction,
                log_mean_weight,
                mean_energy,
                best_energy: step_best,
                acceptance_rate,
            });
        }

        PAResult {
            best_state,
            best_energy,
            final_population: states,
            final_energies: energies,
            step_diagnostics,
            log_partition_ratio,
        }
    }
}

// ---------------------------------------------------------------------------
// Systematic resampling
// ---------------------------------------------------------------------------

/// Systematic resampling: select N indices from a probability distribution.
///
/// Lower variance than multinomial resampling. Uses a single uniform offset
/// plus evenly spaced increments.
fn systematic_resample(probs: &[f64], n: usize, rng: &mut impl Rng) -> Vec<usize> {
    let u0 = rng.next_f64() / n as f64;
    let mut indices = Vec::with_capacity(n);
    let mut cumulative = 0.0f64;
    let mut j = 0;

    for i in 0..n {
        let threshold = u0 + i as f64 / n as f64;
        while j < probs.len() - 1 && cumulative + probs[j] < threshold {
            cumulative += probs[j];
            j += 1;
        }
        indices.push(j);
    }

    debug_assert_eq!(indices.len(), n);
    indices
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for `PopulationAnnealer`.
pub struct PABuilder<S, E, M> {
    objective: Option<E>,
    moves: Option<M>,
    temperatures: Option<Vec<f64>>,
    population_size: usize,
    sweeps_per_step: u64,
    seed: u64,
    _phantom: core::marker::PhantomData<S>,
}

/// Entry point for building a population annealer.
pub fn builder<S>() -> PABuilder<S, (), ()> {
    PABuilder {
        objective: None,
        moves: None,
        temperatures: None,
        population_size: 1000,
        sweeps_per_step: 10,
        seed: 0,
        _phantom: core::marker::PhantomData,
    }
}

impl<S, E, M> PABuilder<S, E, M> {
    /// Set the objective (energy/cost) function.
    pub fn objective<E2: Energy<S>>(self, obj: E2) -> PABuilder<S, E2, M> {
        PABuilder {
            objective: Some(obj),
            moves: self.moves,
            temperatures: self.temperatures,
            population_size: self.population_size,
            sweeps_per_step: self.sweeps_per_step,
            seed: self.seed,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Set the move operator.
    pub fn moves<M2: MoveOperator<S>>(self, m: M2) -> PABuilder<S, E, M2> {
        PABuilder {
            objective: self.objective,
            moves: Some(m),
            temperatures: self.temperatures,
            population_size: self.population_size,
            sweeps_per_step: self.sweeps_per_step,
            seed: self.seed,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Set the cooling temperature schedule (must be strictly decreasing).
    ///
    /// # Errors
    /// Returns [`AnnealError::InvalidParameter`] if fewer than 2 temperatures
    /// or if they are not strictly decreasing.
    pub fn temperatures(mut self, temps: Vec<f64>) -> Result<Self, AnnealError> {
        if temps.len() < 2 {
            return Err(AnnealError::InvalidParameter { name: "temperatures", reason: "need at least 2" });
        }
        for w in temps.windows(2) {
            if w[0] <= w[1] {
                return Err(AnnealError::InvalidParameter { name: "temperatures", reason: "must be strictly decreasing" });
            }
        }
        self.temperatures = Some(temps);
        Ok(self)
    }

    /// Set a geometric cooling schedule.
    ///
    /// # Errors
    /// Returns [`AnnealError::InvalidParameter`] if parameters are invalid.
    pub fn geometric_cooling(self, t_high: f64, t_low: f64, num_steps: usize) -> Result<Self, AnnealError> {
        if t_high <= t_low {
            return Err(AnnealError::InvalidParameter { name: "t_high", reason: "must exceed t_low" });
        }
        if t_low <= 0.0 {
            return Err(AnnealError::InvalidParameter { name: "t_low", reason: "must be positive" });
        }
        if num_steps < 2 {
            return Err(AnnealError::InvalidParameter { name: "num_steps", reason: "need at least 2" });
        }
        let ratio = t_low / t_high;
        let temps: Vec<f64> = (0..num_steps)
            .map(|k| t_high * ratio.powf(k as f64 / (num_steps - 1) as f64))
            .collect();
        self.temperatures(temps)
    }

    /// Set the population size.
    ///
    /// # Errors
    /// Returns [`AnnealError::InvalidParameter`] if `n < 2`.
    pub fn population_size(mut self, n: usize) -> Result<Self, AnnealError> {
        if n < 2 {
            return Err(AnnealError::InvalidParameter { name: "population_size", reason: "must have at least 2 members" });
        }
        self.population_size = n;
        Ok(self)
    }

    /// Set the number of Metropolis sweeps per temperature step.
    pub fn sweeps_per_step(mut self, m: u64) -> Self {
        self.sweeps_per_step = m;
        self
    }

    /// Set the RNG seed for reproducibility.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl<S, E, M> PABuilder<S, E, M>
where
    S: Clone + core::fmt::Debug,
    E: Energy<S>,
    M: MoveOperator<S>,
{
    /// Build the population annealer.
    ///
    /// # Errors
    /// Returns [`AnnealError::MissingField`] if objective, moves, or temperatures
    /// were not set.
    pub fn build(self) -> Result<PopulationAnnealer<S, E, M, DefaultRng>, AnnealError> {
        Ok(PopulationAnnealer {
            objective: self.objective.ok_or(AnnealError::MissingField { field: "objective" })?,
            moves: self.moves.ok_or(AnnealError::MissingField { field: "moves" })?,
            temperatures: self.temperatures.ok_or(AnnealError::MissingField { field: "temperatures" })?,
            population_size: self.population_size,
            sweeps_per_step: self.sweeps_per_step,
            seed: self.seed,
            _phantom: core::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::FnEnergy;
    use crate::moves::GaussianMove;

    fn cooling_schedule(t_high: f64, t_low: f64, steps: usize) -> Vec<f64> {
        let ratio = t_low / t_high;
        (0..steps)
            .map(|k| t_high * ratio.powf(k as f64 / (steps - 1) as f64))
            .collect()
    }

    #[test]
    fn pa_finds_minimum_quadratic() {
        let temps = cooling_schedule(50.0, 0.1, 50);
        let result = builder::<Vec<f64>>()
            .objective(FnEnergy(|x: &Vec<f64>| x.iter().map(|v| v * v).sum()))
            .moves(GaussianMove::new(0.5))
            .temperatures(temps).unwrap()
            .population_size(100).unwrap()
            .sweeps_per_step(10)
            .seed(42)
            .build().unwrap()
            .run(vec![5.0, -3.0, 7.0]);

        assert!(
            result.best_energy < 1.0,
            "PA should find near-origin: E={}",
            result.best_energy
        );
    }

    #[test]
    fn pa_deterministic() {
        let run = |seed| {
            let temps = cooling_schedule(50.0, 0.1, 30);
            builder::<Vec<f64>>()
                .objective(FnEnergy(|x: &Vec<f64>| x.iter().map(|v| v * v).sum()))
                .moves(GaussianMove::new(0.5))
                .temperatures(temps).unwrap()
                .population_size(50).unwrap()
                .sweeps_per_step(5)
                .seed(seed)
                .build().unwrap()
                .run(vec![5.0, -3.0])
        };

        let r1 = run(42);
        let r2 = run(42);
        assert_eq!(r1.best_energy, r2.best_energy, "same seed = same result");

        let r3 = run(43);
        assert_ne!(r1.best_energy, r3.best_energy, "different seeds should differ");
    }

    #[test]
    fn pa_diagnostics_populated() {
        let num_steps = 30;
        let temps = cooling_schedule(50.0, 0.1, num_steps);
        let result = builder::<Vec<f64>>()
            .objective(FnEnergy(|x: &Vec<f64>| x.iter().map(|v| v * v).sum()))
            .moves(GaussianMove::new(0.5))
            .temperatures(temps).unwrap()
            .population_size(50).unwrap()
            .sweeps_per_step(5)
            .seed(42)
            .build().unwrap()
            .run(vec![5.0]);

        // num_steps - 1 transitions
        assert_eq!(result.step_diagnostics.len(), num_steps - 1);

        // All effective fractions should be in (0, 1]
        for diag in &result.step_diagnostics {
            assert!(
                diag.effective_fraction > 0.0 && diag.effective_fraction <= 1.0,
                "ρ out of range: {}",
                diag.effective_fraction
            );
        }

        // Partition function ratio should be finite
        assert!(
            result.log_partition_ratio.is_finite(),
            "log Z ratio should be finite"
        );
    }

    #[test]
    fn pa_effective_fraction_high_with_gentle_cooling() {
        // With many small temperature steps, ρ should stay high
        let temps = cooling_schedule(50.0, 1.0, 100);
        let result = builder::<Vec<f64>>()
            .objective(FnEnergy(|x: &Vec<f64>| x.iter().map(|v| v * v).sum()))
            .moves(GaussianMove::new(0.5))
            .temperatures(temps).unwrap()
            .population_size(200).unwrap()
            .sweeps_per_step(10)
            .seed(42)
            .build().unwrap()
            .run(vec![5.0, -3.0]);

        let min_rho = result
            .step_diagnostics
            .iter()
            .map(|d| d.effective_fraction)
            .fold(f64::INFINITY, f64::min);

        assert!(
            min_rho > 0.1,
            "gentle cooling should maintain ρ > 0.1, got {}",
            min_rho
        );
    }

    #[test]
    fn systematic_resample_preserves_size() {
        let probs = vec![0.1, 0.2, 0.3, 0.15, 0.25];
        let mut rng = DefaultRng::from_seed(42);
        let indices = systematic_resample(&probs, 100, &mut rng);
        assert_eq!(indices.len(), 100);
        // All indices in range
        for &idx in &indices {
            assert!(idx < probs.len());
        }
        // Higher-probability states should appear more often
        let count_2 = indices.iter().filter(|&&i| i == 2).count();
        let count_0 = indices.iter().filter(|&&i| i == 0).count();
        assert!(
            count_2 > count_0,
            "state 2 (p=0.3) should appear more than state 0 (p=0.1)"
        );
    }
}

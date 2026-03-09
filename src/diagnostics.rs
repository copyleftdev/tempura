/// Observability for annealing runs — energy trajectory, acceptance ratio,
/// temperature evolution, and convergence diagnostics.
///
/// # Design (Lamport: make state explicit)
/// Every metric is a concrete value, not an opaque counter.
/// Users can inspect the full trajectory or summary statistics.
///
/// # Performance (Muratori: hot/cold splitting)
/// Hot-path diagnostics (acceptance count, current energy) are updated inline.
/// Cold-path diagnostics (full trajectory recording) are opt-in via feature flags.

/// Lightweight diagnostics accumulated during an annealing run.
///
/// Always collected — zero-allocation in the hot path.
#[derive(Clone, Debug)]
pub struct RunDiagnostics {
    /// Total number of proposals evaluated.
    pub total_proposals: u64,
    /// Number of proposals accepted.
    pub accepted_proposals: u64,
    /// Lowest energy encountered during the run.
    pub best_energy: f64,
    /// Energy at the final step.
    pub final_energy: f64,
    /// Energy at the initial step.
    pub initial_energy: f64,
}

impl RunDiagnostics {
    /// Create diagnostics for a run starting at the given energy.
    pub fn new(initial_energy: f64) -> Self {
        Self {
            total_proposals: 0,
            accepted_proposals: 0,
            best_energy: initial_energy,
            final_energy: initial_energy,
            initial_energy,
        }
    }

    /// Overall acceptance rate across the entire run.
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_proposals == 0 {
            0.0
        } else {
            self.accepted_proposals as f64 / self.total_proposals as f64
        }
    }

    /// Energy improvement ratio: (initial - best) / initial.
    pub fn improvement_ratio(&self) -> f64 {
        if self.initial_energy.abs() < f64::EPSILON {
            return 0.0;
        }
        (self.initial_energy - self.best_energy) / self.initial_energy.abs()
    }

    #[inline(always)]
    pub(crate) fn record_proposal(&mut self, accepted: bool, energy: f64) {
        self.total_proposals += 1;
        if accepted {
            self.accepted_proposals += 1;
            self.final_energy = energy;
            if energy < self.best_energy {
                self.best_energy = energy;
            }
        }
    }
}

/// Detailed trajectory recorder — opt-in, allocates per-step data.
///
/// Records energy, temperature, and acceptance at every step.
/// Useful for visualization and convergence analysis.
#[derive(Clone, Debug)]
pub struct TrajectoryRecorder {
    /// Recorded energy values.
    pub energies: Vec<f64>,
    /// Recorded temperature values.
    pub temperatures: Vec<f64>,
    /// Whether each recorded step was an acceptance.
    pub accepted: Vec<bool>,
    sample_interval: u64,
    step: u64,
}

impl TrajectoryRecorder {
    /// Create a recorder that samples every `interval` steps.
    ///
    /// Use interval=1 for full recording (expensive for 10^6+ steps).
    /// Use interval=100 or 1000 for practical trajectory visualization.
    pub fn new(interval: u64) -> Self {
        assert!(interval > 0, "sample interval must be positive");
        Self {
            energies: Vec::new(),
            temperatures: Vec::new(),
            accepted: Vec::new(),
            sample_interval: interval,
            step: 0,
        }
    }

    /// Pre-allocate capacity for expected number of recorded samples.
    pub fn with_capacity(mut self, expected_steps: u64) -> Self {
        let n = (expected_steps / self.sample_interval + 1) as usize;
        self.energies.reserve(n);
        self.temperatures.reserve(n);
        self.accepted.reserve(n);
        self
    }

    #[inline(always)]
    pub(crate) fn record(&mut self, energy: f64, temperature: f64, was_accepted: bool) {
        self.step += 1;
        if self.step % self.sample_interval == 0 {
            self.energies.push(energy);
            self.temperatures.push(temperature);
            self.accepted.push(was_accepted);
        }
    }

    /// Number of recorded samples.
    pub fn len(&self) -> usize {
        self.energies.len()
    }

    /// Whether no samples have been recorded.
    pub fn is_empty(&self) -> bool {
        self.energies.is_empty()
    }

    /// Windowed acceptance rate at each recorded point.
    ///
    /// Window size is in units of recorded samples (not raw steps).
    pub fn windowed_acceptance_rate(&self, window: usize) -> Vec<f64> {
        if window == 0 || self.accepted.is_empty() {
            return Vec::new();
        }
        let mut rates = Vec::with_capacity(self.accepted.len());
        let mut accept_count = 0u64;
        for (i, &a) in self.accepted.iter().enumerate() {
            if a {
                accept_count += 1;
            }
            if i >= window {
                if self.accepted[i - window] {
                    accept_count -= 1;
                }
            }
            let w = (i + 1).min(window) as f64;
            rates.push(accept_count as f64 / w);
        }
        rates
    }
}

/// The result of an annealing run.
#[derive(Clone, Debug)]
pub struct AnnealResult<S> {
    /// The best state found during the run.
    pub best_state: S,
    /// The energy of the best state.
    pub best_energy: f64,
    /// The final state at termination.
    pub final_state: S,
    /// The energy of the final state.
    pub final_energy: f64,
    /// Summary diagnostics.
    pub diagnostics: RunDiagnostics,
    /// Detailed trajectory (if recording was enabled).
    pub trajectory: Option<TrajectoryRecorder>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_acceptance_rate() {
        let mut d = RunDiagnostics::new(100.0);
        for _ in 0..60 {
            d.record_proposal(true, 90.0);
        }
        for _ in 0..40 {
            d.record_proposal(false, 90.0);
        }
        assert!((d.acceptance_rate() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn diagnostics_best_energy_tracking() {
        let mut d = RunDiagnostics::new(100.0);
        d.record_proposal(true, 80.0);
        d.record_proposal(true, 90.0); // worse, but accepted (SA can go uphill)
        d.record_proposal(true, 70.0);
        assert_eq!(d.best_energy, 70.0);
        assert_eq!(d.final_energy, 70.0);
    }

    #[test]
    fn trajectory_sampling() {
        let mut t = TrajectoryRecorder::new(10);
        for i in 0..100 {
            t.record(100.0 - i as f64, 50.0, i % 2 == 0);
        }
        assert_eq!(t.len(), 10); // 100 steps / 10 interval
    }

    #[test]
    fn windowed_acceptance_rate_basic() {
        let mut t = TrajectoryRecorder::new(1);
        for i in 0..100 {
            t.record(0.0, 1.0, i < 50); // first 50 accepted, last 50 rejected
        }
        let rates = t.windowed_acceptance_rate(20);
        assert_eq!(rates.len(), 100);
        // Early window should be ~1.0
        assert!(rates[30] > 0.9);
        // Late window should be ~0.0
        assert!(rates[90] < 0.1);
    }
}

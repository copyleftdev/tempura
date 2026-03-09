/// 1D Discrete Potential Well — the simplest landscape with analytically
/// known Boltzmann distribution.
///
/// Used by: H-01 (Boltzmann convergence), H-02 (detailed balance)
///
/// States: {0, 1, ..., n-1}
/// Energy: E(x) = (x - center)²
/// Move:   x → x ± 1 (symmetric neighbor proposal, clamped)
///
/// At temperature T, the exact Boltzmann distribution is:
///   π(x) = exp(-(x - center)² / T) / Z(T)
///   Z(T) = Σ_{x=0}^{n-1} exp(-(x - center)² / T)
use crate::energy::Energy;
use crate::moves::MoveOperator;
use crate::rng::Rng;

/// 1D quadratic potential well energy function.
#[derive(Clone, Debug)]
pub struct PotentialWell {
    /// Number of discrete states.
    pub n: usize,
    /// Center of the quadratic well.
    pub center: f64,
}

impl PotentialWell {
    /// Create a potential well with `n` states, centered at `n/2`.
    pub fn new(n: usize) -> Self {
        assert!(n >= 2, "need at least 2 states");
        Self {
            n,
            center: n as f64 / 2.0,
        }
    }

    /// Override the well center position.
    pub fn with_center(mut self, center: f64) -> Self {
        self.center = center;
        self
    }

    /// Compute the exact Boltzmann distribution at temperature T.
    ///
    /// Returns a vector of probabilities π(x) for x = 0..n-1.
    pub fn exact_boltzmann(&self, temperature: f64) -> Vec<f64> {
        assert!(temperature > 0.0, "temperature must be positive");
        let log_weights: Vec<f64> = (0..self.n)
            .map(|x| {
                let dx = x as f64 - self.center;
                -(dx * dx) / temperature
            })
            .collect();

        // Log-sum-exp for numerical stability
        let max_lw = log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = log_weights.iter().map(|&lw| (lw - max_lw).exp()).sum();
        let log_z = max_lw + sum_exp.ln();

        log_weights
            .iter()
            .map(|&lw| (lw - log_z).exp())
            .collect()
    }

    /// Compute the exact partition function Z(T).
    pub fn partition_function(&self, temperature: f64) -> f64 {
        assert!(temperature > 0.0);
        (0..self.n)
            .map(|x| {
                let dx = x as f64 - self.center;
                (-(dx * dx) / temperature).exp()
            })
            .sum()
    }
}

impl Energy<i64> for PotentialWell {
    #[inline(always)]
    fn energy(&self, state: &i64) -> f64 {
        let dx = *state as f64 - self.center;
        dx * dx
    }
}

/// Symmetric ±1 neighbor move for the potential well.
///
/// At interior points, this is truly symmetric: Q(x→x+1) = Q(x→x-1) = 0.5.
/// At boundaries (x=0 or x=n-1), clamping means the proposal may stay put
/// instead of moving outside. For strict symmetry in H-01/H-02 tests,
/// use the reflecting variant that rejects out-of-bounds proposals instead.
#[derive(Clone, Debug)]
pub struct WellNeighborMove {
    /// Number of discrete states in the well.
    pub n: usize,
}

impl WellNeighborMove {
    /// Create a neighbor move for a well with `n` states.
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl MoveOperator<i64> for WellNeighborMove {
    fn propose(&self, state: &i64, rng: &mut impl Rng) -> i64 {
        let step = if rng.next_u64() & 1 == 0 { 1i64 } else { -1i64 };
        let candidate = *state + step;
        // Stay put at boundaries — this preserves symmetry.
        // Reflecting would make Q(0→1) = 1.0 ≠ Q(1→0) = 0.5.
        if candidate < 0 || candidate >= self.n as i64 {
            *state
        } else {
            candidate
        }
    }

    fn is_symmetric(&self) -> bool {
        true // stay-put boundaries preserve symmetry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boltzmann_sums_to_one() {
        let well = PotentialWell::new(20);
        for &t in &[0.5, 1.0, 2.0, 5.0, 10.0, 50.0] {
            let dist = well.exact_boltzmann(t);
            let sum: f64 = dist.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "Boltzmann distribution at T={} sums to {}",
                t,
                sum
            );
        }
    }

    #[test]
    fn boltzmann_peaks_at_center() {
        let well = PotentialWell::new(20);
        let dist = well.exact_boltzmann(1.0);
        let center = 10;
        // Center should have highest probability
        for (i, &p) in dist.iter().enumerate() {
            if i != center {
                assert!(
                    dist[center] >= p,
                    "center {} (p={}) should dominate state {} (p={})",
                    center,
                    dist[center],
                    i,
                    p
                );
            }
        }
    }

    #[test]
    fn high_temp_approaches_uniform() {
        let well = PotentialWell::new(20);
        let dist = well.exact_boltzmann(1e6);
        let uniform = 1.0 / 20.0;
        for &p in &dist {
            assert!(
                (p - uniform).abs() < 0.01,
                "at very high T, distribution should be near-uniform"
            );
        }
    }

    #[test]
    fn energy_quadratic() {
        let well = PotentialWell::new(20);
        assert_eq!(well.energy(&10), 0.0); // center
        assert_eq!(well.energy(&11), 1.0); // 1 away
        assert_eq!(well.energy(&8), 4.0); // 2 away
    }

    #[test]
    fn neighbor_move_stays_in_bounds() {
        let mv = WellNeighborMove::new(20);
        let mut rng = crate::rng::Xoshiro256PlusPlus::from_seed(42);
        let mut state = 0i64;
        for _ in 0..10000 {
            state = mv.propose(&state, &mut rng);
            assert!(state >= 0 && state < 20, "state out of bounds: {}", state);
        }
    }
}

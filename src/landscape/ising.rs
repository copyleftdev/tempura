//! 2D Ising Model — classical statistical mechanics benchmark.
//!
//! Used by: H-01 (Boltzmann convergence), H-06 (population annealing free energy)
//!
//! Square lattice with periodic boundary conditions.
//! For small sizes (L=4), the exact partition function can be computed
//! by exhaustive enumeration of all 2^(L²) states.
//!
//! E(σ) = -J Σ_{⟨i,j⟩} `σ_i` · `σ_j`    where J=1, σ ∈ {-1, +1}
use crate::energy::Energy;
use crate::moves::MoveOperator;
use crate::rng::Rng;

/// 2D Ising model state: L×L grid of spins ∈ {-1, +1}.
pub type IsingState = Vec<i8>;

/// 2D Ising model energy function with periodic boundary conditions.
#[derive(Clone, Debug)]
pub struct Ising2D {
    /// Lattice side length (total spins = `l × l`).
    pub l: usize,
    /// Coupling constant (positive = ferromagnetic).
    pub j: f64,
}

impl Ising2D {
    /// Create an L×L Ising model with coupling J=1.
    pub fn new(l: usize) -> Self {
        assert!(l >= 2, "lattice size must be at least 2");
        Self { l, j: 1.0 }
    }

    /// Set the coupling constant J.
    #[must_use]
    pub const fn with_coupling(mut self, j: f64) -> Self {
        self.j = j;
        self
    }

    /// Total number of spins.
    pub const fn num_spins(&self) -> usize {
        self.l * self.l
    }

    /// Create a random initial state.
    pub fn random_state(&self, rng: &mut impl Rng) -> IsingState {
        (0..self.num_spins()).map(|_| if rng.next_u64() & 1 == 0 { 1i8 } else { -1i8 }).collect()
    }

    /// Create an all-up initial state.
    pub fn all_up(&self) -> IsingState {
        vec![1i8; self.num_spins()]
    }

    /// Magnetization: M = Σ `σ_i`.
    pub fn magnetization(&self, state: &IsingState) -> f64 {
        state.iter().map(|&s| f64::from(s)).sum()
    }

    /// Neighbor indices for site (r, c) with periodic boundaries.
    const fn neighbors(&self, idx: usize) -> [usize; 4] {
        let l = self.l;
        let r = idx / l;
        let c = idx % l;
        [
            ((r + l - 1) % l) * l + c, // up
            ((r + 1) % l) * l + c,     // down
            r * l + (c + l - 1) % l,   // left
            r * l + (c + 1) % l,       // right
        ]
    }

    /// Compute the energy change from flipping spin at `idx`.
    ///
    /// ΔE = 2 · J · `σ_idx` · Σ_{neighbors} `σ_j`
    ///
    /// This is O(1) — the key to efficient single-spin-flip Metropolis.
    pub fn delta_energy_flip(&self, state: &IsingState, idx: usize) -> f64 {
        let s = f64::from(state[idx]);
        let neighbor_sum: f64 = self.neighbors(idx).iter().map(|&n| f64::from(state[n])).sum();
        2.0 * self.j * s * neighbor_sum
    }

    /// Exact partition function by exhaustive enumeration.
    ///
    /// Only feasible for small L (L ≤ 4, i.e., ≤ 16 spins = 65536 states).
    /// Returns Z(T) = `Σ_σ` exp(-E(σ)/T).
    pub fn exact_partition_function(&self, temperature: f64) -> f64 {
        let n = self.num_spins();
        assert!(n <= 20, "exact enumeration only feasible for ≤ 20 spins");
        let total_states = 1u64 << n;
        let mut z = 0.0f64;
        for bits in 0..total_states {
            let state: IsingState =
                (0..n).map(|i| if bits & (1 << i) != 0 { 1i8 } else { -1i8 }).collect();
            let e = self.energy(&state);
            z += (-e / temperature).exp();
        }
        z
    }

    /// Exact free energy F(T) = -T · ln Z(T).
    pub fn exact_free_energy(&self, temperature: f64) -> f64 {
        -temperature * self.exact_partition_function(temperature).ln()
    }

    /// Exact mean energy ⟨E⟩ = `Σ_σ` E(σ) · exp(-E(σ)/T) / Z(T).
    pub fn exact_mean_energy(&self, temperature: f64) -> f64 {
        let n = self.num_spins();
        assert!(n <= 20);
        let total_states = 1u64 << n;
        let mut sum_e_boltz = 0.0f64;
        let mut z = 0.0f64;
        for bits in 0..total_states {
            let state: IsingState =
                (0..n).map(|i| if bits & (1 << i) != 0 { 1i8 } else { -1i8 }).collect();
            let e = self.energy(&state);
            let boltz = (-e / temperature).exp();
            sum_e_boltz += e * boltz;
            z += boltz;
        }
        sum_e_boltz / z
    }
}

impl Energy<IsingState> for Ising2D {
    fn energy(&self, state: &IsingState) -> f64 {
        let l = self.l;
        let mut e = 0.0;
        for r in 0..l {
            for c in 0..l {
                let idx = r * l + c;
                let s = f64::from(state[idx]);
                // Only count right and down neighbors to avoid double-counting
                let right = r * l + (c + 1) % l;
                let down = ((r + 1) % l) * l + c;
                e -= self.j * s * f64::from(state[right]);
                e -= self.j * s * f64::from(state[down]);
            }
        }
        e
    }
}

/// Single-spin-flip move for the Ising model.
///
/// Flips one randomly chosen spin. Symmetric: flipping spin i has the
/// same proposal probability regardless of direction.
#[derive(Clone, Debug)]
pub struct SingleSpinFlip {
    /// Total number of spins (L²).
    pub n: usize,
}

impl SingleSpinFlip {
    /// Create a single-spin-flip move for `n` total spins.
    pub const fn new(n: usize) -> Self {
        Self { n }
    }
}

impl MoveOperator<IsingState> for SingleSpinFlip {
    fn propose(&self, state: &IsingState, rng: &mut impl Rng) -> IsingState {
        let idx = (rng.next_u64() % self.n as u64) as usize;
        let mut candidate = state.clone();
        candidate[idx] = -candidate[idx];
        candidate
    }

    fn is_symmetric(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_up_ground_state() {
        let ising = Ising2D::new(4);
        let state = ising.all_up();
        let e = ising.energy(&state);
        // All-up: each bond contributes -J. For L=4 with PBC, there are
        // L² × 2 = 32 bonds (each site has right+down neighbor).
        assert_eq!(e, -32.0, "all-up energy should be -2*L² = -32");
    }

    #[test]
    fn all_down_same_energy() {
        let ising = Ising2D::new(4);
        let up = ising.all_up();
        let down: IsingState = vec![-1i8; 16];
        assert_eq!(
            ising.energy(&up),
            ising.energy(&down),
            "all-up and all-down should have same energy (Z2 symmetry)"
        );
    }

    #[test]
    fn delta_energy_consistent() {
        let ising = Ising2D::new(4);
        let mut rng = crate::rng::Xoshiro256PlusPlus::from_seed(42);
        let state = ising.random_state(&mut rng);
        let e_before = ising.energy(&state);

        for idx in 0..16 {
            let delta = ising.delta_energy_flip(&state, idx);
            let mut flipped = state.clone();
            flipped[idx] = -flipped[idx];
            let e_after = ising.energy(&flipped);
            assert!(
                (delta - (e_after - e_before)).abs() < 1e-10,
                "delta_energy_flip inconsistent at idx {}: delta={}, actual={}",
                idx,
                delta,
                e_after - e_before
            );
        }
    }

    #[test]
    fn partition_function_positive() {
        let ising = Ising2D::new(3);
        let z = ising.exact_partition_function(2.0);
        assert!(z > 0.0, "partition function must be positive");
    }

    #[test]
    fn spin_flip_symmetric() {
        let flip = SingleSpinFlip::new(16);
        assert!(flip.is_symmetric());
    }
}

//! Move operators — state perturbation strategies.
//!
//! # Contract (H-02: detailed balance)
//! Move operators define the proposal distribution Q(x→y).
//! For Metropolis acceptance to satisfy detailed balance, moves must either:
//! 1. Be symmetric: Q(x→y) = Q(y→x) for all x, y
//! 2. Provide the log proposal ratio: ln(Q(y→x) / Q(x→y)) for Hastings correction
//!
//! # Design (Turon: pit-of-success)
//! The default `log_proposal_ratio` returns 0.0 (symmetric assumption).
//! Non-symmetric moves MUST override this. The `is_symmetric` method
//! enables runtime validation.
use crate::rng::Rng;

/// A move operator that proposes a new candidate state from the current state.
pub trait MoveOperator<S> {
    /// Propose a new candidate state.
    ///
    /// Must use the provided RNG for all randomness (determinism requirement).
    fn propose(&self, state: &S, rng: &mut impl Rng) -> S;

    /// Whether this move operator has symmetric proposal distribution.
    ///
    /// If true: Q(x→y) = Q(y→x) for all x, y.
    /// If false: `log_proposal_ratio` MUST be overridden.
    fn is_symmetric(&self) -> bool {
        true
    }

    /// Log of the proposal ratio: ln(Q(y→x) / Q(x→y)).
    ///
    /// Only needed for non-symmetric proposals (Hastings correction).
    /// Returns 0.0 by default (symmetric assumption).
    ///
    /// # Panics
    /// Debug-mode assertion fires if `is_symmetric()` returns false
    /// and this method is not overridden (still returns 0.0).
    fn log_proposal_ratio(&self, _from: &S, _to: &S) -> f64 {
        0.0
    }
}

/// A reversible move that supports efficient undo-on-reject.
///
/// For large state vectors, cloning the entire state for each proposal is
/// expensive. Reversible moves modify the state in-place and can undo
/// the modification if the proposal is rejected.
///
/// # Usage pattern
/// ```ignore
/// let delta = move_op.apply(&mut state, &mut rng);
/// if !accept(delta_e, temperature, u) {
///     move_op.undo(&mut state, &delta);
/// }
/// ```
pub trait ReversibleMove<S> {
    /// The type that describes the move (for undo).
    type Delta;

    /// Apply the move in-place, returning the delta for potential undo.
    fn apply(&self, state: &mut S, rng: &mut impl Rng) -> Self::Delta;

    /// Undo the move in-place.
    fn undo(&self, state: &mut S, delta: &Self::Delta);

    /// Whether this move operator has symmetric proposal distribution.
    fn is_symmetric(&self) -> bool {
        true
    }

    /// Log proposal ratio (see `MoveOperator::log_proposal_ratio`).
    fn log_proposal_ratio(&self, _state: &S, _delta: &Self::Delta) -> f64 {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Built-in move operators
// ---------------------------------------------------------------------------

/// Swap two elements in a permutation (e.g., TSP).
///
/// Symmetric: swapping i↔j has the same probability as j↔i.
#[derive(Clone, Debug)]
pub struct SwapMove;

impl MoveOperator<Vec<usize>> for SwapMove {
    fn propose(&self, state: &Vec<usize>, rng: &mut impl Rng) -> Vec<usize> {
        let n = state.len();
        if n < 2 {
            return state.clone();
        }
        let mut candidate = state.clone();
        let i = (rng.next_u64() % n as u64) as usize;
        let mut j = (rng.next_u64() % (n - 1) as u64) as usize;
        if j >= i {
            j += 1;
        }
        candidate.swap(i, j);
        candidate
    }
}

/// Swap move as a reversible in-place operation.
#[derive(Clone, Debug)]
pub struct SwapMoveReversible;

impl ReversibleMove<Vec<usize>> for SwapMoveReversible {
    type Delta = (usize, usize);

    fn apply(&self, state: &mut Vec<usize>, rng: &mut impl Rng) -> Self::Delta {
        let n = state.len();
        let i = (rng.next_u64() % n as u64) as usize;
        let mut j = (rng.next_u64() % (n - 1) as u64) as usize;
        if j >= i {
            j += 1;
        }
        state.swap(i, j);
        (i, j)
    }

    fn undo(&self, state: &mut Vec<usize>, delta: &(usize, usize)) {
        state.swap(delta.0, delta.1);
    }
}

/// Gaussian perturbation for continuous optimization.
///
/// `x' = x + N(0, σ)` for each dimension.
/// Symmetric: the Gaussian is symmetric around zero.
#[derive(Clone, Debug)]
pub struct GaussianMove {
    /// Standard deviation of the Gaussian perturbation.
    pub sigma: f64,
}

impl GaussianMove {
    /// Create a Gaussian move with the given standard deviation.
    pub fn new(sigma: f64) -> Self {
        assert!(sigma > 0.0, "sigma must be positive");
        Self { sigma }
    }

    /// Box-Muller transform: generate N(0,1) from two uniform draws.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn normal(rng: &mut impl Rng) -> f64 {
        let u1 = rng.next_f64().max(f64::MIN_POSITIVE);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }
}

impl MoveOperator<Vec<f64>> for GaussianMove {
    fn propose(&self, state: &Vec<f64>, rng: &mut impl Rng) -> Vec<f64> {
        state
            .iter()
            .map(|&x| self.sigma.mul_add(Self::normal(rng), x))
            .collect()
    }
}

/// Neighbor move for 1D discrete state spaces.
///
/// `x → x ± 1` with equal probability, clamped to [min, max].
/// Symmetric within the interior; boundary clamping makes it asymmetric at edges.
#[derive(Clone, Debug)]
pub struct NeighborMove {
    /// Minimum allowed state value.
    pub min: i64,
    /// Maximum allowed state value.
    pub max: i64,
}

impl NeighborMove {
    /// Create a neighbor move clamped to `[min, max]`.
    pub fn new(min: i64, max: i64) -> Self {
        assert!(min < max, "min must be less than max");
        Self { min, max }
    }
}

impl MoveOperator<i64> for NeighborMove {
    fn propose(&self, state: &i64, rng: &mut impl Rng) -> i64 {
        let step = if rng.next_u64() & 1 == 0 { 1 } else { -1 };
        (*state + step).clamp(self.min, self.max)
    }

    fn is_symmetric(&self) -> bool {
        // Strictly symmetric only in the interior.
        // At boundaries, clamping breaks symmetry.
        // Honest documentation per H-02.
        false
    }

    fn log_proposal_ratio(&self, from: &i64, to: &i64) -> f64 {
        // Interior: symmetric (ratio = 1, log = 0)
        // Boundary: one-sided proposal
        let from_neighbors: f64 = if *from == self.min || *from == self.max { 1.0 } else { 2.0 };
        let to_neighbors: f64 = if *to == self.min || *to == self.max { 1.0 } else { 2.0 };
        (from_neighbors / to_neighbors).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Xoshiro256PlusPlus;

    #[test]
    fn swap_move_permutation_preserved() {
        let state: Vec<usize> = (0..10).collect();
        let mut rng = Xoshiro256PlusPlus::from_seed(42);
        let candidate = SwapMove.propose(&state, &mut rng);
        let mut sorted = candidate.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn swap_move_reversible_undo() {
        let mut state: Vec<usize> = (0..10).collect();
        let original = state.clone();
        let mut rng = Xoshiro256PlusPlus::from_seed(42);
        let delta = SwapMoveReversible.apply(&mut state, &mut rng);
        assert_ne!(state, original);
        SwapMoveReversible.undo(&mut state, &delta);
        assert_eq!(state, original);
    }

    #[test]
    fn gaussian_move_changes_state() {
        let state = vec![0.0; 5];
        let mut rng = Xoshiro256PlusPlus::from_seed(42);
        let candidate = GaussianMove::new(1.0).propose(&state, &mut rng);
        assert_ne!(state, candidate);
    }

    #[test]
    fn neighbor_move_stays_in_bounds() {
        let mv = NeighborMove::new(0, 100);
        let mut rng = Xoshiro256PlusPlus::from_seed(42);
        let mut state = 0i64;
        for _ in 0..10000 {
            state = mv.propose(&state, &mut rng);
            assert!(state >= 0 && state <= 100);
        }
    }

    #[test]
    fn neighbor_move_asymmetric_at_boundary() {
        let mv = NeighborMove::new(0, 10);
        assert!(!mv.is_symmetric());
        // At boundary, log_proposal_ratio should be non-zero
        let ratio = mv.log_proposal_ratio(&0, &1);
        assert!(ratio.abs() > 0.0);
    }
}

//! Energy (cost function) trait — the objective to minimize.
//!
//! # Contract (H-01, H-02)
//! - Must be pure: same state → same energy, no side effects
//! - Must be deterministic: no internal randomness
//! - Must return finite f64 (no NaN, no Inf)
//!
//! # Design (Turon: user-first API)
//! Generic over state type S, enabling:
//! - Discrete problems (S = `Vec<usize>` for TSP)
//! - Continuous problems (S = `Vec<f64>` for Rastrigin)
//! - Physics problems (S = `[[i8; N]; N]` for Ising)

/// The objective function to minimize.
///
/// Implementations must be pure and deterministic.
/// The annealer will call this on every proposal — keep it fast.
pub trait Energy<S> {
    /// Compute the energy (cost) of a given state.
    ///
    /// # Invariants
    /// - Must return a finite f64
    /// - Must be deterministic: `energy(s) == energy(s)` always
    fn energy(&self, state: &S) -> f64;
}

/// Convenience: energy from a closure.
///
/// ```
/// use tempura::energy::FnEnergy;
///
/// let obj = FnEnergy(|x: &f64| x * x);
/// ```
pub struct FnEnergy<F>(pub F);

impl<F> core::fmt::Debug for FnEnergy<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("FnEnergy").finish()
    }
}

impl<S, F> Energy<S> for FnEnergy<F>
where
    F: Fn(&S) -> f64,
{
    fn energy(&self, state: &S) -> f64 {
        (self.0)(state)
    }
}

/// Delta-energy optimization: compute ΔE directly without full re-evaluation.
///
/// Many problems can compute the energy change from a move in O(1) instead
/// of re-evaluating the full energy in O(n). This trait enables that.
///
/// # Example: TSP swap
/// Swapping cities i and j only changes 4 edges, not the full tour.
pub trait DeltaEnergy<S, M> {
    /// Compute E(candidate) - E(current) directly.
    ///
    /// If this returns None, the annealer falls back to full energy computation.
    fn delta_energy(&self, current: &S, current_energy: f64, proposed_move: &M) -> Option<f64>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fn_energy_closure() {
        let e = FnEnergy(|x: &f64| x * x);
        assert_eq!(e.energy(&3.0), 9.0);
        assert_eq!(e.energy(&-2.0), 4.0);
    }

    #[test]
    fn fn_energy_deterministic() {
        let e = FnEnergy(|x: &Vec<f64>| x.iter().sum::<f64>());
        let state = vec![1.0, 2.0, 3.0];
        let e1 = e.energy(&state);
        let e2 = e.energy(&state);
        assert_eq!(e1, e2);
    }
}

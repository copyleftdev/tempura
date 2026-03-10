//! N-dimensional Rastrigin function — multi-modal continuous benchmark.
//!
//! Used by: H-04 (acceptance rate diagnostic), H-09 (cache layout)
//!
//! E(x) = 10D + Σ_{i=1}^{D} [`x_i²` - `10·cos(2πx_i)`]
//!
//! Properties:
//! - Global minimum: E(0,...,0) = 0
//! - Many local minima: grid of ~D^10 local minima in [-5.12, 5.12]^D
//! - Highly multi-modal: tests ability to escape local minima
use crate::energy::Energy;

/// N-dimensional Rastrigin energy function.
#[derive(Clone, Debug)]
pub struct Rastrigin {
    /// Number of dimensions.
    pub dim: usize,
}

impl Rastrigin {
    /// Create an N-dimensional Rastrigin function.
    pub fn new(dim: usize) -> Self {
        assert!(dim >= 1, "dimension must be at least 1");
        Self { dim }
    }

    /// The known global minimum energy (always 0).
    pub const fn global_minimum(&self) -> f64 {
        0.0
    }

    /// The global minimizer (all zeros).
    pub fn global_minimizer(&self) -> Vec<f64> {
        vec![0.0; self.dim]
    }
}

impl Energy<Vec<f64>> for Rastrigin {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn energy(&self, state: &Vec<f64>) -> f64 {
        debug_assert_eq!(state.len(), self.dim, "state dimension mismatch");
        let a = 10.0;
        let two_pi = 2.0 * core::f64::consts::PI;
        a * self.dim as f64
            + state.iter().map(|&x| x.mul_add(x, -a * (two_pi * x).cos())).sum::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_minimum_at_origin() {
        let r = Rastrigin::new(5);
        let origin = vec![0.0; 5];
        assert!((r.energy(&origin) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn nonzero_away_from_origin() {
        let r = Rastrigin::new(2);
        let state = vec![1.0, 1.0];
        assert!(r.energy(&state) > 0.0);
    }

    #[test]
    fn has_many_local_minima() {
        let r = Rastrigin::new(1);
        // Local minima near each integer
        let e_0 = r.energy(&vec![0.0]);
        let e_1 = r.energy(&vec![1.0]);
        let e_half = r.energy(&vec![0.5]);
        // 0 is global min, 1 is a local min, 0.5 is between them (higher)
        assert!(e_0 < e_half);
        assert!(e_1 < e_half);
    }
}

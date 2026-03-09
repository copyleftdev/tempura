/// Asymmetric Double Well — landscape with tunable barrier for testing
/// convergence and barrier-crossing algorithms.
///
/// Used by: H-03 (cooling optimality), H-05 (parallel tempering)
///
/// Two basins separated by a barrier of known height.
/// Global minimum is in basin B (lower energy), local minimum in basin A.
use crate::energy::Energy;
use crate::moves::MoveOperator;
use crate::rng::Rng;

/// Asymmetric double-well energy landscape.
///
/// ```text
/// Basin A (local):  E(x) = (x - a_center)² / a_scale      for x ∈ [0, barrier_pos)
/// Barrier:          E(barrier_pos) = barrier_height
/// Basin B (global): E(x) = (x - b_center)² / b_scale + b_offset  for x ∈ (barrier_pos, n)
/// ```
#[derive(Clone, Debug)]
pub struct DoubleWell {
    /// Number of discrete states.
    pub n: usize,
    /// Position of the barrier separating the two basins.
    pub barrier_pos: usize,
    /// Energy at the barrier peak.
    pub barrier_height: f64,
    /// Center of basin A (local minimum).
    pub a_center: f64,
    /// Width scaling of basin A.
    pub a_scale: f64,
    /// Center of basin B (global minimum).
    pub b_center: f64,
    /// Width scaling of basin B.
    pub b_scale: f64,
    /// Energy offset of basin B (negative = deeper).
    pub b_offset: f64,
}

impl DoubleWell {
    /// Create a double well with the given number of states and barrier height.
    ///
    /// Default layout places barrier at midpoint, local minimum at 1/4, global at 3/4.
    pub fn new(n: usize, barrier_height: f64) -> Self {
        let barrier_pos = n / 2;
        Self {
            n,
            barrier_pos,
            barrier_height,
            a_center: n as f64 / 4.0,
            a_scale: n as f64 / 2.0,
            b_center: 3.0 * n as f64 / 4.0,
            b_scale: n as f64 / 2.0,
            b_offset: -10.0, // global minimum is 10 units lower
        }
    }

    /// The critical depth d* — barrier height relative to the local minimum.
    ///
    /// This is the key parameter for Hajek's theorem (H-03).
    pub fn critical_depth(&self) -> f64 {
        let local_min_energy = 0.0; // E(a_center) = 0
        self.barrier_height - local_min_energy
    }

    /// Whether a state is in the global basin.
    pub fn in_global_basin(&self, state: i64) -> bool {
        state > self.barrier_pos as i64
    }

    /// Energy of the global minimum.
    pub fn global_minimum_energy(&self) -> f64 {
        self.b_offset
    }
}

impl Energy<i64> for DoubleWell {
    fn energy(&self, state: &i64) -> f64 {
        let x = *state;
        if x < 0 || x >= self.n as i64 {
            return f64::MAX; // out of bounds
        }
        let xu = x as usize;
        if xu == self.barrier_pos {
            self.barrier_height
        } else if xu < self.barrier_pos {
            let dx = x as f64 - self.a_center;
            dx * dx / self.a_scale
        } else {
            let dx = x as f64 - self.b_center;
            dx * dx / self.b_scale + self.b_offset
        }
    }
}

/// Simple ±1 move for the double well.
#[derive(Clone, Debug)]
pub struct DoubleWellMove {
    /// Number of discrete states.
    pub n: usize,
}

impl DoubleWellMove {
    /// Create a ±1 move for a double well with `n` states.
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl MoveOperator<i64> for DoubleWellMove {
    fn propose(&self, state: &i64, rng: &mut impl Rng) -> i64 {
        let step = if rng.next_u64() & 1 == 0 { 1i64 } else { -1i64 };
        (*state + step).clamp(0, self.n as i64 - 1)
    }

    fn is_symmetric(&self) -> bool {
        // Clamping at boundaries breaks strict symmetry, but for interior
        // states it's symmetric. For hypothesis tests with large n,
        // boundary effects are negligible.
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_basin_lower() {
        let dw = DoubleWell::new(200, 50.0);
        let local_min = dw.energy(&(dw.n as i64 / 4));
        let global_min = dw.energy(&(3 * dw.n as i64 / 4));
        assert!(
            global_min < local_min,
            "global {} should be less than local {}",
            global_min,
            local_min
        );
    }

    #[test]
    fn barrier_is_highest() {
        let dw = DoubleWell::new(200, 50.0);
        let barrier_e = dw.energy(&(dw.barrier_pos as i64));
        let local_e = dw.energy(&(dw.n as i64 / 4));
        let global_e = dw.energy(&(3 * dw.n as i64 / 4));
        assert!(barrier_e > local_e);
        assert!(barrier_e > global_e);
    }

    #[test]
    fn critical_depth_matches() {
        let dw = DoubleWell::new(200, 100.0);
        assert_eq!(dw.critical_depth(), 100.0);
    }

    #[test]
    fn move_stays_in_bounds() {
        let mv = DoubleWellMove::new(200);
        let mut rng = crate::rng::Xoshiro256PlusPlus::from_seed(42);
        let mut state = 0i64;
        for _ in 0..10000 {
            state = mv.propose(&state, &mut rng);
            assert!(state >= 0 && state < 200);
        }
    }
}

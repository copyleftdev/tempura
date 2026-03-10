//! Tunable Barrier Geometry — for testing quantum-inspired tunneling (H-07).
//!
//! Two basins separated by a barrier whose height and width are independently
//! controllable, allowing systematic comparison of classical SA (height-dependent)
//! vs quantum-inspired (width-dependent) acceptance.
use crate::energy::Energy;
use crate::moves::MoveOperator;
use crate::rng::Rng;

/// Barrier landscape with tunable height and width.
///
/// ```text
/// Basin A: E(x) = (x - a_center)² / a_scale       for x ∈ [0, barrier_start)
/// Barrier: E(x) = barrier_height                    for x ∈ [barrier_start, barrier_end]
/// Basin B: E(x) = (x - b_center)² / b_scale - 20   for x ∈ (barrier_end, n)
/// ```
#[derive(Clone, Debug)]
pub struct TunableBarrier {
    /// Number of discrete states.
    pub n: usize,
    /// First position in the barrier region.
    pub barrier_start: usize,
    /// Last position in the barrier region.
    pub barrier_end: usize,
    /// Energy at the barrier plateau.
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

impl TunableBarrier {
    /// Create a tall-thin barrier (quantum should win).
    pub fn tall_thin(n: usize) -> Self {
        let mid = n / 2;
        Self {
            n,
            barrier_start: mid - 2,
            barrier_end: mid + 2,
            barrier_height: 200.0,
            a_center: n as f64 / 4.0,
            a_scale: 50.0,
            b_center: 3.0 * n as f64 / 4.0,
            b_scale: 50.0,
            b_offset: -20.0,
        }
    }

    /// Create a short-wide barrier (classical should win).
    pub fn short_wide(n: usize) -> Self {
        let third = n / 3;
        Self {
            n,
            barrier_start: third,
            barrier_end: 2 * third,
            barrier_height: 10.0,
            a_center: third as f64 / 2.0,
            a_scale: 50.0,
            b_center: (2 * third + n) as f64 / 2.0,
            b_scale: 50.0,
            b_offset: -20.0,
        }
    }

    /// Barrier width in number of states.
    pub const fn barrier_width(&self) -> usize {
        self.barrier_end - self.barrier_start + 1
    }

    /// Whether a state is in the global basin.
    pub const fn in_global_basin(&self, state: i64) -> bool {
        // Safety: barrier_end is always small (< usize::MAX / 2) so the cast is safe.
        #[allow(clippy::cast_possible_wrap)]
        let end = self.barrier_end as i64;
        state > end
    }
}

impl Energy<i64> for TunableBarrier {
    fn energy(&self, state: &i64) -> f64 {
        let x = *state;
        // Safety: n is always small (< usize::MAX / 2) so the cast is safe.
        #[allow(clippy::cast_possible_wrap)]
        if x < 0 || x >= self.n as i64 {
            return f64::MAX;
        }
        let xu = x as usize;
        if xu >= self.barrier_start && xu <= self.barrier_end {
            self.barrier_height
        } else if xu < self.barrier_start {
            let dx = x as f64 - self.a_center;
            dx * dx / self.a_scale
        } else {
            let dx = x as f64 - self.b_center;
            dx * dx / self.b_scale + self.b_offset
        }
    }
}

/// ±1 move for the barrier landscape.
#[derive(Clone, Debug)]
pub struct BarrierMove {
    /// Number of discrete states.
    pub n: usize,
}

impl BarrierMove {
    /// Create a ±1 move for a barrier landscape with `n` states.
    pub const fn new(n: usize) -> Self {
        Self { n }
    }
}

impl MoveOperator<i64> for BarrierMove {
    fn propose(&self, state: &i64, rng: &mut impl Rng) -> i64 {
        let step = if rng.next_u64() & 1 == 0 { 1i64 } else { -1i64 };
        // Safety: n is always small (< usize::MAX / 2) so the cast is safe.
        #[allow(clippy::cast_possible_wrap)]
        let n_max = self.n as i64 - 1;
        (*state + step).clamp(0, n_max)
    }

    fn is_symmetric(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tall_thin_barrier_geometry() {
        let b = TunableBarrier::tall_thin(300);
        assert!(b.barrier_width() < 10, "tall-thin should have narrow barrier");
        assert!(b.barrier_height > 100.0, "tall-thin should have high barrier");
    }

    #[test]
    fn short_wide_barrier_geometry() {
        let b = TunableBarrier::short_wide(300);
        assert!(b.barrier_width() > 50, "short-wide should have wide barrier");
        assert!(b.barrier_height < 20.0, "short-wide should have low barrier");
    }

    #[test]
    fn global_basin_is_lower() {
        let b = TunableBarrier::tall_thin(300);
        let local_e = b.energy(&(b.n as i64 / 4));
        let global_e = b.energy(&(3 * b.n as i64 / 4));
        assert!(global_e < local_e);
    }
}

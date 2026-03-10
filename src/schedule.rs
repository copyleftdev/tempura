//! Cooling schedules — temperature as a function of iteration.
//!
//! # Contract (H-03, H-04)
//! - `temperature(step)` must return a positive finite f64 for all step values
//! - Non-adaptive schedules must be monotonically non-increasing
//! - Logarithmic schedule is the only one with convergence guarantee (Hajek 1988)
//!
//! # Design (Turon: composability)
//! All schedules implement the same trait, enabling drop-in replacement
//! in the Annealer builder.

/// A cooling schedule that maps iteration step → temperature.
///
/// # Invariant
/// `temperature(k) > 0.0` for all `k >= 0`.
pub trait CoolingSchedule {
    /// Return the temperature at the given step.
    fn temperature(&self, step: u64) -> f64;

    /// Whether this schedule is monotonically non-increasing.
    ///
    /// Adaptive schedules return false (temperature may increase).
    fn is_monotonic(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Linear: T_k = T0 - α·k  (clamped to T_min)
// ---------------------------------------------------------------------------

/// Linear cooling: `T_k = T0 - α·k`, clamped to `T_min > 0`.
///
/// Fast cooling but reaches minimum temperature in finite steps.
/// No convergence guarantee.
#[derive(Clone, Debug)]
pub struct Linear {
    /// Initial temperature.
    pub t0: f64,
    /// Cooling rate (temperature decreases by `alpha` per step).
    pub alpha: f64,
    /// Minimum temperature floor (always positive).
    pub t_min: f64,
}

impl Linear {
    /// Create a linear schedule: `T_k = T0 - α·k`.
    pub fn new(t0: f64, alpha: f64) -> Self {
        assert!(t0 > 0.0, "initial temperature must be positive");
        assert!(alpha > 0.0, "cooling rate must be positive");
        Self {
            t0,
            alpha,
            t_min: 1e-10,
        }
    }

    /// Set the minimum temperature floor.
    #[must_use]
    pub fn with_t_min(mut self, t_min: f64) -> Self {
        assert!(t_min > 0.0, "minimum temperature must be positive");
        self.t_min = t_min;
        self
    }
}

impl CoolingSchedule for Linear {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn temperature(&self, step: u64) -> f64 {
        self.alpha.mul_add(-(step as f64), self.t0).max(self.t_min)
    }
}

// ---------------------------------------------------------------------------
// Exponential: T_k = T0 · α^k
// ---------------------------------------------------------------------------

/// Exponential cooling: `T_k = T0 · α^k` where `0 < α < 1`.
///
/// The most commonly used schedule in practice. Geometrically decreasing
/// temperature. No convergence guarantee (H-03), but fast in practice.
#[derive(Clone, Debug)]
pub struct Exponential {
    /// Initial temperature.
    pub t0: f64,
    /// Decay factor per step (`0 < α < 1`).
    pub alpha: f64,
    /// Minimum temperature floor (always positive).
    pub t_min: f64,
}

impl Exponential {
    /// Create an exponential schedule: `T_k = T0 · α^k`.
    pub fn new(t0: f64, alpha: f64) -> Self {
        assert!(t0 > 0.0, "initial temperature must be positive");
        assert!((0.0..1.0).contains(&alpha), "alpha must be in (0, 1)");
        Self {
            t0,
            alpha,
            t_min: 1e-10,
        }
    }

    /// Set the minimum temperature floor.
    #[must_use]
    pub fn with_t_min(mut self, t_min: f64) -> Self {
        assert!(t_min > 0.0, "minimum temperature must be positive");
        self.t_min = t_min;
        self
    }
}

impl CoolingSchedule for Exponential {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn temperature(&self, step: u64) -> f64 {
        (self.t0 * self.alpha.powi(step as i32)).max(self.t_min)
    }
}

// ---------------------------------------------------------------------------
// Logarithmic: T_k = c / ln(1 + k)  (Hajek-optimal)
// ---------------------------------------------------------------------------

/// Logarithmic cooling: `T_k = c / ln(1 + k)`.
///
/// **The only schedule with a convergence guarantee** (Hajek 1988, H-03).
/// Converges to global optimum with probability 1 when `c >= d*` where
/// `d*` is the critical depth of the deepest non-global local minimum.
///
/// Impractically slow for most real problems — use for correctness
/// validation and as a theoretical baseline.
#[derive(Clone, Debug)]
pub struct Logarithmic {
    /// Constant controlling cooling rate. For convergence, `c ≥ d*`.
    pub c: f64,
    /// Minimum temperature floor (always positive).
    pub t_min: f64,
}

impl Logarithmic {
    /// Create a logarithmic schedule with constant `c`.
    ///
    /// For convergence guarantee, `c` should be ≥ the critical depth `d*`.
    pub fn new(c: f64) -> Self {
        assert!(c > 0.0, "constant c must be positive");
        Self { c, t_min: 1e-10 }
    }

    /// Set the minimum temperature floor.
    #[must_use]
    pub fn with_t_min(mut self, t_min: f64) -> Self {
        assert!(t_min > 0.0, "minimum temperature must be positive");
        self.t_min = t_min;
        self
    }
}

impl CoolingSchedule for Logarithmic {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn temperature(&self, step: u64) -> f64 {
        if step == 0 {
            return self.c; // ln(1) = 0, avoid division by zero
        }
        (self.c / (step as f64).ln_1p()).max(self.t_min)
    }
}

// ---------------------------------------------------------------------------
// Fast: T_k = T0 / (1 + k)
// ---------------------------------------------------------------------------

/// Fast cooling: `T_k = T0 / (1 + k)`.
///
/// Cauchy-distribution inspired. Faster convergence than logarithmic,
/// better early exploration than exponential. No convergence guarantee.
#[derive(Clone, Debug)]
pub struct Fast {
    /// Initial temperature.
    pub t0: f64,
    /// Minimum temperature floor (always positive).
    pub t_min: f64,
}

impl Fast {
    /// Create a fast schedule: `T_k = T0 / (1 + k)`.
    pub fn new(t0: f64) -> Self {
        assert!(t0 > 0.0, "initial temperature must be positive");
        Self { t0, t_min: 1e-10 }
    }

    /// Set the minimum temperature floor.
    #[must_use]
    pub fn with_t_min(mut self, t_min: f64) -> Self {
        assert!(t_min > 0.0);
        self.t_min = t_min;
        self
    }
}

impl CoolingSchedule for Fast {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn temperature(&self, step: u64) -> f64 {
        (self.t0 / (1.0 + step as f64)).max(self.t_min)
    }
}

// ---------------------------------------------------------------------------
// Cauchy: T_k = T0 / (1 + k²)
// ---------------------------------------------------------------------------

/// Cauchy cooling: `T_k = T0 / (1 + k²)`.
///
/// Very aggressive cooling. Drops faster than Fast schedule.
/// Useful for problems where early exploration is sufficient.
#[derive(Clone, Debug)]
pub struct Cauchy {
    /// Initial temperature.
    pub t0: f64,
    /// Minimum temperature floor (always positive).
    pub t_min: f64,
}

impl Cauchy {
    /// Create a Cauchy schedule: `T_k = T0 / (1 + k²)`.
    pub fn new(t0: f64) -> Self {
        assert!(t0 > 0.0);
        Self { t0, t_min: 1e-10 }
    }

    /// Set the minimum temperature floor.
    #[must_use]
    pub fn with_t_min(mut self, t_min: f64) -> Self {
        assert!(t_min > 0.0);
        self.t_min = t_min;
        self
    }
}

impl CoolingSchedule for Cauchy {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn temperature(&self, step: u64) -> f64 {
        let k = step as f64;
        (self.t0 / k.mul_add(k, 1.0)).max(self.t_min)
    }
}

// ---------------------------------------------------------------------------
// Adaptive: temperature adjusts to maintain target acceptance rate
// ---------------------------------------------------------------------------

/// Adaptive cooling: temperature adjusts based on observed acceptance rate.
///
/// ```text
/// T_{k+1} = T_k · exp(γ · (r_k - r_target))
/// ```
///
/// where `r_k` is the acceptance rate over the last `window` proposals.
///
/// # Properties (H-04)
/// - NOT monotonic (temperature may increase if acceptance drops)
/// - Self-tuning — no schedule parameters to hand-tune
/// - Targets acceptance rate window for optimal exploration/exploitation balance
///
/// # Stability
/// Temperature is clamped to `[t_min, t_max]` to prevent runaway.
#[derive(Clone, Debug)]
pub struct Adaptive {
    /// Target acceptance rate in `(0, 1)`.
    pub target_rate: f64,
    /// Learning rate for temperature adjustment.
    pub gamma: f64,
    /// Number of proposals per temperature update.
    pub window: usize,
    /// Minimum temperature clamp.
    pub t_min: f64,
    /// Maximum temperature clamp.
    pub t_max: f64,
    // Mutable state
    current_t: f64,
    accept_count: u64,
    total_count: u64,
    window_accept: u64,
    window_total: u64,
}

impl Adaptive {
    /// Create an adaptive schedule targeting a given acceptance rate.
    pub fn new(initial_t: f64, target_rate: f64) -> Self {
        assert!(initial_t > 0.0, "initial temperature must be positive");
        assert!(
            (0.0..1.0).contains(&target_rate),
            "target rate must be in (0, 1)"
        );
        Self {
            target_rate,
            gamma: 1.0,
            window: 500,
            t_min: 1e-10,
            t_max: 1e10,
            current_t: initial_t,
            accept_count: 0,
            total_count: 0,
            window_accept: 0,
            window_total: 0,
        }
    }

    /// Set the learning rate γ for temperature adjustment.
    #[must_use]
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        assert!(gamma > 0.0, "gamma must be positive");
        self.gamma = gamma;
        self
    }

    /// Set the window size (proposals between temperature updates).
    #[must_use]
    pub fn with_window(mut self, window: usize) -> Self {
        assert!(window > 0, "window must be positive");
        self.window = window;
        self
    }

    /// Set temperature bounds `[t_min, t_max]` to prevent runaway.
    #[must_use]
    pub fn with_bounds(mut self, t_min: f64, t_max: f64) -> Self {
        assert!(t_min > 0.0 && t_max > t_min);
        self.t_min = t_min;
        self.t_max = t_max;
        self
    }

    /// Record an acceptance decision and update temperature if window is full.
    pub fn record(&mut self, accepted: bool) {
        self.total_count += 1;
        self.window_total += 1;
        if accepted {
            self.accept_count += 1;
            self.window_accept += 1;
        }

        if self.window_total >= self.window as u64 {
            let rate = self.window_accept as f64 / self.window_total as f64;
            self.current_t *= (self.gamma * (self.target_rate - rate)).exp();
            self.current_t = self.current_t.clamp(self.t_min, self.t_max);
            self.window_accept = 0;
            self.window_total = 0;
        }
    }

    /// Current acceptance rate over the lifetime of the schedule.
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            self.accept_count as f64 / self.total_count as f64
        }
    }

    /// Current temperature.
    pub const fn current_temperature(&self) -> f64 {
        self.current_t
    }
}

impl CoolingSchedule for Adaptive {
    fn temperature(&self, _step: u64) -> f64 {
        self.current_t
    }

    fn is_monotonic(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_monotonic() {
        let s = Linear::new(100.0, 0.1);
        let mut prev = s.temperature(0);
        for k in 1..1000 {
            let t = s.temperature(k);
            assert!(t <= prev, "step {}: {} > {}", k, t, prev);
            assert!(t > 0.0, "step {}: temperature not positive", k);
            prev = t;
        }
    }

    #[test]
    fn exponential_monotonic() {
        let s = Exponential::new(1000.0, 0.995);
        let mut prev = s.temperature(0);
        for k in 1..10000 {
            let t = s.temperature(k);
            assert!(t <= prev);
            assert!(t > 0.0);
            prev = t;
        }
    }

    #[test]
    fn logarithmic_slow_convergence() {
        let s = Logarithmic::new(100.0);
        // At step 10^6, temperature should still be significant
        let t = s.temperature(1_000_000);
        assert!(t > 1.0, "logarithmic should still be warm at 10^6: {}", t);
    }

    #[test]
    fn fast_vs_cauchy_ordering() {
        let fast = Fast::new(100.0);
        let cauchy = Cauchy::new(100.0);
        // Cauchy cools faster than Fast for k > 1
        for k in 2..1000 {
            assert!(
                cauchy.temperature(k) <= fast.temperature(k),
                "Cauchy should be cooler at step {}",
                k
            );
        }
    }

    #[test]
    fn adaptive_increases_on_low_acceptance() {
        let mut s = Adaptive::new(1.0, 0.4);
        let initial_t = s.current_temperature();
        // Record only rejections for one window
        for _ in 0..500 {
            s.record(false);
        }
        assert!(
            s.current_temperature() > initial_t,
            "temperature should increase when acceptance is below target"
        );
    }

    #[test]
    fn adaptive_decreases_on_high_acceptance() {
        let mut s = Adaptive::new(100.0, 0.4);
        let initial_t = s.current_temperature();
        // Record only acceptances for one window
        for _ in 0..500 {
            s.record(true);
        }
        assert!(
            s.current_temperature() < initial_t,
            "temperature should decrease when acceptance is above target"
        );
    }

    #[test]
    fn all_schedules_positive() {
        let schedules: Vec<Box<dyn CoolingSchedule>> = vec![
            Box::new(Linear::new(100.0, 0.01)),
            Box::new(Exponential::new(100.0, 0.999)),
            Box::new(Logarithmic::new(100.0)),
            Box::new(Fast::new(100.0)),
            Box::new(Cauchy::new(100.0)),
        ];
        for (i, s) in schedules.iter().enumerate() {
            for k in 0..100_000 {
                let t = s.temperature(k);
                assert!(t > 0.0, "schedule {} at step {}: T={}", i, k, t);
                assert!(t.is_finite(), "schedule {} at step {}: T={}", i, k, t);
            }
        }
    }
}

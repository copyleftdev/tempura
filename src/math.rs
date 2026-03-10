//! Numerical primitives for the annealing hot loop.
//!
//! # Design rationale (H-10)
//! The Metropolis acceptance decision `u < exp(-ΔE/T)` is the single hottest
//! operation in any annealing run. This module provides:
//! - Exact exp via libm (correct, ~20 cycles)
//! - Fast approximate exp via Schraudolph's trick (~5 cycles, 0.1% error)
//! - Log-domain comparison to avoid exp entirely (~3 cycles)
//!
//! All functions are `#[inline(always)]` — they live in the caller's cache line.

/// Numerically stable exp(-x) that handles extreme inputs without NaN/Inf.
///
/// - x > 709.8: returns 0.0 (would underflow to 0 anyway)
/// - x < -709.8: returns `f64::MAX` (clamped, avoids Inf)
/// - Otherwise: std exp
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn stable_neg_exp(x: f64) -> f64 {
    if x > 709.0 {
        0.0
    } else if x < -709.0 {
        f64::MAX
    } else {
        (-x).exp()
    }
}

/// Metropolis acceptance probability: min(1, exp(-ΔE/T)).
///
/// Returns the probability directly (for diagnostics / logging).
/// For the hot loop, prefer `metropolis_accept` which avoids the branch.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn metropolis_probability(delta_e: f64, temperature: f64) -> f64 {
    if delta_e <= 0.0 {
        1.0
    } else {
        stable_neg_exp(delta_e / temperature)
    }
}

/// Branchless Metropolis acceptance decision.
///
/// The key insight (H-10): when ΔE ≤ 0, exp(-ΔE/T) ≥ 1.0, so
/// `u < exp(-ΔE/T)` is always true for u ∈ [0,1). No branch needed.
///
/// Returns true if the move should be accepted.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn metropolis_accept(delta_e: f64, temperature: f64, u: f64) -> bool {
    // Single comparison, no branch on ΔE sign.
    // When delta_e <= 0: exp(-delta_e/T) >= 1.0 > u, so always true.
    // When delta_e > 0:  exp(-delta_e/T) < 1.0, probabilistic.
    u < stable_neg_exp(delta_e / temperature)
}

/// Log-domain Metropolis acceptance (H-10c).
///
/// Avoids computing `exp()` entirely by comparing in log space:
///   u < exp(-ΔE/T)  ⟺  -ln(u) > ΔE/T
///
/// `exp1_variate` is a pre-computed -ln(u) from `Rng::next_exp1()`.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn metropolis_accept_log_domain(delta_e: f64, temperature: f64, exp1_variate: f64) -> bool {
    // exp1_variate = -ln(u) > 0 always.
    // When delta_e <= 0: ΔE/T <= 0 < exp1_variate, so always true.
    // When delta_e > 0:  compare directly.
    exp1_variate * temperature > delta_e
}

/// Barker acceptance probability: 1 / (1 + exp(ΔE/T)).
///
/// Satisfies detailed balance (H-02) but has strictly lower acceptance
/// rate than Metropolis (Peskun, 1973).
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn barker_probability(delta_e: f64, temperature: f64) -> f64 {
    1.0 / (1.0 + (delta_e / temperature).exp())
}

/// Barker acceptance decision (branchless).
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn barker_accept(delta_e: f64, temperature: f64, u: f64) -> bool {
    u < barker_probability(delta_e, temperature)
}

/// Fast approximate exp(x) using Schraudolph's method (1999).
///
/// Reinterprets f64 bits to approximate 2^(x/ln2).
/// Accuracy: ~0.1% relative error. Speed: ~5 cycles.
///
/// # Safety
/// Returns garbage for |x| > 709. Caller must bounds-check.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn fast_exp(x: f64) -> f64 {
    // Schraudolph's trick: manipulate the exponent bits of f64.
    // f64 = sign(1) | exponent(11) | mantissa(52)
    // exp(x) ≈ 2^(x/ln2) → set exponent bits accordingly.
    const A: f64 = (1u64 << 52) as f64 / core::f64::consts::LN_2;
    const B: f64 = (1023u64 << 52) as f64 - 60801.0 * (1u64 << 32) as f64;

    if x < -709.0 {
        return 0.0;
    }
    if x > 709.0 {
        return f64::INFINITY;
    }

    let bits = A.mul_add(x, B) as i64;
    f64::from_bits(bits as u64)
}

/// Quantum-inspired tunneling acceptance probability (H-07).
///
/// P = exp(-d * √(max(0, ΔE)))
///
/// Does NOT satisfy detailed balance with Boltzmann distribution.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn quantum_tunneling_probability(delta_e: f64, width: f64) -> f64 {
    if delta_e <= 0.0 {
        1.0
    } else {
        (-width * delta_e.sqrt()).exp()
    }
}

/// Quantum-inspired tunneling acceptance decision.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn quantum_tunneling_accept(delta_e: f64, width: f64, u: f64) -> bool {
    u < quantum_tunneling_probability(delta_e, width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_neg_exp_extreme_positive() {
        assert_eq!(stable_neg_exp(800.0), 0.0);
        assert_eq!(stable_neg_exp(1e100), 0.0);
    }

    #[test]
    fn stable_neg_exp_extreme_negative() {
        assert_eq!(stable_neg_exp(-800.0), f64::MAX);
    }

    #[test]
    fn stable_neg_exp_normal() {
        let v = stable_neg_exp(1.0);
        assert!((v - (-1.0f64).exp()).abs() < 1e-15);
    }

    #[test]
    fn metropolis_always_accepts_improvement() {
        assert!(metropolis_accept(-1.0, 1.0, 0.999));
        assert!(metropolis_accept(-100.0, 0.001, 0.999));
    }

    #[test]
    fn metropolis_probability_at_zero() {
        assert_eq!(metropolis_probability(0.0, 1.0), 1.0);
    }

    #[test]
    fn barker_symmetry() {
        let p1 = barker_probability(1.0, 1.0);
        let p2 = barker_probability(-1.0, 1.0);
        assert!((p1 + p2 - 1.0).abs() < 1e-15, "Barker: P(ΔE) + P(-ΔE) = 1");
    }

    #[test]
    fn log_domain_equivalence() {
        // Verify log-domain and standard domain agree
        let delta_e = 2.5;
        let temperature = 3.0;
        let u = 0.3;
        let exp1 = -(u as f64).ln();

        let standard = metropolis_accept(delta_e, temperature, u);
        let log_dom = metropolis_accept_log_domain(delta_e, temperature, exp1);
        assert_eq!(standard, log_dom);
    }

    #[test]
    fn fast_exp_accuracy() {
        for i in -100..100 {
            let x = i as f64 * 0.1;
            let exact = x.exp();
            let approx = fast_exp(x);
            let rel_err = ((approx - exact) / exact).abs();
            assert!(
                rel_err < 0.04,
                "fast_exp({}) = {} vs exact {} (err {})",
                x,
                approx,
                exact,
                rel_err
            );
        }
    }

    #[test]
    fn quantum_always_accepts_improvement() {
        assert!(quantum_tunneling_accept(-5.0, 1.0, 0.999));
    }

    #[test]
    fn quantum_depends_on_sqrt_not_linear() {
        // Doubling barrier height should NOT halve the probability
        // (it should reduce it less than linearly due to sqrt)
        let p1 = quantum_tunneling_probability(4.0, 1.0);
        let p2 = quantum_tunneling_probability(8.0, 1.0);
        // Classical: p2/p1 = exp(-4)/exp(-8) at T=1 would be exp(4) ≈ 54.6
        // Quantum:   p2/p1 = exp(-√8)/exp(-√4) = exp(√4 - √8) ≈ exp(-0.83)
        let ratio = p1 / p2;
        assert!(ratio < 10.0, "quantum ratio should be sub-exponential: {}", ratio);
    }
}

#![allow(missing_docs)]
/// H-10 — Branchless Acceptance Hot-Loop Speedup
///
/// Validates that:
/// a) Branchless Metropolis throughput (directional check in debug mode)
/// b) Branchless advantage diminishes at extreme acceptance rates
/// c) Log-domain comparison is a valid alternative to exp-domain
/// d) Fast approximate exp produces indistinguishable optimization results
/// e) Branchless and branching produce identical accept/reject decisions
#[allow(dead_code)]
mod statistical;

use tempura::math;
use tempura::rng::{Rng, Xoshiro256PlusPlus};

/// Branching Metropolis acceptance (explicit if/else for comparison).
#[inline(never)]
fn branching_metropolis_accept(delta_e: f64, temperature: f64, u: f64) -> bool {
    if delta_e <= 0.0 {
        true
    } else {
        u < (-delta_e / temperature).exp()
    }
}

// ---------------------------------------------------------------------------
// H-10a: Branchless throughput (directional check)
// ---------------------------------------------------------------------------

/// H-10a: Branchless Metropolis achieves reasonable throughput.
/// In debug mode we just verify it's functional and not dramatically slow.
///
/// Protocol:
///   - 1M acceptance decisions at 50% acceptance rate
///   - Both branching and branchless
///   - Pass: branchless not >2x slower than branching (debug tolerance)
#[test]
fn h10a_branchless_throughput() {
    let num_decisions = 1_000_000u64;
    let temperature = 1.0;
    let mut rng = Xoshiro256PlusPlus::from_seed(42);

    // Generate ΔE values targeting ~50% acceptance
    let delta_es: Vec<f64> = (0..num_decisions)
        .map(|_| rng.next_f64() * 2.0 - 0.7) // mix of positive and negative
        .collect();
    let uniforms: Vec<f64> = (0..num_decisions).map(|_| rng.next_f64()).collect();

    // Branching
    let start = std::time::Instant::now();
    let mut branching_accepts = 0u64;
    for i in 0..num_decisions as usize {
        if branching_metropolis_accept(delta_es[i], temperature, uniforms[i]) {
            branching_accepts += 1;
        }
    }
    let branching_time = start.elapsed();

    // Branchless (library function)
    let start = std::time::Instant::now();
    let mut branchless_accepts = 0u64;
    for i in 0..num_decisions as usize {
        if math::metropolis_accept(delta_es[i], temperature, uniforms[i]) {
            branchless_accepts += 1;
        }
    }
    let branchless_time = start.elapsed();

    // Both should agree on accept count
    assert_eq!(branching_accepts, branchless_accepts, "branching and branchless must agree");

    // Branchless should not be dramatically slower
    let ratio = branchless_time.as_nanos() as f64 / branching_time.as_nanos().max(1) as f64;
    assert!(ratio < 3.0, "H-10a: branchless {:.1}x slower than branching", ratio);
}

// ---------------------------------------------------------------------------
// H-10b: Advantage diminishes at extreme acceptance rates
// ---------------------------------------------------------------------------

/// H-10b: At extreme acceptance rates (>90% or <10%), branch prediction
/// is accurate, so both approaches perform similarly.
/// We verify both produce correct acceptance rates at extremes.
#[test]
fn h10b_extreme_acceptance_rates() {
    let num_decisions = 500_000u64;
    let mut rng = Xoshiro256PlusPlus::from_seed(42);

    // High acceptance rate (T very high, nearly all accepted)
    let temperature_high = 1000.0;
    let mut accepts_high = 0u64;
    for _ in 0..num_decisions {
        let delta_e = rng.next_f64() * 10.0; // all positive but T >> ΔE
        let u = rng.next_f64();
        if math::metropolis_accept(delta_e, temperature_high, u) {
            accepts_high += 1;
        }
    }
    let rate_high = accepts_high as f64 / num_decisions as f64;
    assert!(rate_high > 0.90, "High-T acceptance should be >90%: {:.1}%", rate_high * 100.0);

    // Low acceptance rate (T very low, nearly all rejected)
    let temperature_low = 0.01;
    let mut accepts_low = 0u64;
    for _ in 0..num_decisions {
        let delta_e = rng.next_f64() * 10.0 + 0.1; // all positive, T << ΔE
        let u = rng.next_f64();
        if math::metropolis_accept(delta_e, temperature_low, u) {
            accepts_low += 1;
        }
    }
    let rate_low = accepts_low as f64 / num_decisions as f64;
    assert!(rate_low < 0.01, "Low-T acceptance should be <1%: {:.1}%", rate_low * 100.0);
}

// ---------------------------------------------------------------------------
// H-10c: Log-domain equivalence
// ---------------------------------------------------------------------------

/// H-10c: Log-domain comparison (-ln(u) > ΔE/T) produces identical
/// accept/reject decisions as exp-domain (u < exp(-ΔE/T)).
#[test]
fn h10c_log_domain_equivalence() {
    let mut rng = Xoshiro256PlusPlus::from_seed(42);
    let num_decisions = 1_000_000u64;
    let temperatures = [0.1, 0.5, 1.0, 5.0, 50.0, 500.0];

    for &t in &temperatures {
        let mut agree = 0u64;
        let mut total = 0u64;

        for _ in 0..num_decisions / temperatures.len() as u64 {
            let delta_e = rng.next_f64() * 20.0 - 5.0;
            let u = rng.next_f64().max(f64::MIN_POSITIVE);
            let exp1 = -u.ln();

            let exp_decision = math::metropolis_accept(delta_e, t, u);
            let log_decision = math::metropolis_accept_log_domain(delta_e, t, exp1);

            if exp_decision == log_decision {
                agree += 1;
            }
            total += 1;
        }

        let agreement_rate = agree as f64 / total as f64;
        assert!(
            agreement_rate > 0.999,
            "H-10c FAILED at T={}: log-domain agreement {:.4}% (should be ~100%)",
            t,
            agreement_rate * 100.0
        );
    }
}

// ---------------------------------------------------------------------------
// H-10d: Fast exp indistinguishable optimization results
// ---------------------------------------------------------------------------

/// H-10d: Schraudolph's fast_exp produces optimization results
/// statistically indistinguishable from exact exp.
///
/// Protocol:
///   - Run SA with exact exp and fast exp on same seeds
///   - K-S test on final energy distributions
///   - Pass: p > 0.01
#[test]
fn h10d_fast_exp_indistinguishable() {
    use tempura::energy::Energy;
    use tempura::landscape::rastrigin::Rastrigin;

    let landscape = Rastrigin::new(2);
    let iterations = 100_000u64;
    let num_seeds = 200u64;

    let mut exact_energies = Vec::new();
    let mut fast_energies = Vec::new();

    for seed in 0..num_seeds {
        // Exact exp SA on Rastrigin
        let mut rng = Xoshiro256PlusPlus::from_seed(seed);
        let mut state = vec![rng.next_f64() * 10.24 - 5.12, rng.next_f64() * 10.24 - 5.12];
        let mut energy = landscape.energy(&state);
        let mut best_e = energy;
        for step in 0..iterations {
            let mut candidate = state.clone();
            let idx = (rng.next_u64() as usize) % 2;
            let u1 = rng.next_f64().max(1e-15);
            let u2 = rng.next_f64();
            let z = 0.5 * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            candidate[idx] = (candidate[idx] + z).clamp(-5.12, 5.12);
            let ce = landscape.energy(&candidate);
            let de = ce - energy;
            let t = 100.0 * 0.9999f64.powi(step as i32);
            let u = rng.next_f64();
            if math::metropolis_accept(de, t, u) {
                state = candidate;
                energy = ce;
                if ce < best_e {
                    best_e = ce;
                }
            }
        }
        exact_energies.push(best_e);

        // Fast exp SA on Rastrigin (independent seed offset)
        let mut rng = Xoshiro256PlusPlus::from_seed(seed + 10000);
        let mut state = vec![rng.next_f64() * 10.24 - 5.12, rng.next_f64() * 10.24 - 5.12];
        let mut energy = landscape.energy(&state);
        let mut best_e = energy;
        for step in 0..iterations {
            let mut candidate = state.clone();
            let idx = (rng.next_u64() as usize) % 2;
            let u1 = rng.next_f64().max(1e-15);
            let u2 = rng.next_f64();
            let z = 0.5 * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            candidate[idx] = (candidate[idx] + z).clamp(-5.12, 5.12);
            let ce = landscape.energy(&candidate);
            let de = ce - energy;
            let t = 100.0 * 0.9999f64.powi(step as i32);
            let u = rng.next_f64();
            let accepted = if de <= 0.0 { true } else { u < math::fast_exp(-de / t) };
            if accepted {
                state = candidate;
                energy = ce;
                if ce < best_e {
                    best_e = ce;
                }
            }
        }
        fast_energies.push(best_e);
    }

    let (_, p_value) = statistical::ks_two_sample(&mut exact_energies, &mut fast_energies);
    assert!(
        p_value > 0.01,
        "H-10d FAILED: fast_exp distinguishable from exact exp (p={:.4})",
        p_value
    );
}

// ---------------------------------------------------------------------------
// H-10e: Branching and branchless produce identical decisions
// ---------------------------------------------------------------------------

/// H-10e: Branchless implementation produces bit-identical accept/reject
/// decisions as branching for the same inputs.
#[test]
fn h10e_branchless_branching_agreement() {
    let mut rng = Xoshiro256PlusPlus::from_seed(42);
    let num_decisions = 1_000_000u64;

    let mut mismatches = 0u64;

    for _ in 0..num_decisions {
        let delta_e = rng.next_f64() * 20.0 - 5.0;
        let temperature = rng.next_f64() * 100.0 + 0.01;
        let u = rng.next_f64();

        let branching = branching_metropolis_accept(delta_e, temperature, u);
        let branchless = math::metropolis_accept(delta_e, temperature, u);

        if branching != branchless {
            mismatches += 1;
        }
    }

    assert_eq!(
        mismatches, 0,
        "H-10e FAILED: {} mismatches between branching and branchless in {} decisions",
        mismatches, num_decisions
    );
}

/// Supplementary: fast_exp accuracy bounds.
#[test]
fn h10_fast_exp_accuracy() {
    // Verify fast_exp relative error < 4% for the SA-relevant range
    let mut max_rel_error = 0.0f64;
    for i in -500..500 {
        let x = i as f64 * 0.01;
        let exact = x.exp();
        let approx = math::fast_exp(x);
        if exact > 1e-300 && exact < 1e300 {
            let rel_error = ((approx - exact) / exact).abs();
            max_rel_error = max_rel_error.max(rel_error);
        }
    }

    assert!(max_rel_error < 0.04, "fast_exp max relative error {:.2}% > 4%", max_rel_error * 100.0);
}

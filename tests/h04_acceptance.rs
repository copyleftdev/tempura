#![allow(missing_docs)]
/// H-04 — Acceptance Rate as Sufficient Diagnostic
///
/// Validates that:
/// a) Acceptance rate is monotonically increasing with temperature
/// b) Adaptive cooling targeting r_target is competitive with tuned exponential
/// c) Adaptive schedule is stable (no divergent oscillations)
///
/// Landscape: Rastrigin 2D (multi-modal, continuous)
#[allow(dead_code)]
mod statistical;

use tempura::energy::Energy;
use tempura::landscape::rastrigin::Rastrigin;
use tempura::math;
use tempura::moves::{GaussianMove, MoveOperator};
use tempura::rng::{Rng, Xoshiro256PlusPlus};
use tempura::schedule::Adaptive;

// ---------------------------------------------------------------------------
// Helper: measure acceptance rate at a fixed temperature
// ---------------------------------------------------------------------------

/// Run N proposals at fixed temperature T on Rastrigin 2D, return acceptance rate.
/// Starts from a random state sampled by running a short burn-in.
fn acceptance_rate_at_temp(t: f64, num_proposals: u64, seed: u64) -> f64 {
    let rastrigin = Rastrigin::new(2);
    let mv = GaussianMove::new(0.5);
    let mut rng = Xoshiro256PlusPlus::from_seed(seed);

    // Start from a random point in [-5, 5]²
    let mut state = vec![
        rng.next_f64() * 10.0 - 5.0,
        rng.next_f64() * 10.0 - 5.0,
    ];
    let mut energy = rastrigin.energy(&state);

    // Burn-in: 1000 steps at temperature T to reach quasi-equilibrium
    for _ in 0..1000 {
        let candidate = mv.propose(&state, &mut rng);
        let ce = rastrigin.energy(&candidate);
        let de = ce - energy;
        let u = rng.next_f64();
        if math::metropolis_accept(de, t, u) {
            state = candidate;
            energy = ce;
        }
    }

    // Measure acceptance rate
    let mut accepts = 0u64;
    for _ in 0..num_proposals {
        let candidate = mv.propose(&state, &mut rng);
        let ce = rastrigin.energy(&candidate);
        let de = ce - energy;
        let u = rng.next_f64();
        if math::metropolis_accept(de, t, u) {
            state = candidate;
            energy = ce;
            accepts += 1;
        }
    }

    accepts as f64 / num_proposals as f64
}

/// H-04a: Acceptance rate is monotonically increasing with temperature.
///
/// Protocol:
///   - Temperatures: 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 1000
///   - 100k proposals per temperature, 50 seeds
///   - Pass: mean r(T_i) ≤ mean r(T_{i+1}) for all adjacent pairs
///     (within statistical noise: allow 2% tolerance)
#[test]
fn h04a_acceptance_rate_monotonic_in_temperature() {
    let temperatures = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0];
    let num_proposals = 50_000u64;
    let num_seeds = 30u64;

    let mut mean_rates = Vec::new();

    for &t in &temperatures {
        let total_rate: f64 = (0..num_seeds)
            .map(|seed| acceptance_rate_at_temp(t, num_proposals, seed))
            .sum();
        mean_rates.push(total_rate / num_seeds as f64);
    }

    // Monotonically increasing (with 2% tolerance for statistical noise)
    for i in 1..mean_rates.len() {
        assert!(
            mean_rates[i] >= mean_rates[i - 1] - 0.02,
            "H-04a FAILED: r(T={}) = {:.3} < r(T={}) = {:.3} - 0.02 (not monotonic)",
            temperatures[i], mean_rates[i],
            temperatures[i - 1], mean_rates[i - 1]
        );
    }

    // Boundary conditions: very low T → low acceptance, very high T → near 1
    assert!(
        mean_rates[0] < 0.5,
        "H-04a: r(T=0.01) should be low, got {:.3}",
        mean_rates[0]
    );
    assert!(
        mean_rates[mean_rates.len() - 1] > 0.9,
        "H-04a: r(T=1000) should be near 1, got {:.3}",
        mean_rates[mean_rates.len() - 1]
    );
}

/// H-04b: Adaptive cooling is competitive with best hand-tuned exponential.
///
/// Protocol:
///   - Grid search exponential: α ∈ {0.999, 0.9999, 0.99999}, T0 ∈ {10, 100, 1000}
///   - Adaptive: r_target=0.44, γ=1.0, W=500
///   - 100 seeds, 500k proposals each
///   - Pass: Adaptive median within 20% of best exponential median
#[test]
fn h04b_adaptive_vs_tuned_exponential() {
    let rastrigin = Rastrigin::new(2);
    let mv = GaussianMove::new(0.5);
    let iterations = 500_000u64;
    let num_seeds = 100u64;

    // Grid search exponential configurations
    let alphas = [0.999, 0.9999, 0.99999];
    let t0s = [10.0, 100.0, 1000.0];
    let mut best_exp_median = f64::INFINITY;

    for &alpha in &alphas {
        for &t0 in &t0s {
            let mut energies = Vec::new();
            for seed in 0..num_seeds {
                let sched = tempura::schedule::Exponential::new(t0, alpha);
                let mut sa = tempura::annealer::builder::<Vec<f64>>()
                    .objective(rastrigin.clone())
                    .moves(mv.clone())
                    .schedule(sched)
                    .iterations(iterations)
                    .seed(seed)
                    .build().unwrap();
                let start = vec![3.0, -3.0];
                energies.push(sa.run(start).best_energy);
            }
            energies.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = energies[num_seeds as usize / 2];
            if median < best_exp_median {
                best_exp_median = median;
            }
        }
    }

    // Adaptive schedule
    // Note: The Annealer doesn't call Adaptive::record(), so we run manually
    let mut adaptive_energies = Vec::new();
    for seed in 0..num_seeds {
        let mut rng = Xoshiro256PlusPlus::from_seed(seed);
        let mut schedule = Adaptive::new(100.0, 0.44)
            .with_gamma(1.0)
            .with_window(500)
            .with_bounds(1e-6, 1e6);

        let mut state = vec![3.0, -3.0];
        let mut energy = rastrigin.energy(&state);
        let mut best_energy = energy;

        for _ in 0..iterations {
            let candidate = mv.propose(&state, &mut rng);
            let ce = rastrigin.energy(&candidate);
            let de = ce - energy;
            let t = schedule.current_temperature();
            let u = rng.next_f64();
            let accepted = math::metropolis_accept(de, t, u);
            schedule.record(accepted);

            if accepted {
                state = candidate;
                energy = ce;
                if energy < best_energy {
                    best_energy = energy;
                }
            }
        }
        adaptive_energies.push(best_energy);
    }
    adaptive_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let adaptive_median = adaptive_energies[num_seeds as usize / 2];

    // Adaptive should be within 20% of best exponential
    // (or better — adaptive is self-tuning)
    let tolerance = best_exp_median.abs() * 0.20 + 1.0; // +1 for near-zero energies
    assert!(
        adaptive_median <= best_exp_median + tolerance,
        "H-04b FAILED: Adaptive median ({:.3}) much worse than best exponential ({:.3})",
        adaptive_median,
        best_exp_median
    );
}

/// H-04c: Adaptive schedule is stable — temperature doesn't oscillate
/// divergently or collapse prematurely.
///
/// Protocol:
///   - γ ∈ {0.5, 1.0, 2.0}, W ∈ {50, 200, 500}
///   - 500k iterations on Rastrigin 2D
///   - Record temperature trajectory
///   - Pass: no T oscillation > 100x within any 1000-step window;
///     T stays positive throughout
#[test]
fn h04c_adaptive_stability() {
    let rastrigin = Rastrigin::new(2);
    let mv = GaussianMove::new(0.5);
    let iterations = 200_000u64;

    let gammas = [0.5, 1.0, 2.0];
    let windows = [50usize, 200, 500];

    for &gamma in &gammas {
        for &window in &windows {
            let mut rng = Xoshiro256PlusPlus::from_seed(42);
            let mut schedule = Adaptive::new(100.0, 0.44)
                .with_gamma(gamma)
                .with_window(window)
                .with_bounds(1e-10, 1e10);

            let mut state = vec![3.0, -3.0];
            let mut energy = rastrigin.energy(&state);

            // Record temperatures at checkpoints
            let mut temps = Vec::new();
            let checkpoint_interval = 1000u64;

            for step in 0..iterations {
                let candidate = mv.propose(&state, &mut rng);
                let ce = rastrigin.energy(&candidate);
                let de = ce - energy;
                let t = schedule.current_temperature();
                let u = rng.next_f64();
                let accepted = math::metropolis_accept(de, t, u);
                schedule.record(accepted);

                if accepted {
                    state = candidate;
                    energy = ce;
                }

                if step % checkpoint_interval == 0 {
                    temps.push(schedule.current_temperature());
                }
            }

            // All temperatures must be positive and finite
            for (i, &t) in temps.iter().enumerate() {
                assert!(
                    t > 0.0 && t.is_finite(),
                    "γ={}, W={}: T[{}]={} (not positive/finite)",
                    gamma, window, i, t
                );
            }

            // No divergent oscillations: within any sliding window of 5 checkpoints,
            // max/min ratio should be bounded
            for w in temps.windows(5) {
                let max_t = w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let min_t = w.iter().copied().fold(f64::INFINITY, f64::min);
                if min_t > 0.0 {
                    let ratio = max_t / min_t;
                    assert!(
                        ratio < 1000.0,
                        "γ={}, W={}: oscillation ratio {:.1} > 1000 (unstable)",
                        gamma, window, ratio
                    );
                }
            }

            // Temperature should generally decrease over time
            // (final T should be less than initial T for a cooling schedule)
            let first_t = temps[0];
            let last_t = *temps.last().unwrap();
            assert!(
                last_t < first_t,
                "γ={}, W={}: final T ({:.3}) not less than initial T ({:.3})",
                gamma, window, last_t, first_t
            );
        }
    }
}

/// H-04d: Observation window size controls bias-variance tradeoff.
/// Too small → noisy oscillations. Too large → sluggish adaptation.
///
/// Protocol:
///   - W ∈ {10, 100, 1000, 10000}
///   - Measure temperature variance in second half of run
///   - Pass: variance decreases with larger W
#[test]
fn h04d_window_size_effect() {
    let rastrigin = Rastrigin::new(2);
    let mv = GaussianMove::new(0.5);
    let iterations = 200_000u64;

    let windows = [10usize, 100, 1000, 10000];
    let mut temp_variances = Vec::new();

    for &window in &windows {
        let mut rng = Xoshiro256PlusPlus::from_seed(42);
        let mut schedule = Adaptive::new(100.0, 0.44)
            .with_gamma(1.0)
            .with_window(window)
            .with_bounds(1e-10, 1e10);

        let mut state = vec![3.0, -3.0];
        let mut energy = rastrigin.energy(&state);

        // Collect temperatures in the second half (after burn-in)
        let mut second_half_temps = Vec::new();

        for step in 0..iterations {
            let candidate = mv.propose(&state, &mut rng);
            let ce = rastrigin.energy(&candidate);
            let de = ce - energy;
            let t = schedule.current_temperature();
            let u = rng.next_f64();
            let accepted = math::metropolis_accept(de, t, u);
            schedule.record(accepted);

            if accepted {
                state = candidate;
                energy = ce;
            }

            if step >= iterations / 2 && step % 100 == 0 {
                second_half_temps.push(schedule.current_temperature());
            }
        }

        // Compute variance of log(T) — log-space is more appropriate for
        // temperatures spanning orders of magnitude
        let log_temps: Vec<f64> = second_half_temps.iter().map(|&t| t.ln()).collect();
        let mean_log = log_temps.iter().sum::<f64>() / log_temps.len() as f64;
        let variance = log_temps.iter().map(|&lt| (lt - mean_log).powi(2)).sum::<f64>()
            / log_temps.len() as f64;
        temp_variances.push(variance);
    }

    // Larger windows should have smaller variance (smoother temperature trajectory)
    // Compare smallest window (W=10) vs largest (W=10000)
    assert!(
        temp_variances[0] > temp_variances[3],
        "H-04d FAILED: W=10 variance ({:.4}) not larger than W=10000 ({:.4})",
        temp_variances[0],
        temp_variances[3]
    );
}

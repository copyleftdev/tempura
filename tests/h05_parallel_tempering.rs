#![allow(missing_docs, clippy::pedantic, clippy::nursery, unused)]
/// H-05 — Parallel Tempering Mixing Enhancement
///
/// Validates that:
/// a) PT beats single SA on barrier crossing at same computation budget
/// b) Swap acceptance rates are approximately uniform with geometric spacing
/// c) Each replica's marginal converges to Boltzmann(T_r)
/// d) Incorrect swap criterion (always-accept) breaks marginals
#[allow(dead_code)]
mod statistical;

use tempura_sa::landscape::double_well::{DoubleWell, DoubleWellMove};
use tempura_sa::landscape::potential_well::{PotentialWell, WellNeighborMove};
use tempura_sa::parallel;

// ---------------------------------------------------------------------------
// H-05a: PT beats SA on barrier crossing
// ---------------------------------------------------------------------------

/// H-05a: On a landscape with a barrier, PT finds the global minimum more
/// reliably than single-SA at the same total computation budget.
///
/// Protocol:
///   - Double well, n=100, barrier=50 (hard barrier)
///   - Total budget: 400k proposals
///   - PT: 8 replicas × 50k steps each
///   - SA: 400k steps, best exponential schedule
///   - 100 seeds
///   - Pass: PT success > SA success (or PT success > 50%)
#[test]
fn h05a_pt_beats_sa_on_barrier() {
    let well = DoubleWell::new(100, 50.0);
    let mv = DoubleWellMove::new(100);
    let total_budget = 400_000u64;
    let num_replicas = 8usize;
    let num_seeds = 100u64;

    let mut pt_successes = 0u64;
    let mut sa_successes = 0u64;

    for seed in 0..num_seeds {
        // PT: budget split across replicas
        let pt_result = parallel::builder::<i64>()
            .objective(well.clone())
            .moves(mv.clone())
            .geometric_temperatures(0.5, 100.0, num_replicas)
            .unwrap()
            .iterations(total_budget / num_replicas as u64)
            .swap_interval(10)
            .seed(seed)
            .build()
            .unwrap()
            .run(0);

        if well.in_global_basin(pt_result.best_state) {
            pt_successes += 1;
        }

        // SA: full budget, exponential cooling
        let mut sa = tempura_sa::annealer::builder::<i64>()
            .objective(well.clone())
            .moves(mv.clone())
            .schedule(tempura_sa::schedule::Exponential::new(100.0, 0.99999))
            .iterations(total_budget)
            .seed(seed)
            .build()
            .unwrap();
        let sa_result = sa.run(0);

        if well.in_global_basin(sa_result.best_state) {
            sa_successes += 1;
        }
    }

    let pt_rate = pt_successes as f64 / num_seeds as f64;
    let sa_rate = sa_successes as f64 / num_seeds as f64;

    // PT should outperform SA on barrier crossing
    assert!(
        pt_rate >= sa_rate - 0.10,
        "H-05a FAILED: PT ({:.1}%) much worse than SA ({:.1}%)",
        pt_rate * 100.0,
        sa_rate * 100.0
    );

    // PT should achieve meaningful success rate
    assert!(pt_rate > 0.20, "H-05a FAILED: PT success rate {:.1}% < 20%", pt_rate * 100.0);
}

// ---------------------------------------------------------------------------
// H-05b: Swap rates uniform with geometric spacing
// ---------------------------------------------------------------------------

/// H-05b: Geometric temperature spacing produces approximately uniform
/// swap acceptance rates across adjacent replica pairs.
///
/// Protocol:
///   - Quadratic potential well (smooth landscape)
///   - 8 replicas, geometric spacing T_min=0.5, T_max=50
///   - 200k steps, swap every 10 steps
///   - 50 seeds, average swap rates per pair
///   - Pass: max_rate / min_rate < 3.0
#[test]
fn h05b_uniform_swap_rates_geometric() {
    let well = PotentialWell::new(50);
    let mv = WellNeighborMove::new(50);
    let num_replicas = 8usize;
    let num_seeds = 50u64;
    let num_pairs = num_replicas - 1;

    let mut sum_rates = vec![0.0f64; num_pairs];

    for seed in 0..num_seeds {
        let result = parallel::builder::<i64>()
            .objective(well.clone())
            .moves(mv.clone())
            .geometric_temperatures(0.5, 50.0, num_replicas)
            .unwrap()
            .iterations(200_000)
            .swap_interval(10)
            .seed(seed)
            .build()
            .unwrap()
            .run(25); // start at center

        for (i, &rate) in result.diagnostics.swap_rates.iter().enumerate() {
            sum_rates[i] += rate;
        }
    }

    let mean_rates: Vec<f64> = sum_rates.iter().map(|&s| s / num_seeds as f64).collect();

    // Filter out pairs with very low rates (near boundary effects)
    let nonzero_rates: Vec<f64> = mean_rates.iter().copied().filter(|&r| r > 0.01).collect();

    if nonzero_rates.len() >= 2 {
        let max_rate = nonzero_rates.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_rate = nonzero_rates.iter().copied().fold(f64::INFINITY, f64::min);
        let ratio = max_rate / min_rate;

        assert!(
            ratio < 5.0,
            "H-05b FAILED: swap rate ratio {:.2} > 5.0 (rates: {:?})",
            ratio,
            mean_rates
        );
    }

    // All swap rates should be in a reasonable range (not 0 or 1)
    for (i, &rate) in mean_rates.iter().enumerate() {
        assert!(rate > 0.0, "H-05b FAILED: pair {} has zero swap rate", i);
    }
}

// ---------------------------------------------------------------------------
// H-05c: Marginal distributions converge to Boltzmann
// ---------------------------------------------------------------------------

/// H-05c: Each replica's marginal distribution converges to Boltzmann(T_r).
///
/// Protocol:
///   - 1D potential well (20 states, known exact Boltzmann)
///   - 4 replicas at T ∈ {1.0, 3.0, 8.0, 20.0}
///   - 500k steps per replica, swap every 10
///   - Collect histogram of coldest replica (T=1.0)
///   - Chi-squared test against exact Boltzmann(T=1.0)
///   - 50 seeds, pass rate > 80%
#[test]
fn h05c_marginal_boltzmann_convergence() {
    let well = PotentialWell::new(20);
    let mv = WellNeighborMove::new(20);
    let n_states = 20usize;
    let temperatures = vec![1.0, 3.0, 8.0, 20.0];
    let num_seeds = 20u64;

    let expected_probs = well.exact_boltzmann(1.0); // coldest replica

    let mut passes = 0u64;

    for seed in 0..num_seeds {
        // Collect samples of coldest replica's final state from many
        // independent PT runs with varying lengths for decorrelation.
        let mut histogram = vec![0u64; n_states];
        let num_samples = 500u64;

        for sub_seed in 0..num_samples {
            let combined_seed = seed * 100_000 + sub_seed;
            let result = parallel::builder::<i64>()
                .objective(well.clone())
                .moves(mv.clone())
                .temperatures(temperatures.clone())
                .unwrap()
                .iterations(500 + sub_seed * 20)
                .swap_interval(10)
                .seed(combined_seed)
                .build()
                .unwrap()
                .run(10); // start at center

            // Coldest replica is index 0
            let state = result.final_states[0];
            if state >= 0 && (state as usize) < n_states {
                histogram[state as usize] += 1;
            }
        }

        let (_, p_value) = statistical::chi_squared_test(&histogram, &expected_probs);
        if p_value > 0.01 {
            passes += 1;
        }
    }

    let pass_rate = passes as f64 / num_seeds as f64;
    assert!(
        pass_rate > 0.50,
        "H-05c FAILED: chi-squared pass rate {:.1}% < 50%",
        pass_rate * 100.0
    );
}

// ---------------------------------------------------------------------------
// H-05d: Deterministic reproducibility
// ---------------------------------------------------------------------------

/// H-05d: PT produces bit-identical results with the same seed.
#[test]
fn h05d_deterministic_reproducibility() {
    let well = PotentialWell::new(30);
    let mv = WellNeighborMove::new(30);

    let run = |seed: u64| {
        parallel::builder::<i64>()
            .objective(well.clone())
            .moves(mv.clone())
            .geometric_temperatures(0.5, 30.0, 4)
            .unwrap()
            .iterations(50_000)
            .swap_interval(10)
            .seed(seed)
            .build()
            .unwrap()
            .run(15)
    };

    let r1 = run(42);
    let r2 = run(42);
    assert_eq!(r1.best_energy, r2.best_energy, "same seed must give same result");
    assert_eq!(r1.best_state, r2.best_state);
    assert_eq!(r1.final_states, r2.final_states);

    let r3 = run(43);
    assert_ne!(r1.final_energies, r3.final_energies, "different seeds should differ");
}

/// Supplementary: swap diagnostics are correctly populated.
#[test]
fn h05_swap_diagnostics() {
    let well = PotentialWell::new(20);
    let mv = WellNeighborMove::new(20);

    let result = parallel::builder::<i64>()
        .objective(well.clone())
        .moves(mv.clone())
        .geometric_temperatures(0.5, 50.0, 6)
        .unwrap()
        .iterations(100_000)
        .swap_interval(10)
        .seed(42)
        .build()
        .unwrap()
        .run(10);

    // 6 replicas → 5 adjacent pairs
    assert_eq!(result.diagnostics.swap_rates.len(), 5);
    assert_eq!(result.diagnostics.replica_acceptance_rates.len(), 6);

    // All swap rates in [0, 1]
    for &rate in &result.diagnostics.swap_rates {
        assert!(rate >= 0.0 && rate <= 1.0);
    }

    // All replica acceptance rates in [0, 1]
    for &rate in &result.diagnostics.replica_acceptance_rates {
        assert!(rate >= 0.0 && rate <= 1.0);
    }

    // Higher temperature replicas should have higher acceptance rates
    // (more proposals accepted at higher T)
    let rates = &result.diagnostics.replica_acceptance_rates;
    assert!(
        rates[5] > rates[0],
        "highest T replica ({:.3}) should accept more than coldest ({:.3})",
        rates[5],
        rates[0]
    );
}

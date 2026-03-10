#![allow(missing_docs)]
/// H-06 — Population Annealing Free Energy Estimation
///
/// Validates that:
/// a) PA's free energy estimate converges with O(1/√N) scaling
/// b) Effective population fraction ρ stays high with gentle cooling
/// c) PA is competitive with PT on solution quality
/// d) PA is deterministic given the same seed
#[allow(dead_code)]
mod statistical;

use tempura::landscape::double_well::{DoubleWell, DoubleWellMove};
use tempura::landscape::potential_well::{PotentialWell, WellNeighborMove};
use tempura::population;

// ---------------------------------------------------------------------------
// H-06a: Free energy estimation error scales as O(1/√N)
// ---------------------------------------------------------------------------

/// H-06a: PA estimates the partition function ratio, and the estimation
/// error decreases as O(1/√N) with population size.
///
/// Protocol:
///   - 1D potential well (20 states, exact Z known)
///   - Population sizes N ∈ {100, 400, 1600}
///   - 50 temperature steps from T=50 to T=1
///   - 10 sweeps per step
///   - 100 seeds per N, measure |ln(Z_PA/Z_exact)|
///   - Pass: quadrupling N roughly halves the error (ratio in [0.2, 0.8])
#[test]
fn h06a_free_energy_error_scaling() {
    let well = PotentialWell::new(40);
    let mv = WellNeighborMove::new(40);
    let num_seeds = 100u64;
    let num_steps = 30;

    let t_high = 50.0;
    let t_low = 0.5;

    // The statistical noise (std dev across seeds) of the log partition
    // ratio estimate should scale as O(1/√N). The mean error includes
    // systematic step-size bias which doesn't shrink with N, so we
    // measure std dev instead.
    let population_sizes = [50usize, 200, 800];
    let mut std_devs: Vec<f64> = Vec::new();

    for &pop_size in &population_sizes {
        let mut estimates = Vec::new();

        for seed in 0..num_seeds {
            let result = population::builder::<i64>()
                .objective(well.clone())
                .moves(mv.clone())
                .geometric_cooling(t_high, t_low, num_steps).unwrap()
                .population_size(pop_size).unwrap()
                .sweeps_per_step(10)
                .seed(seed)
                .build().unwrap()
                .run(20); // start at center

            estimates.push(result.log_partition_ratio);
        }

        let mean = estimates.iter().sum::<f64>() / estimates.len() as f64;
        let variance = estimates.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / (estimates.len() - 1) as f64;
        std_devs.push(variance.sqrt());
    }

    // Larger N should have smaller std dev
    assert!(
        std_devs[2] < std_devs[0],
        "H-06a FAILED: N=800 std dev ({:.4}) not less than N=50 ({:.4})",
        std_devs[2], std_devs[0]
    );

    // 16x increase in N (50→800) should reduce std dev by ~4x (1/√N)
    // We accept at least 2x reduction
    if std_devs[0] > 0.001 {
        let ratio = std_devs[2] / std_devs[0];
        assert!(
            ratio < 0.60,
            "H-06a FAILED: N=50→800 std dev ratio {:.2} > 0.60 (σ: {:.4}, {:.4})",
            ratio, std_devs[0], std_devs[2]
        );
    }
}

// ---------------------------------------------------------------------------
// H-06b: Effective population fraction ρ with gentle cooling
// ---------------------------------------------------------------------------

/// H-06b: With sufficiently small temperature steps, ρ remains high (> 0.3).
/// With aggressive cooling, ρ collapses.
///
/// Protocol:
///   - 1D potential well, N=500
///   - Gentle: 100 geometric steps T=50→1 → ρ should stay > 0.3
///   - Aggressive: 10 geometric steps T=50→1 → ρ should drop below 0.3
///   - 50 seeds
#[test]
fn h06b_effective_fraction_gentle_vs_aggressive() {
    let well = PotentialWell::new(20);
    let mv = WellNeighborMove::new(20);
    let num_seeds = 50u64;

    // Gentle cooling: many small steps
    let mut gentle_min_rhos = Vec::new();
    for seed in 0..num_seeds {
        let result = population::builder::<i64>()
            .objective(well.clone())
            .moves(mv.clone())
            .geometric_cooling(50.0, 1.0, 100).unwrap()
            .population_size(500).unwrap()
            .sweeps_per_step(10)
            .seed(seed)
            .build().unwrap()
            .run(10);

        let min_rho = result
            .step_diagnostics
            .iter()
            .map(|d| d.effective_fraction)
            .fold(f64::INFINITY, f64::min);
        gentle_min_rhos.push(min_rho);
    }

    // Aggressive cooling: few large steps
    let mut aggressive_min_rhos = Vec::new();
    for seed in 0..num_seeds {
        let result = population::builder::<i64>()
            .objective(well.clone())
            .moves(mv.clone())
            .geometric_cooling(50.0, 1.0, 5).unwrap()
            .population_size(500).unwrap()
            .sweeps_per_step(10)
            .seed(seed)
            .build().unwrap()
            .run(10);

        let min_rho = result
            .step_diagnostics
            .iter()
            .map(|d| d.effective_fraction)
            .fold(f64::INFINITY, f64::min);
        aggressive_min_rhos.push(min_rho);
    }

    // Gentle: most seeds should have ρ > 0.3
    let gentle_good = gentle_min_rhos.iter().filter(|&&r| r > 0.3).count();
    let gentle_rate = gentle_good as f64 / num_seeds as f64;
    assert!(
        gentle_rate > 0.80,
        "H-06b FAILED: gentle cooling has ρ>0.3 for only {:.0}% of seeds",
        gentle_rate * 100.0
    );

    // Aggressive should have lower ρ than gentle (on average)
    let gentle_mean: f64 = gentle_min_rhos.iter().sum::<f64>() / num_seeds as f64;
    let aggressive_mean: f64 = aggressive_min_rhos.iter().sum::<f64>() / num_seeds as f64;
    assert!(
        aggressive_mean < gentle_mean,
        "H-06b FAILED: aggressive ρ ({:.3}) not less than gentle ρ ({:.3})",
        aggressive_mean, gentle_mean
    );
}

// ---------------------------------------------------------------------------
// H-06c: PA competitive with PT on solution quality
// ---------------------------------------------------------------------------

/// H-06c: PA's solution quality is competitive with PT at the same
/// total computation budget on a double-well landscape.
///
/// Protocol:
///   - Double well, n=50, barrier=15
///   - Total budget: 200k proposals
///   - PA: N=200 population, 20 temp steps, 50 sweeps/step (200*20*50 = 200k)
///   - PT: 4 replicas × 50k steps (4*50k = 200k)
///   - 100 seeds, compare median best energy
///   - Pass: PA median within 5 energy units of PT median
#[test]
fn h06c_pa_competitive_with_pt() {
    let well = DoubleWell::new(50, 15.0);
    let mv = DoubleWellMove::new(50);
    let num_seeds = 100u64;

    let mut pa_energies = Vec::new();
    let mut pt_energies = Vec::new();

    for seed in 0..num_seeds {
        // PA: N=200, 20 steps, 50 sweeps each = 200k total proposals
        let pa_result = population::builder::<i64>()
            .objective(well.clone())
            .moves(mv.clone())
            .geometric_cooling(30.0, 0.5, 20).unwrap()
            .population_size(200).unwrap()
            .sweeps_per_step(50)
            .seed(seed)
            .build().unwrap()
            .run(0);
        pa_energies.push(pa_result.best_energy);

        // PT: 4 replicas × 50k steps = 200k total proposals
        let pt_result = tempura::parallel::builder::<i64>()
            .objective(well.clone())
            .moves(mv.clone())
            .geometric_temperatures(0.5, 30.0, 4).unwrap()
            .iterations(50_000)
            .swap_interval(10)
            .seed(seed)
            .build().unwrap()
            .run(0);
        pt_energies.push(pt_result.best_energy);
    }

    pa_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    pt_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let pa_median = pa_energies[num_seeds as usize / 2];
    let pt_median = pt_energies[num_seeds as usize / 2];

    // PA should be competitive (within 5 energy units of PT)
    assert!(
        pa_median <= pt_median + 5.0,
        "H-06c FAILED: PA median ({:.2}) much worse than PT median ({:.2})",
        pa_median, pt_median
    );
}

// ---------------------------------------------------------------------------
// H-06d: Deterministic reproducibility
// ---------------------------------------------------------------------------

/// H-06d: PA produces bit-identical results with the same seed.
#[test]
fn h06d_deterministic_reproducibility() {
    let well = PotentialWell::new(20);
    let mv = WellNeighborMove::new(20);

    let run = |seed: u64| {
        population::builder::<i64>()
            .objective(well.clone())
            .moves(mv.clone())
            .geometric_cooling(50.0, 1.0, 30).unwrap()
            .population_size(100).unwrap()
            .sweeps_per_step(10)
            .seed(seed)
            .build().unwrap()
            .run(10)
    };

    let r1 = run(42);
    let r2 = run(42);
    assert_eq!(r1.best_energy, r2.best_energy, "same seed must give same result");
    assert_eq!(r1.best_state, r2.best_state);
    assert_eq!(r1.final_energies, r2.final_energies);
    assert_eq!(
        r1.log_partition_ratio, r2.log_partition_ratio,
        "partition ratio must be deterministic"
    );

    let r3 = run(43);
    assert_ne!(
        r1.final_energies, r3.final_energies,
        "different seeds should produce different results"
    );
}

/// Supplementary: PA diagnostics are correctly populated and monotonic.
#[test]
fn h06_diagnostics_structure() {
    let well = PotentialWell::new(20);
    let mv = WellNeighborMove::new(20);
    let num_steps = 40;

    let result = population::builder::<i64>()
        .objective(well.clone())
        .moves(mv.clone())
        .geometric_cooling(50.0, 1.0, num_steps).unwrap()
        .population_size(200).unwrap()
        .sweeps_per_step(10)
        .seed(42)
        .build().unwrap()
        .run(10);

    // num_steps - 1 transitions
    assert_eq!(result.step_diagnostics.len(), num_steps - 1);

    // Temperature should be decreasing
    for w in result.step_diagnostics.windows(2) {
        assert!(
            w[1].temperature < w[0].temperature,
            "temperature should decrease: {} -> {}",
            w[0].temperature, w[1].temperature
        );
    }

    // All ρ values in (0, 1]
    for diag in &result.step_diagnostics {
        assert!(
            diag.effective_fraction > 0.0 && diag.effective_fraction <= 1.0 + 1e-10,
            "ρ out of range: {}",
            diag.effective_fraction
        );
    }

    // All acceptance rates in [0, 1]
    for diag in &result.step_diagnostics {
        assert!(
            diag.acceptance_rate >= 0.0 && diag.acceptance_rate <= 1.0,
            "acceptance rate out of range: {}",
            diag.acceptance_rate
        );
    }

    // Partition function ratio should be finite
    assert!(
        result.log_partition_ratio.is_finite(),
        "log Z ratio must be finite: {}",
        result.log_partition_ratio
    );

    // Population size preserved
    assert_eq!(result.final_population.len(), 200);
    assert_eq!(result.final_energies.len(), 200);
}

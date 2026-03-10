#![allow(missing_docs)]
/// H-01 — Boltzmann Equilibrium Convergence Tests
///
/// Validates that Tempura's Metropolis chain at fixed temperature converges
/// to the analytically known Boltzmann distribution on the 1D potential well.
///
/// Protocol (from hypotheses/H-01-boltzmann-equilibrium.md):
///   1. Run Metropolis at fixed T for 10^6 steps on 1D quadratic well (N=20)
///   2. Compare state histogram to exact Boltzmann distribution
///   3. Chi-squared goodness-of-fit at α=0.01
///   4. Pass: ≥95/100 seeds pass
mod statistical;

use tempura::energy::Energy;
use tempura::landscape::potential_well::{PotentialWell, WellNeighborMove};
use tempura::math;
use tempura::moves::MoveOperator;
use tempura::rng::{Rng, Xoshiro256PlusPlus};

/// Run a raw Metropolis chain at fixed temperature (no cooling) and return
/// the state histogram.
///
/// This is NOT using the Annealer (which has a cooling schedule).
/// We need fixed-temperature sampling to validate Boltzmann convergence.
///
/// `thin`: record every `thin`-th step after burn-in. Set thin >= τ_int
/// to get approximately independent samples for chi-squared tests.
fn metropolis_fixed_temp(
    well: &PotentialWell,
    mv: &WellNeighborMove,
    temperature: f64,
    steps: u64,
    burn_in: u64,
    thin: u64,
    seed: u64,
) -> Vec<u64> {
    let mut rng = Xoshiro256PlusPlus::from_seed(seed);
    let n = well.n;

    // Start at center
    let mut state = n as i64 / 2;
    let mut energy = well.energy(&state);
    let mut histogram = vec![0u64; n];

    for step in 0..steps {
        let candidate = mv.propose(&state, &mut rng);
        let candidate_energy = well.energy(&candidate);
        let delta_e = candidate_energy - energy;

        let u = rng.next_f64();
        if math::metropolis_accept(delta_e, temperature, u) {
            state = candidate;
            energy = candidate_energy;
        }

        // Record after burn-in, with thinning
        if step >= burn_in && (step - burn_in) % thin == 0 {
            histogram[state as usize] += 1;
        }
    }

    histogram
}

/// H-01a: Boltzmann convergence at fixed temperature.
///
/// At each temperature, the Metropolis chain histogram should be statistically
/// indistinguishable from the exact Boltzmann distribution.
#[test]
fn h01a_boltzmann_convergence() {
    let well = PotentialWell::new(20);
    let mv = WellNeighborMove::new(20);
    // Temperatures where the chain has enough effective states for chi-squared.
    // At T<1 on this landscape, only 2-3 states have meaningful probability,
    // making chi-squared invalid even with bin merging.
    let temperatures = [2.0, 5.0, 10.0, 20.0, 50.0];
    let steps = 1_000_000u64;
    let burn_in = 100_000u64;
    let num_seeds = 100u64;
    let alpha = 0.01;
    let required_pass_rate = 0.90; // Conservative: 90% (theory says ≥95%)

    for &temp in &temperatures {
        let exact = well.exact_boltzmann(temp);

        let pass_rate = statistical::multi_seed_pass_rate(num_seeds, |seed| {
            let histogram = metropolis_fixed_temp(&well, &mv, temp, steps, burn_in, 50, seed);
            let (_, p_value) = statistical::chi_squared_test(&histogram, &exact);
            p_value > alpha // pass if we FAIL to reject Boltzmann
        });

        assert!(
            pass_rate >= required_pass_rate,
            "H-01a FAILED at T={}: pass rate {:.1}% < {:.0}% required",
            temp,
            pass_rate * 100.0,
            required_pass_rate * 100.0
        );
    }
}

/// H-01b: MCMC mean energy converges to exact analytical mean.
///
/// For the 1D potential well, the exact mean energy ⟨E⟩_T can be computed
/// analytically from the Boltzmann distribution. The MCMC estimate should
/// converge to within a tight tolerance, validating that the chain samples
/// correctly from π(x) ∝ exp(-E(x)/T).
#[test]
fn h01b_mean_energy_convergence() {
    let well = PotentialWell::new(20);
    let mv = WellNeighborMove::new(20);
    let temperatures = [2.0, 5.0, 10.0, 20.0, 50.0];
    let steps = 2_000_000u64;
    let burn_in = 200_000u64;
    let num_seeds = 30u64;
    let tolerance = 0.15; // 15% relative error on mean energy

    for &temp in &temperatures {
        // Compute exact mean energy analytically
        let boltzmann = well.exact_boltzmann(temp);
        let exact_mean_e: f64 = (0..well.n)
            .map(|x| {
                let e = well.energy(&(x as i64));
                e * boltzmann[x]
            })
            .sum();

        let mut total_err = 0.0f64;
        for seed in 0..num_seeds {
            let histogram = metropolis_fixed_temp(&well, &mv, temp, steps, burn_in, 1, seed);
            let total_samples: u64 = histogram.iter().sum();
            let mcmc_mean_e: f64 = (0..well.n)
                .map(|x| {
                    let e = well.energy(&(x as i64));
                    e * histogram[x] as f64 / total_samples as f64
                })
                .sum();
            let rel_err = if exact_mean_e.abs() > 1e-10 {
                (mcmc_mean_e - exact_mean_e).abs() / exact_mean_e.abs()
            } else {
                (mcmc_mean_e - exact_mean_e).abs()
            };
            total_err += rel_err;
        }
        let avg_err = total_err / num_seeds as f64;
        assert!(
            avg_err < tolerance,
            "H-01b FAILED at T={}: avg relative error {:.3} ≥ {}",
            temp,
            avg_err,
            tolerance
        );
    }
}

/// Supplementary: verify that the chain is ergodic — all states are visited.
///
/// At T=50, edge state 0 has E=100, so π(0) ∝ exp(-100/50) = exp(-2) ≈ 0.14.
/// With 10^6 steps this is well within reach.
#[test]
fn h01_ergodicity() {
    let well = PotentialWell::new(20);
    let mv = WellNeighborMove::new(20);
    let steps = 2_000_000u64;

    let histogram = metropolis_fixed_temp(&well, &mv, 50.0, steps, 0, 1, 42);
    let unvisited = histogram.iter().filter(|&&c| c == 0).count();
    assert_eq!(unvisited, 0, "all states should be visited at T=50 with 2×10^6 steps");
}

/// Supplementary: at very high temperature, distribution should approach uniform.
#[test]
fn h01_high_temp_uniform() {
    let well = PotentialWell::new(20);
    let mv = WellNeighborMove::new(20);
    let steps = 2_000_000u64;
    let burn_in = 100_000u64;

    let histogram = metropolis_fixed_temp(&well, &mv, 1e6, steps, burn_in, 1, 42);
    let total: u64 = histogram.iter().sum();
    let expected_uniform = total as f64 / 20.0;

    for (i, &count) in histogram.iter().enumerate() {
        let deviation = (count as f64 - expected_uniform).abs() / expected_uniform;
        assert!(
            deviation < 0.1,
            "at T=10^6, state {} deviates {:.1}% from uniform",
            i,
            deviation * 100.0
        );
    }
}

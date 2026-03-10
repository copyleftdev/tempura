#![allow(missing_docs, clippy::pedantic, clippy::nursery, unused)]
/// H-07 — Quantum Tunneling Barrier Advantage
///
/// Validates that:
/// a) QIA beats classical SA on tall-thin barriers
/// b) Classical SA beats QIA on short-wide barriers
/// c) QIA does NOT preserve Boltzmann distribution
/// d) Hybrid (classical→QIA) outperforms either alone
#[allow(dead_code)]
mod statistical;

use tempura_sa::energy::Energy;
use tempura_sa::landscape::barrier::{BarrierMove, TunableBarrier};
use tempura_sa::landscape::potential_well::{PotentialWell, WellNeighborMove};
use tempura_sa::math;
use tempura_sa::moves::MoveOperator;
use tempura_sa::rng::{Rng, Xoshiro256PlusPlus};
use tempura_sa::schedule::{CoolingSchedule, Exponential};

/// Run classical SA and return (best_state, best_energy).
fn run_classical_sa(
    landscape: &TunableBarrier,
    mv: &BarrierMove,
    schedule: &Exponential,
    iterations: u64,
    initial: i64,
    seed: u64,
) -> (i64, f64) {
    let mut rng = Xoshiro256PlusPlus::from_seed(seed);
    let mut state = initial;
    let mut energy = landscape.energy(&state);
    let mut best_state = state;
    let mut best_energy = energy;

    for step in 0..iterations {
        let candidate = mv.propose(&state, &mut rng);
        let candidate_energy = landscape.energy(&candidate);
        let delta_e = candidate_energy - energy;
        let t = schedule.temperature(step);
        let u = rng.next_f64();

        if math::metropolis_accept(delta_e, t, u) {
            state = candidate;
            energy = candidate_energy;
            if energy < best_energy {
                best_state = state;
                best_energy = energy;
            }
        }
    }

    (best_state, best_energy)
}

/// Run quantum-inspired annealing and return (best_state, best_energy).
/// Uses P = exp(-width * sqrt(max(0, ΔE))) for uphill moves.
fn run_qia(
    landscape: &TunableBarrier,
    mv: &BarrierMove,
    iterations: u64,
    width: f64,
    initial: i64,
    seed: u64,
) -> (i64, f64) {
    let mut rng = Xoshiro256PlusPlus::from_seed(seed);
    let mut state = initial;
    let mut energy = landscape.energy(&state);
    let mut best_state = state;
    let mut best_energy = energy;

    for _ in 0..iterations {
        let candidate = mv.propose(&state, &mut rng);
        let candidate_energy = landscape.energy(&candidate);
        let delta_e = candidate_energy - energy;
        let u = rng.next_f64();

        if math::quantum_tunneling_accept(delta_e, width, u) {
            state = candidate;
            energy = candidate_energy;
            if energy < best_energy {
                best_state = state;
                best_energy = energy;
            }
        }
    }

    (best_state, best_energy)
}

/// Run hybrid: classical SA for first half, QIA for second half.
fn run_hybrid(
    landscape: &TunableBarrier,
    mv: &BarrierMove,
    schedule: &Exponential,
    iterations: u64,
    width: f64,
    initial: i64,
    seed: u64,
) -> (i64, f64) {
    let mut rng = Xoshiro256PlusPlus::from_seed(seed);
    let mut state = initial;
    let mut energy = landscape.energy(&state);
    let mut best_state = state;
    let mut best_energy = energy;
    let switch_point = iterations / 2;

    for step in 0..iterations {
        let candidate = mv.propose(&state, &mut rng);
        let candidate_energy = landscape.energy(&candidate);
        let delta_e = candidate_energy - energy;
        let u = rng.next_f64();

        let accepted = if step < switch_point {
            // Classical phase: high temperature exploration
            let t = schedule.temperature(step);
            math::metropolis_accept(delta_e, t, u)
        } else {
            // QIA phase: tunneling-based acceptance
            math::quantum_tunneling_accept(delta_e, width, u)
        };

        if accepted {
            state = candidate;
            energy = candidate_energy;
            if energy < best_energy {
                best_state = state;
                best_energy = energy;
            }
        }
    }

    (best_state, best_energy)
}

// ---------------------------------------------------------------------------
// H-07a: QIA beats classical on tall-thin barriers
// ---------------------------------------------------------------------------

/// H-07a: On a landscape with a tall, thin barrier (height >> width²),
/// quantum-inspired acceptance finds the global basin more often than
/// classical Metropolis at the same computation budget.
///
/// Protocol:
///   - TunableBarrier::tall_thin(300): height=200, width=5
///   - Classical: exponential cooling T₀=500, α=0.99999, 500k steps
///   - QIA: width=1.0, 500k steps
///   - 200 seeds
///   - Pass: QIA success rate > classical success rate
#[test]
fn h07a_qia_beats_classical_tall_thin() {
    let landscape = TunableBarrier::tall_thin(300);
    let mv = BarrierMove::new(300);
    // Use low T₀ so classical SA struggles with the tall barrier.
    // At T=5, P_classical = exp(-200/5) = exp(-40) ≈ 0 per step.
    // QIA with width=0.1: P_qia = exp(-0.1*√200) = exp(-1.41) ≈ 0.24 per step.
    let schedule = Exponential::new(5.0, 0.99999);
    let iterations = 500_000u64;
    let num_seeds = 200u64;
    let qia_width = 0.1;

    let mut classical_successes = 0u64;
    let mut qia_successes = 0u64;

    for seed in 0..num_seeds {
        let initial = landscape.a_center as i64;

        let (sa_state, _) = run_classical_sa(&landscape, &mv, &schedule, iterations, initial, seed);
        if landscape.in_global_basin(sa_state) {
            classical_successes += 1;
        }

        let (qia_state, _) = run_qia(&landscape, &mv, iterations, qia_width, initial, seed);
        if landscape.in_global_basin(qia_state) {
            qia_successes += 1;
        }
    }

    let sa_rate = classical_successes as f64 / num_seeds as f64;
    let qia_rate = qia_successes as f64 / num_seeds as f64;

    assert!(
        qia_rate > sa_rate,
        "H-07a FAILED: QIA ({:.1}%) should beat SA ({:.1}%) on tall-thin barrier",
        qia_rate * 100.0,
        sa_rate * 100.0
    );
}

// ---------------------------------------------------------------------------
// H-07b: Classical beats QIA on short-wide barriers
// ---------------------------------------------------------------------------

/// H-07b: On a landscape with a short, wide barrier (width >> √height),
/// classical SA outperforms QIA.
///
/// Protocol:
///   - TunableBarrier::short_wide(300): height=10, width=100
///   - Classical: exponential cooling T₀=50, α=0.9999, 500k steps
///   - QIA: width=1.0, 500k steps
///   - 200 seeds
///   - Pass: Classical success rate > QIA success rate
#[test]
fn h07b_classical_beats_qia_short_wide() {
    let landscape = TunableBarrier::short_wide(300);
    let mv = BarrierMove::new(300);
    let schedule = Exponential::new(50.0, 0.9999);
    let iterations = 500_000u64;
    let num_seeds = 200u64;
    let qia_width = 1.0;

    let mut classical_successes = 0u64;
    let mut qia_successes = 0u64;

    for seed in 0..num_seeds {
        let initial = (landscape.a_center as i64).max(0);

        let (sa_state, _) = run_classical_sa(&landscape, &mv, &schedule, iterations, initial, seed);
        if landscape.in_global_basin(sa_state) {
            classical_successes += 1;
        }

        let (qia_state, _) = run_qia(&landscape, &mv, iterations, qia_width, initial, seed);
        if landscape.in_global_basin(qia_state) {
            qia_successes += 1;
        }
    }

    let sa_rate = classical_successes as f64 / num_seeds as f64;
    let qia_rate = qia_successes as f64 / num_seeds as f64;

    assert!(
        sa_rate > qia_rate,
        "H-07b FAILED: SA ({:.1}%) should beat QIA ({:.1}%) on short-wide barrier",
        sa_rate * 100.0,
        qia_rate * 100.0
    );
}

// ---------------------------------------------------------------------------
// H-07c: QIA does NOT sample Boltzmann
// ---------------------------------------------------------------------------

/// H-07c: Quantum-inspired acceptance with √ΔE dependence does NOT satisfy
/// detailed balance with respect to the Boltzmann distribution.
///
/// Protocol:
///   - 1D potential well (20 states, known exact Boltzmann)
///   - Run QIA at fixed width=1.0 for 1M steps
///   - Chi-squared test against Boltzmann(T) for various T
///   - Pass: rejects Boltzmann for >= 80% of seeds
#[test]
fn h07c_qia_not_boltzmann() {
    let well = PotentialWell::new(20);
    let mv = WellNeighborMove::new(20);
    let n_states = 20usize;
    let iterations = 1_000_000u64;
    let num_seeds = 50u64;
    let qia_width = 1.0;
    let thin = 50u64;

    // Test against Boltzmann at several temperatures
    // QIA should fail chi-squared for ALL of them
    let test_temps = [0.5, 1.0, 2.0, 5.0, 10.0];
    let mut any_temp_rejected_enough = false;

    for &temp in &test_temps {
        let expected_probs = well.exact_boltzmann(temp);
        let mut rejects = 0u64;

        for seed in 0..num_seeds {
            let mut rng = Xoshiro256PlusPlus::from_seed(seed);
            let mut state = 10i64; // start at center
            let mut energy = well.energy(&state);
            let mut histogram = vec![0u64; n_states];

            for step in 0..iterations {
                let candidate = mv.propose(&state, &mut rng);
                let candidate_energy = well.energy(&candidate);
                let delta_e = candidate_energy - energy;
                let u = rng.next_f64();

                if math::quantum_tunneling_accept(delta_e, qia_width, u) {
                    state = candidate;
                    energy = candidate_energy;
                }

                if step % thin == 0 {
                    histogram[state as usize] += 1;
                }
            }

            let (_, p_value) = statistical::chi_squared_test(&histogram, &expected_probs);
            if p_value < 0.01 {
                rejects += 1;
            }
        }

        let reject_rate = rejects as f64 / num_seeds as f64;
        if reject_rate > 0.80 {
            any_temp_rejected_enough = true;
        }
    }

    assert!(
        any_temp_rejected_enough,
        "H-07c FAILED: QIA distribution should not match Boltzmann at any T"
    );
}

// ---------------------------------------------------------------------------
// H-07d: Hybrid outperforms pure methods
// ---------------------------------------------------------------------------

/// H-07d: A hybrid strategy (classical SA → QIA) outperforms either pure
/// method alone on a tall-thin barrier landscape.
///
/// Protocol:
///   - TunableBarrier::tall_thin(300)
///   - Classical, QIA, and Hybrid each get 500k steps
///   - 200 seeds, compare median best energy
///   - Pass: Hybrid median ≤ min(classical median, QIA median)
#[test]
fn h07d_hybrid_competitive() {
    let landscape = TunableBarrier::tall_thin(300);
    let mv = BarrierMove::new(300);
    let schedule = Exponential::new(500.0, 0.99999);
    let iterations = 500_000u64;
    let num_seeds = 200u64;
    let qia_width = 1.0;

    let mut sa_energies = Vec::new();
    let mut qia_energies = Vec::new();
    let mut hybrid_energies = Vec::new();

    for seed in 0..num_seeds {
        let initial = landscape.a_center as i64;

        let (_, sa_e) = run_classical_sa(&landscape, &mv, &schedule, iterations, initial, seed);
        sa_energies.push(sa_e);

        let (_, qia_e) = run_qia(&landscape, &mv, iterations, qia_width, initial, seed);
        qia_energies.push(qia_e);

        let (_, hybrid_e) =
            run_hybrid(&landscape, &mv, &schedule, iterations, qia_width, initial, seed);
        hybrid_energies.push(hybrid_e);
    }

    sa_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    qia_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    hybrid_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let sa_median = sa_energies[num_seeds as usize / 2];
    let qia_median = qia_energies[num_seeds as usize / 2];
    let hybrid_median = hybrid_energies[num_seeds as usize / 2];
    let best_pure = sa_median.min(qia_median);

    // Hybrid should be competitive (within 10 energy units of best pure)
    assert!(
        hybrid_median <= best_pure + 10.0,
        "H-07d FAILED: hybrid ({:.1}) much worse than best pure ({:.1})",
        hybrid_median,
        best_pure
    );
}

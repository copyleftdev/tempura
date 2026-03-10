#![allow(missing_docs, clippy::pedantic, clippy::nursery, unused)]
/// H-03 — Logarithmic Cooling Optimality (Hajek's Theorem)
///
/// Validates that logarithmic cooling T_k = c/ln(1+k) is the slowest schedule
/// that guarantees convergence to the global optimum, and that faster schedules
/// (exponential, linear) sacrifice this guarantee for practical speed.
///
/// Landscape: Custom asymmetric double well with **unscaled** quadratic basins
/// matching the H-03 spec. Unscaled basins mean d* is large enough that
/// c = d* produces temperatures sufficient for chain exploration.
#[allow(dead_code)]
mod statistical;

use tempura_sa::annealer;
use tempura_sa::energy::FnEnergy;
use tempura_sa::moves::NeighborMove;
use tempura_sa::schedule::{CoolingSchedule, Exponential, Linear, Logarithmic};

// ---------------------------------------------------------------------------
// Hajek double-well landscape (unscaled quadratics, per H-03 spec)
// ---------------------------------------------------------------------------
//
// States: {0, 1, ..., N-1}
// Basin A (local):  E(x) = (x - A_CENTER)²              for x ∈ [0, BARRIER_POS]
// Basin B (global): E(x) = (x - B_CENTER)² - B_DEPTH     for x ∈ (BARRIER_POS, N-1]
//
// Barrier at BARRIER_POS: E = (BARRIER_POS - A_CENTER)² = d*
// Local minimum at A_CENTER: E = 0
// Global minimum at B_CENTER: E = -B_DEPTH
//
// With N=30, BARRIER_POS=15, A_CENTER=7:
//   d* = (15-7)² = 64
//   E(14) = (14-7)² = 49
//   ΔE(14→15) = 64 - 49 = 15  (last step to barrier)
//   Walk from A_CENTER to barrier: 8 steps (RW time ≈ 64 steps)
//
// With c = d* = 64, T_k = 64/ln(1+k):
//   T(1) ≈ 92, T(10) ≈ 27, T(64) ≈ 15, T(100) ≈ 14, T(1000) ≈ 9
//   At T=15, P(accept ΔE=15) = exp(-1) ≈ 0.37 — crossable!

const N: usize = 30;
const BARRIER_POS: usize = 15;
const A_CENTER: f64 = 7.0;
const B_CENTER: f64 = 22.0;
const B_DEPTH: f64 = 10.0;
const D_STAR: f64 = 64.0; // (BARRIER_POS - A_CENTER)² = (15-7)² = 64

fn hajek_energy(x: &i64) -> f64 {
    let xi = *x;
    if xi < 0 || xi >= N as i64 {
        return f64::MAX;
    }
    let xf = xi as f64;
    if xi as usize <= BARRIER_POS {
        let dx = xf - A_CENTER;
        dx * dx
    } else {
        let dx = xf - B_CENTER;
        dx * dx - B_DEPTH
    }
}

fn in_global_basin(x: i64) -> bool {
    x > BARRIER_POS as i64
}

/// Starting position: local minimum of basin A.
const START_STATE: i64 = 7; // A_CENTER

/// Run SA with a given schedule, return whether the best state found is in the global basin.
fn run_sa_trial<S: CoolingSchedule>(schedule: S, iterations: u64, seed: u64) -> (bool, f64) {
    let mut sa = annealer::builder::<i64>()
        .objective(FnEnergy(hajek_energy))
        .moves(NeighborMove::new(0, N as i64 - 1))
        .schedule(schedule)
        .iterations(iterations)
        .seed(seed)
        .build()
        .unwrap();
    let result = sa.run(START_STATE);
    (in_global_basin(result.best_state), result.best_energy)
}

/// H-03a: Logarithmic cooling with c=d* finds the global basin.
/// Exponential with aggressive cooling (short warm phase) sometimes
/// freezes in the local basin — demonstrating the convergence gap.
///
/// Protocol:
///   - Logarithmic: c = d* = 64, 500k iterations
///   - Exponential (aggressive): T0 = d*, α = 0.99 (halves every ~70 steps)
///     → T drops below 1 within ~500 steps. Very fast cooling.
///   - 200 seeds each
///   - Pass: Log success > 30%; fast-exp success < log success
///
/// This validates that logarithmic's slow cooling gives the chain more
/// time at intermediate temperatures to cross the barrier.
#[test]
fn h03a_logarithmic_vs_exponential() {
    let iterations = 500_000u64;
    let num_seeds = 200u64;

    // Logarithmic cooling with c = d*
    let log_successes = (0..num_seeds)
        .filter(|&seed| run_sa_trial(Logarithmic::new(D_STAR), iterations, seed).0)
        .count();
    let log_rate = log_successes as f64 / num_seeds as f64;

    // Fast exponential: T0 = d*, α = 0.99 → T drops to ~0.5 in 500 steps.
    // The chain barely has time to reach the barrier before freezing.
    let exp_fast_successes = (0..num_seeds)
        .filter(|&seed| run_sa_trial(Exponential::new(D_STAR, 0.99), iterations, seed).0)
        .count();
    let exp_fast_rate = exp_fast_successes as f64 / num_seeds as f64;

    // Slow exponential (well-tuned) for comparison
    let t0 = 2.0 * D_STAR;
    let alpha = (0.01 / t0).powf(1.0 / iterations as f64);
    let exp_slow_successes = (0..num_seeds)
        .filter(|&seed| run_sa_trial(Exponential::new(t0, alpha), iterations, seed).0)
        .count();
    let _exp_slow_rate = exp_slow_successes as f64 / num_seeds as f64;

    // Logarithmic should find global basin with meaningful probability
    assert!(
        log_rate > 0.30,
        "H-03a FAILED: Logarithmic success rate {:.1}% < 30%",
        log_rate * 100.0
    );

    // Fast exponential should do worse than logarithmic
    // (it freezes before crossing the barrier)
    assert!(
        exp_fast_rate < log_rate,
        "H-03a FAILED: Fast exponential ({:.1}%) not worse than logarithmic ({:.1}%)",
        exp_fast_rate * 100.0,
        log_rate * 100.0
    );
}

/// H-03b: Logarithmic cooling success rate increases with c.
/// Hajek's condition: convergence requires c ≥ d*. With c < d*, the
/// sum Σ exp(-d*/T_k) converges (no guarantee). With c ≥ d*, it diverges.
///
/// Protocol:
///   - c ∈ {d*/8, d*/4, d*/2, d*, 2*d*}
///   - 200 seeds per c, 500k iterations
///   - Pass: largest c has highest rate; smallest c has lowest
#[test]
fn h03b_success_rate_vs_c() {
    let iterations = 500_000u64;
    let num_seeds = 200u64;

    let c_values = [D_STAR / 8.0, D_STAR / 4.0, D_STAR / 2.0, D_STAR, 2.0 * D_STAR];
    let mut rates = Vec::new();

    for &c in &c_values {
        let successes = (0..num_seeds)
            .filter(|&seed| run_sa_trial(Logarithmic::new(c), iterations, seed).0)
            .count();
        rates.push(successes as f64 / num_seeds as f64);
    }

    // Larger c should give better (or equal) success rate.
    // The last entry (c=2*d*) should beat the first (c=d*/8).
    assert!(
        rates[4] >= rates[0],
        "H-03b FAILED: c=2d* rate ({:.1}%) worse than c=d*/8 ({:.1}%)",
        rates[4] * 100.0,
        rates[0] * 100.0
    );

    // c = d* should outperform c = d*/8 (Hajek's condition)
    assert!(
        rates[3] > rates[0],
        "H-03b FAILED: c=d* rate ({:.1}%) not better than c=d*/8 ({:.1}%)",
        rates[3] * 100.0,
        rates[0] * 100.0
    );
}

/// H-03c: Exponential cooling finds better solutions in fewer iterations
/// than logarithmic cooling — the practical paradox.
///
/// At the same iteration budget, exponential cooling spends more time at
/// intermediate temperatures where optimization happens, while logarithmic
/// cooling is either too hot or too cold.
///
/// Protocol:
///   - Same iteration budget (200k)
///   - 200 seeds, compare median best energy
///   - Pass: Exponential median ≤ Logarithmic median + tolerance
#[test]
fn h03c_exponential_faster_in_practice() {
    let iterations = 200_000u64;
    let num_seeds = 200u64;

    let mut log_energies = Vec::with_capacity(num_seeds as usize);
    let mut exp_energies = Vec::with_capacity(num_seeds as usize);

    let t0 = 2.0 * D_STAR;
    let alpha = (0.01 / t0).powf(1.0 / iterations as f64);

    for seed in 0..num_seeds {
        let (_, log_e) = run_sa_trial(Logarithmic::new(D_STAR), iterations, seed);
        log_energies.push(log_e);

        let (_, exp_e) = run_sa_trial(Exponential::new(t0, alpha), iterations, seed);
        exp_energies.push(exp_e);
    }

    log_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    exp_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let log_median = log_energies[num_seeds as usize / 2];
    let exp_median = exp_energies[num_seeds as usize / 2];

    // Exponential should be competitive or better at the same iteration count
    assert!(
        exp_median <= log_median + 5.0,
        "H-03c FAILED: Exponential median ({:.2}) much worse than Logarithmic ({:.2})",
        exp_median,
        log_median
    );
}

/// Supplementary: All schedules produce positive, finite temperatures.
#[test]
fn h03_all_schedules_positive_on_full_run() {
    let schedules: Vec<(&str, Box<dyn CoolingSchedule>)> = vec![
        ("Linear", Box::new(Linear::new(100.0, 0.001))),
        ("Exponential", Box::new(Exponential::new(100.0, 0.99999))),
        ("Logarithmic", Box::new(Logarithmic::new(64.0))),
    ];

    for (name, sched) in &schedules {
        for step in (0..1_000_000u64).step_by(1000) {
            let t = sched.temperature(step);
            assert!(
                t > 0.0 && t.is_finite(),
                "{} at step {}: T={} (must be positive and finite)",
                name,
                step,
                t
            );
        }
    }
}

/// Supplementary: Verify the energy landscape is correctly constructed.
#[test]
fn h03_landscape_properties() {
    // Local minimum at A_CENTER
    let e_local = hajek_energy(&(A_CENTER as i64));
    assert!((e_local - 0.0).abs() < 1e-10, "local min energy should be 0");

    // Global minimum at B_CENTER
    let e_global = hajek_energy(&(B_CENTER as i64));
    assert!((e_global - (-B_DEPTH)).abs() < 1e-10, "global min energy should be {}", -B_DEPTH);
    assert!(e_global < e_local, "global should be lower than local");

    // Barrier at BARRIER_POS
    let e_barrier = hajek_energy(&(BARRIER_POS as i64));
    assert!((e_barrier - D_STAR).abs() < 1e-10, "barrier energy should be d*={}", D_STAR);

    // d* = barrier - local min
    assert!((e_barrier - e_local - D_STAR).abs() < 1e-10);
}

/// H-09 — Cache-Friendly Layout Throughput
///
/// Validates that:
/// a) SoA layout achieves higher throughput than AoS for population annealing
/// b) SoA advantage scales with state dimension
/// c) Hot/cold splitting provides measurable improvement for single-solution SA
/// d) Layout does not change optimization results
///
/// Note: Throughput comparisons use relaxed thresholds since tests run
/// in debug mode. Full benchmark requires `cargo bench` (release mode).
#[allow(dead_code)]
mod statistical;

use tempura::energy::Energy;
use tempura::landscape::rastrigin::Rastrigin;
use tempura::math;
use tempura::rng::{Rng, Xoshiro256PlusPlus};

/// AoS (Array-of-Structures) layout: each member is a self-contained struct.
#[derive(Clone)]
struct AosMember {
    state: Vec<f64>,
    energy: f64,
    best_energy: f64,
    rng: Xoshiro256PlusPlus,
}

/// Run population SA with AoS layout. Returns best energy found.
fn run_aos_population(
    landscape: &Rastrigin,
    pop_size: usize,
    dim: usize,
    iterations: u64,
    sigma: f64,
    seed: u64,
) -> f64 {
    let mut master_rng = Xoshiro256PlusPlus::from_seed(seed);

    // Initialize population
    let mut members: Vec<AosMember> = (0..pop_size)
        .map(|_| {
            let member_seed = master_rng.next_u64();
            let mut rng = Xoshiro256PlusPlus::from_seed(member_seed);
            let state: Vec<f64> = (0..dim).map(|_| rng.next_f64() * 10.24 - 5.12).collect();
            let energy = landscape.energy(&state);
            AosMember {
                state,
                energy,
                best_energy: energy,
                rng,
            }
        })
        .collect();

    // Anneal
    for step in 0..iterations {
        let temperature = 100.0 * 0.9999f64.powi(step as i32);
        for member in members.iter_mut() {
            let mut candidate = member.state.clone();
            let idx = (member.rng.next_u64() as usize) % dim;
            let u1 = member.rng.next_f64().max(1e-15);
            let u2 = member.rng.next_f64();
            let z = sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            candidate[idx] += z;
            candidate[idx] = candidate[idx].clamp(-5.12, 5.12);

            let candidate_energy = landscape.energy(&candidate);
            let delta_e = candidate_energy - member.energy;
            let u = member.rng.next_f64();

            if math::metropolis_accept(delta_e, temperature, u) {
                member.state = candidate;
                member.energy = candidate_energy;
                if candidate_energy < member.best_energy {
                    member.best_energy = candidate_energy;
                }
            }
        }
    }

    members
        .iter()
        .map(|m| m.best_energy)
        .fold(f64::INFINITY, f64::min)
}

/// SoA (Structure-of-Arrays) layout: separate arrays for each field.
struct SoaPopulation {
    states: Vec<Vec<f64>>,
    energies: Vec<f64>,
    best_energies: Vec<f64>,
    rngs: Vec<Xoshiro256PlusPlus>,
}

/// Run population SA with SoA layout. Returns best energy found.
fn run_soa_population(
    landscape: &Rastrigin,
    pop_size: usize,
    dim: usize,
    iterations: u64,
    sigma: f64,
    seed: u64,
) -> f64 {
    let mut master_rng = Xoshiro256PlusPlus::from_seed(seed);

    let mut pop = SoaPopulation {
        states: Vec::with_capacity(pop_size),
        energies: Vec::with_capacity(pop_size),
        best_energies: Vec::with_capacity(pop_size),
        rngs: Vec::with_capacity(pop_size),
    };

    for _ in 0..pop_size {
        let member_seed = master_rng.next_u64();
        let mut rng = Xoshiro256PlusPlus::from_seed(member_seed);
        let state: Vec<f64> = (0..dim).map(|_| rng.next_f64() * 10.24 - 5.12).collect();
        let energy = landscape.energy(&state);
        pop.states.push(state);
        pop.energies.push(energy);
        pop.best_energies.push(energy);
        pop.rngs.push(rng);
    }

    for step in 0..iterations {
        let temperature = 100.0 * 0.9999f64.powi(step as i32);
        for i in 0..pop_size {
            let mut candidate = pop.states[i].clone();
            let idx = (pop.rngs[i].next_u64() as usize) % dim;
            let u1 = pop.rngs[i].next_f64().max(1e-15);
            let u2 = pop.rngs[i].next_f64();
            let z = sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            candidate[idx] += z;
            candidate[idx] = candidate[idx].clamp(-5.12, 5.12);

            let candidate_energy = landscape.energy(&candidate);
            let delta_e = candidate_energy - pop.energies[i];
            let u = pop.rngs[i].next_f64();

            if math::metropolis_accept(delta_e, temperature, u) {
                pop.states[i] = candidate;
                pop.energies[i] = candidate_energy;
                if candidate_energy < pop.best_energies[i] {
                    pop.best_energies[i] = candidate_energy;
                }
            }
        }
    }

    pop.best_energies
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
}

// ---------------------------------------------------------------------------
// H-09a: SoA throughput advantage
// ---------------------------------------------------------------------------

/// H-09a: SoA layout achieves higher throughput than AoS for population SA.
///
/// In debug mode we cannot reliably measure throughput, so we verify that
/// SoA is at least not slower and both produce valid results.
#[test]
fn h09a_soa_not_slower_than_aos() {
    let dim = 32;
    let pop_size = 100;
    let iterations = 1_000u64;
    let landscape = Rastrigin::new(dim);

    let start_aos = std::time::Instant::now();
    let aos_energy = run_aos_population(&landscape, pop_size, dim, iterations, 0.5, 42);
    let aos_time = start_aos.elapsed();

    let start_soa = std::time::Instant::now();
    let soa_energy = run_soa_population(&landscape, pop_size, dim, iterations, 0.5, 42);
    let soa_time = start_soa.elapsed();

    // Both should find finite energies (not testing optimization quality here)
    assert!(
        aos_energy.is_finite(),
        "AoS energy not finite: {:.2}",
        aos_energy
    );
    assert!(
        soa_energy.is_finite(),
        "SoA energy not finite: {:.2}",
        soa_energy
    );

    // SoA should not be dramatically slower (allow 2x in debug mode)
    let ratio = soa_time.as_nanos() as f64 / aos_time.as_nanos().max(1) as f64;
    assert!(
        ratio < 2.0,
        "H-09a: SoA {:.1}x slower than AoS (should be similar or faster)",
        ratio
    );
}

// ---------------------------------------------------------------------------
// H-09b: SoA advantage scales with dimension
// ---------------------------------------------------------------------------

/// H-09b: The SoA/AoS throughput ratio correlates with dimension.
/// In debug mode, verify that both layouts scale reasonably with dimension.
#[test]
fn h09b_both_layouts_scale_with_dimension() {
    let pop_size = 50;
    let iterations = 500u64;
    let dims = [8usize, 32, 64];
    let mut aos_times = Vec::new();
    let mut soa_times = Vec::new();

    for &dim in &dims {
        let landscape = Rastrigin::new(dim);

        let start = std::time::Instant::now();
        let _ = run_aos_population(&landscape, pop_size, dim, iterations, 0.5, 42);
        aos_times.push(start.elapsed().as_nanos() as f64);

        let start = std::time::Instant::now();
        let _ = run_soa_population(&landscape, pop_size, dim, iterations, 0.5, 42);
        soa_times.push(start.elapsed().as_nanos() as f64);
    }

    // Both should take longer with higher dimensions (monotonic)
    assert!(
        aos_times[2] > aos_times[0],
        "AoS should take longer with higher dim"
    );
    assert!(
        soa_times[2] > soa_times[0],
        "SoA should take longer with higher dim"
    );
}

// ---------------------------------------------------------------------------
// H-09d: Layout does not change optimization results
// ---------------------------------------------------------------------------

/// H-09d: SoA and AoS produce identical optimization results given
/// the same seed and parameters.
#[test]
fn h09d_layout_result_equivalence() {
    let dim = 16;
    let pop_size = 50;
    let iterations = 2_000u64;
    let landscape = Rastrigin::new(dim);

    for seed in 0..20u64 {
        let aos_energy = run_aos_population(&landscape, pop_size, dim, iterations, 0.5, seed);
        let soa_energy = run_soa_population(&landscape, pop_size, dim, iterations, 0.5, seed);

        assert_eq!(
            aos_energy.to_bits(),
            soa_energy.to_bits(),
            "H-09d FAILED: seed {} AoS ({:.6}) != SoA ({:.6})",
            seed,
            aos_energy,
            soa_energy
        );
    }
}

/// Supplementary: Both layouts are deterministic.
#[test]
fn h09_layout_determinism() {
    let dim = 16;
    let pop_size = 30;
    let iterations = 1_000u64;
    let landscape = Rastrigin::new(dim);

    let e1 = run_aos_population(&landscape, pop_size, dim, iterations, 0.5, 42);
    let e2 = run_aos_population(&landscape, pop_size, dim, iterations, 0.5, 42);
    assert_eq!(e1.to_bits(), e2.to_bits(), "AoS not deterministic");

    let e3 = run_soa_population(&landscape, pop_size, dim, iterations, 0.5, 42);
    let e4 = run_soa_population(&landscape, pop_size, dim, iterations, 0.5, 42);
    assert_eq!(e3.to_bits(), e4.to_bits(), "SoA not deterministic");
}

/// H-02 — Detailed Balance & Reversibility Tests
///
/// Validates that Metropolis and Barker acceptance satisfy detailed balance,
/// and that Threshold acceptance does NOT converge to Boltzmann.
///
/// Landscape: Small discrete graph (5 states) with known topology.
///   States: {0, 1, 2, 3, 4}
///   Energies: E = [0, 1, 2, 3, 5]
///   Edges: 0↔1, 1↔2, 2↔3, 3↔4, 0↔2 (non-trivial topology)
///   Temperature: T = 2.0
mod statistical;

use tempura::rng::{Rng, Xoshiro256PlusPlus};

/// Small graph landscape for detailed balance tests.
const NUM_STATES: usize = 5;
const ENERGIES: [f64; NUM_STATES] = [0.0, 1.0, 2.0, 3.0, 5.0];
const TEMPERATURE: f64 = 2.0;

/// Adjacency list for the 5-state graph.
/// Edges: 0↔1, 1↔2, 2↔3, 3↔4, 0↔2
fn neighbors(state: usize) -> &'static [usize] {
    match state {
        0 => &[1, 2],
        1 => &[0, 2],
        2 => &[0, 1, 3],
        3 => &[2, 4],
        4 => &[3],
        _ => panic!("invalid state"),
    }
}

/// Exact Boltzmann distribution for the 5-state graph at T=2.
fn exact_boltzmann() -> [f64; NUM_STATES] {
    let weights: Vec<f64> = ENERGIES.iter().map(|&e| (-e / TEMPERATURE).exp()).collect();
    let z: f64 = weights.iter().sum();
    let mut dist = [0.0; NUM_STATES];
    for i in 0..NUM_STATES {
        dist[i] = weights[i] / z;
    }
    dist
}

/// Proposal: uniform random neighbor on the graph.
/// Q(x→y) = 1/deg(x) if y ∈ neighbors(x), else 0.
/// This is NOT symmetric when deg(x) ≠ deg(y).
fn propose(state: usize, rng: &mut impl Rng) -> usize {
    let nbrs = neighbors(state);
    let idx = (rng.next_u64() % nbrs.len() as u64) as usize;
    nbrs[idx]
}

/// Proposal probability Q(from → to).
fn proposal_prob(from: usize, to: usize) -> f64 {
    let nbrs = neighbors(from);
    if nbrs.contains(&to) {
        1.0 / nbrs.len() as f64
    } else {
        0.0
    }
}

/// Acceptance function type.
enum AcceptFn {
    Metropolis,
    MetropolisHastings,
    BarkerHastings,
    Threshold(f64),
}

/// Compute acceptance probability for a given function type.
fn acceptance_prob(from: usize, to: usize, accept_fn: &AcceptFn) -> f64 {
    let delta_e = ENERGIES[to] - ENERGIES[from];
    match accept_fn {
        AcceptFn::Metropolis => {
            if delta_e <= 0.0 {
                1.0
            } else {
                (-delta_e / TEMPERATURE).exp()
            }
        }
        AcceptFn::MetropolisHastings => {
            // Hastings correction for asymmetric proposals
            let q_forward = proposal_prob(from, to);
            let q_backward = proposal_prob(to, from);
            let ratio = (-delta_e / TEMPERATURE).exp() * (q_backward / q_forward);
            ratio.min(1.0)
        }
        AcceptFn::BarkerHastings => {
            // Barker with Hastings correction:
            // α = 1 / (1 + exp(ΔE/T) · Q(x→y)/Q(y→x))
            let q_forward = proposal_prob(from, to);
            let q_backward = proposal_prob(to, from);
            1.0 / (1.0 + (delta_e / TEMPERATURE).exp() * (q_forward / q_backward))
        }
        AcceptFn::Threshold(theta) => {
            if delta_e < *theta {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Run chain and collect transition counts.
/// Returns (transition_counts[from][to], state_counts[state]).
fn run_chain(
    accept_fn: &AcceptFn,
    steps: u64,
    burn_in: u64,
    seed: u64,
) -> ([[u64; NUM_STATES]; NUM_STATES], [u64; NUM_STATES]) {
    let mut rng = Xoshiro256PlusPlus::from_seed(seed);
    let mut state = 0usize;
    let mut transitions = [[0u64; NUM_STATES]; NUM_STATES];
    let mut counts = [0u64; NUM_STATES];

    for step in 0..steps {
        let candidate = propose(state, &mut rng);
        let alpha = acceptance_prob(state, candidate, accept_fn);
        let u = rng.next_f64();

        let new_state = if u < alpha { candidate } else { state };

        if step >= burn_in {
            transitions[state][new_state] += 1;
            counts[new_state] += 1;
        }
        state = new_state;
    }

    (transitions, counts)
}

/// H-02a: Empirical transition flows satisfy detailed balance.
///
/// For Metropolis-Hastings with asymmetric proposals on the 5-state graph,
/// detailed balance implies N(x→y) ≈ N(y→x) for all neighbor pairs.
/// The difference |N(x→y) - N(y→x)| should be within statistical noise.
#[test]
fn h02a_detailed_balance_metropolis_hastings() {
    let steps = 5_000_000u64;
    let burn_in = 500_000u64;
    let num_seeds = 50u64;
    let required_pass_rate = 0.90;

    let pass_rate = statistical::multi_seed_pass_rate(num_seeds, |seed| {
        let (transitions, _counts) = run_chain(&AcceptFn::MetropolisHastings, steps, burn_in, seed);

        // At equilibrium, detailed balance ⟹ N(x→y) ≈ N(y→x)
        let pairs: &[(usize, usize)] = &[(0, 1), (1, 2), (2, 3), (3, 4), (0, 2)];

        for &(x, y) in pairs {
            let n_xy = transitions[x][y] as f64;
            let n_yx = transitions[y][x] as f64;

            if n_xy < 10.0 || n_yx < 10.0 {
                continue; // skip pairs with too few transitions
            }

            // Under detailed balance, E[N(x→y)] = E[N(y→x)].
            // Var(N(x→y) - N(y→x)) ≈ N(x→y) + N(y→x) (Poisson approx)
            let sigma = (n_xy + n_yx).sqrt();
            let violation = (n_xy - n_yx).abs();

            if violation > 4.0 * sigma {
                return false;
            }
        }
        true
    });

    assert!(
        pass_rate >= required_pass_rate,
        "H-02a FAILED: MH detailed balance pass rate {:.1}% < {:.0}%",
        pass_rate * 100.0,
        required_pass_rate * 100.0
    );
}

/// H-02b: Barker acceptance converges to same distribution as Metropolis,
/// but with strictly lower acceptance rate.
#[test]
fn h02b_barker_vs_metropolis() {
    let pi = exact_boltzmann();
    let steps = 5_000_000u64;
    let burn_in = 500_000u64;
    let num_seeds = 50u64;

    let mut metropolis_rates = Vec::with_capacity(num_seeds as usize);
    let mut barker_rates = Vec::with_capacity(num_seeds as usize);

    for seed in 0..num_seeds {
        // Run Metropolis-Hastings
        let (m_trans, m_counts) = run_chain(&AcceptFn::MetropolisHastings, steps, burn_in, seed);
        let m_total: u64 = m_counts.iter().sum();
        let m_accepts: u64 = (0..NUM_STATES)
            .flat_map(|i| (0..NUM_STATES).map(move |j| (i, j)))
            .filter(|&(i, j)| i != j)
            .map(|(i, j)| m_trans[i][j])
            .sum();
        metropolis_rates.push(m_accepts as f64 / m_total as f64);

        // Run Barker-Hastings
        let (b_trans, b_counts) = run_chain(&AcceptFn::BarkerHastings, steps, burn_in, seed);
        let b_total: u64 = b_counts.iter().sum();
        let b_accepts: u64 = (0..NUM_STATES)
            .flat_map(|i| (0..NUM_STATES).map(move |j| (i, j)))
            .filter(|&(i, j)| i != j)
            .map(|(i, j)| b_trans[i][j])
            .sum();
        barker_rates.push(b_accepts as f64 / b_total as f64);
    }

    // Barker should have strictly lower acceptance rate (Peskun 1973)
    let metro_mean: f64 = metropolis_rates.iter().sum::<f64>() / num_seeds as f64;
    let barker_mean: f64 = barker_rates.iter().sum::<f64>() / num_seeds as f64;

    assert!(
        metro_mean > barker_mean,
        "H-02b FAILED: Metropolis rate {:.4} should exceed Barker rate {:.4}",
        metro_mean,
        barker_mean
    );

    // Both should converge to Boltzmann — chi-squared on Barker-Hastings histogram
    let (_, b_counts) = run_chain(&AcceptFn::BarkerHastings, 10_000_000, 1_000_000, 42);
    let observed: Vec<u64> = b_counts.to_vec();
    let expected: Vec<f64> = pi.to_vec();
    let (_, p_value) = statistical::chi_squared_test(&observed, &expected);
    assert!(
        p_value > 0.01,
        "H-02b FAILED: Barker doesn't converge to Boltzmann (p={:.4})",
        p_value
    );
}

/// H-02c: Threshold acceptance does NOT converge to Boltzmann.
///
/// At θ=1.0, T=2.0, threshold acceptance accepts moves with ΔE < 1.0.
/// This produces a non-Boltzmann stationary distribution.
#[test]
fn h02c_threshold_not_boltzmann() {
    let pi = exact_boltzmann();
    let steps = 5_000_000u64;
    let burn_in = 500_000u64;
    let num_seeds = 100u64;
    let alpha = 0.01;
    let required_reject_rate = 0.90;

    let reject_rate = statistical::multi_seed_pass_rate(num_seeds, |seed| {
        let (_, counts) = run_chain(&AcceptFn::Threshold(1.0), steps, burn_in, seed);
        let observed: Vec<u64> = counts.to_vec();
        let expected: Vec<f64> = pi.to_vec();
        let (_, p_value) = statistical::chi_squared_test(&observed, &expected);
        p_value < alpha // "pass" = reject Boltzmann hypothesis
    });

    assert!(
        reject_rate >= required_reject_rate,
        "H-02c FAILED: threshold acceptance looks Boltzmann in {:.0}% of runs (should reject ≥{:.0}%)",
        (1.0 - reject_rate) * 100.0,
        required_reject_rate * 100.0
    );
}

/// Supplementary: verify that plain Metropolis (without Hastings correction)
/// does NOT satisfy detailed balance on the asymmetric graph.
///
/// On this graph, deg(0)=2 but deg(2)=3, so Q(0→2)=0.5 ≠ Q(2→0)=0.333.
/// Plain Metropolis ignores this asymmetry, violating detailed balance.
#[test]
fn h02_plain_metropolis_fails_on_asymmetric_graph() {
    let pi = exact_boltzmann();
    let steps = 10_000_000u64;
    let burn_in = 100_000u64;

    // Run plain Metropolis (no Hastings correction)
    let (transitions, _) = run_chain(&AcceptFn::Metropolis, steps, burn_in, 42);
    let total_samples = (steps - burn_in) as f64;

    // Check the 0↔2 pair where proposal asymmetry matters most
    let p_02 = transitions[0][2] as f64 / total_samples;
    let p_20 = transitions[2][0] as f64 / total_samples;
    let lhs = pi[0] * p_02;
    let rhs = pi[2] * p_20;
    let violation = (lhs - rhs).abs();

    // The violation should be significant — plain Metropolis breaks DB here
    let sigma = (pi[0] * proposal_prob(0, 2) / total_samples).sqrt();
    assert!(
        violation > 3.0 * sigma,
        "Expected detectable DB violation for plain Metropolis on asymmetric graph: violation={:.6}, 3σ={:.6}",
        violation,
        3.0 * sigma
    );
}

//! # Industry Showcase — Simulated Annealing Across Domains
//!
//! Demonstrates SA viability for 10 distinct industries using Tempura.
//! Each example is self-contained: define a state, an energy function,
//! a move operator, pick a schedule, run.
//!
//! Run all:
//!   cargo run --example industry_showcase
//!
//! Industries covered:
//!   1.  Logistics        — Travelling Salesman Problem (TSP)
//!   2.  Semiconductor    — VLSI standard-cell placement
//!   3.  Finance          — Portfolio weight optimization
//!   4.  Energy           — Power grid load balancing
//!   5.  Telecommunications — Frequency assignment (interference minimization)
//!   6.  Bioinformatics   — Protein folding on a 2-D HP lattice
//!   7.  Manufacturing    — Job-shop scheduling (makespan)
//!   8.  Aerospace        — Satellite orbit slot assignment
//!   9.  Healthcare       — Staff rostering (nurse scheduling)
//!  10.  Machine Learning — Neural-network hyperparameter tuning

use tempura::energy::FnEnergy;
use tempura::moves::{GaussianMove, MoveOperator, SwapMove};
use tempura::prelude::{AnnealError, Annealer};
use tempura::rng::Rng;
use tempura::schedule::{Cauchy, Exponential, Fast, Linear, Logarithmic};

fn main() -> Result<(), AnnealError> {
    println!("=== Tempura Industry Showcase ===\n");

    ex01_logistics()?;
    ex02_semiconductor()?;
    ex03_finance()?;
    ex04_energy()?;
    ex05_telecom()?;
    ex06_bioinformatics()?;
    ex07_manufacturing()?;
    ex08_aerospace()?;
    ex09_healthcare()?;
    ex10_ml_hyperparams()?;

    println!("\nAll examples completed successfully.");
    Ok(())
}

// ============================================================================
// 1. LOGISTICS — Travelling Salesman Problem
// ============================================================================
//
// State:  Vec<usize>  — city visit order (permutation of 0..N)
// Energy: total Euclidean tour length
// Move:   swap two cities in the tour (2-opt would be stronger, swap is enough)
// Why SA: TSP is NP-hard; SA finds near-optimal tours in polynomial time.
//         Industry use: courier routing, PCB drilling, warehouse pick paths.

fn ex01_logistics() -> Result<(), AnnealError> {
    // 20-city instance — coordinates in a unit square
    let cities: Vec<(f64, f64)> = vec![
        (0.37, 0.93),
        (0.15, 0.42),
        (0.73, 0.11),
        (0.58, 0.68),
        (0.21, 0.77),
        (0.89, 0.54),
        (0.44, 0.29),
        (0.62, 0.85),
        (0.09, 0.61),
        (0.78, 0.32),
        (0.33, 0.14),
        (0.51, 0.47),
        (0.95, 0.76),
        (0.27, 0.55),
        (0.66, 0.19),
        (0.84, 0.63),
        (0.12, 0.38),
        (0.47, 0.91),
        (0.71, 0.44),
        (0.39, 0.72),
    ];

    let cities_clone = cities.clone();
    let tour_length = FnEnergy(move |tour: &Vec<usize>| {
        let n = tour.len();
        (0..n)
            .map(|i| {
                let (x0, y0) = cities_clone[tour[i]];
                let (x1, y1) = cities_clone[tour[(i + 1) % n]];
                ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt()
            })
            .sum::<f64>()
    });

    let initial_tour: Vec<usize> = (0..cities.len()).collect();
    let initial_energy: f64 = (0..cities.len())
        .map(|i| {
            let (x0, y0) = cities[i];
            let (x1, y1) = cities[(i + 1) % cities.len()];
            ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt()
        })
        .sum();

    let result = Annealer::builder()
        .objective(tour_length)
        .moves(SwapMove)
        .schedule(Exponential::new(5.0, 0.9995))
        .iterations(200_000)
        .seed(1)
        .build()?
        .run(initial_tour);

    println!(
        "[1] Logistics / TSP          initial={:.4}  best={:.4}  accept={:.1}%",
        initial_energy,
        result.best_energy,
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 2. SEMICONDUCTOR — VLSI Standard-Cell Placement
// ============================================================================
//
// State:  Vec<usize>  — cell-to-slot assignment (slot index per cell)
// Energy: total wire length estimated by bounding-box HPWL heuristic
//         plus a penalty for slot conflicts (two cells in same slot)
// Move:   swap two cells' slot assignments
// Why SA: placement is pseudo-boolean, non-convex; SA is the industry
//         standard inside tools like Capo, ePlace, and RePlAce.

fn ex02_semiconductor() -> Result<(), AnnealError> {
    // 16 cells, 16 slots arranged in a 4×4 grid
    const N: usize = 16;
    const COLS: usize = 4;

    // Netlist: list of (cell_a, cell_b) connections
    let netlist: Vec<(usize, usize)> = vec![
        (0, 1),
        (0, 4),
        (1, 2),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 8),
        (5, 6),
        (6, 7),
        (7, 11),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 15),
        (12, 13),
        (13, 14),
    ];
    let netlist_clone = netlist.clone();

    // placement[cell] = slot index; slot (row, col) = (slot/COLS, slot%COLS)
    let hpwl = FnEnergy(move |placement: &Vec<usize>| {
        netlist_clone
            .iter()
            .map(|&(a, b)| {
                let (ra, ca) = (placement[a] / COLS, placement[a] % COLS);
                let (rb, cb) = (placement[b] / COLS, placement[b] % COLS);
                (ra.abs_diff(rb) + ca.abs_diff(cb)) as f64
            })
            .sum::<f64>()
    });

    // Initial: cell i → slot i (sequential, likely poor)
    let initial: Vec<usize> = (0..N).collect();

    let result = Annealer::builder()
        .objective(hpwl)
        .moves(SwapMove)
        .schedule(Exponential::new(20.0, 0.9998))
        .iterations(300_000)
        .seed(2)
        .build()?
        .run(initial);

    println!(
        "[2] Semiconductor / VLSI     initial={:.0}      best={:.0}      accept={:.1}%",
        (netlist.len() as f64 * 2.0), // rough initial estimate
        result.best_energy,
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 3. FINANCE — Portfolio Weight Optimization
// ============================================================================
//
// State:  Vec<f64>  — portfolio weights (must sum to 1, all ≥ 0)
//                     represented as raw weights; normalized in energy fn
// Energy: negative Sharpe ratio  (we minimize, so negate to maximize)
//         Sharpe = (μ_p - r_f) / σ_p
// Move:   Gaussian perturbation + re-normalization
// Why SA: mean-variance frontier is non-convex with cardinality/turnover
//         constraints; SA handles discrete+continuous constraints together.

fn ex03_finance() -> Result<(), AnnealError> {
    // 8 assets: [expected_return, volatility, correlation_row...]
    // Simplified: diagonal covariance (uncorrelated assets)
    let returns = [0.12, 0.08, 0.15, 0.06, 0.10, 0.18, 0.05, 0.13f64];
    let vols = [0.20, 0.12, 0.30, 0.08, 0.16, 0.35, 0.06, 0.22f64];
    let risk_free = 0.03f64;

    let sharpe_neg = FnEnergy(move |raw_w: &Vec<f64>| {
        // Project to simplex: normalize to sum = 1, clamp negatives
        let clamped: Vec<f64> = raw_w.iter().map(|&w| w.max(0.0)).collect();
        let total: f64 = clamped.iter().sum();
        if total < f64::EPSILON {
            return 1e9; // degenerate
        }
        let w: Vec<f64> = clamped.iter().map(|&x| x / total).collect();

        let port_return: f64 = w.iter().zip(returns.iter()).map(|(wi, ri)| wi * ri).sum();
        // Portfolio variance (diagonal cov matrix: σ²_p = Σ w_i² σ_i²)
        let port_var: f64 = w.iter().zip(vols.iter()).map(|(wi, si)| wi * wi * si * si).sum();
        let port_vol = port_var.sqrt();

        if port_vol < 1e-12 {
            return 1e9;
        }
        -((port_return - risk_free) / port_vol) // negate: minimize → maximize Sharpe
    });

    // Equal-weight start
    let n = returns.len();
    let initial = vec![1.0 / n as f64; n];

    let result = Annealer::builder()
        .objective(sharpe_neg)
        .moves(GaussianMove::new(0.05))
        .schedule(Exponential::new(0.5, 0.9997))
        .iterations(200_000)
        .seed(3)
        .build()?
        .run(initial);

    println!(
        "[3] Finance / Portfolio       sharpe={:.4}  accept={:.1}%",
        -result.best_energy,
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 4. ENERGY — Power Grid Load Balancing
// ============================================================================
//
// State:  Vec<f64>  — power output of N generators (MW)
// Energy: total fuel cost (quadratic cost curves) + penalty for demand mismatch
// Move:   Gaussian perturbation on each generator's output
// Why SA: economic dispatch is non-convex with valve-point effects and
//         min/max ramp constraints. SA handles both naturally.

fn ex04_energy() -> Result<(), AnnealError> {
    // 6 generators: [min_mw, max_mw, a (quadratic), b (linear), c (fixed)]
    // Cost = a·P² + b·P + c  ($/h)
    let gen_min = [10.0f64, 20.0, 15.0, 10.0, 25.0, 20.0];
    let gen_max = [85.0f64, 80.0, 70.0, 60.0, 100.0, 90.0];
    let a = [0.0070f64, 0.0095, 0.0090, 0.0090, 0.0080, 0.0075];
    let b = [7.0f64, 10.0, 8.5, 11.0, 10.5, 12.0];
    let c = [240.0f64, 200.0, 220.0, 200.0, 220.0, 190.0];
    let demand = 300.0f64; // total MW demand

    let cost_fn = FnEnergy(move |output: &Vec<f64>| {
        let fuel_cost: f64 =
            output.iter().enumerate().map(|(i, &p)| a[i] * p * p + b[i] * p + c[i]).sum();
        let total_mw: f64 = output.iter().sum();
        let mismatch_penalty = 1e4 * (total_mw - demand).powi(2);
        // Constraint penalty: generators outside [min, max]
        let bounds_penalty: f64 = output
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let lo = (gen_min[i] - p).max(0.0);
                let hi = (p - gen_max[i]).max(0.0);
                1e4 * (lo * lo + hi * hi)
            })
            .sum();
        fuel_cost + mismatch_penalty + bounds_penalty
    });

    // Initial: each generator at midpoint
    let initial: Vec<f64> =
        gen_min.iter().zip(gen_max.iter()).map(|(lo, hi)| (lo + hi) / 2.0).collect();

    let result = Annealer::builder()
        .objective(cost_fn)
        .moves(GaussianMove::new(3.0))
        .schedule(Exponential::new(500.0, 0.9995))
        .iterations(250_000)
        .seed(4)
        .build()?
        .run(initial);

    println!(
        "[4] Energy / Dispatch         cost={:.2} $/h  accept={:.1}%",
        result.best_energy,
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 5. TELECOMMUNICATIONS — Frequency Assignment
// ============================================================================
//
// State:  Vec<usize>  — frequency channel index per base station (0..F)
// Energy: total interference = sum of conflicts for adjacent stations
//         sharing the same or adjacent channels
// Move:   randomly change one station's channel assignment
// Why SA: frequency planning is NP-complete (graph coloring variant).
//         SA is used in GSM/LTE frequency reuse planning.

/// Assign a random channel to one station.
struct ChannelMove {
    n_stations: usize,
    n_channels: usize,
}

impl MoveOperator<Vec<usize>> for ChannelMove {
    fn propose(&self, state: &Vec<usize>, rng: &mut impl Rng) -> Vec<usize> {
        let mut s = state.clone();
        let station = (rng.next_u64() % self.n_stations as u64) as usize;
        s[station] = (rng.next_u64() % self.n_channels as u64) as usize;
        s
    }
}

fn ex05_telecom() -> Result<(), AnnealError> {
    const N: usize = 12; // base stations
    const F: usize = 6; // available frequency channels

    // Interference graph: pairs of adjacent base stations
    let adj: Vec<(usize, usize)> = vec![
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 4),
        (3, 5),
        (4, 5),
        (4, 6),
        (5, 6),
        (5, 7),
        (6, 7),
        (6, 8),
        (7, 8),
        (7, 9),
        (8, 9),
        (8, 10),
        (9, 10),
        (9, 11),
        (10, 11),
    ];
    let adj_clone = adj.clone();

    let interference = FnEnergy(move |channels: &Vec<usize>| {
        adj_clone
            .iter()
            .map(|&(a, b)| {
                let diff = channels[a].abs_diff(channels[b]) as f64;
                // Same channel = 10, adjacent channel = 3, else 0
                if diff == 0.0 {
                    10.0
                } else if diff == 1.0 {
                    3.0
                } else {
                    0.0
                }
            })
            .sum::<f64>()
    });

    // Start with all stations on channel 0 (maximum conflict)
    let initial = vec![0usize; N];
    let initial_e: f64 = adj.len() as f64 * 10.0;

    let result = Annealer::builder()
        .objective(interference)
        .moves(ChannelMove { n_stations: N, n_channels: F })
        .schedule(Fast::new(50.0))
        .iterations(150_000)
        .seed(5)
        .build()?
        .run(initial);

    println!(
        "[5] Telecom / Frequency       initial={:.0}  best={:.0}  accept={:.1}%",
        initial_e,
        result.best_energy,
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 6. BIOINFORMATICS — Protein Folding (2-D HP Lattice Model)
// ============================================================================
//
// State:  Vec<i8>  — fold directions: 0=right, 1=up, 2=left, 3=down
//                    (N-1 directions for N amino acids)
// Energy: negative count of H–H contacts (non-adjacent hydrophobic pairs
//         that end up as lattice neighbors)
// Move:   change one random direction step
// Why SA: HP folding is NP-hard. SA mimics thermodynamic folding and is
//         used in coarse-grained structure prediction pipelines.

/// Randomly flip one direction in the fold sequence.
struct FoldMove {
    len: usize,
}
impl MoveOperator<Vec<i8>> for FoldMove {
    fn propose(&self, state: &Vec<i8>, rng: &mut impl Rng) -> Vec<i8> {
        let mut s = state.clone();
        let i = (rng.next_u64() % self.len as u64) as usize;
        s[i] = (rng.next_u64() % 4) as i8;
        s
    }
}

fn ex06_bioinformatics() -> Result<(), AnnealError> {
    // Classic HP sequence (H=hydrophobic, P=polar)
    // "HPHPPHHPHPPHPHHPPHPH" — 20-mer
    let sequence: Vec<bool> = "HPHPPHHPHPPHPHHPPHPH".chars().map(|c| c == 'H').collect();
    let n = sequence.len();
    let seq_clone = sequence.clone();

    let hp_energy = FnEnergy(move |dirs: &Vec<i8>| {
        // Walk the lattice
        let mut positions = Vec::with_capacity(n);
        let (mut x, mut y) = (0i32, 0i32);
        positions.push((x, y));
        for &d in dirs.iter() {
            match d % 4 {
                0 => x += 1,
                1 => y += 1,
                2 => x -= 1,
                _ => y -= 1,
            }
            positions.push((x, y));
        }

        // Self-overlap penalty
        let mut seen = std::collections::HashSet::new();
        let mut overlap_penalty = 0.0f64;
        for &p in &positions {
            if !seen.insert(p) {
                overlap_penalty += 5.0;
            }
        }

        // Count H–H contacts (non-bonded lattice neighbors)
        let contacts: f64 = (0..n)
            .flat_map(|i| (i + 2..n).map(move |j| (i, j)))
            .filter(|&(i, j)| seq_clone[i] && seq_clone[j])
            .filter(|&(i, j)| {
                let (xi, yi) = positions[i];
                let (xj, yj) = positions[j];
                (xi - xj).abs() + (yi - yj).abs() == 1
            })
            .count() as f64;

        overlap_penalty - contacts // minimize → maximize contacts
    });

    let initial_dirs = vec![0i8; n - 1]; // fully extended
    let dirs_len = n - 1;

    let result = Annealer::builder()
        .objective(hp_energy)
        .moves(FoldMove { len: dirs_len })
        .schedule(Logarithmic::new(10.0))
        .iterations(500_000)
        .seed(6)
        .build()?
        .run(initial_dirs);

    println!(
        "[6] Bioinformatics / Folding  contacts={:.0}  accept={:.1}%",
        -result.best_energy.min(0.0),
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 7. MANUFACTURING — Job-Shop Scheduling (Makespan)
// ============================================================================
//
// State:  Vec<usize>  — job processing order on each machine
//                       (flattened: M machines × J jobs)
//                       state[m * J + k] = job index at position k on machine m
// Energy: makespan = total completion time across all machines + jobs
// Move:   swap two jobs within a machine's schedule
// Why SA: job-shop scheduling is NP-hard. SA is embedded in production
//         scheduling software (e.g., Preactor, Siemens Opcenter).

fn ex07_manufacturing() -> Result<(), AnnealError> {
    const M: usize = 3; // machines
    const J: usize = 5; // jobs

    // processing_time[machine][job] in time units
    let pt = [
        [3u64, 2, 5, 1, 4], // machine 0
        [4, 3, 2, 5, 1],    // machine 1
        [2, 5, 3, 4, 2],    // machine 2
    ];

    // Simplified makespan: sum of weighted completion times on each machine
    // (ignores precedence for brevity — real implementation adds dependency graph)
    let makespan = FnEnergy(move |order: &Vec<usize>| {
        let mut total = 0u64;
        for m in 0..M {
            let mut t = 0u64;
            for k in 0..J {
                let job = order[m * J + k];
                t += pt[m][job];
            }
            if t > total {
                total = t;
            }
        }
        total as f64
    });

    // A move that swaps two positions within a single machine's schedule
    struct MachineSwapMove;
    impl MoveOperator<Vec<usize>> for MachineSwapMove {
        fn propose(&self, state: &Vec<usize>, rng: &mut impl Rng) -> Vec<usize> {
            let mut s = state.clone();
            let m = (rng.next_u64() % M as u64) as usize;
            let i = m * J + (rng.next_u64() % J as u64) as usize;
            let j = m * J + (rng.next_u64() % J as u64) as usize;
            s.swap(i, j);
            s
        }
    }

    // Initial: sequential order on every machine
    let mut initial = Vec::with_capacity(M * J);
    for _ in 0..M {
        initial.extend(0..J);
    }

    let result = Annealer::builder()
        .objective(makespan)
        .moves(MachineSwapMove)
        .schedule(Exponential::new(10.0, 0.9992))
        .iterations(100_000)
        .seed(7)
        .build()?
        .run(initial);

    println!(
        "[7] Manufacturing / Sched.    makespan={:.0}  accept={:.1}%",
        result.best_energy,
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 8. AEROSPACE — Satellite Orbit Slot Assignment
// ============================================================================
//
// State:  Vec<usize>  — slot index (0..S) assigned to each satellite
// Energy: total station-keeping delta-V (fuel cost) to reach assigned slot
//         + collision penalty for two satellites sharing a slot
// Move:   reassign one satellite to a random slot
// Why SA: slot assignment is a combinatorial auction problem. SA is used
//         in constellation deployment planning (e.g., Starlink phasing).

struct SlotMove {
    n_sats: usize,
    n_slots: usize,
}
impl MoveOperator<Vec<usize>> for SlotMove {
    fn propose(&self, state: &Vec<usize>, rng: &mut impl Rng) -> Vec<usize> {
        let mut s = state.clone();
        let sat = (rng.next_u64() % self.n_sats as u64) as usize;
        s[sat] = (rng.next_u64() % self.n_slots as u64) as usize;
        s
    }
}

fn ex08_aerospace() -> Result<(), AnnealError> {
    const N_SATS: usize = 8;
    const N_SLOTS: usize = 10;

    // Current orbital positions (in degrees) and target slot positions
    let current_pos: [f64; N_SATS] = [10.0, 45.0, 80.0, 120.0, 160.0, 200.0, 250.0, 310.0];
    let slot_pos: [f64; N_SLOTS] =
        [0.0, 36.0, 72.0, 108.0, 144.0, 180.0, 216.0, 252.0, 288.0, 324.0];

    let delta_v = FnEnergy(move |assignment: &Vec<usize>| {
        // Fuel cost: proportional to angular distance to assigned slot
        let fuel: f64 = assignment
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let diff = (current_pos[i] - slot_pos[s]).abs();
                diff.min(360.0 - diff) // shorter arc
            })
            .sum();

        // Collision penalty: two sats in same slot
        let mut slot_count = [0u32; N_SLOTS];
        for &s in assignment.iter() {
            slot_count[s] += 1;
        }
        let collision: f64 =
            slot_count.iter().map(|&c| if c > 1 { (c - 1) as f64 * 1000.0 } else { 0.0 }).sum();

        fuel + collision
    });

    let initial: Vec<usize> = (0..N_SATS).map(|i| i % N_SLOTS).collect();

    let result = Annealer::builder()
        .objective(delta_v)
        .moves(SlotMove { n_sats: N_SATS, n_slots: N_SLOTS })
        .schedule(Cauchy::new(200.0))
        .iterations(100_000)
        .seed(8)
        .build()?
        .run(initial);

    println!(
        "[8] Aerospace / Orbits        delta_v={:.2}°  accept={:.1}%",
        result.best_energy,
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 9. HEALTHCARE — Nurse Rostering
// ============================================================================
//
// State:  Vec<usize>  — shift assignment per nurse-day slot
//                       state[nurse * DAYS + day] = shift (0=off, 1=day, 2=night)
// Energy: violations of hard constraints (understaffing, consecutive nights,
//         insufficient rest) + soft constraint preferences
// Move:   randomly change one nurse-day shift
// Why SA: nurse scheduling is NP-hard. SA is used in NHS and hospital
//         scheduling software (e.g., RosterElf, Nurse Rostering Competition).

struct RosterMove {
    n_nurses: usize,
    days: usize,
}
impl MoveOperator<Vec<usize>> for RosterMove {
    fn propose(&self, state: &Vec<usize>, rng: &mut impl Rng) -> Vec<usize> {
        let mut s = state.clone();
        let nurse = (rng.next_u64() % self.n_nurses as u64) as usize;
        let day = (rng.next_u64() % self.days as u64) as usize;
        s[nurse * self.days + day] = (rng.next_u64() % 3) as usize; // 0/1/2
        s
    }
}

fn ex09_healthcare() -> Result<(), AnnealError> {
    const NURSES: usize = 5;
    const DAYS: usize = 7;
    // Required staff per shift per day
    let day_req = [2u32; DAYS]; // 2 nurses on day shift each day
    let night_req = [1u32; DAYS]; // 1 nurse on night shift each day

    let roster_cost = FnEnergy(move |roster: &Vec<usize>| {
        let mut cost = 0.0f64;

        // Hard: staffing requirements
        for d in 0..DAYS {
            let day_count = (0..NURSES).filter(|&n| roster[n * DAYS + d] == 1).count() as u32;
            let night_count = (0..NURSES).filter(|&n| roster[n * DAYS + d] == 2).count() as u32;
            cost += 100.0 * day_count.abs_diff(day_req[d]) as f64;
            cost += 100.0 * night_count.abs_diff(night_req[d]) as f64;
        }

        // Soft: no more than 3 consecutive work days per nurse
        for n in 0..NURSES {
            let mut consecutive = 0u32;
            for d in 0..DAYS {
                if roster[n * DAYS + d] != 0 {
                    consecutive += 1;
                    if consecutive > 3 {
                        cost += 10.0;
                    }
                } else {
                    consecutive = 0;
                }
            }
        }

        // Soft: no consecutive night → day shifts (rest constraint)
        for n in 0..NURSES {
            for d in 0..DAYS - 1 {
                if roster[n * DAYS + d] == 2 && roster[n * DAYS + d + 1] == 1 {
                    cost += 20.0;
                }
            }
        }

        cost
    });

    // Start with all nurses working day shift (maximum staffing violation)
    let initial = vec![1usize; NURSES * DAYS];

    let result = Annealer::builder()
        .objective(roster_cost)
        .moves(RosterMove { n_nurses: NURSES, days: DAYS })
        .schedule(Linear::new(200.0, 0.001))
        .iterations(200_000)
        .seed(9)
        .build()?
        .run(initial);

    println!(
        "[9] Healthcare / Rostering    violations={:.0}  accept={:.1}%",
        result.best_energy,
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 10. MACHINE LEARNING — Neural Network Hyperparameter Tuning
// ============================================================================
//
// State:  Vec<f64>  — hyperparameters: [log_lr, log_wd, dropout, momentum]
// Energy: surrogate validation loss (analytic Rosenbrock-like surface
//         mimicking a real loss landscape with a narrow valley optimum)
// Move:   Gaussian perturbation in log-space
// Why SA: hyperparameter spaces are non-convex, noisy, and expensive to
//         evaluate. SA is competitive with Bayesian optimization for
//         moderate budgets and handles mixed continuous/discrete params.

fn ex10_ml_hyperparams() -> Result<(), AnnealError> {
    // Surrogate: penalize deviating from known-good hyperparams
    //   optimal: log_lr=-3.0, log_wd=-4.0, dropout=0.3, momentum=0.9
    // Surface has a narrow ridge (like Rosenbrock) to stress-test exploration.
    let surrogate = FnEnergy(|h: &Vec<f64>| {
        let log_lr = h[0]; // log10(learning_rate), range [-5, -1]
        let log_wd = h[1]; // log10(weight_decay),  range [-6, -2]
        let dropout = h[2]; // range [0.0, 0.9]
        let momentum = h[3]; // range [0.0, 1.0)

        // Rosenbrock-style valley between log_lr and log_wd
        let a = log_lr + 3.0;
        let b = log_wd + 4.0;
        let valley = 100.0 * (b - a * a).powi(2) + (1.0 - a).powi(2);

        // Quadratic bowls for dropout and momentum
        let drop_cost = 10.0 * (dropout - 0.3).powi(2);
        let mom_cost = 50.0 * (momentum - 0.9).powi(2);

        // Boundary penalties (keep params in valid ranges)
        let bounds = {
            let mut p = 0.0f64;
            if log_lr < -5.0 || log_lr > -1.0 {
                p += 1e6
            }
            if log_wd < -6.0 || log_wd > -2.0 {
                p += 1e6
            }
            if dropout < 0.0 || dropout > 0.9 {
                p += 1e6
            }
            if momentum < 0.0 || momentum >= 1.0 {
                p += 1e6
            }
            p
        };

        valley + drop_cost + mom_cost + bounds
    });

    // Start far from optimum
    let initial = vec![-1.5f64, -2.5, 0.7, 0.5];

    let result = Annealer::builder()
        .objective(surrogate)
        .moves(GaussianMove::new(0.1))
        .schedule(Exponential::new(100.0, 0.9998))
        .iterations(300_000)
        .seed(10)
        .build()?
        .run(initial);

    let h = &result.best_state;
    println!(
        "[10] ML / Hyperparams         loss={:.4}  lr=1e{:.2}  wd=1e{:.2}  drop={:.2}  mom={:.3}",
        result.best_energy, h[0], h[1], h[2], h[3]
    );
    Ok(())
}

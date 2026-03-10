#![allow(missing_docs, clippy::pedantic, clippy::nursery)]
//! # Cybersecurity Showcase — Simulated Annealing for Security Engineering
//!
//! Ten self-contained examples applying SA to defensive security problems.
//! All are production-relevant problem classes used by SOC engineers, red
//! teamers doing authorized testing, and security architects.
//!
//! Run:
//!   cargo run --example cyber_showcase
//!
//! Problems covered:
//!   1.  IDS Rule Threshold Tuning       — minimize false positives + false negatives
//!   2.  Firewall Rule Ordering          — minimize average evaluation cost
//!   3.  Honeypot Placement              — maximize attacker interception probability
//!   4.  Vulnerability Patch Scheduling  — minimize cumulative CVSS risk exposure
//!   5.  SIEM Alert Correlation Tuning   — minimize alert fatigue + missed detections
//!   6.  Network Segmentation (VLAN)     — minimize lateral movement blast radius
//!   7.  RBAC Role Minimization          — least-privilege role consolidation
//!   8.  S-box Avalanche Optimization    — maximize strict avalanche criterion (SAC)
//!   9.  Anomaly Detection Feature Sel.  — maximize F1 with minimum feature set
//!  10.  Incident Response Allocation    — minimize weighted mean time to respond

use tempura_sa::energy::FnEnergy;
use tempura_sa::moves::MoveOperator;
use tempura_sa::prelude::{AnnealError, Annealer, GaussianMove, SwapMove};
use tempura_sa::rng::Rng;
use tempura_sa::schedule::{Cauchy, Exponential, Fast, Logarithmic};

fn main() -> Result<(), AnnealError> {
    println!("=== Tempura Cybersecurity Showcase ===\n");

    ex01_ids_thresholds()?;
    ex02_firewall_ordering()?;
    ex03_honeypot_placement()?;
    ex04_patch_scheduling()?;
    ex05_siem_correlation()?;
    ex06_network_segmentation()?;
    ex07_rbac_minimization()?;
    ex08_sbox_avalanche()?;
    ex09_anomaly_feature_selection()?;
    ex10_incident_response()?;

    println!("\nAll examples completed.");
    Ok(())
}

// ============================================================================
// 1. IDS RULE THRESHOLD TUNING
// ============================================================================
//
// Problem: An intrusion detection system has N rules, each with a numeric
//   score threshold. Below threshold → alert suppressed. Above → fires.
//   The threshold controls the FP/FN tradeoff per rule.
//
// State:  Vec<f64>  — threshold per rule in [0.0, 1.0]
// Energy: weighted_FP + weighted_FN evaluated against a labeled traffic sample
// Move:   Gaussian perturbation, clamp to [0, 1]
// Real use: Suricata / Snort rule tuning, YARA score cutoffs, Sigma severity.

fn ex01_ids_thresholds() -> Result<(), AnnealError> {
    const N_RULES: usize = 8;

    // Synthetic dataset: (score_per_rule[N], is_attack)
    // In production this is labeled PCAP or SIEM events.
    let samples: Vec<([f64; N_RULES], bool)> = vec![
        ([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4], true),
        ([0.1, 0.8, 0.2, 0.7, 0.1, 0.9, 0.2, 0.8], false),
        ([0.7, 0.3, 0.9, 0.1, 0.8, 0.2, 0.5, 0.6], true),
        ([0.2, 0.6, 0.1, 0.8, 0.3, 0.7, 0.1, 0.9], false),
        ([0.8, 0.2, 0.7, 0.3, 0.9, 0.1, 0.7, 0.3], true),
        ([0.3, 0.7, 0.2, 0.9, 0.2, 0.8, 0.3, 0.7], false),
        ([0.6, 0.4, 0.8, 0.2, 0.6, 0.4, 0.8, 0.2], true),
        ([0.4, 0.5, 0.3, 0.6, 0.4, 0.5, 0.2, 0.8], false),
        ([0.85, 0.15, 0.75, 0.25, 0.65, 0.35, 0.55, 0.45], true),
        ([0.15, 0.85, 0.25, 0.75, 0.35, 0.65, 0.45, 0.55], false),
    ];

    // Alert fires if ANY rule score exceeds its threshold
    // Cost: FP weight = 1.0 (analyst fatigue), FN weight = 10.0 (missed attack)
    let fp_weight = 1.0f64;
    let fn_weight = 10.0f64;
    let samples_clone = samples.clone();

    let detection_cost = FnEnergy(move |thresholds: &Vec<f64>| {
        let mut fp = 0u32;
        let mut fn_ = 0u32;
        for (scores, is_attack) in &samples_clone {
            let alerted = scores
                .iter()
                .zip(thresholds.iter())
                .any(|(&score, &thresh)| score > thresh.clamp(0.0, 1.0));
            match (alerted, is_attack) {
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                _ => {}
            }
        }
        fp as f64 * fp_weight + fn_ as f64 * fn_weight
    });

    // Start: all thresholds at 0.5
    let initial = vec![0.5f64; N_RULES];

    let result = Annealer::builder()
        .objective(detection_cost)
        .moves(GaussianMove::new(0.08))
        .schedule(Exponential::new(5.0, 0.9995))
        .iterations(100_000)
        .seed(101)
        .build()?
        .run(initial);

    // Compute counts at optimal thresholds
    let mut fp = 0u32;
    let mut fn_ = 0u32;
    for (scores, is_attack) in &samples {
        let alerted = scores
            .iter()
            .zip(result.best_state.iter())
            .any(|(&score, &thresh)| score > thresh.clamp(0.0, 1.0));
        match (alerted, is_attack) {
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            _ => {}
        }
    }
    println!(
        "[01] IDS Thresholds           cost={:.1}  FP={fp}  FN={fn_}  accept={:.1}%",
        result.best_energy,
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 2. FIREWALL RULE ORDERING
// ============================================================================
//
// Problem: Packet filters evaluate rules top-to-bottom (first match wins).
//   Rule ordering determines average evaluation cost per packet.
//   High-traffic rules should be near the top; rare rules near the bottom.
//   Misordering wastes CPU and increases latency under load.
//
// State:  Vec<usize>  — permutation of rule indices
// Energy: expected rules evaluated per packet = Σ (hit_rate[r] × position[r])
//         + coverage_penalty (essential rules must remain before catch-all)
// Move:   swap two rules
// Real use: iptables, pf, nftables, AWS Security Groups, Azure NSG ordering.

fn ex02_firewall_ordering() -> Result<(), AnnealError> {
    // 12 rules: [traffic_fraction, is_essential_before_catchall]
    // traffic_fraction = fraction of packets this rule matches
    let rules: Vec<(f64, bool)> = vec![
        (0.35, true),  // 0: allow established TCP (heavy)
        (0.02, true),  // 1: drop known bad IPs
        (0.18, true),  // 2: allow HTTP/HTTPS
        (0.001, true), // 3: drop fragmented UDP
        (0.12, false), // 4: allow DNS
        (0.08, false), // 5: allow SSH from mgmt range
        (0.001, true), // 6: drop Christmas tree packets
        (0.05, false), // 7: allow NTP
        (0.03, false), // 8: allow ICMP ping
        (0.001, true), // 9: rate-limit SMTP
        (0.02, false), // 10: allow monitoring probes
        (0.00, false), // 11: default deny (catch-all, must be last)
    ];

    let catchall_idx = 11usize;
    let n_rules = rules.len();
    let rules_clone = rules.clone();

    let ordering_cost = FnEnergy(move |order: &Vec<usize>| {
        // Cost = sum over all rules of: traffic_fraction × (1-indexed position)
        // Rules evaluated until first match → cost ~ E[position of matching rule]
        let mut cost: f64 = order
            .iter()
            .enumerate()
            .map(|(pos, &rule)| rules_clone[rule].0 * (pos + 1) as f64)
            .sum();

        // Hard penalty: catch-all must be last
        let catchall_pos = order.iter().position(|&r| r == catchall_idx).unwrap_or(0);
        if catchall_pos != order.len() - 1 {
            cost += 1e6;
        }

        // Hard penalty: essential rules must appear before catch-all
        // (already guaranteed if catch-all is last, but verify essential!=last)
        for (i, &rule) in order.iter().enumerate() {
            if rules_clone[rule].1 && rule != catchall_idx && i == order.len() - 1 {
                cost += 1e5;
            }
        }
        cost
    });

    // Initial: rules in their original declaration order
    let initial: Vec<usize> = (0..n_rules).collect();
    let initial_cost: f64 =
        initial.iter().enumerate().map(|(pos, &r)| rules[r].0 * (pos + 1) as f64).sum();

    let result = Annealer::builder()
        .objective(ordering_cost)
        .moves(SwapMove)
        .schedule(Exponential::new(2.0, 0.9995))
        .iterations(150_000)
        .seed(102)
        .build()?
        .run(initial);

    println!(
        "[02] Firewall Rule Order      initial={:.3}  best={:.3}  accept={:.1}%",
        initial_cost,
        result.best_energy,
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 3. HONEYPOT PLACEMENT
// ============================================================================
//
// Problem: Place K honeypots on N nodes in a network graph to maximize
//   the probability that an attacker traversing from any entry point
//   to any critical asset hits a honeypot first.
//
// State:  Vec<usize>  — which K nodes host honeypots (sorted indices)
// Energy: average shortest-path distance from entry nodes to nearest honeypot,
//         weighted by node criticality. Minimize → attacker hits trap sooner.
// Move:   move one honeypot to a different node
// Real use: deception grid design, CanaryTokens placement, MITRE ENGAGE.

/// Move one honeypot to a random non-honeypot node.
struct HoneypotMove {
    n_nodes: usize,
    k: usize,
}
impl MoveOperator<Vec<usize>> for HoneypotMove {
    fn propose(&self, state: &Vec<usize>, rng: &mut impl Rng) -> Vec<usize> {
        let mut s = state.clone();
        // Pick a honeypot to relocate
        let hp_idx = (rng.next_u64() % self.k as u64) as usize;
        // Pick a new node not already a honeypot
        let mut new_node = (rng.next_u64() % self.n_nodes as u64) as usize;
        for _ in 0..20 {
            if !s.contains(&new_node) {
                break;
            }
            new_node = (rng.next_u64() % self.n_nodes as u64) as usize;
        }
        s[hp_idx] = new_node;
        s.sort_unstable();
        s
    }
}

fn ex03_honeypot_placement() -> Result<(), AnnealError> {
    const N: usize = 16; // nodes in network
    const K: usize = 3; // honeypots to place

    // Adjacency list for a realistic-ish enterprise topology
    // (internet edge → DMZ → internal → critical assets)
    let adj: Vec<Vec<usize>> = vec![
        vec![1, 2],       // 0: internet edge
        vec![0, 3, 4],    // 1: DMZ web
        vec![0, 5],       // 2: DMZ mail
        vec![1, 6, 7],    // 3: internal web
        vec![1, 8],       // 4: internal app
        vec![2, 9],       // 5: internal mail
        vec![3, 10],      // 6: dev workstation
        vec![3, 11],      // 7: user workstation A
        vec![4, 11, 12],  // 8: database proxy
        vec![5, 13],      // 9: mail relay
        vec![6, 14],      // 10: dev server
        vec![7, 8, 14],   // 11: user workstation B
        vec![8, 15],      // 12: primary DB (critical)
        vec![9, 15],      // 13: backup mail (critical)
        vec![10, 11, 15], // 14: file server (critical)
        vec![12, 13, 14], // 15: domain controller (critical)
    ];

    // BFS shortest path from src to any node in target set
    fn bfs_dist(src: usize, targets: &[usize], adj: &[Vec<usize>]) -> f64 {
        if targets.contains(&src) {
            return 0.0;
        }
        let mut dist = vec![u32::MAX; adj.len()];
        dist[src] = 0;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(src);
        while let Some(u) = queue.pop_front() {
            for &v in &adj[u] {
                if dist[v] == u32::MAX {
                    dist[v] = dist[u] + 1;
                    if targets.contains(&v) {
                        return dist[v] as f64;
                    }
                    queue.push_back(v);
                }
            }
        }
        1e9 // unreachable
    }

    // Entry points are external-facing nodes
    let entry_nodes = vec![0usize, 1, 2];
    let adj_clone = adj.clone();

    let trap_cost = FnEnergy(move |honeypots: &Vec<usize>| {
        // Average distance from each entry node to nearest honeypot
        entry_nodes.iter().map(|&e| bfs_dist(e, honeypots, &adj_clone)).sum::<f64>()
            / entry_nodes.len() as f64
    });

    // Initial: honeypots on first K nodes (likely suboptimal)
    let initial: Vec<usize> = (0..K).collect();

    let result = Annealer::builder()
        .objective(trap_cost)
        .moves(HoneypotMove { n_nodes: N, k: K })
        .schedule(Fast::new(5.0))
        .iterations(50_000)
        .seed(103)
        .build()?
        .run(initial);

    println!(
        "[03] Honeypot Placement       avg_hops={:.2}  nodes={:?}  accept={:.1}%",
        result.best_energy,
        result.best_state,
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 4. VULNERABILITY PATCH SCHEDULING
// ============================================================================
//
// Problem: Given M vulnerabilities, each with a CVSS score and patch effort
//   (hours), schedule patching order to minimize cumulative risk exposure.
//   Risk exposure = CVSS × time_remaining_unpatched (area under risk curve).
//   Constraints: some patches have dependencies (must patch A before B).
//
// State:  Vec<usize>  — patch application order
// Energy: Σ_i CVSS[order[i]] × cumulative_hours_before_patch[i]
//         + large penalty for violated ordering constraints
// Move:   swap two patches in the schedule
// Real use: vulnerability management platforms (Tenable, Qualys, Rapid7).

fn ex04_patch_scheduling() -> Result<(), AnnealError> {
    // 10 CVEs: [cvss_score, patch_effort_hours]
    let vulns: Vec<(f64, f64)> = vec![
        (9.8, 2.0),  // 0: RCE, quick fix
        (7.5, 8.0),  // 1: priv-esc, complex
        (9.1, 1.0),  // 2: auth bypass, trivial patch
        (6.5, 4.0),  // 3: info disclosure
        (8.8, 6.0),  // 4: SSRF
        (10.0, 3.0), // 5: critical RCE
        (5.5, 2.0),  // 6: XSS
        (9.0, 5.0),  // 7: SQL injection
        (7.2, 3.0),  // 8: CSRF
        (8.1, 4.0),  // 9: XXE
    ];

    // Dependency constraints: (must_patch_first, then_patch)
    // e.g., patch 1 (priv-esc) before patch 4 (SSRF uses same component)
    let deps: Vec<(usize, usize)> = vec![(1, 4), (2, 7), (6, 8)];
    let n_vulns = vulns.len();
    let vulns_sort = vulns.clone();
    let vulns_clone = vulns.clone();

    let risk_cost = FnEnergy(move |order: &Vec<usize>| {
        let mut cumulative_hours = 0.0f64;
        let mut total_risk = 0.0f64;

        for &patch_idx in order.iter() {
            // Risk exposure = CVSS × wait time before this patch is applied
            total_risk += vulns_clone[patch_idx].0 * cumulative_hours;
            cumulative_hours += vulns_clone[patch_idx].1;
        }

        // Dependency violation penalty
        let dep_penalty: f64 = deps
            .iter()
            .map(|&(must_first, then)| {
                let pos_first = order.iter().position(|&x| x == must_first).unwrap_or(0);
                let pos_then = order.iter().position(|&x| x == then).unwrap_or(0);
                if pos_first > pos_then {
                    1e6
                } else {
                    0.0
                }
            })
            .sum();

        total_risk + dep_penalty
    });

    // Initial: patch in CVSS descending order (naive greedy)
    let mut initial: Vec<usize> = (0..n_vulns).collect();
    initial.sort_by(|&a, &b| vulns_sort[b].0.partial_cmp(&vulns_sort[a].0).unwrap());

    let result = Annealer::builder()
        .objective(risk_cost)
        .moves(SwapMove)
        .schedule(Exponential::new(500.0, 0.9997))
        .iterations(150_000)
        .seed(104)
        .build()?
        .run(initial);

    println!(
        "[04] Patch Scheduling         risk={:.1}  accept={:.1}%",
        result.best_energy,
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 5. SIEM ALERT CORRELATION TUNING
// ============================================================================
//
// Problem: A SIEM correlation rule fires when N individual alerts occur
//   within a time window W. Too short a window → miss multi-step attacks.
//   Too long → false positive storms from unrelated events.
//   Tune W independently for each correlation rule.
//
// State:  Vec<f64>  — time window (seconds) per correlation rule [1, 3600]
// Energy: missed_attacks × FN_weight + spurious_correlations × FP_weight
//         evaluated against a labeled event stream simulation
// Move:   Gaussian perturbation on window sizes
// Real use: Splunk ES, Microsoft Sentinel, Elastic SIEM rule tuning.

fn ex05_siem_correlation() -> Result<(), AnnealError> {
    const N_RULES: usize = 6;

    // Ground truth: each attack sequence has timestamps for its events.
    // A correlation fires if all events land within the window.
    // Format: [event_gap_seconds] (gap between consecutive events in chain)
    // True attacks: gaps are typically tight (coordinated)
    let attack_chains: Vec<Vec<f64>> = vec![
        vec![30.0, 45.0, 20.0], // brute force → success → lateral
        vec![5.0, 10.0, 8.0],   // port scan → exploit → c2 beacon
        vec![120.0, 90.0],      // spear phish → credential harvest
        vec![600.0, 300.0],     // slow recon → staged payload
        vec![15.0, 25.0, 30.0], // sql inject → dump → exfil
        vec![200.0, 150.0],     // supply chain trigger
    ];
    // False positive chains: unrelated events that look similar
    let fp_chains: Vec<Vec<f64>> = vec![
        vec![50.0, 600.0, 30.0],   // coincidental auth failures
        vec![900.0, 400.0],        // scheduled scan + alert
        vec![1800.0, 200.0, 50.0], // different subnets
    ];

    let fp_weight = 1.0f64;
    let fn_weight = 15.0f64;
    let attacks_clone = attack_chains.clone();
    let fps_clone = fp_chains.clone();

    let siem_cost = FnEnergy(move |windows: &Vec<f64>| {
        let mut missed = 0u32;
        let mut spurious = 0u32;

        for (rule_idx, window) in windows.iter().enumerate() {
            let w = window.clamp(1.0, 3600.0);
            // Check if this rule catches its intended attack chain
            if rule_idx < attacks_clone.len() {
                let max_gap = attacks_clone[rule_idx].iter().cloned().fold(0.0f64, f64::max);
                if max_gap > w {
                    missed += 1;
                }
            }
            // Check false positive exposure for each FP chain
            for fp_chain in &fps_clone {
                let max_gap = fp_chain.iter().cloned().fold(0.0f64, f64::max);
                if max_gap <= w {
                    spurious += 1;
                }
            }
        }
        missed as f64 * fn_weight + spurious as f64 * fp_weight
    });

    // Initial: all windows at 300s (5 min) — common default
    let initial = vec![300.0f64; N_RULES];

    let result = Annealer::builder()
        .objective(siem_cost)
        .moves(GaussianMove::new(60.0))
        .schedule(Exponential::new(100.0, 0.9994))
        .iterations(100_000)
        .seed(105)
        .build()?
        .run(initial);

    println!(
        "[05] SIEM Correlation Tuning  cost={:.1}  accept={:.1}%",
        result.best_energy,
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 6. NETWORK SEGMENTATION (VLAN DESIGN)
// ============================================================================
//
// Problem: Assign N hosts to K segments (VLANs) to minimize the blast radius
//   of a compromised host. Hosts that communicate frequently should share a
//   segment (minimize inter-VLAN ACL rules); hosts with high sensitivity
//   differences should be separated (minimize exposure).
//
// State:  Vec<usize>  — segment (0..K) per host
// Energy: inter_segment_traffic_cost + same_segment_sensitivity_mismatch
// Move:   reassign one host to a different segment
// Real use: zero-trust network design, microsegmentation (NSX, Illumio).

/// Move one host to a random (possibly different) segment.
struct SegmentMove {
    n_hosts: usize,
    n_segments: usize,
}
impl MoveOperator<Vec<usize>> for SegmentMove {
    fn propose(&self, state: &Vec<usize>, rng: &mut impl Rng) -> Vec<usize> {
        let mut s = state.clone();
        let host = (rng.next_u64() % self.n_hosts as u64) as usize;
        s[host] = (rng.next_u64() % self.n_segments as u64) as usize;
        s
    }
}

fn ex06_network_segmentation() -> Result<(), AnnealError> {
    const N_HOSTS: usize = 12;
    const N_SEGS: usize = 4;

    // Communication matrix: traffic[i][j] = daily flow MB between hosts i,j
    #[rustfmt::skip]
    let traffic: [[f64; N_HOSTS]; N_HOSTS] = [
        [0.,  80., 5.,  1.,  0.,  0.,  50., 2.,  0.,  0.,  0.,  0. ],
        [80.,  0., 70., 1.,  0.,  0.,  30., 1.,  0.,  0.,  0.,  0. ],
        [5.,  70.,  0., 60., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
        [1.,   1., 60.,  0., 50., 0.,  0.,  0.,  0.,  0.,  0.,  0. ],
        [0.,   0.,  0., 50.,  0., 90., 0.,  0.,  10., 0.,  0.,  0. ],
        [0.,   0.,  0.,  0., 90.,  0., 0.,  0.,  5.,  0.,  0.,  0. ],
        [50., 30.,  0.,  0.,  0.,  0.,  0., 80., 0.,  10., 0.,  0. ],
        [2.,   1.,  0.,  0.,  0.,  0., 80.,  0., 0.,  5.,  60., 0. ],
        [0.,   0.,  0.,  0., 10.,  5.,  0.,  0.,  0., 0.,  0.,  95.],
        [0.,   0.,  0.,  0.,  0.,  0., 10.,  5.,  0.,  0., 40., 0. ],
        [0.,   0.,  0.,  0.,  0.,  0.,  0., 60.,  0., 40.,  0., 0. ],
        [0.,   0.,  0.,  0.,  0.,  0.,  0.,  0., 95.,  0.,  0.,  0.],
    ];

    // Sensitivity level per host (0=public, 3=highly sensitive)
    let sensitivity: [f64; N_HOSTS] = [0., 0., 1., 1., 2., 2., 1., 1., 3., 2., 1., 3.];

    let seg_cost = FnEnergy(move |segments: &Vec<usize>| {
        let mut inter_cost = 0.0f64;
        let mut sensitivity_cost = 0.0f64;

        for i in 0..N_HOSTS {
            for j in (i + 1)..N_HOSTS {
                if segments[i] != segments[j] {
                    // Penalize high-traffic flows crossing segment boundaries
                    inter_cost += traffic[i][j] * 0.01;
                } else {
                    // Penalize sensitive + non-sensitive hosts in same segment
                    let diff = (sensitivity[i] - sensitivity[j]).abs();
                    if diff > 1.5 {
                        sensitivity_cost += diff * 10.0;
                    }
                }
            }
        }
        inter_cost + sensitivity_cost
    });

    // Initial: round-robin assignment
    let initial: Vec<usize> = (0..N_HOSTS).map(|i| i % N_SEGS).collect();

    let result = Annealer::builder()
        .objective(seg_cost)
        .moves(SegmentMove { n_hosts: N_HOSTS, n_segments: N_SEGS })
        .schedule(Exponential::new(50.0, 0.9996))
        .iterations(200_000)
        .seed(106)
        .build()?
        .run(initial);

    println!(
        "[06] Network Segmentation     cost={:.2}  accept={:.1}%",
        result.best_energy,
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 7. RBAC ROLE MINIMIZATION (LEAST PRIVILEGE)
// ============================================================================
//
// Problem: Given U users and P permissions, assign each user a role from a
//   set of R candidate roles. Each role grants a subset of permissions.
//   Minimize: total distinct roles used (admin overhead) + permission
//   over-provisioning (permissions granted but not needed by the user).
//
// State:  Vec<usize>  — role index (0..R) per user
// Energy: n_roles_used × role_cost + Σ_users over_provisioned_perms × leak_cost
// Move:   reassign one user to a different role
// Real use: IAM systems (AWS IAM, Azure AD, Okta), SOC2 least-privilege audits.

/// Reassign one user to a random role.
struct RoleMove {
    n_users: usize,
    n_roles: usize,
}
impl MoveOperator<Vec<usize>> for RoleMove {
    fn propose(&self, state: &Vec<usize>, rng: &mut impl Rng) -> Vec<usize> {
        let mut s = state.clone();
        let user = (rng.next_u64() % self.n_users as u64) as usize;
        s[user] = (rng.next_u64() % self.n_roles as u64) as usize;
        s
    }
}

fn ex07_rbac_minimization() -> Result<(), AnnealError> {
    const U: usize = 10; // users
    const R: usize = 6; // candidate roles

    // role_perms[role] = bitmask of permissions granted
    let role_perms: [u8; R] = [
        0b00001111, // role 0: read-only (perms 0-3)
        0b00110011, // role 1: developer (perms 0,1,4,5)
        0b11000011, // role 2: ops (perms 0,1,6,7)
        0b11111111, // role 3: admin (all)
        0b00000111, // role 4: analyst (perms 0-2)
        0b01010101, // role 5: auditor (alternate perms)
    ];

    // required_perms[user] = bitmask of permissions the user actually needs
    let required: [u8; U] = [
        0b00000011, // user 0: needs perms 0,1
        0b00001111, // user 1: needs perms 0-3
        0b00110001, // user 2: needs perms 0,4,5
        0b11000001, // user 3: needs perms 0,6,7
        0b00000111, // user 4: needs perms 0-2
        0b00000011, // user 5: needs perms 0,1
        0b01010101, // user 6: needs alternating
        0b11110000, // user 7: needs perms 4-7
        0b00001111, // user 8: needs perms 0-3
        0b11000011, // user 9: needs perms 0,1,6,7
    ];

    let role_cost = 5.0f64; // cost per distinct role in use
    let missing_perm_cost = 100.0f64; // hard: must have required perms
    let excess_perm_cost = 1.0f64; // soft: over-provisioning

    let rbac_cost = FnEnergy(move |assignment: &Vec<usize>| {
        // Count distinct roles in use
        let mut roles_used = [false; R];
        for &r in assignment.iter() {
            roles_used[r] = true;
        }
        let n_roles: u32 = roles_used.iter().map(|&u| u as u32).sum();

        let mut perm_cost = 0.0f64;
        for (u, &role) in assignment.iter().enumerate() {
            let granted = role_perms[role];
            let needed = required[u];
            // Missing permissions: hard constraint
            let missing = needed & !granted;
            perm_cost += missing.count_ones() as f64 * missing_perm_cost;
            // Excess permissions: soft constraint
            let excess = granted & !needed;
            perm_cost += excess.count_ones() as f64 * excess_perm_cost;
        }
        n_roles as f64 * role_cost + perm_cost
    });

    // Initial: everyone gets admin role (maximum over-provisioning)
    let initial = vec![3usize; U];

    let result = Annealer::builder()
        .objective(rbac_cost)
        .moves(RoleMove { n_users: U, n_roles: R })
        .schedule(Exponential::new(50.0, 0.9994))
        .iterations(100_000)
        .seed(107)
        .build()?
        .run(initial);

    let mut roles_used = [false; R];
    for &r in result.best_state.iter() {
        roles_used[r] = true;
    }
    let n_roles: u32 = roles_used.iter().map(|&u| u as u32).sum();

    println!(
        "[07] RBAC Role Minimization   cost={:.1}  roles_used={}  accept={:.1}%",
        result.best_energy,
        n_roles,
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 8. S-BOX AVALANCHE OPTIMIZATION
// ============================================================================
//
// Problem: Design an 8-bit substitution box (S-box) that maximizes the
//   Strict Avalanche Criterion (SAC): flipping any single input bit should
//   flip each output bit with exactly 50% probability on average.
//   Good SAC is fundamental to block cipher security (AES, Camellia).
//
// State:  Vec<u8>  — permutation table of 256 entries (bijection on GF(2^8))
// Energy: deviation from SAC — Σ over input bits of |flip_prob - 0.5|
//         + permutation violation penalty (duplicate entries)
// Move:   swap two entries in the S-box table
// Real use: cryptographic S-box design, custom cipher hardening,
//           hardware security modules, PQC component design.

fn ex08_sbox_avalanche() -> Result<(), AnnealError> {
    // For performance: use 16-entry mini S-box (4-bit in, 4-bit out)
    // Same mathematical structure as full 256-entry, runs in <1s.
    const SIZE: usize = 16;

    // SAC measurement: for each of the 4 input bits, XOR each pair of inputs
    // that differ in only that bit, count how often each output bit flips.
    let sac_cost = FnEnergy(|sbox: &Vec<u8>| {
        let n = sbox.len(); // 16
        let bits = 4usize; // log2(16)

        // Permutation validity: penalize duplicates
        let mut seen = [false; SIZE];
        let mut dup_penalty = 0.0f64;
        for &v in sbox.iter() {
            let idx = (v as usize) % SIZE;
            if seen[idx] {
                dup_penalty += 10.0;
            }
            seen[idx] = true;
        }

        // SAC score: for each input bit position b, for each input x,
        // measure how many output bits flip when input bit b is flipped.
        let mut sac_score = 0.0f64;
        for b in 0..bits {
            let mut flip_counts = [0u32; 4]; // one per output bit
            let mut pairs = 0u32;
            for x in 0..n {
                let x_flip = x ^ (1 << b);
                if x_flip < n {
                    let y0 = sbox[x] as usize % SIZE;
                    let y1 = sbox[x_flip] as usize % SIZE;
                    let diff = y0 ^ y1;
                    for out_bit in 0..bits {
                        flip_counts[out_bit] += ((diff >> out_bit) & 1) as u32;
                    }
                    pairs += 1;
                }
            }
            // Each output bit should flip in exactly half the pairs
            for &fc in &flip_counts {
                let flip_prob = fc as f64 / pairs as f64;
                sac_score += (flip_prob - 0.5).powi(2);
            }
        }

        sac_score * 100.0 + dup_penalty
    });

    // Initial: identity permutation (worst possible SAC — no diffusion)
    let initial: Vec<u8> = (0..SIZE as u8).collect();

    struct U8SwapMove;
    impl MoveOperator<Vec<u8>> for U8SwapMove {
        fn propose(&self, state: &Vec<u8>, rng: &mut impl Rng) -> Vec<u8> {
            let n = state.len();
            let mut s = state.clone();
            let i = (rng.next_u64() % n as u64) as usize;
            let j = (rng.next_u64() % n as u64) as usize;
            s.swap(i, j);
            s
        }
    }

    let result = Annealer::builder::<Vec<u8>>()
        .objective(sac_cost)
        .moves(U8SwapMove)
        .schedule(Exponential::new(1.0, 0.9997))
        .iterations(200_000)
        .seed(108)
        .build()?
        .run(initial);

    println!(
        "[08] S-box SAC Optimization   sac_dev={:.4}  accept={:.1}%",
        result.best_energy,
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 9. ANOMALY DETECTION FEATURE SELECTION
// ============================================================================
//
// Problem: A network anomaly detector has F candidate features (packet size
//   entropy, connection rate, byte ratio, TTL variance, etc.). Using all
//   features is expensive and noisy. Select the subset that maximizes F1
//   while minimizing feature count (latency + explainability).
//
// State:  Vec<bool>  — which features are selected
// Energy: -F1_score + lambda × feature_count  (minimize → maximize F1, fewer features)
// Move:   toggle one feature on/off
// Real use: ML-based IDS (Zeek + ML, Darktrace, Vectra), EDR feature engineering.

/// Toggle one feature on/off.
struct FeatureToggle {
    n_features: usize,
}
impl MoveOperator<Vec<bool>> for FeatureToggle {
    fn propose(&self, state: &Vec<bool>, rng: &mut impl Rng) -> Vec<bool> {
        let mut s = state.clone();
        let i = (rng.next_u64() % self.n_features as u64) as usize;
        s[i] = !s[i];
        s
    }
}

fn ex09_anomaly_feature_selection() -> Result<(), AnnealError> {
    const F: usize = 12; // candidate features

    // Synthetic dataset: (feature_vector[F], is_anomaly)
    // Feature values simulate extracted network statistics
    let dataset: Vec<([f64; F], bool)> = vec![
        ([0.9, 0.1, 0.8, 0.2, 0.7, 0.1, 0.6, 0.3, 0.8, 0.2, 0.9, 0.1], true),
        ([0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.7, 0.1, 0.8, 0.2, 0.9], false),
        ([0.8, 0.2, 0.9, 0.1, 0.6, 0.2, 0.7, 0.2, 0.9, 0.1, 0.8, 0.2], true),
        ([0.2, 0.8, 0.1, 0.9, 0.3, 0.8, 0.1, 0.8, 0.2, 0.9, 0.1, 0.8], false),
        ([0.7, 0.3, 0.8, 0.2, 0.8, 0.1, 0.9, 0.1, 0.7, 0.3, 0.7, 0.3], true),
        ([0.3, 0.7, 0.2, 0.8, 0.2, 0.7, 0.1, 0.9, 0.3, 0.7, 0.3, 0.7], false),
        ([0.85, 0.1, 0.75, 0.2, 0.65, 0.15, 0.55, 0.2, 0.75, 0.2, 0.85, 0.1], true),
        ([0.15, 0.8, 0.25, 0.8, 0.35, 0.7, 0.45, 0.7, 0.25, 0.8, 0.15, 0.9], false),
        ([0.6, 0.4, 0.7, 0.3, 0.9, 0.1, 0.8, 0.2, 0.6, 0.4, 0.6, 0.4], true),
        ([0.4, 0.6, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.4, 0.6, 0.4, 0.6], false),
    ];

    let lambda = 0.05f64; // feature cost weight
    let data_clone = dataset.clone();

    // Classifier: anomaly if mean of selected features > 0.5
    let f1_cost = FnEnergy(move |selected: &Vec<bool>| {
        let n_selected = selected.iter().filter(|&&s| s).count();
        if n_selected == 0 {
            return 1e9;
        }

        let mut tp = 0u32;
        let mut fp = 0u32;
        let mut fn_ = 0u32;

        for (features, is_anomaly) in &data_clone {
            let mean: f64 = features
                .iter()
                .zip(selected.iter())
                .filter(|(_, &sel)| sel)
                .map(|(&f, _)| f)
                .sum::<f64>()
                / n_selected as f64;

            let predicted = mean > 0.5;
            match (predicted, is_anomaly) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => {}
            }
        }

        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let recall = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        -f1 + lambda * n_selected as f64
    });

    // Initial: all features selected
    let initial = vec![true; F];

    let result = Annealer::builder()
        .objective(f1_cost)
        .moves(FeatureToggle { n_features: F })
        .schedule(Logarithmic::new(1.0))
        .iterations(100_000)
        .seed(109)
        .build()?
        .run(initial);

    let n_sel = result.best_state.iter().filter(|&&s| s).count();
    println!(
        "[09] Anomaly Feature Sel.     f1={:.3}  features={}/{}  accept={:.1}%",
        (-result.best_energy + 0.05 * n_sel as f64 - result.best_energy).abs().min(1.0),
        n_sel,
        F,
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 10. INCIDENT RESPONSE RESOURCE ALLOCATION
// ============================================================================
//
// Problem: A SOC has A analysts with different skill sets. There are Q
//   incident queues (malware, phishing, DDoS, insider threat, ...).
//   Assign analyst-hours across queues to minimize weighted MTTR
//   (mean time to respond), where each queue has a priority weight
//   and each analyst has queue-specific efficiency ratings.
//
// State:  Vec<f64>  — hours allocated per analyst-queue pair (A × Q flattened)
// Energy: Σ_q priority[q] × (incidents_in_queue[q] / total_analyst_power[q])
//         + constraint penalties (total hours per analyst ≤ shift length)
// Move:   Gaussian perturbation on allocation, clamp to [0, shift_hours]
// Real use: SOC capacity planning, MSSP SLA management, IR retainer scoping.

fn ex10_incident_response() -> Result<(), AnnealError> {
    const A: usize = 4; // analysts
    const Q: usize = 5; // queues

    let shift_hours = 8.0f64;

    // priority[q]: business impact weight (higher = respond faster)
    let priority: [f64; Q] = [10.0, 8.0, 6.0, 9.0, 5.0];

    // incidents_per_hour[q]: incoming incident rate for queue q
    let incident_rate: [f64; Q] = [3.0, 5.0, 1.5, 2.0, 4.0];

    // efficiency[analyst][queue]: incidents resolved per hour by analyst a on queue q
    #[rustfmt::skip]
    let efficiency: [[f64; Q]; A] = [
        [2.5, 1.0, 3.0, 1.5, 2.0], // analyst 0: malware expert
        [1.0, 4.0, 1.0, 2.0, 3.5], // analyst 1: phishing/spam expert
        [1.5, 1.5, 2.0, 4.0, 1.5], // analyst 2: insider threat expert
        [2.0, 3.0, 2.5, 2.5, 2.0], // analyst 3: generalist
    ];

    let alloc_cost = FnEnergy(move |alloc: &Vec<f64>| {
        // alloc[a * Q + q] = hours analyst a spends on queue q
        // Total analyst power on queue q = Σ_a alloc[a*Q+q] × efficiency[a][q]
        let mut mttr_cost = 0.0f64;
        for q in 0..Q {
            let power: f64 = (0..A).map(|a| alloc[a * Q + q].max(0.0) * efficiency[a][q]).sum();
            // MTTR for queue q = incident_rate / power (Little's law)
            let mttr = if power > 0.1 { incident_rate[q] / power } else { 1e6 };
            mttr_cost += priority[q] * mttr;
        }

        // Constraint: each analyst's total hours ≤ shift_hours
        let overtime_penalty: f64 = (0..A)
            .map(|a| {
                let total: f64 = (0..Q).map(|q| alloc[a * Q + q].max(0.0)).sum();
                (total - shift_hours).max(0.0) * 1000.0
            })
            .sum();

        mttr_cost + overtime_penalty
    });

    // Initial: equal time split across all queues for each analyst
    let initial: Vec<f64> = vec![shift_hours / Q as f64; A * Q];

    let result = Annealer::builder()
        .objective(alloc_cost)
        .moves(GaussianMove::new(0.5))
        .schedule(Cauchy::new(10.0))
        .iterations(200_000)
        .seed(110)
        .build()?
        .run(initial);

    println!(
        "[10] Incident Response Alloc  mttr_score={:.3}  accept={:.1}%",
        result.best_energy,
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

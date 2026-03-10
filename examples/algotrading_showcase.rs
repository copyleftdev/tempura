//! # Algorithmic Trading Showcase — SA for Quantitative Finance
//!
//! Ten in-depth examples applying SA to real algo-trading problem classes.
//! All use synthetic but statistically realistic data (deterministic seeds).
//! Financial formulas are production-grade: Almgren-Chriss execution cost,
//! CVaR via sorted scenarios, Kelly criterion, Sortino, Information Ratio.
//!
//! Run:
//!   cargo run --example algotrading_showcase
//!
//! Problems:
//!   1.  Mean-Variance + Transaction Cost   — multi-period rebalancing
//!   2.  Technical Strategy Parameter Sweep — SMA/RSI/ATR signal tuning
//!   3.  Optimal Order Execution            — Almgren-Chriss implementation shortfall
//!   4.  Pairs Trading Threshold Tuning     — z-score entry/exit/stop calibration
//!   5.  Risk Parity Allocation             — equal risk contribution weighting
//!   6.  Regime-Aware Strategy Switching    — bull/bear/vol threshold optimization
//!   7.  Kelly Criterion Multi-Strategy     — optimal fractional sizing across signals
//!   8.  Market Making Spread Optimization  — bid-ask + inventory policy
//!   9.  Cross-Asset Momentum Combination   — multi-horizon signal weight blending
//!  10.  CVaR-Constrained Portfolio         — tail-risk-budgeted return maximization

#![allow(clippy::excessive_precision)]

use tempura::energy::FnEnergy;
use tempura::prelude::{AnnealError, Annealer, GaussianMove};
use tempura::rng::{Rng, Xoshiro256PlusPlus};
use tempura::schedule::{Adaptive, Exponential, Logarithmic};

fn main() -> Result<(), AnnealError> {
    println!("=== Tempura Algorithmic Trading Showcase ===\n");

    ex01_mv_transaction_cost()?;
    ex02_technical_strategy()?;
    ex03_order_execution()?;
    ex04_pairs_trading()?;
    ex05_risk_parity()?;
    ex06_regime_switching()?;
    ex07_kelly_multistrategy()?;
    ex08_market_making()?;
    ex09_momentum_combination()?;
    ex10_cvar_portfolio()?;

    println!("\nAll examples completed.");
    Ok(())
}

// ============================================================================
// Shared utilities
// ============================================================================

/// Generate N synthetic daily log-returns for T days.
/// Uses a factor model: return_i = beta_i * market_factor + idio_i * sigma_i
/// Deterministic: same seed, same data every call.
fn generate_returns(n_assets: usize, t_days: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Xoshiro256PlusPlus::from_seed(seed);
    let market_vol = 0.012f64; // ~19% annualised market vol
    let mut returns = vec![vec![0.0f64; t_days]; n_assets];

    // Asset parameters: [beta, idiosyncratic_vol, drift]
    let params: Vec<(f64, f64, f64)> = (0..n_assets)
        .map(|i| {
            let beta = 0.5 + (i as f64 / n_assets as f64) * 1.0; // 0.5 to 1.5
            let ivol = 0.006 + (i as f64 / n_assets as f64) * 0.012;
            let drift = 0.00015 * (1.0 + i as f64 % 3 as f64 * 0.5); // small positive drift
            (beta, ivol, drift)
        })
        .collect();

    for t in 0..t_days {
        let market = normal_sample(&mut rng) * market_vol;
        for i in 0..n_assets {
            let (beta, ivol, drift) = params[i];
            returns[i][t] = drift + beta * market + normal_sample(&mut rng) * ivol;
        }
    }
    returns
}

/// Box-Muller transform for N(0,1) samples.
#[inline(always)]
fn normal_sample(rng: &mut impl Rng) -> f64 {
    let u1 = rng.next_f64().max(1e-15);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
}

/// Annualised Sharpe ratio from a return series.
fn sharpe(rets: &[f64]) -> f64 {
    if rets.len() < 2 {
        return 0.0;
    }
    let mu = mean(rets);
    let sigma = std_dev(rets);
    if sigma < 1e-12 {
        return 0.0;
    }
    mu / sigma * (252.0f64).sqrt()
}

/// Annualised Sortino ratio (downside deviation denominator).
fn sortino(rets: &[f64]) -> f64 {
    if rets.len() < 2 {
        return 0.0;
    }
    let mu = mean(rets);
    let downside_var: f64 =
        rets.iter().map(|&r| r.min(0.0).powi(2)).sum::<f64>() / rets.len() as f64;
    let downside_std = downside_var.sqrt();
    if downside_std < 1e-12 {
        return 0.0;
    }
    mu / downside_std * (252.0f64).sqrt()
}

/// Historical CVaR at confidence level alpha (e.g. 0.95).
/// Returns the expected loss in the worst (1-alpha) fraction of days.
#[allow(dead_code)]
fn cvar(rets: &[f64], alpha: f64) -> f64 {
    let mut sorted = rets.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cutoff = ((1.0 - alpha) * sorted.len() as f64).ceil() as usize;
    let tail = &sorted[..cutoff.max(1)];
    -mean(tail) // return as a positive loss number
}

/// Max drawdown from a return series.
fn max_drawdown(rets: &[f64]) -> f64 {
    let mut peak = 1.0f64;
    let mut nav = 1.0f64;
    let mut dd = 0.0f64;
    for &r in rets {
        nav *= 1.0 + r;
        if nav > peak {
            peak = nav;
        }
        let current_dd = (peak - nav) / peak;
        if current_dd > dd {
            dd = current_dd;
        }
    }
    dd
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance(xs: &[f64]) -> f64 {
    let mu = mean(xs);
    xs.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / xs.len() as f64
}

fn std_dev(xs: &[f64]) -> f64 {
    variance(xs).sqrt()
}

/// Project weight vector onto the probability simplex (Σw=1, w≥0).
fn simplex_project(w: &[f64]) -> Vec<f64> {
    let clamped: Vec<f64> = w.iter().map(|&x| x.max(0.0)).collect();
    let total: f64 = clamped.iter().sum();
    if total < 1e-15 {
        let n = w.len();
        return vec![1.0 / n as f64; n];
    }
    clamped.iter().map(|&x| x / total).collect()
}

// ============================================================================
// 1. MEAN-VARIANCE + TRANSACTION COST OPTIMISATION
// ============================================================================
//
// Classic Markowitz extended with:
//   - Proportional transaction costs (bid-ask spread + market impact)
//   - L1 turnover constraint (limit gross rebalancing)
//   - Cardinality soft penalty (prefer fewer active positions)
//
// Energy: -μ_p + λ_risk * σ²_p + λ_cost * turnover + λ_card * n_positions
//
// State:  Vec<f64> — raw weights (projected to simplex internally)
//
// Real use: daily/weekly rebalancing of equity long-only funds, ETF portfolio
//           construction, factor tilt optimisation.

fn ex01_mv_transaction_cost() -> Result<(), AnnealError> {
    const N: usize = 10;
    const T: usize = 252; // 1 year of daily returns

    let rets = generate_returns(N, T, 1_001);

    // Pre-compute asset stats
    let asset_means: Vec<f64> = (0..N).map(|i| mean(&rets[i])).collect();
    let asset_means_report = asset_means.clone();
    // Covariance matrix (sample)
    let mut cov = vec![vec![0.0f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            let mu_i = asset_means[i];
            let mu_j = asset_means[j];
            cov[i][j] =
                (0..T).map(|t| (rets[i][t] - mu_i) * (rets[j][t] - mu_j)).sum::<f64>() / T as f64;
        }
    }

    // Previous portfolio weights (for transaction cost calculation)
    let prev_weights = vec![1.0 / N as f64; N];
    let lambda_risk = 3.0f64; // risk aversion
    let lambda_cost = 10.0f64; // transaction cost per unit turnover
    let lambda_card = 0.02f64; // cardinality penalty per position
    let tc_rate = 0.001f64; // 10 bps one-way transaction cost

    let energy = FnEnergy(move |raw: &Vec<f64>| {
        let w = simplex_project(raw);

        // Portfolio return
        let port_ret: f64 = w.iter().zip(asset_means.iter()).map(|(wi, ri)| wi * ri).sum();

        // Portfolio variance via covariance matrix
        let port_var: f64 =
            (0..N).map(|i| (0..N).map(|j| w[i] * w[j] * cov[i][j]).sum::<f64>()).sum();

        // Transaction cost: L1 turnover from previous weights × tc_rate
        let turnover: f64 = w.iter().zip(prev_weights.iter()).map(|(wi, pw)| (wi - pw).abs()).sum();

        // Cardinality: count non-trivial positions
        let n_positions = w.iter().filter(|&&wi| wi > 0.005).count() as f64;

        -port_ret * 252.0
            + lambda_risk * port_var * 252.0
            + lambda_cost * turnover * tc_rate
            + lambda_card * n_positions
    });

    let initial = vec![1.0 / N as f64; N];

    let result = Annealer::builder()
        .objective(energy)
        .moves(GaussianMove::new(0.05))
        .schedule(Exponential::new(0.1, 0.9997))
        .iterations(300_000)
        .seed(1_001)
        .build()?
        .run(initial);

    let w = simplex_project(&result.best_state);
    let port_ret: f64 =
        w.iter().zip(asset_means_report.iter()).map(|(wi, ri)| wi * ri * 252.0).sum();
    let n_pos = w.iter().filter(|&&wi| wi > 0.005).count();

    println!(
        "[01] MV + TxCost   ret={:.2}%  positions={}  sharpe_energy={:.4}  accept={:.1}%",
        port_ret * 100.0,
        n_pos,
        result.best_energy,
        result.diagnostics.acceptance_rate() * 100.0,
    );
    Ok(())
}

// ============================================================================
// 2. TECHNICAL STRATEGY PARAMETER SWEEP
// ============================================================================
//
// Optimise a multi-signal strategy on synthetic OHLCV data:
//   Signal = α₁·SMA_crossover + α₂·RSI_signal + α₃·ATR_breakout
//
// State:  [fast_sma, slow_sma, rsi_period, rsi_lo, rsi_hi, atr_mult, stop_atr]
//         All continuous — Gaussian perturbation in param space.
//
// Energy: -Sortino ratio of backtest P&L
//          + penalty for in-sample overfitting proxy (too-short periods)
//
// Real use: systematic CTA signal development, equity long/short strategy
//           research, crypto market-making signal calibration.

fn ex02_technical_strategy() -> Result<(), AnnealError> {
    const T: usize = 504; // ~2 years daily bars

    // Synthetic price series: geometric Brownian motion with occasional jumps
    let prices: Vec<f64> = {
        let mut rng = Xoshiro256PlusPlus::from_seed(2_001);
        let mut p = 100.0f64;
        let mut ps = Vec::with_capacity(T);
        for _ in 0..T {
            let drift = 0.0003;
            let vol = 0.015;
            let jump = if rng.next_f64() < 0.02 {
                // 2% chance of ±5% jump
                (if rng.next_f64() < 0.5 { 1.0 } else { -1.0 }) * 0.05
            } else {
                0.0
            };
            p *= (drift + vol * normal_sample(&mut rng) + jump).exp();
            ps.push(p);
        }
        ps
    };

    let prices_clone = prices.clone();

    // Backtest: generate daily return series given strategy params.
    // params: [fast(2-20), slow(10-100), rsi_p(5-30), rsi_lo(20-40),
    //          rsi_hi(60-80), atr_mult(0.5-3), stop_atr(0.5-3)]
    let energy = FnEnergy(move |params: &Vec<f64>| {
        let fast = (params[0].abs() % 18.0 + 2.0) as usize; // 2..20
        let slow = (params[1].abs() % 90.0 + fast as f64 + 2.0) as usize; // fast+2..fast+92
        let rsi_p = (params[2].abs() % 25.0 + 5.0) as usize; // 5..30
        let rsi_lo = params[3].clamp(20.0, 40.0);
        let rsi_hi = params[4].clamp(60.0, 80.0);

        if slow >= prices_clone.len() || rsi_p >= prices_clone.len() {
            return 1e9;
        }
        let warmup = slow.max(rsi_p) + 1;
        if warmup + 10 >= prices_clone.len() {
            return 1e9;
        }

        // Compute daily returns
        let daily_ret: Vec<f64> =
            (1..prices_clone.len()).map(|t| prices_clone[t] / prices_clone[t - 1] - 1.0).collect();

        // SMA crossover signal
        let sma = |period: usize, from: usize| -> f64 {
            prices_clone[from - period..from].iter().sum::<f64>() / period as f64
        };

        // RSI computation
        let rsi = |period: usize, from: usize| -> f64 {
            let gains: f64 =
                (from - period..from).map(|t| daily_ret[t].max(0.0)).sum::<f64>() / period as f64;
            let losses: f64 = (from - period..from).map(|t| (-daily_ret[t]).max(0.0)).sum::<f64>()
                / period as f64;
            if losses < 1e-12 {
                return 100.0;
            }
            100.0 - 100.0 / (1.0 + gains / losses)
        };

        // Strategy P&L
        let mut strategy_rets = Vec::new();
        let mut position = 0.0f64; // -1, 0, +1

        for t in warmup..prices_clone.len() - 1 {
            let fast_ma = sma(fast, t);
            let slow_ma = sma(slow, t);
            let rsi_val = rsi(rsi_p, t);

            // Combined signal
            let trend_signal: f64 = if fast_ma > slow_ma * 1.001 {
                1.0
            } else if fast_ma < slow_ma * 0.999 {
                -1.0
            } else {
                0.0
            };
            let rsi_signal: f64 = if rsi_val < rsi_lo {
                1.0
            } else if rsi_val > rsi_hi {
                -1.0
            } else {
                0.0
            };

            // Combined: trend must agree with RSI or either is strong
            let new_pos: f64 = if trend_signal == rsi_signal {
                trend_signal
            } else if trend_signal.abs() > 0.5 && rsi_signal == 0.0 {
                trend_signal * 0.5
            } else {
                0.0
            };

            // Transaction cost on position change
            let tc = (new_pos - position).abs() * 0.001;
            strategy_rets.push(position * daily_ret[t] - tc);
            position = new_pos;
        }

        if strategy_rets.is_empty() {
            return 1e9;
        }

        // Penalise too-short periods (overfitting proxy)
        let complexity_penalty = 0.05 / fast as f64 + 0.02 / rsi_p as f64;

        -sortino(&strategy_rets) + complexity_penalty
    });

    // Start with common defaults: fast=10, slow=30, rsi=14, lo=30, hi=70
    let initial = vec![10.0f64, 30.0, 14.0, 30.0, 70.0];

    let result = Annealer::builder()
        .objective(energy)
        .moves(GaussianMove::new(2.0))
        .schedule(Exponential::new(1.0, 0.9997))
        .iterations(200_000)
        .seed(2_001)
        .build()?
        .run(initial);

    let p = &result.best_state;
    let fast = (p[0].abs() % 18.0 + 2.0) as usize;
    let slow = (p[1].abs() % 90.0 + fast as f64 + 2.0) as usize;
    let rsi_p = (p[2].abs() % 25.0 + 5.0) as usize;

    println!(
        "[02] Tech Strategy  sortino={:.3}  fast={fast}  slow={slow}  rsi={rsi_p}  lo={:.0}  hi={:.0}  accept={:.1}%",
        -result.best_energy,
        p[3].clamp(20.0, 40.0),
        p[4].clamp(60.0, 80.0),
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 3. OPTIMAL ORDER EXECUTION (ALMGREN-CHRISS)
// ============================================================================
//
// Liquidate X shares over T intervals. Split the order across buckets to
// minimise implementation shortfall:
//   IS = permanent_impact + temporary_impact + timing_risk
//
// Almgren-Chriss model:
//   permanent_impact  = η · Σ_t (n_t / V) · n_t       (η: perm. impact coeff)
//   temporary_impact  = ε · Σ_t (n_t / V)^γ            (ε: temp. impact coeff)
//   timing_risk       = σ · Σ_t n_t² / V²              (execution risk variance)
//
// State:  Vec<f64> — fraction to execute in each of T buckets (Σ=1)
// Move:   Gaussian perturbation + simplex projection
//
// Real use: institutional equity execution, large-block trades, algo TWAP/VWAP
//           optimisation (Two Sigma, Citadel desk tools).

fn ex03_order_execution() -> Result<(), AnnealError> {
    const BUCKETS: usize = 20; // execution intervals (e.g. 20 × 15-min buckets)
    let total_shares = 1_000_000.0f64; // shares to sell
    let adv = 10_000_000.0f64; // average daily volume
    let price = 50.0f64; // current mid-price
    let sigma_daily = 0.015f64; // daily vol (1.5%)
    let sigma_bucket = sigma_daily / (BUCKETS as f64).sqrt();

    // Almgren-Chriss parameters
    let eta = 0.1f64; // permanent impact coefficient
    let eps = 0.05f64; // temporary impact coefficient
    let gamma = 0.6f64; // temporary impact concavity (0.5 = square-root law)
    let lambda = 1e-6f64; // risk aversion ($ per unit variance)

    let energy = FnEnergy(move |raw: &Vec<f64>| {
        let fractions = simplex_project(raw);
        let mut remaining = total_shares;

        let mut perm_impact = 0.0f64;
        let mut temp_impact = 0.0f64;
        let mut exec_risk = 0.0f64;

        for &frac in &fractions {
            let n_t = frac * total_shares;
            let participation = n_t / adv;

            // Costs in dollar terms
            perm_impact += eta * participation * n_t * price;
            temp_impact += eps * participation.powf(gamma) * n_t * price;
            exec_risk += lambda * (remaining * sigma_bucket * price).powi(2);

            remaining -= n_t;
        }
        let total_cost = perm_impact + temp_impact + exec_risk;

        // Penalty for negative executions (can't buy while liquidating)
        let neg_penalty: f64 = raw.iter().map(|&x| (-x).max(0.0) * 1e8).sum();

        total_cost + neg_penalty
    });

    // TWAP baseline: equal split across all buckets
    let initial = vec![1.0 / BUCKETS as f64; BUCKETS];
    let twap_cost = {
        let n_t = total_shares / BUCKETS as f64;
        let participation = n_t / adv;
        let perm = BUCKETS as f64 * eta * participation * n_t * price;
        let temp = BUCKETS as f64 * eps * participation.powf(gamma) * n_t * price;
        perm + temp
    };

    let result = Annealer::builder()
        .objective(energy)
        .moves(GaussianMove::new(0.03))
        .schedule(Exponential::new(1e6, 0.9997))
        .iterations(300_000)
        .seed(3_001)
        .build()?
        .run(initial);

    let fracs = simplex_project(&result.best_state);
    let front_loading: f64 = fracs[..BUCKETS / 2].iter().sum::<f64>() * 100.0;

    println!(
        "[03] Order Execution  TWAP=${:.0}  SA=${:.0}  saving={:.1}%  front_loaded={:.1}%  accept={:.1}%",
        twap_cost,
        result.best_energy,
        (twap_cost - result.best_energy) / twap_cost * 100.0,
        front_loading,
        result.diagnostics.acceptance_rate() * 100.0,
    );
    Ok(())
}

// ============================================================================
// 4. PAIRS TRADING THRESHOLD OPTIMISATION
// ============================================================================
//
// Given a cointegrated pair (spread = log(P_A) - β·log(P_B)):
//   Entry long-spread: spread < -entry_z * σ
//   Entry short-spread: spread > +entry_z * σ
//   Exit: |spread| < exit_z * σ
//   Stop: |spread| > stop_z * σ  (cut losing trade)
//
// State:  Vec<f64>  — [entry_z, exit_z, stop_z, lookback_days]
// Energy: -Information Ratio of the pairs strategy P&L
//
// Real use: stat arb desks (Renaissance, D.E. Shaw), equity market-neutral,
//           ETF arbitrage, crypto triangular arbitrage threshold tuning.

fn ex04_pairs_trading() -> Result<(), AnnealError> {
    const T: usize = 504;

    // Synthetic cointegrated pair: spread = log(A) - beta*log(B)
    // True spread is mean-reverting AR(1) with half-life ~20 days
    let spread: Vec<f64> = {
        let mut rng = Xoshiro256PlusPlus::from_seed(4_001);
        let half_life = 20.0f64;
        let phi = (-1.0 / half_life).exp(); // AR(1) coefficient
        let sigma_noise = 0.8f64;
        let mut s = 0.0f64;
        (0..T)
            .map(|_| {
                s = phi * s + sigma_noise * normal_sample(&mut rng);
                s
            })
            .collect()
    };

    let spread_clone = spread.clone();

    let energy = FnEnergy(move |params: &Vec<f64>| {
        let entry_z = params[0].abs().clamp(0.5, 4.0);
        let exit_z = params[1].abs().clamp(0.0, entry_z - 0.1);
        let stop_z = params[2].abs().clamp(entry_z + 0.1, 8.0);
        let lookback = (params[3].abs() as usize).clamp(10, 60);

        if lookback + 5 >= spread_clone.len() {
            return 1e9;
        }

        let mut rets = Vec::new();
        let mut position = 0.0f64; // +1 = long spread, -1 = short spread, 0 = flat
        let tc = 0.002f64; // round-trip transaction cost

        for t in lookback..spread_clone.len() {
            // Rolling statistics over lookback window
            let window = &spread_clone[t - lookback..t];
            let mu = mean(window);
            let sd = std_dev(window);
            if sd < 1e-10 {
                continue;
            }

            let z = (spread_clone[t] - mu) / sd;
            let mut daily_pnl;

            if position == 0.0 {
                // Entry rules
                if z < -entry_z {
                    position = 1.0; // go long spread
                    daily_pnl = -tc;
                } else if z > entry_z {
                    position = -1.0; // go short spread
                    daily_pnl = -tc;
                } else {
                    daily_pnl = 0.0;
                }
            } else {
                // P&L = change in spread × position direction
                let spread_chg = spread_clone[t] - spread_clone[t - 1];
                daily_pnl = position * spread_chg;

                // Exit rules
                if z.abs() < exit_z || z.signum() == position.signum() * -1.0 && z.abs() > stop_z {
                    daily_pnl -= tc; // exit cost
                    position = 0.0;
                }
            }
            rets.push(daily_pnl);
        }

        if rets.is_empty() {
            return 1e9;
        }

        // Information Ratio (Sharpe without the risk-free adjustment)
        let ir = sharpe(&rets);
        -ir + 0.1 / entry_z // mild regulariser: prefer tighter entry thresholds
    });

    // Canonical starting point
    let initial = vec![2.0f64, 0.5, 3.5, 30.0];

    let result = Annealer::builder()
        .objective(energy)
        .moves(GaussianMove::new(0.2))
        .schedule(Exponential::new(0.5, 0.9996))
        .iterations(150_000)
        .seed(4_001)
        .build()?
        .run(initial);

    let p = &result.best_state;
    println!(
        "[04] Pairs Trading  IR={:.3}  entry_z={:.2}  exit_z={:.2}  stop_z={:.2}  lookback={:.0}d  accept={:.1}%",
        -result.best_energy,
        p[0].abs().clamp(0.5, 4.0),
        p[1].abs().clamp(0.0, 3.9),
        p[2].abs().clamp(2.1, 8.0),
        p[3].abs().clamp(10.0, 60.0),
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 5. RISK PARITY ALLOCATION
// ============================================================================
//
// Allocate weights so each asset contributes equally to total portfolio risk.
// Risk contribution of asset i: RC_i = w_i · (Σw)_i / (w'Σw)
// Perfect risk parity: RC_i = 1/N for all i.
//
// Energy: Σ_i (RC_i - 1/N)²  (sum of squared deviations from equal contribution)
//         + barrier term if any weight is negative or too concentrated
//
// Real use: Bridgewater All Weather, AQR Risk Parity funds, multi-asset
//           risk-budgeting overlays, factor-risk allocation.

fn ex05_risk_parity() -> Result<(), AnnealError> {
    const N: usize = 8;
    const T: usize = 252;

    let rets = generate_returns(N, T, 5_001);
    let asset_means: Vec<f64> = (0..N).map(|i| mean(&rets[i])).collect();
    let mut cov = vec![vec![0.0f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            cov[i][j] = (0..T)
                .map(|t| (rets[i][t] - asset_means[i]) * (rets[j][t] - asset_means[j]))
                .sum::<f64>()
                / T as f64;
        }
    }

    let energy = FnEnergy(move |raw: &Vec<f64>| {
        let w = simplex_project(raw);

        // Portfolio variance
        let port_var: f64 =
            (0..N).map(|i| (0..N).map(|j| w[i] * w[j] * cov[i][j]).sum::<f64>()).sum();
        if port_var < 1e-20 {
            return 1e9;
        }

        // Marginal contribution to risk for each asset: (Σw)_i
        let mcr: Vec<f64> = (0..N).map(|i| (0..N).map(|j| cov[i][j] * w[j]).sum::<f64>()).collect();

        // Risk contribution: RC_i = w_i * MCR_i / port_vol
        let port_vol = port_var.sqrt();
        let target_rc = 1.0 / N as f64;

        // Sum of squared deviations from equal risk contribution
        let rc_dev: f64 = (0..N)
            .map(|i| {
                let rc_i = w[i] * mcr[i] / port_vol;
                (rc_i - target_rc).powi(2)
            })
            .sum();

        // Concentration penalty (no single asset > 40%)
        let concentration: f64 = w.iter().map(|&wi| (wi - 0.4).max(0.0) * 100.0).sum();

        rc_dev * 1000.0 + concentration
    });

    // Initial: equal weight (good starting point for risk parity)
    let initial = vec![1.0 / N as f64; N];

    let result = Annealer::builder()
        .objective(energy)
        .moves(GaussianMove::new(0.03))
        .schedule(Exponential::new(0.1, 0.9997))
        .iterations(250_000)
        .seed(5_001)
        .build()?
        .run(initial);

    let w = simplex_project(&result.best_state);
    let w_max = w.iter().cloned().fold(0.0f64, f64::max);
    let w_min = w.iter().cloned().fold(f64::MAX, f64::min);

    println!(
        "[05] Risk Parity    rc_dev={:.6}  w_min={:.1}%  w_max={:.1}%  accept={:.1}%",
        result.best_energy / 1000.0,
        w_min * 100.0,
        w_max * 100.0,
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 6. REGIME-AWARE STRATEGY SWITCHING
// ============================================================================
//
// A strategy switches between three modes based on detected market regime:
//   BULL:    long equity + short vol  (trend following)
//   BEAR:    short equity + long bonds  (defensive)
//   HIGH_VOL: market-neutral + long vol  (dispersion)
//
// Regime detection uses two signals:
//   Trend: 50-day return vs threshold_bull / threshold_bear
//   Vol:   20-day realised vol vs vol_threshold
//
// State:  [threshold_bull, threshold_bear, vol_threshold, mode_persistence]
// Energy: -Sharpe of the switching strategy over synthetic multi-regime data
//
// Real use: macro CTA strategies, risk-on/risk-off switching overlays,
//           systematic global macro (Man AHL, Winton).

fn ex06_regime_switching() -> Result<(), AnnealError> {
    const T: usize = 756; // 3 years

    // Synthetic multi-regime market: 3 regimes with known transitions
    let (equity_rets, vol_rets, bond_rets): (Vec<f64>, Vec<f64>, Vec<f64>) = {
        let mut rng = Xoshiro256PlusPlus::from_seed(6_001);
        let mut regime = 0usize; // 0=bull, 1=bear, 2=high_vol
        let mut eq_rets = Vec::with_capacity(T);
        let mut vx_rets = Vec::with_capacity(T);
        let mut bd_rets = Vec::with_capacity(T);

        for _ in 0..T {
            // Regime transition (Markov chain)
            let r = rng.next_f64();
            regime = match regime {
                0 => {
                    if r < 0.97 {
                        0
                    } else if r < 0.99 {
                        1
                    } else {
                        2
                    }
                }
                1 => {
                    if r < 0.93 {
                        1
                    } else if r < 0.98 {
                        2
                    } else {
                        0
                    }
                }
                _ => {
                    if r < 0.90 {
                        2
                    } else if r < 0.95 {
                        1
                    } else {
                        0
                    }
                }
            };

            let (eq_drift, eq_vol, vx_drift, vx_vol, bd_drift) = match regime {
                0 => (0.0008, 0.010, -0.003, 0.020, 0.0001), // bull
                1 => (-0.0010, 0.018, 0.005, 0.040, 0.0004), // bear
                _ => (0.0000, 0.025, 0.008, 0.060, 0.0002),  // high_vol
            };

            eq_rets.push(eq_drift + eq_vol * normal_sample(&mut rng));
            vx_rets.push(vx_drift + vx_vol * normal_sample(&mut rng));
            bd_rets.push(bd_drift + 0.003 * normal_sample(&mut rng));
        }
        (eq_rets, vx_rets, bd_rets)
    };

    let energy = FnEnergy(move |params: &Vec<f64>| {
        let bull_thresh = params[0].clamp(0.001, 0.05); // 50d return > this → bull
        let bear_thresh = params[1].clamp(-0.05, -0.001);
        let vol_thresh = params[2].clamp(0.005, 0.03); // 20d rv > this → high_vol
        let persist = (params[3].abs() as usize).clamp(1, 10); // regime confirmation days

        let lookback_trend = 50usize;
        let lookback_vol = 20usize;
        let warmup = lookback_trend + persist;
        if warmup >= T {
            return 1e9;
        }

        // Pre-compute signals
        let trend_50: Vec<f64> = (lookback_trend..T)
            .map(|t| equity_rets[t - lookback_trend..t].iter().sum::<f64>())
            .collect();

        let rv_20: Vec<f64> =
            (lookback_vol..T).map(|t| std_dev(&equity_rets[t - lookback_vol..t])).collect();

        let mut strat_rets = Vec::new();
        let mut current_regime = 0usize;
        let mut signal_count = 0usize;

        for t in warmup..T {
            let trend = trend_50[t - lookback_trend];
            let rv = rv_20[t - lookback_vol];

            let raw_regime = if rv > vol_thresh {
                2
            } else if trend > bull_thresh {
                0
            } else if trend < bear_thresh {
                1
            } else {
                current_regime
            };

            // Regime persistence: only switch after persist consecutive signals
            if raw_regime != current_regime {
                signal_count += 1;
                if signal_count >= persist {
                    current_regime = raw_regime;
                    signal_count = 0;
                }
            } else {
                signal_count = 0;
            }

            // Portfolio weights by regime
            let (w_eq, w_vx, w_bd): (f64, f64, f64) = match current_regime {
                0 => (0.80, -0.10, 0.30), // bull: long equity, short vol, some bonds
                1 => (-0.30, 0.30, 0.80), // bear: short equity, long vol, bonds
                _ => (0.10, 0.50, 0.40),  // high_vol: neutral equity, long vol
            };

            // Daily P&L (simplified: no rebalancing cost)
            let daily = w_eq * equity_rets[t] + w_vx * vol_rets[t] + w_bd * bond_rets[t];
            strat_rets.push(daily);
        }

        if strat_rets.is_empty() {
            return 1e9;
        }
        -sharpe(&strat_rets) + max_drawdown(&strat_rets) * 2.0 // penalise drawdown
    });

    let initial = vec![0.01f64, -0.01, 0.015, 3.0];

    let result = Annealer::builder()
        .objective(energy)
        .moves(GaussianMove::new(0.005))
        .schedule(Exponential::new(0.5, 0.9997))
        .iterations(200_000)
        .seed(6_001)
        .build()?
        .run(initial);

    let p = &result.best_state;
    println!(
        "[06] Regime Switching  sharpe={:.3}  bull={:.3}  bear={:.3}  vol_th={:.3}  persist={:.0}  accept={:.1}%",
        -result.best_energy,
        p[0].clamp(0.001, 0.05),
        p[1].clamp(-0.05, -0.001),
        p[2].clamp(0.005, 0.03),
        p[3].abs().clamp(1.0, 10.0),
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 7. KELLY CRITERION MULTI-STRATEGY SIZING
// ============================================================================
//
// Fractional Kelly across K strategies with correlated P&L.
// Full Kelly maximises E[log(wealth)] but is too volatile in practice.
// Fractional Kelly: f_i = κ · kelly_i, where κ ∈ (0, 1].
//
// Energy: -E[log(1 + Σ_i f_i · ret_i)]   (Kelly objective)
//         + leverage_penalty · max(0, Σ|f_i| - max_gross)
//         + margin_penalty   · max(0, Σ f_i^- - max_short)
//
// State:  Vec<f64>  — fraction of wealth in each strategy (can be negative = short)
//
// Real use: multi-strategy hedge funds, crypto quant books, portfolio of signals.

fn ex07_kelly_multistrategy() -> Result<(), AnnealError> {
    const K: usize = 6; // strategies
    const T: usize = 252;

    // Synthetic strategy returns (each has different Sharpe, correlation)
    let strat_rets: Vec<Vec<f64>> = {
        let mut rng = Xoshiro256PlusPlus::from_seed(7_001);
        let sharpes = [1.2, 0.8, 1.5, 0.6, 1.0, 0.9f64];
        let vols = [0.08, 0.12, 0.06, 0.15, 0.10, 0.11f64];
        let daily_vol: Vec<f64> = vols.iter().map(|&v| v / (252.0f64).sqrt()).collect();
        let daily_mu: Vec<f64> =
            sharpes.iter().zip(daily_vol.iter()).map(|(&sr, &dv)| sr * dv).collect();

        // Correlation factor (common market factor)
        let beta = [0.3, 0.5, 0.1, 0.6, 0.4, 0.2f64];

        let mut all: Vec<Vec<f64>> = vec![Vec::with_capacity(T); K];
        for _ in 0..T {
            let mkt = normal_sample(&mut rng) * 0.01;
            for s in 0..K {
                let idio =
                    normal_sample(&mut rng) * daily_vol[s] * (1.0 - beta[s] * beta[s]).sqrt();
                all[s].push(daily_mu[s] + beta[s] * mkt + idio);
            }
        }
        all
    };

    let max_gross = 2.0f64; // max gross leverage
    let max_net = 0.5f64; // max net long
    let kelly_pen = 10.0f64;

    let energy = FnEnergy(move |fracs: &Vec<f64>| {
        // Kelly objective: E[log(1 + Σ f_i r_i)]
        let log_wealth: f64 = (0..T)
            .map(|t| {
                let portfolio_ret: f64 =
                    fracs.iter().zip(strat_rets.iter()).map(|(&f, s)| f * s[t]).sum();
                // Guard against ruin (log is undefined at ≤ -1)
                (1.0 + portfolio_ret).max(0.01).ln()
            })
            .sum::<f64>()
            / T as f64;

        // Constraints
        let gross_lev: f64 = fracs.iter().map(|&f| f.abs()).sum();
        let net_lev: f64 = fracs.iter().sum::<f64>();
        let leverage_pen = kelly_pen * (gross_lev - max_gross).max(0.0).powi(2);
        let net_pen = kelly_pen * (net_lev.abs() - max_net).max(0.0).powi(2);

        -log_wealth + leverage_pen + net_pen
    });

    // Initial: equal fractional Kelly (rough heuristic start)
    let initial = vec![0.1f64; K];

    let result = Annealer::builder()
        .objective(energy)
        .moves(GaussianMove::new(0.05))
        .schedule(Exponential::new(0.1, 0.9997))
        .iterations(200_000)
        .seed(7_001)
        .build()?
        .run(initial);

    let gross: f64 = result.best_state.iter().map(|&f| f.abs()).sum();
    let net: f64 = result.best_state.iter().sum::<f64>();

    println!(
        "[07] Kelly Multi-Strat  log_ret={:.4}/day  gross={:.2}x  net={:.2}x  accept={:.1}%",
        -result.best_energy,
        gross,
        net,
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 8. MARKET MAKING SPREAD OPTIMISATION
// ============================================================================
//
// A market maker quotes bid and ask around mid-price. Optimise:
//   bid_spread:  ticks below mid to quote bid
//   ask_spread:  ticks above mid to quote ask
//   inv_limit:   inventory limit before skewing quotes
//   skew_factor: how aggressively to skew when at inventory limit
//
// Revenue  = (ask_spread + bid_spread)/2 × fill_rate
// Risk     = inventory_risk × sigma²
// Adverse  = probability of being "picked off" by informed traders
//
// Glosten-Milgrom inspired model for adverse selection probability.
//
// Real use: HFT market making (Virtu, Citadel Securities), DEX liquidity
//           provision, options market making, crypto perpetual swap making.

fn ex08_market_making() -> Result<(), AnnealError> {
    const T: usize = 10_000; // ticks in simulation

    // Simulate order flow: fraction of trades are informed (adverse selection)
    let order_flow: Vec<(bool, f64)> = {
        // (is_buy, is_informed)
        let mut rng = Xoshiro256PlusPlus::from_seed(8_001);
        (0..T)
            .map(|_| {
                let is_buy = rng.next_f64() < 0.5;
                let _is_informed = rng.next_f64() < 0.15; // 15% informed traders (consumed for RNG advance)
                let price_move = normal_sample(&mut rng) * 0.1; // mid-price moves
                (is_buy, price_move)
            })
            .collect()
    };
    let sigma_tick = 0.1f64; // mid-price vol per tick (ticks)

    let energy = FnEnergy(move |params: &Vec<f64>| {
        let half_spread = params[0].abs().clamp(0.1, 5.0); // ticks
        let inv_limit = params[1].abs().clamp(1.0, 20.0); // max inventory
        let skew_factor = params[2].abs().clamp(0.0, 1.0); // [0,1]
        let fill_prob_base = params[3].abs().clamp(0.1, 0.9); // base fill prob

        let mut inventory = 0.0f64;
        let mut pnl = 0.0f64;
        let mut n_trades = 0u32;

        for &(is_buy, price_move) in &order_flow {
            // Inventory-based quote skewing (reduce fill probability when at limit)
            let inv_ratio = inventory.abs() / inv_limit;
            let skew = skew_factor * inv_ratio * inventory.signum();

            // Effective spread for this side
            let effective_spread = half_spread + skew;

            // Fill probability decreases as spread widens
            let fill_prob = (fill_prob_base * (-effective_spread * 0.3).exp()).clamp(0.0, 0.95);

            // Adverse selection: informed traders always fill at the expense of maker
            let adverse = price_move.abs() > half_spread * 0.8;

            // Trade happens
            let trade_occurs = !adverse; // simplified: informed traders consume the spread
            if trade_occurs && is_buy {
                // We sell at ask: +spread, -1 inventory
                pnl += effective_spread;
                inventory -= 1.0;
                n_trades += 1;
            } else if trade_occurs && !is_buy {
                // We buy at bid: +spread, +1 inventory
                pnl += effective_spread;
                inventory += 1.0;
                n_trades += 1;
            }

            // Adverse selection loss
            if adverse {
                pnl -= price_move.abs() * fill_prob * 0.5;
            }

            // Inventory carrying cost (mark-to-market risk)
            pnl -= 0.5 * sigma_tick.powi(2) * inventory.powi(2) * 0.001;

            // Hard inventory limit hit — forced liquidation penalty
            if inventory.abs() > inv_limit * 1.5 {
                pnl -= 5.0 * inventory.abs();
                inventory *= 0.5; // partial liquidation
            }
        }

        // Normalise by trade count
        if n_trades == 0 {
            return 1e9;
        }
        -(pnl / T as f64) * 10_000.0 // in bps
    });

    let initial = vec![1.0f64, 5.0, 0.3, 0.5];

    let result = Annealer::builder()
        .objective(energy)
        .moves(GaussianMove::new(0.1))
        .schedule(Adaptive::new(500.0, 0.35))
        .iterations(150_000)
        .seed(8_001)
        .build()?
        .run(initial);

    let p = &result.best_state;
    println!(
        "[08] Market Making   pnl={:.2}bps/tick  spread={:.2}  inv_lim={:.1}  skew={:.2}  accept={:.1}%",
        -result.best_energy,
        p[0].abs().clamp(0.1, 5.0),
        p[1].abs().clamp(1.0, 20.0),
        p[2].abs().clamp(0.0, 1.0),
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 9. CROSS-ASSET MOMENTUM SIGNAL COMBINATION
// ============================================================================
//
// Blend momentum signals at multiple horizons across asset classes.
// Final signal = Σ_h w_h · sign(ret_{t-1, t-1-h}) for each horizon h.
// Portfolio = long/short assets by combined signal strength.
//
// Horizons: [1d, 5d, 21d, 63d, 126d, 252d]
// Asset classes: equity, bonds, commodities, currencies (6 assets total)
//
// State:  Vec<f64>  — weights per horizon (6 weights, projected to unit norm)
// Energy: -Sharpe of the multi-asset momentum strategy
//          + turnover penalty (higher-frequency signals → more rebalancing)
//
// Real use: CTA trend following (AHL, Winton), multi-asset momentum factor
//           construction (AQR MOM), signal ensemble weighting.

fn ex09_momentum_combination() -> Result<(), AnnealError> {
    const N_ASSETS: usize = 6;
    const T: usize = 756; // 3 years
    let horizons: [usize; 6] = [1, 5, 21, 63, 126, 252];

    // Generate synthetic multi-asset returns with momentum structure
    let returns: Vec<Vec<f64>> = {
        let mut rng = Xoshiro256PlusPlus::from_seed(9_001);
        let vols = [0.012, 0.006, 0.018, 0.015, 0.010, 0.013f64]; // daily
        let drifts = [0.0003, 0.0001, 0.0002, -0.0001, 0.0002, 0.0001f64];
        let momentum_strength = [0.04, 0.06, 0.03, 0.05, 0.04, 0.05f64]; // autocorrelation

        let mut rets: Vec<Vec<f64>> = vec![Vec::with_capacity(T); N_ASSETS];
        let mut prev: Vec<f64> = vec![0.0; N_ASSETS];

        for _ in 0..T {
            for i in 0..N_ASSETS {
                let r = drifts[i]
                    + momentum_strength[i] * prev[i]  // mild momentum
                    + vols[i] * normal_sample(&mut rng);
                rets[i].push(r);
                prev[i] = r;
            }
        }
        rets
    };

    let max_horizon = *horizons.iter().max().unwrap();
    let turnover_costs = [0.001, 0.0003, 0.0001, 0.00005, 0.00003, 0.00002f64]; // per horizon

    let energy = FnEnergy(move |raw_weights: &Vec<f64>| {
        // L2-normalise horizon weights → unit sphere projection
        let norm: f64 = raw_weights.iter().map(|&w| w * w).sum::<f64>().sqrt().max(1e-12);
        let hw: Vec<f64> = raw_weights.iter().map(|&w| w / norm).collect();

        let mut strategy_rets = Vec::new();
        let mut prev_positions = vec![0.0f64; N_ASSETS];

        for t in max_horizon..T {
            // Compute blended signal for each asset
            let signals: Vec<f64> = (0..N_ASSETS)
                .map(|i| {
                    let raw_sig: f64 = hw
                        .iter()
                        .zip(horizons.iter())
                        .map(|(&w, &h)| {
                            if t < h {
                                return 0.0;
                            }
                            let horizon_ret: f64 = returns[i][t - h..t].iter().sum();
                            w * horizon_ret.signum()
                        })
                        .sum();
                    raw_sig.clamp(-1.0, 1.0)
                })
                .collect();

            // Cross-sectional normalise: equal vol allocation
            let sig_norm: f64 = signals.iter().map(|&s| s * s).sum::<f64>().sqrt().max(1e-12);
            let positions: Vec<f64> = signals.iter().map(|&s| s / sig_norm).collect();

            // Daily P&L
            let daily: f64 = positions.iter().zip(returns.iter()).map(|(&pos, r)| pos * r[t]).sum();

            // Turnover cost (blended across horizons)
            let turnover: f64 = positions
                .iter()
                .zip(prev_positions.iter())
                .map(|(&p, &pp)| (p - pp).abs())
                .sum::<f64>();
            let tc: f64 =
                hw.iter().zip(turnover_costs.iter()).map(|(&w, &tc)| w.abs() * tc).sum::<f64>()
                    * turnover;

            strategy_rets.push(daily - tc);
            prev_positions = positions;
        }

        if strategy_rets.is_empty() {
            return 1e9;
        }
        -sharpe(&strategy_rets)
    });

    // Equal weight start
    let initial = vec![1.0f64; horizons.len()];

    let result = Annealer::builder()
        .objective(energy)
        .moves(GaussianMove::new(0.3))
        .schedule(Exponential::new(0.5, 0.9997))
        .iterations(200_000)
        .seed(9_001)
        .build()?
        .run(initial);

    let norm: f64 = result.best_state.iter().map(|&w| w * w).sum::<f64>().sqrt();
    let final_w: Vec<f64> = result.best_state.iter().map(|&w| w / norm).collect();

    println!(
        "[09] Momentum Combo  sharpe={:.3}  w=[{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}]  accept={:.1}%",
        -result.best_energy,
        final_w[0],
        final_w[1],
        final_w[2],
        final_w[3],
        final_w[4],
        final_w[5],
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

// ============================================================================
// 10. CVaR-CONSTRAINED PORTFOLIO OPTIMISATION
// ============================================================================
//
// Maximise expected return subject to a CVaR budget:
//   max  μ_p
//   s.t. CVaR_{95%}(portfolio) ≤ CVaR_budget
//        Σ w_i = 1, w_i ≥ 0  (long-only)
//
// CVaR (Conditional Value-at-Risk, aka Expected Shortfall) is coherent,
// convex in weights — but the SA formulation handles it naturally alongside
// non-convex constraints (cardinality, sector limits, ESG scores).
//
// Energy: -μ_p + λ · max(0, CVaR(w) - budget)²
//
// Real use: insurance / pension fund tail-risk budgeting, UCITS fund VaR
//           compliance, bank trading book FRTB CVaR optimisation.

fn ex10_cvar_portfolio() -> Result<(), AnnealError> {
    const N: usize = 12;
    const T: usize = 504; // 2 years for CVaR estimation

    let rets = generate_returns(N, T, 10_001);
    let asset_means: Vec<f64> = (0..N).map(|i| mean(&rets[i]) * 252.0).collect();
    let asset_means_report = asset_means.clone();

    // Scenario matrix: row = day, col = asset return
    let scenarios: Vec<Vec<f64>> = (0..T).map(|t| (0..N).map(|i| rets[i][t]).collect()).collect();
    let scenarios_report = scenarios.clone();

    let cvar_budget = 0.02f64; // 2% daily CVaR at 95% confidence
    let lambda = 500.0f64; // hard constraint penalty weight

    // ESG scores: each asset has a score 0-10; require portfolio score ≥ 6
    let esg_scores: [f64; N] = [8.0, 5.0, 7.0, 9.0, 4.0, 6.0, 8.0, 3.0, 7.0, 9.0, 6.0, 5.0];
    let esg_min = 6.0f64;

    let energy = FnEnergy(move |raw: &Vec<f64>| {
        let w = simplex_project(raw);

        // Expected annualised return
        let port_ret: f64 = w.iter().zip(asset_means.iter()).map(|(wi, ri)| wi * ri).sum();

        // Historical CVaR: sort portfolio scenario returns, take worst 5%
        let mut port_scenario_rets: Vec<f64> = scenarios
            .iter()
            .map(|day| w.iter().zip(day.iter()).map(|(wi, ri)| wi * ri).sum())
            .collect();
        port_scenario_rets.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let cutoff = ((1.0 - 0.95) * T as f64).ceil() as usize;
        let tail = &port_scenario_rets[..cutoff.max(1)];
        let portfolio_cvar = -mean(tail); // positive number

        // CVaR constraint penalty
        let cvar_penalty = lambda * (portfolio_cvar - cvar_budget).max(0.0).powi(2);

        // ESG constraint: weighted average score ≥ esg_min
        let port_esg: f64 = w.iter().zip(esg_scores.iter()).map(|(wi, si)| wi * si).sum();
        let esg_penalty = lambda * (esg_min - port_esg).max(0.0).powi(2);

        // Concentration: no single asset > 25%
        let conc_penalty: f64 = w.iter().map(|&wi| lambda * (wi - 0.25).max(0.0).powi(2)).sum();

        -port_ret + cvar_penalty + esg_penalty + conc_penalty
    });

    // Equal weight start
    let initial = vec![1.0 / N as f64; N];

    let result = Annealer::builder()
        .objective(energy)
        .moves(GaussianMove::new(0.04))
        .schedule(Logarithmic::new(0.1))
        .iterations(400_000)
        .seed(10_001)
        .build()?
        .run(initial);

    let w = simplex_project(&result.best_state);

    // Report key metrics
    let port_ret: f64 = w.iter().zip(asset_means_report.iter()).map(|(wi, ri)| wi * ri).sum();
    let mut sc_rets: Vec<f64> = scenarios_report
        .iter()
        .map(|day| w.iter().zip(day.iter()).map(|(wi, ri)| wi * ri).sum())
        .collect();
    sc_rets.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cutoff = ((0.05) * T as f64).ceil() as usize;
    let final_cvar = -mean(&sc_rets[..cutoff.max(1)]);
    let esg: f64 = w.iter().zip(esg_scores.iter()).map(|(wi, si)| wi * si).sum();

    println!(
        "[10] CVaR Portfolio  ret={:.2}%  CVaR={:.2}%  ESG={:.1}  accept={:.1}%",
        port_ret * 100.0,
        final_cvar * 100.0,
        esg,
        result.diagnostics.acceptance_rate() * 100.0
    );
    Ok(())
}

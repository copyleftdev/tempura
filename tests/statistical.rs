#![allow(missing_docs)]
/// Shared statistical test utilities for hypothesis testing.
///
/// Provides chi-squared goodness-of-fit, Kolmogorov-Smirnov tests,
/// and multi-seed aggregation used across all hypothesis test files.

/// Chi-squared goodness-of-fit test with bin merging.
///
/// Compares observed counts against expected probabilities.
/// Bins with expected count < `min_expected` (default 5) are merged with
/// adjacent bins — this is standard statistical practice to ensure the
/// chi-squared approximation is valid.
///
/// Returns (chi2_statistic, p_value).
pub fn chi_squared_test(observed: &[u64], expected_probs: &[f64]) -> (f64, f64) {
    use statrs::distribution::{ChiSquared, ContinuousCDF};

    assert_eq!(observed.len(), expected_probs.len());
    let n: u64 = observed.iter().sum();
    let n_f = n as f64;
    let min_expected = 5.0;

    // Merge bins with expected count < min_expected
    let mut merged_obs: Vec<f64> = Vec::new();
    let mut merged_exp: Vec<f64> = Vec::new();
    let mut acc_obs = 0.0f64;
    let mut acc_exp = 0.0f64;

    for (&obs, &prob) in observed.iter().zip(expected_probs.iter()) {
        acc_obs += obs as f64;
        acc_exp += prob * n_f;
        if acc_exp >= min_expected {
            merged_obs.push(acc_obs);
            merged_exp.push(acc_exp);
            acc_obs = 0.0;
            acc_exp = 0.0;
        }
    }
    // Merge any remaining into the last bin
    if acc_exp > 0.0 {
        if let Some(last) = merged_exp.last_mut() {
            *last += acc_exp;
            *merged_obs.last_mut().unwrap() += acc_obs;
        } else {
            // All bins merged into one — test is degenerate
            return (0.0, 1.0);
        }
    }

    if merged_obs.len() < 2 {
        return (0.0, 1.0); // need at least 2 bins for chi-squared
    }

    let chi2: f64 = merged_obs
        .iter()
        .zip(merged_exp.iter())
        .map(|(&obs, &exp)| {
            let diff = obs - exp;
            diff * diff / exp
        })
        .sum();

    let dof = (merged_obs.len() - 1) as f64;
    let dist = ChiSquared::new(dof).unwrap();
    let p_value = 1.0 - dist.cdf(chi2);

    (chi2, p_value)
}

/// Run a test function across multiple seeds and return the pass rate.
///
/// `test_fn(seed) -> bool` where true = pass.
pub fn multi_seed_pass_rate<F>(num_seeds: u64, test_fn: F) -> f64
where
    F: Fn(u64) -> bool,
{
    let passes = (0..num_seeds).filter(|&seed| test_fn(seed)).count();
    passes as f64 / num_seeds as f64
}

/// Kolmogorov-Smirnov two-sample test.
///
/// Tests whether two samples come from the same distribution.
/// Returns (ks_statistic, approximate_p_value).
pub fn ks_two_sample(a: &mut [f64], b: &mut [f64]) -> (f64, f64) {
    a.sort_by(|x, y| x.partial_cmp(y).unwrap());
    b.sort_by(|x, y| x.partial_cmp(y).unwrap());

    let na = a.len() as f64;
    let nb = b.len() as f64;

    let mut i = 0usize;
    let mut j = 0usize;
    let mut d_max = 0.0f64;

    while i < a.len() && j < b.len() {
        let fa = (i + 1) as f64 / na;
        let fb = (j + 1) as f64 / nb;
        if a[i] <= b[j] {
            d_max = d_max.max((fa - j as f64 / nb).abs());
            i += 1;
        } else {
            d_max = d_max.max((i as f64 / na - fb).abs());
            j += 1;
        }
    }
    // Handle remaining elements
    while i < a.len() {
        let fa = (i + 1) as f64 / na;
        d_max = d_max.max((fa - 1.0).abs());
        i += 1;
    }
    while j < b.len() {
        let fb = (j + 1) as f64 / nb;
        d_max = d_max.max((1.0 - fb).abs());
        j += 1;
    }

    // Approximate p-value using asymptotic formula
    let ne = (na * nb) / (na + nb);
    let lambda = (ne.sqrt() + 0.12 + 0.11 / ne.sqrt()) * d_max;
    let p_value = 2.0 * (-2.0 * lambda * lambda).exp(); // Kolmogorov distribution approx
    let p_value = p_value.clamp(0.0, 1.0);

    (d_max, p_value)
}

/// Compute integrated autocorrelation time of a time series.
///
/// Uses the windowed estimator with automatic truncation.
pub fn integrated_autocorrelation_time(series: &[f64]) -> f64 {
    let n = series.len();
    if n < 10 {
        return 1.0;
    }
    let mean: f64 = series.iter().sum::<f64>() / n as f64;
    let var: f64 = series.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    if var < 1e-15 {
        return 1.0;
    }

    let mut tau = 0.5; // start with gamma(0)/2
    let max_lag = n / 2;
    for lag in 1..max_lag {
        let gamma: f64 = (0..n - lag)
            .map(|i| (series[i] - mean) * (series[i + lag] - mean))
            .sum::<f64>()
            / n as f64;
        let rho = gamma / var;
        if rho < 0.0 {
            break; // truncate at first negative autocorrelation
        }
        tau += rho;
    }
    2.0 * tau // convention: τ_int = 2 * Σ ρ(k)
}

//! Convergence diagnostics for Markov chain Monte Carlo draws.
//!
//! The functions compute the autocorrelation of one chain, the effective sample
//! size of one chain, and the split potential scale reduction factor of several
//! chains \[1\].
//!
//! # Autocorrelation
//!
//! For a chain `x_1, ..., x_n` with mean `m`, [`autocorrelation`] at lag `k` is
//!
//! ```text
//! rho_k = sum_{t=1}^{n-k} (x_t - m)(x_{t+k} - m)  /  sum_{t=1}^{n} (x_t - m)²
//! ```
//!
//! The denominator is the lag 0 sum of squares, not scaled by `n - k`, so the
//! estimate is biased towards 0 at large lags. It is `None` when `k >= n` or the
//! chain is constant.
//!
//! # Effective sample size
//!
//! [`effective_sample_size`] is
//!
//! ```text
//! ESS = n / max(1, 1 + 2 sum_{k=1}^{K} rho_k)
//! ```
//!
//! where `K + 1` is the first lag with `rho <= 0`, or the sum runs to lag `n - 1`
//! if none is. The truncation at the first non-positive value keeps the noisy
//! tail of the autocorrelation out of the sum. The `max` keeps the result at or
//! below `n`, so a negatively correlated chain does not report more draws than it
//! has. The result is `None` for a constant chain and for a chain of fewer than
//! two draws, as [`split_rhat`] is: a sampler that never moves has not
//! converged, and a number would read as if it had.
//!
//! # Split R-hat
//!
//! [`split_rhat`] cuts each chain into a first and a second half of length
//! `n = floor(shortest / 2)`, where `shortest` is the length of the shortest chain.
//! Longer chains are truncated to `2 n` draws. Over the `m` half-chains with
//! means `mean_j` and variances `s_j²` (divisor `n - 1`), with grand mean
//! `mean`:
//!
//! ```text
//! B = n / (m - 1) * sum_j (mean_j - mean)²        (between)
//! W = (1 / m) * sum_j s_j²                         (within)
//! var_plus = (n - 1)/n * W + B / n
//! R-hat = sqrt(var_plus / W)
//! ```
//!
//! The result is at least 1. Splitting makes the statistic react to a trend within
//! a chain as well as to disagreement between chains. The means and variances use
//! Welford's update. The result is `None` for fewer than two chains, fewer than
//! four draws in the shortest chain, or a within-chain variance that is zero or
//! not finite.
//!
//! # References
//!
//! 1. Gelman A, Carlin JB, Stern HS, Dunson DB, Vehtari A, Rubin DB. Bayesian
//!    Data Analysis. 3rd ed. Boca Raton: CRC Press; 2013.

/// The lag-`lag` autocorrelation of a chain, with the lag-0 sum of squares in
/// the denominator. `None` when the lag is not shorter than the chain or the
/// chain is constant.
#[must_use]
pub fn autocorrelation(chain: &[f64], lag: usize) -> Option<f64> {
    if lag >= chain.len() {
        return None;
    }
    let (mean, squares) = centred_squares(chain)?;
    Some(lagged_products(chain, lag, mean) / squares)
}

/// Effective sample size `n / (1 + 2 sum rho_k)`, summing autocorrelations
/// from lag 1 until the first that is not positive.
///
/// `None` for a constant chain or one of fewer than two draws, where the
/// autocorrelation does not exist.
#[must_use]
pub fn effective_sample_size(chain: &[f64]) -> Option<f64> {
    let n = chain.len() as f64;
    let (mean, squares) = centred_squares(chain)?;
    let mut rho_sum = 0.0;
    for lag in 1..chain.len() {
        let rho = lagged_products(chain, lag, mean) / squares;
        if rho <= 0.0 {
            break;
        }
        rho_sum += rho;
    }
    Some(n / (1.0 + 2.0 * rho_sum).max(1.0))
}

/// Split R-hat (Gelman et al., Bayesian Data Analysis, 3rd ed., 2013): each
/// chain is cut into halves, and the potential scale reduction is computed
/// over the half-chains. Chains are truncated to the shortest even length.
///
/// `None` for fewer than two chains, fewer than four draws per chain, or
/// half-chains without variance. The result is at least 1.
#[must_use]
pub fn split_rhat(chains: &[&[f64]]) -> Option<f64> {
    let shortest = chains.iter().map(|chain| chain.len()).min()?;
    let half = shortest / 2;
    if chains.len() < 2 || half < 2 {
        return None;
    }
    let halves: Vec<(f64, f64)> = chains
        .iter()
        .flat_map(|chain| [&chain[..half], &chain[half..2 * half]])
        .map(mean_and_variance)
        .collect();
    let m = halves.len() as f64;
    let n = half as f64;
    let grand = halves.iter().map(|&(mean, _)| mean).sum::<f64>() / m;
    let between = n * halves
        .iter()
        .map(|&(mean, _)| (mean - grand) * (mean - grand))
        .sum::<f64>()
        / (m - 1.0);
    let within = halves.iter().map(|&(_, variance)| variance).sum::<f64>() / m;
    if !(within.is_finite() && within > 0.0 && between.is_finite()) {
        return None;
    }
    let pooled = (n - 1.0) / n * within + between / n;
    Some((pooled / within).sqrt().max(1.0))
}

/// Mean and `n - 1` variance by Welford's update.
fn mean_and_variance(values: &[f64]) -> (f64, f64) {
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for (count, &value) in (1_u32..).zip(values) {
        let delta = value - mean;
        mean += delta / f64::from(count);
        m2 += delta * (value - mean);
    }
    (mean, m2 / (values.len() as f64 - 1.0))
}

fn centred_squares(chain: &[f64]) -> Option<(f64, f64)> {
    let mean = chain.iter().sum::<f64>() / chain.len() as f64;
    let squares: f64 = chain.iter().map(|&v| (v - mean) * (v - mean)).sum();
    (squares > 0.0).then_some((mean, squares))
}

fn lagged_products(chain: &[f64], lag: usize, mean: f64) -> f64 {
    chain
        .iter()
        .zip(&chain[lag..])
        .map(|(&a, &b)| (a - mean) * (b - mean))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_alternating_chain_is_anticorrelated() {
        let chain = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        assert!((autocorrelation(&chain, 1).unwrap() + 5.0 / 6.0).abs() < 1e-15);
        assert_eq!(autocorrelation(&chain, 6), None);
        assert_eq!(autocorrelation(&[2.0; 4], 1), None);
        assert!((effective_sample_size(&chain).unwrap() - 6.0).abs() < 1e-15);
    }

    #[test]
    fn a_persistent_chain_has_fewer_effective_draws() {
        let chain: Vec<f64> = (0..100).map(|i| f64::from(i / 10)).collect();
        assert!(effective_sample_size(&chain).unwrap() < 20.0);
        assert_eq!(effective_sample_size(&[3.0; 5]), None);
        assert_eq!(effective_sample_size(&[3.0]), None);
        assert_eq!(effective_sample_size(&[]), None);
    }

    #[test]
    fn split_rhat_flags_chains_that_disagree() {
        let a: Vec<f64> = (0..40).map(|i| f64::from(i % 7)).collect();
        let b: Vec<f64> = (0..40).map(|i| f64::from((i + 3) % 7)).collect();
        let shifted: Vec<f64> = b.iter().map(|v| v + 10.0).collect();
        let mixed = split_rhat(&[&a, &b]).unwrap();
        assert!((1.0..1.1).contains(&mixed));
        assert!(split_rhat(&[&a, &shifted]).unwrap() > 2.0);
        assert_eq!(split_rhat(&[&a]), None);
        assert_eq!(split_rhat(&[&a[..3], &b]), None);
    }
}

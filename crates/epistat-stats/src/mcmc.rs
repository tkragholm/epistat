//! Convergence diagnostics for Markov chain Monte Carlo draws.

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
/// A constant chain or one of fewer than two draws counts as `n`
/// independent draws.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn effective_sample_size(chain: &[f64]) -> f64 {
    let n = chain.len() as f64;
    let Some((mean, squares)) = centred_squares(chain).filter(|_| chain.len() >= 2) else {
        return n;
    };
    let mut rho_sum = 0.0;
    for lag in 1..chain.len() {
        let rho = lagged_products(chain, lag, mean) / squares;
        if rho <= 0.0 {
            break;
        }
        rho_sum += rho;
    }
    n / (1.0 + 2.0 * rho_sum).max(1.0)
}

/// Split R-hat (Gelman et al., Bayesian Data Analysis, 3rd ed., 2013): each
/// chain is cut into halves, and the potential scale reduction is computed
/// over the half-chains. Chains are truncated to the shortest even length.
///
/// `None` for fewer than two chains, fewer than four draws per chain, or
/// half-chains without variance. The result is at least 1.
#[must_use]
#[allow(clippy::cast_precision_loss)]
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
#[allow(clippy::cast_precision_loss)]
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

#[allow(clippy::cast_precision_loss)]
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
        assert!((effective_sample_size(&chain) - 6.0).abs() < 1e-15);
    }

    #[test]
    fn a_persistent_chain_has_fewer_effective_draws() {
        let chain: Vec<f64> = (0..100).map(|i| f64::from(i / 10)).collect();
        assert!(effective_sample_size(&chain) < 20.0);
        assert!((effective_sample_size(&[3.0; 5]) - 5.0).abs() < 1e-15);
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

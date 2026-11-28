//! Posterior storage and summaries for MTP.

use crate::utils::{calculate_quantile, defined, usize_to_f64};

/// A single posterior draw from the MTP parameter space.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct MtpPosteriorDraw {
    /// Coefficients of the binary part.
    pub alpha: Vec<f64>,
    /// Coefficients of the marginal mean.
    pub beta: Vec<f64>,
    /// The skewness of the log-skew-normal positive part, 0 for log-normal.
    pub kappa: f64,
    /// The scale of the positive part on the log scale.
    pub omega_sq: f64,
}

/// Posterior draw collection.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct MtpPosteriorSamples {
    /// The retained draws in sampling order.
    pub draws: Vec<MtpPosteriorDraw>,
}

impl MtpPosteriorSamples {
    /// Whether there are no draws.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.draws.is_empty()
    }

    /// The number of draws.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.draws.len()
    }
}

/// Scalar posterior summary statistics.
///
/// Each value is `None` when it is undefined: there are no draws, or a draw
/// is NaN.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default)]
pub struct ParameterSummary {
    /// The posterior mean.
    pub mean: Option<f64>,
    /// The posterior standard deviation.
    pub std_dev: Option<f64>,
    /// The 2.5% posterior quantile.
    pub q025: Option<f64>,
    /// The posterior median.
    pub q50: Option<f64>,
    /// The 97.5% posterior quantile.
    pub q975: Option<f64>,
}

/// Posterior summary for the MTP parameter blocks.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct MtpPosteriorSummary {
    /// Summaries of the binary-part coefficients.
    pub alpha: Vec<ParameterSummary>,
    /// Summaries of the mean coefficients.
    pub beta: Vec<ParameterSummary>,
    /// The summary of kappa, `None` when it is fixed.
    pub kappa: Option<ParameterSummary>,
    /// The summary of omega squared.
    pub omega_sq: Option<ParameterSummary>,
    /// The number of draws summarised.
    pub draw_count: usize,
}

/// Compute posterior summaries for all stored parameter blocks.
#[must_use]
pub fn summarize_posterior(samples: &MtpPosteriorSamples) -> MtpPosteriorSummary {
    let draw_count = samples.len();
    if draw_count == 0 {
        return MtpPosteriorSummary {
            draw_count,
            ..MtpPosteriorSummary::default()
        };
    }

    let alpha_len = samples.draws.first().map_or(0, |draw| draw.alpha.len());
    let beta_len = samples.draws.first().map_or(0, |draw| draw.beta.len());

    // One scratch buffer reused across every coefficient and block, instead of
    // allocating a fresh `Vec` (twice) per coefficient.
    let mut buffer: Vec<f64> = Vec::with_capacity(draw_count);

    let mut alpha = Vec::with_capacity(alpha_len);
    for index in 0..alpha_len {
        buffer.clear();
        buffer.extend(samples.draws.iter().map(|draw| draw.alpha[index]));
        alpha.push(summarize_scalar_in_place(&mut buffer));
    }

    let mut beta = Vec::with_capacity(beta_len);
    for index in 0..beta_len {
        buffer.clear();
        buffer.extend(samples.draws.iter().map(|draw| draw.beta[index]));
        beta.push(summarize_scalar_in_place(&mut buffer));
    }

    buffer.clear();
    buffer.extend(samples.draws.iter().map(|draw| draw.kappa));
    let kappa = Some(summarize_scalar_in_place(&mut buffer));

    buffer.clear();
    buffer.extend(samples.draws.iter().map(|draw| draw.omega_sq));
    let omega_sq = Some(summarize_scalar_in_place(&mut buffer));

    MtpPosteriorSummary {
        alpha,
        beta,
        kappa,
        omega_sq,
        draw_count,
    }
}

/// Summarize a scalar parameter from a caller-owned buffer: mean and variance
/// in one pass, then sort `values` in place for the quantiles (no internal
/// `to_vec`).
#[must_use]
fn summarize_scalar_in_place(values: &mut [f64]) -> ParameterSummary {
    if values.is_empty() {
        return ParameterSummary::default();
    }

    let n = usize_to_f64(values.len());
    let mean = values.iter().sum::<f64>() / n;
    let variance = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / n.max(1.0);

    values.sort_by(f64::total_cmp);

    ParameterSummary {
        mean: defined(mean),
        std_dev: defined(variance.sqrt()),
        q025: calculate_quantile(values, 0.025),
        q50: calculate_quantile(values, 0.5),
        q975: calculate_quantile(values, 0.975),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarize_empty_samples() {
        let summary = summarize_posterior(&MtpPosteriorSamples::default());
        assert_eq!(summary.draw_count, 0);
        assert_eq!(summary.alpha.len(), 0);
        assert_eq!(summary.beta.len(), 0);
        assert!(summary.kappa.is_none());
    }

    #[test]
    fn summarize_non_empty_samples() {
        let samples = MtpPosteriorSamples {
            draws: vec![
                MtpPosteriorDraw {
                    alpha: vec![0.0],
                    beta: vec![1.0],
                    kappa: 0.1,
                    omega_sq: 0.8,
                },
                MtpPosteriorDraw {
                    alpha: vec![2.0],
                    beta: vec![3.0],
                    kappa: 0.3,
                    omega_sq: 1.2,
                },
            ],
        };

        let summary = summarize_posterior(&samples);
        assert_eq!(summary.draw_count, 2);
        assert_eq!(summary.alpha.len(), 1);
        assert!((summary.alpha[0].mean.unwrap() - 1.0).abs() < 1.0e-12);
    }
}

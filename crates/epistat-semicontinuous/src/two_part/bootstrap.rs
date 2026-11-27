//! Row-resampling bootstrap and its percentile summaries.

use epistat_core::{Bootstrap, Interval, Level};
use epistat_inference::interval::{Percentile, percentile};
use epistat_linalg::Mat;
use epistat_rng::Stream;

use super::{TwoPartOptions, fit_two_part_weighted};
use crate::Error;
use crate::data::CostData;
use crate::matrix_ops::{select_rows_ref, select_values_ref};
use crate::utils::{mean_vector, std_vector};

/// The coefficients of the replicates of [`bootstrap`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct BootstrapDraws {
    /// The logit coefficients of each replicate that converged.
    pub betas_logit: Vec<Mat<f64>>,
    /// The gamma coefficients of each replicate that converged.
    pub betas_gamma: Vec<Mat<f64>>,
    /// The replicates that did not converge and were skipped.
    pub failures: usize,
}

/// The mean, standard deviation and percentile intervals of bootstrap
/// coefficients, from [`summary`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct BootstrapSummary {
    /// The mean of each coefficient.
    pub mean: Mat<f64>,
    /// The standard deviation of each coefficient, with the `n - 1` divisor.
    pub se: Mat<f64>,
    /// The percentile interval of each coefficient.
    pub intervals: Vec<Interval>,
}

/// Bootstrap the two-part model by resampling rows with replacement.
///
/// Replicate `r` draws `n` rows with replacement from one
/// [`Stream`](epistat_rng::Stream) seeded with `draws.seed`, the replicates
/// drawn in turn, and fits them with `options` and the weights of the drawn
/// rows, without the strategy's retries. A replicate that does not converge,
/// saturates or gives values that are not finite is skipped when
/// `max_failures` is given, and the bootstrap fails once more than
/// `max_failures` were skipped; with `None` the first such replicate fails it.
///
/// # Errors
///
/// [`Error::Input`] for no replicates; [`Error::Unsupported`] for data with
/// clusters, which a row bootstrap would split; [`Error::TooManyFailures`];
/// or the error of a replicate that cannot be skipped.
pub fn bootstrap(
    data: &CostData<'_>,
    options: &TwoPartOptions,
    draws: Bootstrap,
    max_failures: Option<usize>,
) -> Result<BootstrapDraws, Error> {
    if draws.replicates == 0 {
        return Err(crate::error::domain("replicates", 0.0, "at least 1"));
    }
    if data.clusters().is_some() {
        return Err(Error::Unsupported {
            what: "clusters in a row bootstrap",
        });
    }
    let options = *options;
    let (x, y) = (data.x(), data.y_column());
    let n = data.n_rows();
    let ones = Mat::from_fn(n, 1, |_, _| 1.0);
    let weights = data.weights_column().unwrap_or(ones.as_ref());

    let mut rng = Stream::seeded(draws.seed);
    let mut betas_logit = Vec::with_capacity(draws.replicates);
    let mut betas_gamma = Vec::with_capacity(draws.replicates);
    let mut failures = 0;

    for _ in 0..draws.replicates {
        let indices = (0..n).map(|_| rng.below(n)).collect::<Vec<_>>();
        let x_sample = select_rows_ref(x, &indices);
        let y_sample = select_values_ref(y, &indices);
        let w_sample = select_values_ref(weights, &indices);
        match fit_two_part_weighted(
            x_sample.as_ref(),
            y_sample.as_ref(),
            w_sample.as_ref(),
            options,
            None,
        ) {
            Ok(model) => {
                betas_logit.push(model.beta_logit);
                betas_gamma.push(model.beta_gamma);
            }
            Err(Error::NotConverged { .. } | Error::Saturated | Error::NotFinite)
                if max_failures.is_some() =>
            {
                failures += 1;
                let allowed = max_failures.unwrap_or(0);
                if failures > allowed {
                    return Err(Error::TooManyFailures { failures, allowed });
                }
            }
            Err(err) => return Err(err),
        }
    }

    Ok(BootstrapDraws {
        betas_logit,
        betas_gamma,
        failures,
    })
}

/// Percentile intervals of bootstrap coefficients, one per coefficient.
///
/// The limits are R's type 7 quantiles
/// (`quantile(x, c(alpha / 2, 1 - alpha / 2), type = 7)`) of the draws, so they
/// can fall between two draws. A coefficient whose draws hold a NaN has
/// limits of `None`.
#[must_use]
pub fn percentile_intervals(betas: &[Mat<f64>], level: Level) -> Vec<Interval> {
    if betas.is_empty() {
        return Vec::new();
    }
    let n = betas[0].nrows();
    let mut intervals = Vec::with_capacity(n);
    let mut values: Vec<f64> = Vec::with_capacity(betas.len());
    for col in 0..n {
        values.clear();
        values.extend(betas.iter().map(|b| b[(col, 0)]));
        intervals.push(
            percentile(&mut values, level, Percentile::Type7)
                .unwrap_or_else(|_| Interval::new(None, None, level)),
        );
    }
    intervals
}

/// The mean, standard deviation and percentile intervals of bootstrap
/// coefficients; empty for no draws.
#[must_use]
pub fn summary(betas: &[Mat<f64>], level: Level) -> BootstrapSummary {
    if betas.is_empty() {
        return BootstrapSummary {
            mean: Mat::zeros(0, 1),
            se: Mat::zeros(0, 1),
            intervals: Vec::new(),
        };
    }
    let mean = mean_vector(betas);
    let se = std_vector(betas, &mean);
    let intervals = percentile_intervals(betas, level);
    BootstrapSummary {
        mean,
        se,
        intervals,
    }
}

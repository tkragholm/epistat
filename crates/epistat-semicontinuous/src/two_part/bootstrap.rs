//! Row-resampling bootstrap and its percentile summaries.

use epistat_core::{Bootstrap, Interval, Level, Matrix};
use epistat_inference::interval::{Percentile, percentile};
use epistat_linalg::Mat;
use epistat_rng::Stream;

use super::{TwoPartOptions, fit_two_part_weighted};
use crate::Error;
use crate::data::CostData;
use crate::matrix_ops::{select_rows_ref, select_values_ref};
use crate::utils::usize_to_f64;
use crate::variance::Variance;

/// The coefficients of the replicates of [`bootstrap`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct BootstrapDraws {
    /// The logit coefficients, one row per replicate that converged and one
    /// column per coefficient.
    pub logit: Matrix,
    /// The gamma coefficients, as `logit`.
    pub gamma: Matrix,
    /// The replicates that did not converge and were skipped.
    pub failures: usize,
}

/// The mean, standard deviation and percentile intervals of bootstrap
/// coefficients, from [`summary`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct BootstrapSummary {
    /// The mean of each coefficient.
    pub mean: Vec<f64>,
    /// The standard deviation of each coefficient over the draws, with the
    /// `n - 1` divisor; `None` with fewer than two draws.
    pub se: Vec<Option<f64>>,
    /// The percentile interval of each coefficient.
    pub intervals: Vec<Interval>,
}

/// Bootstrap the two-part model by resampling rows with replacement.
///
/// Replicate `r` draws `n` rows with replacement from one
/// [`Stream`](epistat_rng::Stream) seeded with `draws.seed`, the replicates
/// drawn in turn, and fits them with `options` and the weights of the drawn
/// rows, without the strategy's retries and without covariances. A replicate
/// that does not converge, saturates or gives values that are not finite is
/// skipped when `max_failures` is given, and the bootstrap fails once more
/// than `max_failures` were skipped; with `None` the first such replicate
/// fails it.
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
    let inference = Variance::None.resolve(data)?;
    let (x, y) = (data.x(), data.y_column());
    let n = data.n_rows();
    let p = data.n_columns();
    let ones = Mat::from_fn(n, 1, |_, _| 1.0);
    let weights = data.weights_column().unwrap_or(ones.as_ref());

    let mut rng = Stream::seeded(draws.seed.value());
    let mut logit = Vec::with_capacity(draws.replicates * p);
    let mut gamma = Vec::with_capacity(draws.replicates * p);
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
            None,
            options,
            inference,
            None,
        ) {
            Ok(model) => {
                logit.extend_from_slice(&model.logit.coefficients);
                gamma.extend_from_slice(&model.gamma.coefficients);
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

    let rows = logit.len() / p.max(1);
    Ok(BootstrapDraws {
        logit: Matrix::new(rows, p, logit).map_err(|_| Error::NotFinite)?,
        gamma: Matrix::new(rows, p, gamma).map_err(|_| Error::NotFinite)?,
        failures,
    })
}

/// Percentile intervals of bootstrap coefficients, one per column of
/// `draws` (one row per replicate).
///
/// The limits are R's type 7 quantiles
/// (`quantile(x, c(alpha / 2, 1 - alpha / 2), type = 7)`) of the draws, so they
/// can fall between two draws. Without draws the list is empty.
#[must_use]
pub fn percentile_intervals(draws: &Matrix, level: Level) -> Vec<Interval> {
    if draws.n_rows() == 0 {
        return Vec::new();
    }
    let mut values: Vec<f64> = Vec::with_capacity(draws.n_rows());
    draws
        .columns()
        .map(|column| {
            values.clear();
            values.extend(column);
            percentile(&mut values, level, Percentile::Type7)
                .unwrap_or_else(|_| Interval::new(None, None, level))
        })
        .collect()
}

/// The mean, standard deviation and percentile intervals of bootstrap
/// coefficients, one row of `draws` per replicate; empty for no draws.
#[must_use]
pub fn summary(draws: &Matrix, level: Level) -> BootstrapSummary {
    let r = draws.n_rows();
    if r == 0 {
        return BootstrapSummary {
            mean: Vec::new(),
            se: Vec::new(),
            intervals: Vec::new(),
        };
    }
    let count = usize_to_f64(r);
    let mut mean = Vec::with_capacity(draws.n_columns());
    let mut se = Vec::with_capacity(draws.n_columns());
    for column in draws.columns() {
        let values: Vec<f64> = column.collect();
        let mut sum = 0.0;
        for &value in &values {
            sum += value;
        }
        let m = sum / count;
        let mut squares = 0.0_f64;
        for &value in &values {
            let diff = value - m;
            squares = diff.mul_add(diff, squares);
        }
        mean.push(m);
        se.push((r > 1).then(|| (squares / (count - 1.0)).sqrt()));
    }
    let intervals = percentile_intervals(draws, level);
    BootstrapSummary {
        mean,
        se,
        intervals,
    }
}


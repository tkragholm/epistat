//! Which covariance a cost model reports, and the pieces every model shares
//! to form it.

use epistat_core::{Covariance, VarianceKind, WeightKind};
use epistat_linalg::{Mat, MatRef};

use crate::Error;
use crate::data::CostData;

/// The covariance a fit of [`two_part`](crate::two_part),
/// [`tweedie`](crate::tweedie) or [`lognormal`](crate::lognormal) computes.
///
/// The model-based covariance is the inverse information scaled by the
/// estimated dispersion, as R's `summary.glm` and `summary.lm` report it. The
/// sandwich is HC0 without clusters and CR0, the clusters' score sums
/// without a small-sample factor, with them, as `sandwich::vcovCL(type =
/// "HC0", cadjust = FALSE)`. Clustered rows or sampling weights make the
/// model-based covariance wrong, so [`Variance::Auto`] chooses the sandwich
/// for them; a caller who wants the model-based one on clustered data says so
/// with [`Variance::ModelBased`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Variance {
    /// The sandwich when the data carry clusters or sampling weights, the
    /// model-based covariance otherwise.
    #[default]
    Auto,
    /// The model-based covariance, refused with sampling weights.
    ModelBased,
    /// The sandwich, clustered when the data carry clusters.
    Sandwich,
    /// No covariance and no standard errors, as for a bootstrap replicate
    /// whose coefficients are all that is read.
    None,
}

impl Variance {
    /// The covariance this choice gives for `data`, `None` for none.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for [`Variance::ModelBased`] with sampling
    /// weights.
    pub(crate) fn resolve(self, data: &CostData<'_>) -> Result<Inference, Error> {
        let sampling = data
            .weights()
            .is_some_and(|w| w.kind() == WeightKind::Sampling);
        let frequency = data
            .weights()
            .is_some_and(|w| w.kind() == WeightKind::Frequency);
        let variance = match self {
            Self::Auto if sampling || data.clusters().is_some() => Some(VarianceKind::Robust),
            Self::Auto => Some(VarianceKind::ModelBased),
            Self::ModelBased if sampling => {
                return Err(Error::Unsupported {
                    what: "a model-based covariance with sampling weights",
                });
            }
            Self::ModelBased => Some(VarianceKind::ModelBased),
            Self::Sandwich => Some(VarianceKind::Robust),
            Self::None => None,
        };
        Ok(Inference {
            variance,
            frequency,
        })
    }
}

/// The resolved choice of a fit: the covariance to compute and how the rows
/// count in the residual degrees of freedom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Inference {
    /// The covariance, `None` for none.
    pub variance: Option<VarianceKind>,
    /// Whether the weights are frequency weights, which count `sum(w)` rows.
    pub frequency: bool,
}

impl Inference {
    /// Whether the covariance is the sandwich.
    pub(crate) fn robust(self) -> bool {
        self.variance == Some(VarianceKind::Robust)
    }
}

/// The rows that enter a fit, those with a positive weight, and the residual
/// degrees of freedom `n_used - p`, or `sum(w) - p` for frequency weights;
/// `None` for the degrees of freedom when they are not positive.
pub(crate) fn residual_df(
    weights: MatRef<'_, f64>,
    frequency: bool,
    p: usize,
) -> (usize, Option<f64>) {
    let used = (0..weights.nrows())
        .filter(|&i| weights[(i, 0)] > 0.0)
        .count();
    let rows = if frequency {
        (0..weights.nrows()).map(|i| weights[(i, 0)]).sum::<f64>()
    } else {
        crate::utils::usize_to_f64(used)
    };
    let df = rows - crate::utils::usize_to_f64(p);
    (used, (df > 0.0).then_some(df))
}

/// The Pearson dispersion `sum_i w_i r_i^2 / df` for the Pearson residuals
/// `r_i`, R's `summary.glm` estimate; `None` without degrees of freedom.
pub(crate) fn pearson_dispersion(
    weights: MatRef<'_, f64>,
    residual: impl Fn(usize) -> f64,
    df: Option<f64>,
) -> Option<f64> {
    let df = df?;
    let mut sum = 0.0;
    for i in 0..weights.nrows() {
        let w = weights[(i, 0)];
        if w > 0.0 {
            let r = residual(i);
            sum += w * r * r;
        }
    }
    let dispersion = sum / df;
    dispersion.is_finite().then_some(dispersion)
}

/// `scale` times the faer matrix `m`, as a [`Covariance`] with a non-finite
/// entry undefined.
pub(crate) fn to_covariance(m: &Mat<f64>, scale: f64) -> Covariance {
    let dim = m.nrows();
    let values = (0..dim * dim)
        .map(|k| {
            let value = scale * m[(k / dim, k % dim)];
            value.is_finite().then_some(value)
        })
        .collect();
    Covariance::new(dim, values).expect("a square matrix with finite entries")
}

/// The standard errors of `covariance`, `None` for an undefined or negative
/// variance, and all `None` for `p` coefficients without a covariance.
pub(crate) fn standard_errors(covariance: Option<&Covariance>, p: usize) -> Vec<Option<f64>> {
    covariance.map_or_else(|| vec![None; p], Covariance::standard_errors)
}

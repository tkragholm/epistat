//! Sandwich covariances of a fitted generalised linear model, and risk ratios
//! and risk differences of a binary outcome by modified Poisson regression.
//!
//! # Sandwich covariance
//!
//! For a model fitted by [`Glm::fit`] with the working weights `W_i = w_i
//! μ'(η_i)² / V(μ_i)` of its last iteration and the working residuals
//! `e_i = (y_i - μ_i) / μ'(η_i)`, the score of row `i` is `s_i = W_i e_i x_i`
//! \[1, 2\]. The bread is `A^-1 = (X'WX)^-1`, which the fit stores as its
//! covariance divided by its dispersion, and the meat `M` sums the scores
//! row by row or within clusters first:
//!
//! ```text
//! M = Σ_i r_i² x_i x_i'              r_i = W_i e_i
//! M = Σ_g s_g s_g'                   s_g = Σ_{i in g} r_i x_i
//! V = c A^-1 M A^-1
//! ```
//!
//! This is `sandwich::estfun.glm`, whose scores also use the weights of the
//! last iteration and the residuals at the final means, and `bread.glm`.
//! The factor `c` follows `sandwich::vcovHC` without clusters and
//! `sandwich::vcovCL(cadjust = TRUE)` with them, with `N` the number of rows,
//! `K` the rank of the model and `G` the number of clusters:
//!
//! ```text
//! type   no clusters                    clusters
//! Hc0    1                              1
//! Hc1    N / (N - K)                    G/(G-1) (N-1)/(N-K)
//! Hc3    r_i replaced by r_i/(1-h_i)    not defined
//! ```
//!
//! where `h_i = W_i x_i' A^-1 x_i` is the leverage. `Hc0` with clusters has no
//! small-sample factor. The covariance is `None` throughout when `N <= K` or
//! when there is one cluster, where the factor does not exist, and `None` for
//! an aliased coefficient. A dispersion in the fit is divided out, so the
//! sandwich is the same for a family with an estimated dispersion.
//!
//! # Modified Poisson regression
//!
//! [`risk_regression`] fits the Poisson family with a log link to a 0/1
//! outcome \[3\]. The coefficients are log risk ratios when the log-linear mean
//! `P(y = 1 | x) = exp(x'β)` holds. The Poisson variance is wrong for a
//! binary outcome, so the standard errors are those of the sandwich. The
//! estimate is consistent without the constraint `x'β <= 0`, but a fitted
//! probability above 1 shows that the model does not hold there, and
//! [`RiskFit::n_fitted_above_one`] counts such rows.
//!
//! For risk differences the model is binomial with an identity link. It is
//! fitted by the iteratively reweighted least squares of [`Glm`], from the
//! family's starting means and, if the first step leaves the unit interval,
//! from the least-squares coefficients. It fails with a [`GlmError`] when that
//! does not give fitted probabilities inside `(0, 1)`, which is the case when
//! the maximum lies on the boundary. The same sandwich is used.
//!
//! # References
//!
//! 1. Huber PJ. The behavior of maximum likelihood estimates under nonstandard
//!    conditions. In: Proceedings of the Fifth Berkeley Symposium on
//!    Mathematical Statistics and Probability, vol 1. Berkeley: University of
//!    California Press; 1967. p. 221-33.
//! 2. Zeileis A, Köll S, Graham N. Various versatile variances: an
//!    object-oriented implementation of clustered covariances in R. J Stat
//!    Softw. 2020;95(1):1-36.
//! 3. Zou G. A modified Poisson regression approach to prospective studies
//!    with binary data. Am J Epidemiol. 2004;159(7):702-6.
//! 4. Wacholder S. Binomial regression in GLIM: estimating risk ratios and
//!    risk differences. Am J Epidemiol. 1986;123(1):174-84.

use epistat_inference::Clusters;
use epistat_inference::sandwich::{meat, sandwich_with_inverse};
use epistat_linalg::design::select_columns;
use epistat_linalg::{Mat, MatRef};

use crate::{Family, Glm, GlmError, Link, fit::Fit};

/// The small-sample form of the sandwich.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SandwichType {
    /// No correction.
    Hc0,
    /// `N/(N-K)`, and with clusters `G/(G-1) (N-1)/(N-K)`.
    Hc1,
    /// Each score divided by `1 - h_i`. Not defined with clusters.
    Hc3,
}

/// The inverse of `X'WX` over the columns that are not aliased, from the
/// covariance of a fit and its dispersion.
fn bread(fit: &Fit, columns: &[usize]) -> Result<Mat<f64>, GlmError> {
    let p = fit.coefficients.len();
    let dispersion = fit
        .dispersion
        .filter(|&d| d > 0.0)
        .ok_or(GlmError::Missing("positive dispersion"))?;
    Ok(Mat::from_fn(columns.len(), columns.len(), |a, b| {
        fit.covariance[columns[a] * p + columns[b]].map_or(f64::NAN, |v| v / dispersion)
    }))
}

/// The columns that are not aliased.
pub(crate) fn kept_columns(fit: &Fit) -> Vec<usize> {
    (0..fit.coefficients.len())
        .filter(|&j| fit.coefficients[j].is_some())
        .collect()
}

/// The sandwich `c A^-1 M A^-1` from the pieces: the design without aliased
/// columns, the inverse of its information, the row scores `r_i`, the
/// leverages (for `Hc3`) and `N`, `K`.
fn assemble(
    reduced: MatRef<'_, f64>,
    inverse: MatRef<'_, f64>,
    scores: &[f64],
    leverage: Option<&[f64]>,
    clusters: Option<&Clusters>,
    kind: SandwichType,
    rows: usize,
    rank: usize,
) -> Result<Option<Mat<f64>>, GlmError> {
    let factor = match (kind, clusters) {
        (SandwichType::Hc0 | SandwichType::Hc3, _) => 1.0,
        (SandwichType::Hc1, None) => {
            if rows <= rank {
                return Ok(None);
            }
            rows as f64 / (rows - rank) as f64
        }
        (SandwichType::Hc1, Some(clusters)) => {
            let g = clusters.count();
            if g < 2 || rows <= rank {
                return Ok(None);
            }
            g as f64 / (g - 1) as f64 * (rows - 1) as f64 / (rows - rank) as f64
        }
    };
    let meat = if kind == SandwichType::Hc3 {
        let leverage = leverage.ok_or(GlmError::Missing("leverage"))?;
        meat(reduced, |i| scores[i] / (1.0 - leverage[i]), clusters)?
    } else {
        meat(reduced, |i| scores[i], clusters)?
    };
    let v = sandwich_with_inverse(inverse, meat.as_ref());
    Ok(Some(Mat::from_fn(v.nrows(), v.ncols(), |a, b| {
        factor * v[(a, b)]
    })))
}

/// The sandwich covariance of the coefficients of `fit`, row-major `p x p`,
/// `None` in the rows and columns of aliased coefficients and everywhere when
/// the small-sample factor does not exist.
///
/// `model` is the [`Glm`] that produced `fit`, `x` and `y` its data, and
/// `clusters` the cluster of each row, which sums the scores within a cluster
/// first. See the module for the formulas.
///
/// # Errors
///
/// [`GlmError::Length`] for inputs of the wrong length,
/// [`GlmError::Missing`] when the fit has no positive dispersion,
/// [`GlmError::Clusters`] for `Hc3` with clusters, and
/// [`GlmError::Inference`] when a cluster sum or the sandwich cannot be
/// formed.
pub fn robust_covariance(
    model: &Glm,
    fit: &Fit,
    x: MatRef<'_, f64>,
    y: &[f64],
    clusters: Option<&Clusters>,
    kind: SandwichType,
) -> Result<Vec<Option<f64>>, GlmError> {
    let (n, p) = (x.nrows(), x.ncols());
    for (what, expected, found) in [
        ("y", n, y.len()),
        ("fitted", n, fit.fitted.len()),
        ("coefficients", p, fit.coefficients.len()),
    ] {
        if found != expected {
            return Err(GlmError::Length {
                what,
                expected,
                found,
            });
        }
    }
    if kind == SandwichType::Hc3 && clusters.is_some() {
        return Err(GlmError::Clusters("HC3 is defined for unclustered rows"));
    }
    let columns = kept_columns(fit);
    let keep: Vec<bool> = (0..p).map(|j| fit.coefficients[j].is_some()).collect();
    let reduced = select_columns(x, &keep);
    let inverse = bread(fit, &columns)?;
    let scores: Vec<f64> = (0..n)
        .map(|i| {
            let w = fit.working_weights[i];
            if w > 0.0 {
                w * (y[i] - fit.fitted[i]) / model.link.derivative(fit.linear_predictor[i])
            } else {
                0.0
            }
        })
        .collect();
    let leverage: Option<Vec<f64>> = (kind == SandwichType::Hc3).then(|| {
        (0..n)
            .map(|i| {
                let mut quad = 0.0;
                for a in 0..columns.len() {
                    for b in 0..columns.len() {
                        quad += reduced[(i, a)] * inverse[(a, b)] * reduced[(i, b)];
                    }
                }
                fit.working_weights[i] * quad
            })
            .collect()
    });
    let rank = fit.rank;
    let Some(v) = assemble(
        reduced.as_ref(),
        inverse.as_ref(),
        &scores,
        leverage.as_deref(),
        clusters,
        kind,
        n,
        rank,
    )?
    else {
        return Ok(vec![None; p * p]);
    };
    let mut out = vec![None; p * p];
    for (a, &i) in columns.iter().enumerate() {
        for (b, &j) in columns.iter().enumerate() {
            out[i * p + j] = Some(v[(a, b)]);
        }
    }
    Ok(out)
}

/// The scale on which a risk is compared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RiskScale {
    /// Risk ratios: the Poisson family with a log link.
    Ratio,
    /// Risk differences: the binomial family with an identity link.
    Difference,
}

/// A risk regression and its sandwich covariance.
#[derive(Clone, Debug)]
pub struct RiskFit {
    /// The fit. Its own covariance is the model-based one, which is not valid
    /// for a binary outcome under the Poisson family.
    pub fit: Fit,
    /// The sandwich covariance, row-major `p x p`.
    pub robust_covariance: Vec<Option<f64>>,
    /// The number of rows with a fitted value above 1.
    pub n_fitted_above_one: usize,
    /// The largest fitted value.
    pub max_fitted: f64,
}

/// Risk ratios or risk differences of a binary outcome `y` in `[0, 1]` on the
/// columns of `x`, with the sandwich covariance of `kind`.
///
/// `weights` are prior weights and `offset` is added to the linear predictor.
/// `clusters` sums the scores within a cluster, for rows that are siblings or
/// repeated measures. See the module for the model and its assumptions.
///
/// # Errors
///
/// [`GlmError::Response`] for a response outside `[0, 1]`, and the errors of
/// [`Glm::fit`] and [`robust_covariance`].
pub fn risk_regression(
    scale: RiskScale,
    x: MatRef<'_, f64>,
    y: &[f64],
    weights: Option<&[f64]>,
    offset: Option<&[f64]>,
    clusters: Option<&Clusters>,
    kind: SandwichType,
) -> Result<RiskFit, GlmError> {
    if let Some((index, &value)) = y
        .iter()
        .enumerate()
        .find(|&(_, &v)| !(0.0..=1.0).contains(&v))
    {
        return Err(GlmError::Response { index, value });
    }
    let (model, fit) = match scale {
        RiskScale::Ratio => {
            let model = Glm::new(Family::Poisson, Link::Log);
            let fit = model.fit(x, y, weights, offset, None)?;
            (model, fit)
        }
        RiskScale::Difference => {
            let model = Glm::new(Family::Binomial, Link::Identity);
            match model.fit(x, y, weights, offset, None) {
                Ok(fit) => (model, fit),
                Err(GlmError::NoValidStart | GlmError::StepHalving) => {
                    let ols = Glm::new(Family::Gaussian, Link::Identity)
                        .fit(x, y, weights, offset, None)?;
                    let start: Vec<f64> =
                        ols.coefficients.iter().map(|c| c.unwrap_or(0.0)).collect();
                    let fit = model.fit(x, y, weights, offset, Some(&start))?;
                    (model, fit)
                }
                Err(error) => return Err(error),
            }
        }
    };
    let robust_covariance = robust_covariance(&model, &fit, x, y, clusters, kind)?;
    let n_fitted_above_one = fit.fitted.iter().filter(|&&m| m > 1.0).count();
    let max_fitted = fit.fitted.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(RiskFit {
        fit,
        robust_covariance,
        n_fitted_above_one,
        max_fitted,
    })
}

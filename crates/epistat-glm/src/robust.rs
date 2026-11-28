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
//! Cr1    N / (N - 1)                    G/(G-1)
//! Hc3    r_i replaced by r_i/(1-h_i)    not defined
//! ```
//!
//! where `h_i = W_i x_i' A^-1 x_i` is the leverage. `Hc0` with clusters has no
//! small-sample factor, and `Cr1` is `sandwich::vcovCL(type = "HC0", cadjust =
//! TRUE)`, every row its own cluster when there are none. The factors are
//! those of [`SandwichAdjustment`]. The covariance is `None` throughout where the
//! factor does not exist (`N <= K` for `Hc1`, fewer than two clusters), and
//! `None` for an aliased coefficient. A dispersion in the fit is divided out,
//! so the sandwich is the same for a family with an estimated dispersion.
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
//! from the least-squares coefficients. It fails with an [`Error`] when that
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

use epistat_core::Covariance;
use epistat_inference::Clusters;
use epistat_inference::sandwich::{SandwichAdjustment, meat, sandwich_with_inverse};
use epistat_linalg::design::select_columns;
use epistat_linalg::{Mat, MatRef};

use crate::fit::covariance;
use crate::{Error, Family, Glm, GlmData, Link, fit::Fit};

/// The small-sample form of the sandwich.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SandwichType {
    /// No correction.
    Hc0,
    /// `N/(N-K)`, and with clusters `G/(G-1) (N-1)/(N-K)`.
    Hc1,
    /// Each score divided by `1 - h_i`. Not defined with clusters.
    Hc3,
    /// `G/(G-1)` with clusters, `N/(N-1)` without.
    Cr1,
}

impl SandwichType {
    /// The factor `c` for `rows` rows, a model of rank `rank` and `clusters`
    /// clusters (`None` without clusters), or `None` where it does not exist.
    pub(crate) fn factor(self, rows: usize, rank: usize, clusters: Option<usize>) -> Option<f64> {
        let rule = match self {
            Self::Hc0 | Self::Hc3 => SandwichAdjustment::None,
            Self::Hc1 => SandwichAdjustment::Hc1,
            Self::Cr1 => SandwichAdjustment::ClusterCount,
        };
        rule.factor(rows, rank, clusters)
    }
}

/// `c V` over the kept `columns`, placed in a `p x p` covariance with `None`
/// in the rows and columns of the others, or `None` throughout when the factor
/// `c` does not exist.
pub(crate) fn place(
    v: &Mat<f64>,
    factor: Option<f64>,
    columns: &[usize],
    p: usize,
) -> Result<Covariance, Error> {
    let mut out = vec![None; p * p];
    if let Some(factor) = factor {
        for (a, &i) in columns.iter().enumerate() {
            for (b, &j) in columns.iter().enumerate() {
                out[i * p + j] = Some(factor * v[(a, b)]);
            }
        }
    }
    covariance(p, out)
}

/// The inverse of `X'WX` over the columns that are not aliased, from the
/// covariance of a fit and its dispersion.
fn bread(fit: &Fit, columns: &[usize]) -> Result<Mat<f64>, Error> {
    let dispersion = fit
        .dispersion
        .filter(|&d| d > 0.0)
        .ok_or(Error::Undefined("a positive dispersion"))?;
    Ok(Mat::from_fn(columns.len(), columns.len(), |a, b| {
        fit.covariance
            .get(columns[a], columns[b])
            .map_or(f64::NAN, |v| v / dispersion)
    }))
}

/// The columns that are not aliased.
pub(crate) fn kept_columns(fit: &Fit) -> Vec<usize> {
    (0..fit.coefficients.len())
        .filter(|&j| fit.coefficients[j].is_some())
        .collect()
}

/// The meat `M` from the pieces: the design without aliased columns, the row
/// scores `r_i` and the leverages, for `Hc3`.
fn meat_of(
    reduced: MatRef<'_, f64>,
    scores: &[f64],
    leverage: Option<&[f64]>,
    clusters: Option<&Clusters>,
) -> Result<Mat<f64>, Error> {
    Ok(match leverage {
        Some(leverage) => meat(reduced, |i| scores[i] / (1.0 - leverage[i]), clusters)?,
        None => meat(reduced, |i| scores[i], clusters)?,
    })
}

/// The sandwich covariance of the coefficients of `fit`, `p x p`, `None` in
/// the rows and columns of aliased coefficients and everywhere when the
/// small-sample factor does not exist.
///
/// `model` is the [`Glm`] that produced `fit`, `x` and `y` its data, and
/// `clusters` the cluster of each row, which sums the scores within a cluster
/// first. See the module for the formulas.
///
/// # Errors
///
/// [`Error::Input`] for inputs of the wrong length or values that are not
/// finite, [`Error::Undefined`]
/// when the fit has no positive dispersion, [`Error::Unsupported`] for `Hc3`
/// with clusters, and [`Error::Inference`] when a cluster sum or the sandwich
/// cannot be formed.
pub fn robust_covariance(
    model: &Glm,
    fit: &Fit,
    x: MatRef<'_, f64>,
    y: &[f64],
    clusters: Option<&Clusters>,
    kind: SandwichType,
) -> Result<Covariance, Error> {
    let (n, p) = (x.nrows(), x.ncols());
    for (what, expected, found) in [
        ("y", n, y.len()),
        ("fitted", n, fit.fitted.len()),
        ("coefficients", p, fit.coefficients.len()),
    ] {
        crate::error::length(what, expected, found)?;
    }
    crate::error::finite("y", y)?;
    crate::error::finite_matrix("x", x)?;
    if kind == SandwichType::Hc3 && clusters.is_some() {
        return Err(Error::Unsupported {
            what: "HC3 with clusters",
        });
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
    let factor = kind.factor(n, fit.rank, clusters.map(Clusters::count));
    let middle = meat_of(reduced.as_ref(), &scores, leverage.as_deref(), clusters)?;
    let v = sandwich_with_inverse(inverse.as_ref(), middle.as_ref());
    place(&v, factor, &columns, p)
}

/// The scale on which a risk is compared.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RiskScale {
    /// Risk ratios: the Poisson family with a log link.
    Ratio,
    /// Risk differences: the binomial family with an identity link.
    Difference,
}

/// A risk regression and its sandwich covariance.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct RiskFit {
    /// The fit. Its own covariance is the model-based one, which is not valid
    /// for a binary outcome under the Poisson family.
    pub fit: Fit,
    /// The sandwich covariance, `p x p`.
    pub robust_covariance: Covariance,
    /// The number of rows with a fitted value above 1.
    pub n_fitted_above_one: usize,
    /// The largest fitted value.
    pub max_fitted: f64,
}

/// Risk ratios or risk differences of a binary outcome in `[0, 1]`, the
/// response of `data`, with the sandwich covariance of `kind`.
///
/// The prior weights and offset of `data` enter the fit; its starting
/// coefficients are not used. `clusters` sums the scores within a cluster, for
/// rows that are siblings or repeated measures. See the module for the model
/// and its assumptions.
///
/// # Errors
///
/// [`Error::Response`] for a response outside `[0, 1]`, and the errors of
/// [`Glm::fit`] and [`robust_covariance`].
pub fn risk_regression(
    scale: RiskScale,
    data: &GlmData<'_>,
    clusters: Option<&Clusters>,
    kind: SandwichType,
) -> Result<RiskFit, Error> {
    let (x, y) = (data.x(), data.y());
    if let Some((row, &value)) = y
        .iter()
        .enumerate()
        .find(|&(_, &v)| !(0.0..=1.0).contains(&v))
    {
        return Err(Error::Response { row, value });
    }
    let data = data.with_start(None)?;
    let (model, fit) = match scale {
        RiskScale::Ratio => {
            let model = Glm::new(Family::Poisson, Link::Log);
            let fit = model.fit(&data)?;
            (model, fit)
        }
        RiskScale::Difference => {
            let model = Glm::new(Family::Binomial, Link::Identity);
            match model.fit(&data) {
                Ok(fit) => (model, fit),
                Err(Error::NoValidStart | Error::StepHalving) => {
                    let ols = Glm::new(Family::Gaussian, Link::Identity).fit(&data)?;
                    let start: Vec<f64> =
                        ols.coefficients.iter().map(|c| c.unwrap_or(0.0)).collect();
                    let fit = model.fit(&data.with_start(Some(&start))?)?;
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

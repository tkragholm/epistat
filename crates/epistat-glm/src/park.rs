//! Park tests of the variance function of a fitted model.
//!
//! A model for a mean `μ` with variance `Var(Y | X) = κ μ^λ` has squared raw
//! residuals with `E[r²] = κ μ^λ`, so `ln E[r²] = ln κ + λ ln μ`. Both tests
//! estimate `λ` from that line: about 0 for a Gaussian variance, 1 for a
//! Poisson, 2 for a gamma and 3 for an inverse Gaussian variance \[1\].
//!
//! # Modified Park test
//!
//! [`modified_park_test`] is the test of Manning and Mullahy \[1\]. With the
//! fitted means `μ` and the raw residuals `r = y - μ`, it fits
//!
//! ```text
//! E[r²] = exp(γ + λ ln μ)
//! ```
//!
//! by a gamma GLM with a log link, by [`Glm::fit`] with R's defaults, and
//! reports `λ`. It is the fit of R's
//! `glm(r2 ~ log(mu), family = Gamma(link = "log"))`, with the same iteration
//! counts and, to 1e-11, the same coefficients. The residual is on the raw
//! scale of the outcome, not the log scale, which is the point of the
//! modification: the log of a squared residual is a biased estimate of the log
//! of its mean, and the gamma fit with a log link models the mean directly.
//!
//! Optional prior weights, which the original paper does not use, are passed on
//! as the prior weights of the gamma fit, so that the residuals of a weighted
//! model count as the model counted them. The standard error is the
//! model-based one scaled by the Pearson dispersion, as R's `summary.glm`
//! reports it. A row enters when its fitted mean is positive and finite, its
//! weight is positive and its squared residual is positive and finite. R's
//! default stopping rule is on the deviance, so the coefficients are about 1e-7
//! from the root of the likelihood equations. [`modified_park_test_with`] takes
//! the controls of the fit for a tighter rule.
//!
//! # Classic Park test
//!
//! [`classic_park_test`] is the original test \[2\]: ordinary least squares of
//! `ln r²` on `ln μ`, over the rows with a positive mean and a non-zero residual.
//! Its slope estimates `λ` only when the errors of the log-squared residuals
//! have a mean that does not depend on `μ`, which holds for a gamma variance
//! and not in general, so it is biased where the modified test is not \[1\]. The
//! standard error is the least-squares one on `n - 2` degrees of freedom.
//!
//! # References
//!
//! 1. Manning WG, Mullahy J. Estimating log models: to transform or not to
//!    transform? J Health Econ. 2001;20(4):461-94.
//!    doi:10.1016/S0167-6296(01)00086-8
//! 2. Park RE. Estimation with heteroscedastic error terms. Econometrica.
//!    1966;34(4):888.
//! 3. R Core Team. Fitting generalized linear models: `glm` and `summary.glm`.
//!    Help pages of the stats package of R.

use epistat_linalg::Mat;

use crate::{Family, Glm, GlmError, Link};

/// The result of [`modified_park_test`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParkTest {
    /// The intercept `γ`.
    pub intercept: f64,
    /// The estimated power `λ` of the mean in the variance function.
    pub power: f64,
    /// The model-based standard error of `power`, with the Pearson dispersion.
    pub standard_error: f64,
    /// The rows used.
    pub rows: usize,
    /// Whether the gamma fit converged within its iteration limit.
    pub converged: bool,
    /// The IRLS iterations of the gamma fit.
    pub iterations: usize,
}

/// The result of [`classic_park_test`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassicParkTest {
    /// The intercept of `ln r²` on `ln μ`.
    pub intercept: f64,
    /// The slope, the estimate of the power `λ`.
    pub power: f64,
    /// The least-squares standard error of the slope.
    pub standard_error: f64,
    /// The rows used.
    pub rows: usize,
}

/// The modified Park test of Manning and Mullahy, with R's controls for the
/// gamma fit. See the module documentation.
///
/// Returns `Ok(None)` when fewer than three rows qualify or `ln μ` does not vary
/// among them, which leave the slope undefined.
///
/// # Errors
///
/// [`GlmError::Length`] when the inputs differ in length, [`GlmError::Weight`]
/// for a weight that is negative or not finite, and the errors of
/// [`Glm::fit`].
pub fn modified_park_test(
    fitted: &[f64],
    outcome: &[f64],
    weights: Option<&[f64]>,
) -> Result<Option<ParkTest>, GlmError> {
    modified_park_test_with(Glm::new(Family::Gamma, Link::Log), fitted, outcome, weights)
}

/// [`modified_park_test`] with the tolerance and iteration limit of `controls`.
/// Its family and link are replaced by the gamma family and the log link.
///
/// # Errors
///
/// As [`modified_park_test`].
pub fn modified_park_test_with(
    controls: Glm,
    fitted: &[f64],
    outcome: &[f64],
    weights: Option<&[f64]>,
) -> Result<Option<ParkTest>, GlmError> {
    let n = fitted.len();
    check(n, outcome.len(), "outcome")?;
    if let Some(w) = weights {
        check(n, w.len(), "weights")?;
        if let Some((index, &value)) = w
            .iter()
            .enumerate()
            .find(|&(_, &v)| !(v.is_finite() && v >= 0.0))
        {
            return Err(GlmError::Weight { index, value });
        }
    }
    let rows = usable(fitted, outcome, weights);
    if rows.len() < 3 {
        return Ok(None);
    }
    let design = Mat::from_fn(rows.len(), 2, |i, j| {
        if j == 0 { 1.0 } else { fitted[rows[i]].ln() }
    });
    let squared: Vec<f64> = rows
        .iter()
        .map(|&i| {
            let r = outcome[i] - fitted[i];
            r * r
        })
        .collect();
    let prior: Option<Vec<f64>> = weights.map(|w| rows.iter().map(|&i| w[i]).collect());
    let model = Glm {
        family: Family::Gamma,
        link: Link::Log,
        ..controls
    };
    let fit = model.fit(design.as_ref(), &squared, prior.as_deref(), None, None)?;
    let (Some(intercept), Some(power), Some(Some(standard_error))) = (
        fit.coefficients[0],
        fit.coefficients[1],
        fit.standard_errors().get(1).copied(),
    ) else {
        return Ok(None);
    };
    Ok(Some(ParkTest {
        intercept,
        power,
        standard_error,
        rows: rows.len(),
        converged: fit.converged,
        iterations: fit.iterations,
    }))
}

/// The classic Park test: least squares of `ln r²` on `ln μ`. See the module
/// documentation.
///
/// Returns `Ok(None)` when fewer than three rows qualify or `ln μ` does not
/// vary.
///
/// # Errors
///
/// [`GlmError::Length`] when the inputs differ in length.
pub fn classic_park_test(
    fitted: &[f64],
    outcome: &[f64],
) -> Result<Option<ClassicParkTest>, GlmError> {
    check(fitted.len(), outcome.len(), "outcome")?;
    let rows = usable(fitted, outcome, None);
    if rows.len() < 3 {
        return Ok(None);
    }
    let n = rows.len() as f64;
    let x: Vec<f64> = rows.iter().map(|&i| fitted[i].ln()).collect();
    let y: Vec<f64> = rows
        .iter()
        .map(|&i| {
            let r = outcome[i] - fitted[i];
            (r * r).ln()
        })
        .collect();
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let sxx: f64 = x.iter().map(|v| (v - mean_x) * (v - mean_x)).sum();
    if sxx <= 0.0 || !sxx.is_finite() {
        return Ok(None);
    }
    let sxy: f64 = x
        .iter()
        .zip(&y)
        .map(|(a, b)| (a - mean_x) * (b - mean_y))
        .sum();
    let power = sxy / sxx;
    let intercept = mean_y - power * mean_x;
    let rss: f64 = x
        .iter()
        .zip(&y)
        .map(|(a, b)| {
            let e = b - intercept - power * a;
            e * e
        })
        .sum();
    Ok(Some(ClassicParkTest {
        intercept,
        power,
        standard_error: (rss / (n - 2.0) / sxx).sqrt(),
        rows: rows.len(),
    }))
}

/// The rows with a positive finite mean, a positive weight and a positive
/// finite squared residual.
fn usable(fitted: &[f64], outcome: &[f64], weights: Option<&[f64]>) -> Vec<usize> {
    (0..fitted.len())
        .filter(|&i| {
            let r = outcome[i] - fitted[i];
            let squared = r * r;
            fitted[i] > 0.0
                && fitted[i].is_finite()
                && weights.is_none_or(|w| w[i] > 0.0)
                && squared > 0.0
                && squared.is_finite()
        })
        .collect()
}

fn check(expected: usize, found: usize, what: &'static str) -> Result<(), GlmError> {
    if expected == found {
        Ok(())
    } else {
        Err(GlmError::Length {
            what,
            expected,
            found,
        })
    }
}

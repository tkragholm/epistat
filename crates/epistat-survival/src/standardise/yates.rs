//! Population-averaged predictions and contrasts of a fitted model, as `yates`
//! of survival.
//!
//! # The estimate
//!
//! A model gives a prediction for every covariate row. The population
//! marginal mean (PMM) of a level of a term is the mean of the predictions over
//! a population of rows in which the term is set to that level \[1, 2\]. With
//! `eta_i = (x_i - c)' beta` the linear predictor of row `i`, centred at `c`
//! (for a Cox model `c` is the mean of the covariates of the fit, since the
//! linear predictor of a Cox model is defined up to a constant), `g` the
//! inverse link and `w_i` the weights of the population,
//!
//! ```text
//! m_l = sum over i of w_i g(eta_i) / sum over i of w_i.
//! ```
//!
//! With `g` the identity this is the mean linear predictor, which is the column
//! means of the design times `beta`. With `g = exp` and a Cox model it is the
//! mean relative risk, `predict = "risk"` of `yates`; with the inverse link of a
//! generalised linear model it is the mean response. The caller builds the
//! population: the fitted rows with the term set to the level (`"data"`), all
//! combinations of the categorical adjusting variables (`"factorial"`), those
//! combinations for the categorical variables and the fitted rows for the
//! others (`"sas"`), or any other set of rows, with weights if wanted
//! ([`YatesLevel`]).
//!
//! # The variance and the tests
//!
//! The covariance of the PMMs is the delta method, `G V G'`, with `V` the
//! covariance of the coefficients and row `l` of `G` the gradient of `m_l`,
//!
//! ```text
//! sum over i of w_i g'(eta_i) (x_i - c)' / sum over i of w_i.
//! ```
//!
//! For the identity link this is the variance of `survival::yates`, exactly. There
//! the centre `c` is a constant offset of the estimate and not part of the
//! gradient, which is that of the uncentred rows, `x_i` in place of `x_i - c`:
//! only the contrasts of the means of a Cox model are identified, and they do
//! not depend on the centre. For the other links `yates` simulates coefficient
//! vectors from `N(beta, V)`, moves the centre with each vector, and takes the
//! empirical covariance of the `nsim` PMMs, so that its values carry a Monte
//! Carlo error of about `1 / sqrt(2 nsim)` relative, and the gradient above
//! has `x_i - c`. The delta method has no Monte Carlo error. It is the first
//! order of the covariance of the simulated PMMs, which differs from it by terms
//! in the curvature of `g` that grow with the uncertainty of the coefficients:
//! for a logistic model with a sparse level the two standard errors differ by
//! 15 per cent in the test data.
//!
//! A test is the Wald test of `C m = 0` with the contrast matrix `C`:
//! `chi^2 = (C m)' (C S C')^+ (C m)` with `S` the covariance of the PMMs and
//! `^+` the generalised inverse that drops the eigenvalues not above
//! `sqrt(eps)` times the largest, and `df` the number kept, as `qform` of
//! `yates`. [`YatesTest::Global`] is the test that all PMMs are equal,
//! [`YatesTest::Pairwise`] one test for every pair of levels, and
//! [`YatesTest::VersusMean`] one test of each level against the mean of the levels.
//!
//! # Where this differs from `yates`
//!
//! - The design matrices of the populations are the caller's. No formula, factor
//!   coding or model frame exists here, so `population = "data"`, `"factorial"`
//!   and `"sas"` are made by building the rows that R would build.
//! - Every coefficient must be defined. `yates` drops aliased columns and then
//!   checks which predictions are estimable; here a model with an aliased
//!   column is refused, since a population row's estimability depends on the
//!   rows of the fit, which this function does not see.
//! - The variance of a non-linear prediction is the delta method, not a
//!   simulation.
//! - `predict = "survival"`, a restricted mean of the PMM survival curve, is
//!   not provided. The standardised survival at chosen times, with its
//!   covariance across levels, is [`standardise_cox`](super::standardise_cox),
//!   and [`Standardised::tests`](super::Standardised::tests) tests its equality.
//! - `test = "trend"` is not provided: in `survival` 3.8 it refers to a
//!   variable that is not defined and fails.
//! - `method = "sgtt"` is not provided.
//!
//! # References
//!
//! 1. Yates F. The analysis of multiple classifications with unequal numbers in
//!    the different classes. J Am Stat Assoc 1934;29(185):51-66.
//!    doi:10.1080/01621459.1934.10502686
//! 2. Searle SR, Speed FM, Milliken GA. Population marginal means in the linear
//!    model: an alternative to least squares means. Am Stat 1980;34(4):216-21.
//!    doi:10.1080/00031305.1980.10483031
//! 3. Therneau TM. Population contrasts. Vignette of the survival package,
//!    version 3.8.

use epistat_core::{Covariance, InputError, check};
use epistat_linalg::eigen::pinv_quadratic_form;
use epistat_stats::dist::chi_square::sf as chi_square_sf;
use epistat_stats::dist::normal;

use super::{Counterfactual, ordered_reduce};
use crate::covariance;
use crate::error::{Error, Result};

/// The inverse of a link function: how the linear predictor becomes the
/// quantity averaged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum InverseLink {
    /// The linear predictor itself, `predict = "linear"` or `"link"`.
    #[default]
    Identity,
    /// `exp(eta)`: the risk of a Cox model, the mean of a log link.
    Exp,
    /// `1 / (1 + exp(-eta))`, the logit link.
    Logistic,
    /// The standard normal distribution function, the probit link.
    Probit,
    /// `1 - exp(-exp(eta))`, the complementary log-log link.
    CLogLog,
    /// `1 / eta`, the inverse link.
    Inverse,
    /// `eta^2`, the square-root link.
    Square,
    /// `1 / sqrt(eta)`, the `1/mu^2` link.
    InverseSquareRoot,
}

impl InverseLink {
    /// The mean and its derivative with respect to the linear predictor.
    fn apply(self, eta: f64) -> (f64, f64) {
        match self {
            Self::Identity => (eta, 1.0),
            Self::Exp => {
                let e = eta.exp();
                (e, e)
            }
            Self::Logistic => {
                let m = 1.0 / (1.0 + (-eta).exp());
                (m, m * (1.0 - m))
            }
            Self::Probit => (
                normal::cdf(eta),
                (-0.5 * eta * eta).exp() / (2.0 * std::f64::consts::PI).sqrt(),
            ),
            Self::CLogLog => {
                let e = eta.exp();
                (-(-e).exp_m1(), e * (-e).exp())
            }
            Self::Inverse => (1.0 / eta, -1.0 / (eta * eta)),
            Self::Square => (eta * eta, 2.0 * eta),
            Self::InverseSquareRoot => {
                let r = 1.0 / eta.sqrt();
                (r, -0.5 * r / eta)
            }
        }
    }
}

/// The rows of the population for one level of the term.
#[derive(Debug, Clone, Copy)]
pub struct YatesLevel<'a> {
    /// The rows: the design matrix itself, or a base design with columns set to
    /// the level.
    pub population: Counterfactual<'a>,
    /// The base design of [`Counterfactual::Columns`], with one row per row
    /// of the population. Ignored for [`Counterfactual::Design`].
    pub base: Option<epistat_linalg::MatRef<'a, f64>>,
    /// A weight for each row, for example the case weights of the fit. Equal
    /// weights when `None`.
    pub weights: Option<&'a [f64]>,
}

/// Which tests between the levels to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum YatesTest {
    /// All PMMs are equal.
    #[default]
    Global,
    /// One test for each pair of levels.
    Pairwise,
    /// One test for each level against the mean of the levels.
    VersusMean,
}

/// `YatesTest`, under its name in 0.3.
#[deprecated(since = "0.4.0", note = "renamed `YatesTest`")]
pub type Test = YatesTest;

/// The options of [`yates`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct YatesOptions {
    /// What the predictions are. `Identity`, `predict = "linear"`.
    pub link: InverseLink,
    /// The covariate vector subtracted from every row before the linear
    /// predictor is formed: the means of the covariates of a Cox fit, as
    /// `coxph` centres at, [`cox::reference_means`](crate::cox::reference_means)
    /// with `Centring::Coxph`, or `None` for a model with an intercept.
    pub centre: Option<Vec<f64>>,
    /// The tests. `Global`, as `yates`.
    pub test: YatesTest,
}

/// One Wald test of a contrast of the PMMs.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct WaldTest {
    /// `global`, `i vs j` or `i vs mean`, with levels counted from 1.
    pub label: String,
    /// The chi-square statistic.
    pub chi_square: f64,
    /// The degrees of freedom: the rank of the covariance of the contrasts.
    pub df: usize,
    /// The p-value, `None` with no degrees of freedom.
    pub p_value: Option<f64>,
}

/// The result of [`yates`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Yates {
    /// The population marginal mean of each level.
    pub estimate: Vec<f64>,
    /// Its standard error.
    pub se: Vec<f64>,
    /// The covariance of the means, `n_levels` by `n_levels`.
    pub covariance: Covariance,
    /// The tests.
    pub tests: Vec<WaldTest>,
}

/// Population marginal means of a model and the tests between them.
///
/// `coefficients` and `covariance` (row-major, `p` by `p`) are those of the
/// fit, `levels` the population for each level of the term.
///
/// # Errors
///
/// An input error for no levels, a covariance that is not `p` by `p`, finite
/// and symmetric, a coefficient that is not finite (an aliased column included),
/// a level whose design has another number of columns than `p`, a `centre` or
/// weights of the wrong length or not finite, weights that are negative or
/// sum to zero, a design value that is not finite, and a linear predictor at
/// which the inverse link is not finite.
pub fn yates(
    coefficients: &[f64],
    covariance: &[f64],
    levels: &[YatesLevel<'_>],
    opts: &YatesOptions,
) -> Result<Yates> {
    let p = coefficients.len();
    if levels.is_empty() {
        return Err(InputError::Empty { what: "levels" }.into());
    }
    if let Some(row) = coefficients.iter().position(|b| !b.is_finite()) {
        return Err(InputError::NotFinite {
            what: "coefficients",
            row,
            column: None,
        }
        .into());
    }
    if covariance.len() != p * p {
        return Err(InputError::Length {
            what: "covariance",
            expected: p * p,
            found: covariance.len(),
        }
        .into());
    }
    if let Some(at) = covariance.iter().position(|v| !v.is_finite()) {
        return Err(InputError::NotFinite {
            what: "covariance",
            row: at / p,
            column: Some(at % p),
        }
        .into());
    }
    for a in 0..p {
        for b in 0..a {
            let scale = covariance[a * p + a].abs().max(covariance[b * p + b].abs());
            if (covariance[a * p + b] - covariance[b * p + a]).abs()
                > f64::EPSILON.sqrt() * scale.max(f64::MIN_POSITIVE)
            {
                return Err(Error::value("covariance: must be symmetric"));
            }
        }
    }
    if let Some(c) = &opts.centre {
        if c.len() != p {
            return Err(InputError::Length {
                what: "centre",
                expected: p,
                found: c.len(),
            }
            .into());
        }
        if let Some(row) = c.iter().position(|v| !v.is_finite()) {
            return Err(InputError::NotFinite {
                what: "centre",
                row,
                column: None,
            }
            .into());
        }
    }
    let n_levels = levels.len();
    let mut estimate = Vec::with_capacity(n_levels);
    let mut gradient = vec![0.0; n_levels * p];
    for (l, level) in levels.iter().enumerate() {
        let (m, grad) = one_level(l, level, coefficients, opts)?;
        estimate.push(m);
        gradient[l * p..(l + 1) * p].copy_from_slice(&grad);
    }
    // G V G'
    let mut cov = vec![0.0; n_levels * n_levels];
    for a in 0..n_levels {
        for b in 0..n_levels {
            let mut total = 0.0;
            for i in 0..p {
                let mut inner = 0.0;
                for j in 0..p {
                    inner += covariance[i * p + j] * gradient[b * p + j];
                }
                total += gradient[a * p + i] * inner;
            }
            cov[a * n_levels + b] = total;
        }
    }
    let std_err = (0..n_levels)
        .map(|l| cov[l * n_levels + l].max(0.0).sqrt())
        .collect();
    let tests = wald_tests(&estimate, &cov, opts.test)?;
    Ok(Yates {
        estimate,
        se: std_err,
        covariance: covariance::from_values(n_levels, &cov)?,
        tests,
    })
}

/// The mean and its gradient for one level.
fn one_level(
    l: usize,
    level: &YatesLevel<'_>,
    beta: &[f64],
    opts: &YatesOptions,
) -> Result<(f64, Vec<f64>)> {
    let p = beta.len();
    let (rows, over, design): (
        usize,
        Option<Vec<Option<f64>>>,
        epistat_linalg::MatRef<'_, f64>,
    ) = match (&level.population, level.base) {
        (Counterfactual::Design(d), _) => (d.nrows(), None, *d),
        (Counterfactual::Columns(set), Some(base)) => {
            let mut over = vec![None; p];
            for &(col, value) in *set {
                let message = || {
                    format!(
                        "level {l}: column {col} is outside the {p} columns or its value is not finite"
                    )
                };
                if col >= p {
                    return Err(Error::out_of_range(message()));
                }
                if !value.is_finite() {
                    return Err(Error::value(message()));
                }
                over[col] = Some(value);
            }
            (base.nrows(), Some(over), base)
        }
        (Counterfactual::Columns(_), None) => {
            return Err(Error::missing(format!(
                "level {l}: a population of set columns needs its base design"
            )));
        }
    };
    if design.ncols() != p {
        return Err(Error::mismatch(format!(
            "level {l}: the design has {} columns, the model {p}",
            design.ncols()
        )));
    }
    if rows == 0 {
        return Err(Error::no_information(format!(
            "level {l}: the population is empty"
        )));
    }
    let total_weight = match level.weights {
        Some(w) => {
            let message =
                || format!("level {l}: weights must be one non-negative finite value per row");
            if w.len() != rows {
                return Err(Error::mismatch(message()));
            }
            if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(Error::value(message()));
            }
            w.iter().sum::<f64>()
        }
        None => rows as f64,
    };
    if total_weight <= 0.0 {
        return Err(Error::no_information(format!(
            "level {l}: the weights sum to zero"
        )));
    }
    let centre = opts.centre.as_deref();
    let link = opts.link;
    let sums = ordered_reduce(
        rows,
        |range| {
            // sum of w g(eta), then sum of w g'(eta) (x - c), p values.
            let mut acc = vec![0.0; 1 + p];
            let mut x = vec![0.0; p];
            let mut raw = vec![0.0; p];
            for i in range {
                for (a, xa) in x.iter_mut().enumerate() {
                    let v = match &over {
                        Some(over) => over[a].unwrap_or_else(|| design[(i, a)]),
                        None => design[(i, a)],
                    };
                    if !v.is_finite() {
                        return Err(Error::value(format!(
                            "level {l}: a design value is not finite"
                        )));
                    }
                    raw[a] = v;
                    *xa = v - centre.map_or(0.0, |c| c[a]);
                }
                let mut eta = 0.0;
                for (xa, b) in x.iter().zip(beta) {
                    eta += xa * b;
                }
                let (m, d) = link.apply(eta);
                if !m.is_finite() || !d.is_finite() {
                    return Err(Error::undefined(
                        "inverse link",
                        format!(
                            "level {l}: the inverse link is not finite at the linear predictor {eta}"
                        ),
                    ));
                }
                let w = level.weights.map_or(1.0, |w| w[i]);
                acc[0] += w * m;
                // For the identity link the centre is a constant offset, as `yates`
                // has it, and the gradient is that of the uncentred row.
                let g_row = if link == InverseLink::Identity {
                    &raw
                } else {
                    &x
                };
                for (a, xa) in g_row.iter().enumerate() {
                    acc[1 + a] += w * d * xa;
                }
            }
            Ok(acc)
        },
        |a, b| {
            for (x, y) in a.iter_mut().zip(&b) {
                *x += *y;
            }
        },
    )?;
    let mean = sums[0] / total_weight;
    let grad = sums[1..].iter().map(|v| v / total_weight).collect();
    Ok((mean, grad))
}

/// The Wald tests of the equality of estimates with covariance `cov`.
///
/// # Errors
///
/// An input error for fewer than two estimates (a pairwise or mean test needs
/// two), a covariance of the wrong size and a value that is not finite.
pub fn wald_tests(estimate: &[f64], cov: &[f64], kind: YatesTest) -> Result<Vec<WaldTest>> {
    let n = estimate.len();
    check::non_empty("estimate", n)?;
    if n < 2 {
        return Err(Error::no_information(
            "tests: at least two levels are needed",
        ));
    }
    if cov.len() != n * n {
        return Err(InputError::Length {
            what: "cov",
            expected: n * n,
            found: cov.len(),
        }
        .into());
    }
    if let Some(at) = cov.iter().position(|v| !v.is_finite()) {
        return Err(InputError::NotFinite {
            what: "cov",
            row: at / n,
            column: Some(at % n),
        }
        .into());
    }
    if let Some(row) = estimate.iter().position(|v| !v.is_finite()) {
        return Err(InputError::NotFinite {
            what: "estimate",
            row,
            column: None,
        }
        .into());
    }
    let mut contrasts: Vec<(String, Vec<f64>)> = Vec::new();
    match kind {
        YatesTest::Global => {
            let rows: Vec<Vec<f64>> = (0..n - 1)
                .map(|i| {
                    let mut row = vec![0.0; n];
                    row[i] = 1.0;
                    row[n - 1] = -1.0;
                    row
                })
                .collect();
            return Ok(vec![wald("global".to_string(), &rows, estimate, cov)]);
        }
        YatesTest::Pairwise => {
            for i in 0..n - 1 {
                for j in i + 1..n {
                    let mut row = vec![0.0; n];
                    row[i] = 1.0;
                    row[j] = -1.0;
                    contrasts.push((format!("{} vs {}", i + 1, j + 1), row));
                }
            }
            if contrasts.len() == 1 {
                contrasts[0].0 = "global".to_string();
            }
        }
        YatesTest::VersusMean => {
            let k = n as f64;
            for i in 0..n {
                let mut row = vec![-1.0 / k; n];
                row[i] = (k - 1.0) / k;
                contrasts.push((format!("{} vs mean", i + 1), row));
            }
        }
    }
    Ok(contrasts
        .into_iter()
        .map(|(label, row)| wald(label, &[row], estimate, cov))
        .collect())
}

/// `chi^2 = (C m)' (C S C')^+ (C m)`.
fn wald(label: String, c: &[Vec<f64>], estimate: &[f64], cov: &[f64]) -> WaldTest {
    let n = estimate.len();
    let k = c.len();
    let cm: Vec<f64> = c
        .iter()
        .map(|row| row.iter().zip(estimate).map(|(a, b)| a * b).sum())
        .collect();
    let mut csc = vec![0.0; k * k];
    for a in 0..k {
        for b in 0..k {
            let mut total = 0.0;
            for i in 0..n {
                let mut inner = 0.0;
                for j in 0..n {
                    inner += cov[i * n + j] * c[b][j];
                }
                total += c[a][i] * inner;
            }
            csc[a * k + b] = total;
        }
    }
    let (chi_square, df) = quadratic_form(&csc, &cm);
    WaldTest {
        label,
        chi_square,
        df,
        p_value: (df > 0).then(|| chi_square_sf(chi_square, df as f64)),
    }
}

/// `b' M^+ b` for a symmetric `M` of order `b.len()`, row-major, with the
/// eigenvalues not above `sqrt(eps)` times the largest dropped, and the
/// number kept. A zero matrix gives 0 and 0.
fn quadratic_form(m: &[f64], b: &[f64]) -> (f64, usize) {
    pinv_quadratic_form(m, b, |top| top * f64::EPSILON.sqrt()).unwrap_or((0.0, 0))
}

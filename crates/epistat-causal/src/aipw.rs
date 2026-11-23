//! Augmented inverse probability weighting for binary and continuous outcomes.
//!
//! # Algorithm
//!
//! With treatment `A`, outcome `Y`, the outcome regressions `mu_a(x) = E[Y | A =
//! a, X = x]` and the propensity score `e(x) = P(A = 1 | X = x)`, the augmented
//! estimator of the mean outcome under treatment `a` in the population of the
//! estimand is a ratio of weighted sums, `theta_a = sum w_i U_ai / sum w_i D_i`,
//! with case weights `w_i` and
//!
//! ```text
//!        D_i   U_1i                                U_0i
//! ATE    1     mu1 + A (Y - mu1) / e               mu0 + (1 - A)(Y - mu0) / (1 - e)
//! ATT    A     A Y                                 A mu0 + (1 - A) e / (1 - e) (Y - mu0)
//! ATC    1-A   (1 - A) mu1 + A (1 - e) / e (Y - mu1)    (1 - A) Y
//! ```
//!
//! The ATE row is the estimator of Robins, Rotnitzky and Zhao (1994) and Bang
//! and Robins (2005), `mean_a = (1/n) sum [mu_a + 1(A = a)(Y - mu_a) / pi_a]`.
//! The difference of the ATT means is `[A (Y - mu0) - (1 - A) e / (1 - e) (Y -
//! mu0)] / P(A = 1)`, the estimator of Hahn (1998). The estimator is consistent
//! if either the outcome model or the propensity model is right, and efficient
//! if both are.
//!
//! The influence function of a ratio `sum w U / sum w D` is `w_i (U_i - theta
//! D_i) / sum w D`, which for the ATE is the efficient influence function
//! `psi_i - tau` over `n`, and for the ATT and ATC contains the term `- tau A /
//! P(A = 1)` that comes from estimating the size of the population. The
//! influence of the difference is the difference of the influences, and of the
//! ratio `theta_1 / theta_0` it is `(IF_1 - ratio IF_0) / theta_0`.
//!
//! # Variance
//!
//! [`Variance::Efficient`], from [`aipw`]: the sum of the squared influences,
//! `sum IF_i^2`, with the influences of the rows of a cluster summed first and
//! the factor `G / (G - 1)` for `G` clusters, as
//! `epistat_inference::influence::clustered_variance`. It is the variance of
//! the efficient estimator when both models are right, and the right one for
//! predictions that come from cross-fitting. It takes the predictions as fixed,
//! so with models that were fitted to the same data it is not valid when only
//! one model is right.
//!
//! [`Variance::Stacked`], from [`aipw_glm`]: the estimating equations of the
//! three fitted models (outcome among the treated, outcome among the controls,
//! propensity) are stacked with the equation of the estimator. With parameters
//! `beta` solving `sum s_i(beta) = 0`, and `G = - sum d s_i / d beta`, the
//! influence of a row gains `(sum_j w_j dU_j / d beta / sum w D)' G^-1 s_i` for
//! each model, and the variance is that of the sum. For the canonical links used
//! here the derivative of the score is the same observed and expected, so the
//! stacked variance is exact. It is valid when only one of the two models is
//! right (`known.nuisance = FALSE` of `riskRegression::ate`).
//!
//! # Edge cases
//!
//! A propensity of exactly 0 or 1, or outside the open interval, is an error. The
//! propensity may be truncated at limits of the caller's with
//! [`AipwOptions::truncation`], which changes the estimand and is off by default;
//! [`Aipw::n_trimmed`] counts the rows it moved, and the derivative of a
//! truncated propensity with respect to the propensity model is taken as 0. All
//! persons in one arm, a target group of weight 0, and a fitted propensity
//! model that does not converge, has an aliased column or has fitted
//! probabilities numerically 0 or 1 (separation) are errors. The estimate may lie
//! outside the range of the outcome, a known property of the estimator. The ratio
//! is `None` when the mean of the controls is 0.
//!
//! # References
//!
//! 1. Robins JM, Rotnitzky A, Zhao LP. Estimation of regression coefficients when
//!    some regressors are not always observed. J Am Stat Assoc.
//!    1994;89(427):846-66.
//! 2. Bang H, Robins JM. Doubly robust estimation in missing data and causal
//!    inference models. Biometrics. 2005;61(4):962-73.
//! 3. Hahn J. On the role of the propensity score in efficient semiparametric
//!    estimation of average treatment effects. Econometrica. 1998;66(2):315-31.
//! 4. Ozenne BMH, Scheike TH, Staerk L, Gerds TA. On the estimation of average
//!    treatment effects with right-censored time to event outcome and competing
//!    risks. Biom J. 2020;62(3):751-63.

use std::collections::BTreeMap;

use epistat_glm::{Family, Glm, Link};
use epistat_linalg::{Attempt, Factor, Mat, MatRef, weighted_gram};

use crate::error::{Error, Result};

/// The population the effect refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Target {
    /// The whole population, the average treatment effect.
    #[default]
    Ate,
    /// The treated, the average treatment effect in the treated.
    Att,
    /// The controls, the average treatment effect in the controls.
    Atc,
}

/// Which variance an estimate carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variance {
    /// The variance of the efficient influence function, with the predictions
    /// taken as given.
    Efficient,
    /// The variance that accounts for the estimation of the fitted models.
    Stacked,
}

/// What the estimators do besides the data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AipwOptions {
    /// One minus the level of the confidence limits. The default is 0.05.
    pub alpha: f64,
    /// Limits `(lower, upper)`, with `0 < lower < upper < 1`, at which the
    /// propensity is truncated. The default is no truncation.
    pub truncation: Option<(f64, f64)>,
}

impl Default for AipwOptions {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            truncation: None,
        }
    }
}

/// An estimate with its standard error, normal confidence limits and Wald test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    /// The estimate.
    pub estimate: f64,
    /// The standard error.
    pub se: f64,
    /// The lower confidence limit.
    pub lower: f64,
    /// The upper confidence limit.
    pub upper: f64,
    /// The two-sided p-value of the Wald test of 0, or of 1 for a ratio. `None`
    /// when the standard error is 0.
    pub p_value: Option<f64>,
}

/// The predictions of the models and the data of the core estimator.
#[derive(Debug, Clone, Copy)]
pub struct AipwInput<'a> {
    /// The outcome, 0 or 1 or continuous.
    pub outcome: &'a [f64],
    /// Whether each person is treated.
    pub treatment: &'a [bool],
    /// `E[Y | A = 1, X]` of each person.
    pub mu1: &'a [f64],
    /// `E[Y | A = 0, X]` of each person.
    pub mu0: &'a [f64],
    /// `P(A = 1 | X)` of each person.
    pub propensity: &'a [f64],
    /// Case weights, 1 for each person when `None`.
    pub weight: Option<&'a [f64]>,
    /// The cluster of each person, each person a cluster when `None`.
    pub cluster: Option<&'a [u32]>,
}

/// The result of an augmented estimator.
#[derive(Debug, Clone)]
pub struct Aipw {
    /// The population of the estimate.
    pub target: Target,
    /// The mean outcome had everyone been treated.
    pub mean1: Estimate,
    /// The mean outcome had no one been treated.
    pub mean0: Estimate,
    /// The difference `mean1 - mean0`.
    pub difference: Estimate,
    /// The ratio `mean1 / mean0` with the delta-method standard error, `None`
    /// when `mean0` is 0.
    pub ratio: Option<Estimate>,
    /// The influence function of the difference, one value per person, so that
    /// the variance is the sum of squares of the sums over clusters.
    pub influence: Vec<f64>,
    /// The variance of the estimates.
    pub variance: Variance,
    /// The number of persons whose propensity was truncated.
    pub n_trimmed: usize,
    /// The smallest propensity used, after truncation.
    pub min_propensity: f64,
    /// The largest propensity used, after truncation.
    pub max_propensity: f64,
}

/// What the estimators share: the terms of the ratios, and the influences of the
/// means with the models taken as fixed.
struct Core {
    omega: Vec<f64>,
    /// The propensity used, after truncation.
    e: Vec<f64>,
    clipped: Vec<bool>,
    /// The sum of the weights of the target, `sum w D`.
    size: f64,
    theta: [f64; 2],
    /// The influences of the means 1 and 0.
    influence: [Vec<f64>; 2],
}

fn check_len(name: &str, found: usize, n: usize) -> Result<()> {
    if found == n {
        Ok(())
    } else {
        Err(Error::input(format!(
            "{name}: {found} values for {n} persons"
        )))
    }
}

fn check_alpha(alpha: f64) -> Result<()> {
    if alpha > 0.0 && alpha < 1.0 {
        Ok(())
    } else {
        Err(Error::input(format!("alpha: {alpha} is not in (0, 1)")))
    }
}

fn compensated(values: impl IntoIterator<Item = f64>) -> f64 {
    epistat_stats::sum::compensated(values)
}

fn core(input: &AipwInput<'_>, target: Target, opts: &AipwOptions) -> Result<Core> {
    let n = input.treatment.len();
    if n == 0 {
        return Err(Error::input("treatment: no persons"));
    }
    check_alpha(opts.alpha)?;
    check_len("outcome", input.outcome.len(), n)?;
    check_len("mu1", input.mu1.len(), n)?;
    check_len("mu0", input.mu0.len(), n)?;
    check_len("propensity", input.propensity.len(), n)?;
    if let Some(w) = input.weight {
        check_len("weight", w.len(), n)?;
    }
    if let Some(c) = input.cluster {
        check_len("cluster", c.len(), n)?;
    }
    if let Some((lo, hi)) = opts.truncation
        && !(0.0 < lo && lo < hi && hi < 1.0)
    {
        return Err(Error::input(format!(
            "truncation: ({lo}, {hi}) are not limits with 0 < lower < upper < 1"
        )));
    }
    for (name, v) in [
        ("outcome", input.outcome),
        ("mu1", input.mu1),
        ("mu0", input.mu0),
    ] {
        if let Some(i) = v.iter().position(|x| !x.is_finite()) {
            return Err(Error::input(format!("{name}: value {i} is not finite")));
        }
    }
    let omega: Vec<f64> = input.weight.map_or_else(|| vec![1.0; n], <[f64]>::to_vec);
    if let Some(i) = omega.iter().position(|&w| !(w.is_finite() && w >= 0.0)) {
        return Err(Error::input(format!(
            "weight: value {i} is negative or not finite"
        )));
    }
    let mut e = Vec::with_capacity(n);
    let mut clipped = Vec::with_capacity(n);
    for (i, &p) in input.propensity.iter().enumerate() {
        if !(p > 0.0 && p < 1.0) {
            return Err(Error::input(format!(
                "propensity: person {i} has {p}, outside (0, 1)"
            )));
        }
        let q = opts.truncation.map_or(p, |(lo, hi)| p.clamp(lo, hi));
        clipped.push(q.to_bits() != p.to_bits());
        e.push(q);
    }
    let (mut treated, mut controls) = (false, false);
    for (&w, &t) in omega.iter().zip(input.treatment) {
        if w > 0.0 {
            treated |= t;
            controls |= !t;
        }
    }
    if !(treated && controls) {
        return Err(Error::input("treatment: both groups need persons"));
    }
    let (mut u1, mut u0, mut d) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let a = f64::from(u8::from(input.treatment[i]));
        let (y, m1, m0, p) = (input.outcome[i], input.mu1[i], input.mu0[i], e[i]);
        let ate1 = m1 + a * (y - m1) / p;
        let ate0 = m0 + (1.0 - a) * (y - m0) / (1.0 - p);
        match target {
            Target::Ate => {
                (u1[i], u0[i], d[i]) = (ate1, ate0, 1.0);
            }
            Target::Att => {
                (u1[i], u0[i], d[i]) = (a * y, a * m0 + (1.0 - a) * p / (1.0 - p) * (y - m0), a);
            }
            Target::Atc => {
                (u1[i], u0[i], d[i]) = (
                    (1.0 - a) * m1 + a * (1.0 - p) / p * (y - m1),
                    (1.0 - a) * y,
                    1.0 - a,
                );
            }
        }
    }
    let size = compensated((0..n).map(|i| omega[i] * d[i]));
    if size <= 0.0 {
        return Err(Error::input("the target population has weight 0"));
    }
    let theta = [
        compensated((0..n).map(|i| omega[i] * u1[i])) / size,
        compensated((0..n).map(|i| omega[i] * u0[i])) / size,
    ];
    let influence = [
        (0..n)
            .map(|i| omega[i] * (u1[i] - theta[0] * d[i]) / size)
            .collect(),
        (0..n)
            .map(|i| omega[i] * (u0[i] - theta[1] * d[i]) / size)
            .collect(),
    ];
    Ok(Core {
        omega,
        e,
        clipped,
        size,
        theta,
        influence,
    })
}

/// The variance of a sum of influences: the sum of squares, or with clusters
/// `G / (G - 1)` times the sum of squares of the sums over clusters.
fn variance_of(influence: &[f64], cluster: Option<&[u32]>) -> Result<f64> {
    let v = match cluster {
        None => compensated(influence.iter().map(|x| x * x)),
        Some(c) => {
            let mut sums: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
            for (&g, &x) in c.iter().zip(influence) {
                sums.entry(g).or_default().push(x);
            }
            let g = sums.len();
            if g < 2 {
                return Err(Error::input("cluster: the variance needs two clusters"));
            }
            let squares = compensated(sums.values().map(|v| {
                let s = compensated(v.iter().copied());
                s * s
            }));
            g as f64 / (g as f64 - 1.0) * squares
        }
    };
    if v.is_finite() {
        Ok(v.max(0.0))
    } else {
        Err(Error::input("the variance is not finite"))
    }
}

fn estimate(value: f64, variance: f64, null: f64, alpha: f64) -> Result<Estimate> {
    let se = variance.sqrt();
    let (lower, upper) = epistat_inference::interval::normal(value, se, alpha)
        .map_err(|e| Error::input(format!("alpha: {e}")))?;
    let p_value =
        (se > 0.0).then(|| 2.0 * epistat_stats::normal::cdf(-((value - null) / se).abs()));
    Ok(Estimate {
        estimate: value,
        se,
        lower,
        upper,
        p_value,
    })
}

fn finish(
    c: &Core,
    influence: &[Vec<f64>; 2],
    cluster: Option<&[u32]>,
    target: Target,
    variance: Variance,
    opts: &AipwOptions,
) -> Result<Aipw> {
    let n = c.omega.len();
    let diff: Vec<f64> = (0..n).map(|i| influence[0][i] - influence[1][i]).collect();
    let mean1 = estimate(
        c.theta[0],
        variance_of(&influence[0], cluster)?,
        0.0,
        opts.alpha,
    )?;
    let mean0 = estimate(
        c.theta[1],
        variance_of(&influence[1], cluster)?,
        0.0,
        opts.alpha,
    )?;
    let difference = estimate(
        c.theta[0] - c.theta[1],
        variance_of(&diff, cluster)?,
        0.0,
        opts.alpha,
    )?;
    let ratio = if c.theta[1] == 0.0 {
        None
    } else {
        let r = c.theta[0] / c.theta[1];
        let if_r: Vec<f64> = (0..n)
            .map(|i| (influence[0][i] - r * influence[1][i]) / c.theta[1])
            .collect();
        Some(estimate(r, variance_of(&if_r, cluster)?, 1.0, opts.alpha)?)
    };
    Ok(Aipw {
        target,
        mean1,
        mean0,
        difference,
        ratio,
        influence: diff,
        variance,
        n_trimmed: c.clipped.iter().filter(|&&x| x).count(),
        min_propensity: c.e.iter().copied().fold(f64::INFINITY, f64::min),
        max_propensity: c.e.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    })
}

/// The augmented estimator on given predictions of the outcome and propensity
/// models, with the variance of the efficient influence function.
///
/// # Errors
///
/// [`Error::Input`] for inputs of different lengths, no persons, a value that is
/// not finite, a negative weight, a propensity outside `(0, 1)`, truncation
/// limits that are not `0 < lower < upper < 1`, `alpha` outside `(0, 1)`, a
/// group without persons of positive weight, a target population of weight 0, and
/// fewer than two clusters.
pub fn aipw(input: &AipwInput<'_>, target: Target, opts: &AipwOptions) -> Result<Aipw> {
    let c = core(input, target, opts)?;
    finish(
        &c,
        &c.influence,
        input.cluster,
        target,
        Variance::Efficient,
        opts,
    )
}

/// The data of [`aipw_glm`].
#[derive(Debug, Clone, Copy)]
pub struct GlmInput<'a> {
    /// The outcome.
    pub outcome: &'a [f64],
    /// Whether each person is treated.
    pub treatment: &'a [bool],
    /// The design of the outcome models, one row per person, with the intercept
    /// as a column. The same design is used in each arm.
    pub outcome_design: MatRef<'a, f64>,
    /// The design of the propensity model, with the intercept as a column.
    pub propensity_design: MatRef<'a, f64>,
    /// Case weights, 1 for each person when `None`.
    pub weight: Option<&'a [f64]>,
    /// The cluster of each person.
    pub cluster: Option<&'a [u32]>,
}

/// `g^-1` and the variance function of a family with its canonical link.
/// The variance function of a family, `V(mu)`.
type VarianceFunction = fn(f64) -> f64;

fn canonical(family: Family) -> Result<(Link, VarianceFunction)> {
    match family {
        Family::Gaussian => Ok((Link::Identity, |_| 1.0)),
        Family::Binomial => Ok((Link::Logit, |m| m * (1.0 - m))),
        Family::Poisson => Ok((Link::Log, |m| m)),
        _ => Err(Error::input(
            "family: the outcome model is Gaussian, binomial or Poisson with its canonical link",
        )),
    }
}

struct Fitted {
    /// The prediction for every row, from the coefficients.
    mean: Vec<f64>,
}

fn fit_model(
    glm: &Glm,
    x: MatRef<'_, f64>,
    y: &[f64],
    weights: &[f64],
    what: &str,
) -> Result<Fitted> {
    let model = glm
        .fit(x, y, Some(weights), None, None)
        .map_err(|e| Error::input(format!("{what} model: {e}")))?;
    if !model.converged {
        return Err(Error::input(format!("{what} model: no convergence")));
    }
    let mut coefficients = Vec::with_capacity(model.coefficients.len());
    for (j, c) in model.coefficients.iter().enumerate() {
        coefficients.push(
            c.ok_or_else(|| Error::input(format!("{what} model: design column {j} is aliased")))?,
        );
    }
    let mean = (0..x.nrows())
        .map(|i| {
            let eta: f64 = (0..x.ncols()).map(|j| x[(i, j)] * coefficients[j]).sum();
            glm.link.inverse(eta)
        })
        .collect();
    Ok(Fitted { mean })
}

/// For one model: with `G = X' diag(gram) X`, the contribution to the influence of
/// each person of the estimating equations of the model, `score_i (v_m' x_i)` with
/// `v_m = G^-1 X' rhs_m`, for the two means `m`.
fn block(
    x: MatRef<'_, f64>,
    gram: &[f64],
    rhs: [&[f64]; 2],
    score: &[f64],
    what: &str,
) -> Result<[Vec<f64>; 2]> {
    let (n, p) = (x.nrows(), x.ncols());
    let g = weighted_gram(x, gram).map_err(|e| Error::input(format!("{what} model: {e}")))?;
    let factor = Factor::new(g.as_ref(), &Attempt::SYMMETRIC)
        .map_err(|e| Error::input(format!("{what} model: {e}")))?;
    let b = Mat::from_fn(p, 2, |j, m| {
        epistat_stats::sum::compensated((0..n).map(|i| x[(i, j)] * rhs[m][i]))
    });
    let v = factor
        .solve(b.as_ref())
        .map_err(|e| Error::input(format!("{what} model: {e}")))?;
    let out = |m: usize| -> Vec<f64> {
        (0..n)
            .map(|i| {
                let dot: f64 = (0..p).map(|j| v[(j, m)] * x[(i, j)]).sum();
                score[i] * dot
            })
            .collect()
    };
    Ok([out(0), out(1)])
}

/// The augmented estimator with generalised linear models for the outcome and a
/// logistic model for the treatment, and the variance that accounts for the
/// estimation of the models.
///
/// The outcome model is fitted to each arm separately, `family` with its
/// canonical link: Gaussian, binomial or Poisson. The propensity model is a
/// logistic regression of the treatment on `propensity_design`. Case weights
/// enter every fit and the estimator.
///
/// # Errors
///
/// The errors of [`aipw`], and [`Error::Input`] for a design with another number
/// of rows than the data or without columns, an outcome family other than the
/// three above, a model that does not converge or has an aliased column, a
/// response outside the support of the family, and a propensity model whose
/// fitted probabilities are numerically 0 or 1.
pub fn aipw_glm(
    input: &GlmInput<'_>,
    family: Family,
    target: Target,
    opts: &AipwOptions,
) -> Result<Aipw> {
    let n = input.treatment.len();
    check_len("outcome", input.outcome.len(), n)?;
    for (name, m) in [
        ("outcome_design", input.outcome_design),
        ("propensity_design", input.propensity_design),
    ] {
        if m.nrows() != n {
            return Err(Error::input(format!(
                "{name}: {} rows for {n} persons",
                m.nrows()
            )));
        }
        if m.ncols() == 0 {
            return Err(Error::input(format!("{name}: no columns")));
        }
        if !epistat_linalg::is_finite(m) {
            return Err(Error::input(format!("{name}: a value is not finite")));
        }
    }
    let (link, variance_function) = canonical(family)?;
    let omega: Vec<f64> = input.weight.map_or_else(|| vec![1.0; n], <[f64]>::to_vec);
    if omega.len() != n {
        return Err(Error::input(format!(
            "weight: {} values for {n} persons",
            omega.len()
        )));
    }
    let a: Vec<f64> = input
        .treatment
        .iter()
        .map(|&t| f64::from(u8::from(t)))
        .collect();
    let arm = |treated: bool| -> Vec<f64> {
        (0..n)
            .map(|i| {
                if input.treatment[i] == treated {
                    omega[i]
                } else {
                    0.0
                }
            })
            .collect()
    };
    let (x, z) = (input.outcome_design, input.propensity_design);
    let propensity = fit_model(
        &Glm::new(Family::Binomial, Link::Logit),
        z,
        &a,
        &omega,
        "propensity",
    )?;
    let eps10 = 10.0 * f64::EPSILON;
    if let Some(i) = propensity
        .mean
        .iter()
        .position(|&p| p < eps10 || p > 1.0 - eps10)
    {
        return Err(Error::input(format!(
            "propensity model: person {i} has fitted probability {} (separation)",
            propensity.mean[i]
        )));
    }
    let glm = Glm::new(family, link);
    let m1 = fit_model(&glm, x, input.outcome, &arm(true), "outcome (treated)")?;
    let m0 = fit_model(&glm, x, input.outcome, &arm(false), "outcome (controls)")?;
    let data = AipwInput {
        outcome: input.outcome,
        treatment: input.treatment,
        mu1: &m1.mean,
        mu0: &m0.mean,
        propensity: &propensity.mean,
        weight: input.weight,
        cluster: input.cluster,
    };
    let c = core(&data, target, opts)?;

    // Derivatives of U_1 and U_0 per person with respect to the outcome models
    // (a scalar times the design row) and to the propensity model, divided by the
    // size of the target.
    let (mut d1b, mut d0b) = (vec![0.0; n], vec![0.0; n]);
    let (mut d1e, mut d0e) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let (y, p) = (input.outcome[i], c.e[i]);
        let (v1, v0) = (variance_function(m1.mean[i]), variance_function(m0.mean[i]));
        let (ai, wi) = (a[i], c.omega[i] / c.size);
        let odds = p / (1.0 - p);
        let inverse_odds = (1.0 - p) / p;
        d1b[i] = wi
            * v1
            * match target {
                Target::Ate => 1.0 - ai / p,
                Target::Atc => (1.0 - ai) - ai * inverse_odds,
                Target::Att => 0.0,
            };
        d0b[i] = wi
            * v0
            * match target {
                Target::Ate => 1.0 - (1.0 - ai) / (1.0 - p),
                Target::Att => ai - (1.0 - ai) * odds,
                Target::Atc => 0.0,
            };
        if !c.clipped[i] {
            if target != Target::Att {
                d1e[i] = -wi * ai * (y - m1.mean[i]) * inverse_odds;
            }
            if target != Target::Atc {
                d0e[i] = wi * (1.0 - ai) * (y - m0.mean[i]) * odds;
            }
        }
    }
    let score = |treated: bool, mu: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|i| {
                if input.treatment[i] == treated {
                    c.omega[i] * (input.outcome[i] - mu[i])
                } else {
                    0.0
                }
            })
            .collect()
    };
    let gram = |treated: bool, mu: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|i| {
                if input.treatment[i] == treated {
                    c.omega[i] * variance_function(mu[i])
                } else {
                    0.0
                }
            })
            .collect()
    };
    let zeros = vec![0.0; n];
    // The propensity model's own equations, on the fitted probabilities.
    let ps_score: Vec<f64> = (0..n)
        .map(|i| c.omega[i] * (a[i] - propensity.mean[i]))
        .collect();
    let ps_gram: Vec<f64> = (0..n)
        .map(|i| c.omega[i] * propensity.mean[i] * (1.0 - propensity.mean[i]))
        .collect();
    let b1 = block(
        x,
        &gram(true, &m1.mean),
        [&d1b, &zeros],
        &score(true, &m1.mean),
        "outcome (treated)",
    )?;
    let b0 = block(
        x,
        &gram(false, &m0.mean),
        [&zeros, &d0b],
        &score(false, &m0.mean),
        "outcome (controls)",
    )?;
    let be = block(z, &ps_gram, [&d1e, &d0e], &ps_score, "propensity")?;
    let [e1, e0] = be;
    let mut influence = c.influence.clone();
    for i in 0..n {
        influence[0][i] += b1[0][i] + e1[i];
        influence[1][i] += b0[1][i] + e0[i];
    }
    finish(
        &c,
        &influence,
        input.cluster,
        target,
        Variance::Stacked,
        opts,
    )
}

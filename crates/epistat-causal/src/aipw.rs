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
//! The influence function of a ratio `sum w U / sum w D` is `n w_i (U_i - theta
//! D_i) / sum w D` per person, which for the ATE is the efficient influence
//! function `psi_i - tau`, and for the ATT and ATC contains the term `- tau A /
//! P(A = 1)` that comes from estimating the size of the population. The
//! influence of the difference is the difference of the influences, and of the
//! ratio `theta_1 / theta_0` it is `(IF_1 - ratio IF_0) / theta_0`.
//!
//! # Variance
//!
//! [`Aipw::influence`] holds the influence functions of the two means on the
//! per-unit scale, each person a unit coded by its row number. Every variance
//! is `Influence::variance` of them, with the influences of a cluster summed
//! first and centred, every person its own cluster when there are none. The
//! rule follows the R references. Without clusters it is
//! `ClusterRule::NoSmallSample`, the variance of a mean `sum IF^2 / n^2` with
//! the `n` divisor, as `riskRegression::ate` reports it (the standard error
//! `sqrt(sum iid^2)` of its influence functions on the dfbeta scale). With
//! clusters it is `ClusterRule::SmallSample`, the factor `G / (G - 1)` for `G`
//! clusters, as `sandwich::vcovCL` and `WeightIt` apply it. Fewer than two
//! clusters leave the standard errors `None`.
//!
//! [`Variance::Efficient`], from [`aipw`], is the variance of the efficient
//! estimator when both models are right, and the right one for predictions
//! that come from cross-fitting. It takes the predictions as fixed, so with
//! models that were fitted to the same data it is not valid when only one
//! model is right.
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
//! persons in one arm, a target group of weight 0, an estimand other than the
//! ATE, ATT and ATC, and a fitted propensity model that does not converge, has
//! an aliased column or has fitted probabilities numerically 0 or 1
//! (separation) are errors. The estimate may lie outside the range of the
//! outcome, a known property of the estimator. The ratio is `None` when the
//! mean of the controls is 0.
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

use epistat_core::{
    CaseWeights, Coefficients, Contrast, Covariance, Estimand, Glance, Level, Tidy, VarianceKind,
};
use epistat_glm::{Family, Glm, Link};
use epistat_inference::Influence;
use epistat_linalg::{Attempt, Factor, Mat, MatRef, weighted_gram};
use epistat_stats::weighted::ProbabilityBounds;

use crate::error::{Error, Model, Result, domain_at, empty};
use crate::shared::{
    case_weights, check_arms, check_design, check_estimand, check_finite, check_intercept,
    check_len, check_probabilities, check_separation, clusters, compensated,
    contrast_and_covariance, fit_glm, glm_data, influence, is_binary, means_table, row_keys, rule,
};

/// Which variance an estimate carries.
#[non_exhaustive]
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
    /// The population the effect refers to: the ATE (the default), ATT or ATC,
    /// `riskRegression::ate`'s `estimator` population.
    pub estimand: Estimand,
    /// The level of the confidence limits, by default 95%.
    pub level: Level,
    /// Bounds at which the propensity is truncated, with
    /// `0 < lower < upper < 1`. The default is no truncation.
    pub truncation: Option<ProbabilityBounds>,
}

impl Default for AipwOptions {
    fn default() -> Self {
        Self {
            estimand: Estimand::Ate,
            level: Level::default(),
            truncation: None,
        }
    }
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
    /// Case weights, 1 for each person when `None`; a person of weight 0
    /// leaves the estimate.
    pub weight: Option<CaseWeights<'a>>,
    /// The cluster of each person, each person a cluster when `None`.
    pub cluster: Option<&'a [u32]>,
}

/// The result of an augmented estimator.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct Aipw {
    /// The population of the estimate.
    pub estimand: Estimand,
    /// The mean outcome had everyone been treated (`mean1`), had no one been
    /// treated (`mean0`), their difference and their ratio, the ratio with the
    /// delta-method standard error, its interval on the scale of the ratio and
    /// the p-value of a ratio of 1; `None` when `mean0` is 0.
    pub contrast: Contrast,
    /// The influence functions of `mean1` (column 0) and `mean0` (column 1),
    /// per person on the per-unit scale, keyed by row, with the rule of the
    /// standard errors: `NoSmallSample` without clusters, `SmallSample` with
    /// them. Their `variance` with the clusters `Clusters::per_row(cluster)`
    /// (or none) reproduces the standard errors.
    pub influence: Influence,
    /// The variance of the estimates.
    pub variance: Variance,
    /// The number of persons whose propensity was truncated.
    pub n_trimmed: usize,
    /// The smallest propensity used, after truncation.
    pub min_propensity: f64,
    /// The largest propensity used, after truncation.
    pub max_propensity: f64,
    /// The persons, and those of them with a positive weight.
    n: (usize, usize),
    covariance: Covariance,
}

/// The table of the two means `mean1` and `mean0` on the outcome's scale,
/// with the influence-function covariance of [`Aipw::influence`], the only
/// one the estimator has ([`VarianceKind::Robust`]).
impl Tidy for Aipw {
    type Error = Error;

    fn coefficients(&self, level: Level, variance: VarianceKind) -> Result<Coefficients> {
        let theta = [self.contrast.mean1.estimate, self.contrast.mean0.estimate];
        means_table(theta, &self.covariance, level, variance)
    }

    fn default_variance(&self) -> VarianceKind {
        VarianceKind::Robust
    }

    fn glance(&self) -> Glance {
        Glance::new(self.n.0, self.n.1, true)
    }
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
    /// The influences of the means 1 and 0, on the dfbeta scale.
    influence: [Vec<f64>; 2],
}

fn core(input: &AipwInput<'_>, opts: &AipwOptions) -> Result<Core> {
    let n = input.treatment.len();
    if n == 0 {
        return Err(empty("treatment"));
    }
    check_estimand(opts.estimand)?;
    check_len("outcome", input.outcome.len(), n)?;
    check_len("mu1", input.mu1.len(), n)?;
    check_len("mu0", input.mu0.len(), n)?;
    check_len("propensity", input.propensity.len(), n)?;
    if let Some(c) = input.cluster {
        check_len("cluster", c.len(), n)?;
    }
    if let Some(bounds) = opts.truncation
        && !(0.0 < bounds.lower() && bounds.lower() < bounds.upper() && bounds.upper() < 1.0)
    {
        return Err(Error::invalid(
            "truncation",
            "the limits must have 0 < lower < upper < 1",
        ));
    }
    check_finite("outcome", input.outcome)?;
    check_finite("mu1", input.mu1)?;
    check_finite("mu0", input.mu0)?;
    let omega = case_weights("weight", input.weight, n)?;
    let mut e = Vec::with_capacity(n);
    let mut clipped = Vec::with_capacity(n);
    for (i, &p) in input.propensity.iter().enumerate() {
        if !p.is_finite() {
            check_finite("propensity", input.propensity)?;
        }
        if !(p > 0.0 && p < 1.0) {
            return Err(domain_at("propensity", p, "(0, 1)", i));
        }
        let q = opts.truncation.map_or(p, |bounds| bounds.apply(p));
        clipped.push(q.to_bits() != p.to_bits());
        e.push(q);
    }
    check_arms(&omega, input.treatment)?;
    let (mut u1, mut u0, mut d) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let a = f64::from(u8::from(input.treatment[i]));
        let (y, m1, m0, p) = (input.outcome[i], input.mu1[i], input.mu0[i], e[i]);
        let ate1 = m1 + a * (y - m1) / p;
        let ate0 = m0 + (1.0 - a) * (y - m0) / (1.0 - p);
        match opts.estimand {
            Estimand::Att => {
                (u1[i], u0[i], d[i]) = (a * y, a * m0 + (1.0 - a) * p / (1.0 - p) * (y - m0), a);
            }
            Estimand::Atc => {
                (u1[i], u0[i], d[i]) = (
                    (1.0 - a) * m1 + a * (1.0 - p) / p * (y - m1),
                    (1.0 - a) * y,
                    1.0 - a,
                );
            }
            _ => {
                (u1[i], u0[i], d[i]) = (ate1, ate0, 1.0);
            }
        }
    }
    let size = compensated((0..n).map(|i| omega[i] * d[i]));
    if size <= 0.0 {
        return Err(Error::invalid(
            "weight",
            "the target population has weight 0",
        ));
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

fn finish(
    c: &Core,
    dfbeta: &[Vec<f64>; 2],
    cluster: Option<&[u32]>,
    variance: Variance,
    opts: &AipwOptions,
) -> Result<Aipw> {
    let n = c.omega.len();
    let influence = influence(
        [&dfbeta[0], &dfbeta[1]],
        row_keys(0..n)?,
        rule(cluster.is_some()),
    )?;
    let clusters = clusters(cluster, n)?;
    let (contrast, covariance) =
        contrast_and_covariance(c.theta, &influence, clusters.as_ref(), opts.level)?;
    Ok(Aipw {
        estimand: opts.estimand,
        contrast,
        influence,
        variance,
        n_trimmed: c.clipped.iter().filter(|&&x| x).count(),
        min_propensity: c.e.iter().copied().fold(f64::INFINITY, f64::min),
        max_propensity: c.e.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        n: (n, c.omega.iter().filter(|&&w| w > 0.0).count()),
        covariance,
    })
}

/// The augmented estimator on given predictions of the outcome and propensity
/// models, with the variance of the efficient influence function.
///
/// For a binary outcome, every outcome 0 or 1, the predictions `mu1` and
/// `mu0` are probabilities and one outside `[0, 1]`, such as a log odds or a
/// percent, is refused.
///
/// # Errors
///
/// [`Error::Input`] for inputs of different lengths, no persons, a value that is
/// not finite, a propensity outside `(0, 1)` and, for a binary outcome, a
/// prediction outside `[0, 1]`;
/// [`Error::Invalid`] for truncation limits that are not `0 < lower < upper <
/// 1`, an estimand other than the ATE, ATT and ATC, a group without persons
/// of positive weight and a target population of weight 0.
pub fn aipw(input: &AipwInput<'_>, opts: &AipwOptions) -> Result<Aipw> {
    let c = core(input, opts)?;
    if is_binary(input.outcome) {
        check_probabilities("mu1", input.mu1)?;
        check_probabilities("mu0", input.mu0)?;
    }
    finish(&c, &c.influence, input.cluster, Variance::Efficient, opts)
}

/// The data of [`aipw_glm`] and of [`crate::tmle::tmle_glm`].
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
    /// Case weights, 1 for each person when `None`; a person of weight 0
    /// leaves the estimate.
    pub weight: Option<CaseWeights<'a>>,
    /// The cluster of each person.
    pub cluster: Option<&'a [u32]>,
}

/// The variance function of a family, `V(mu)`.
type VarianceFunction = fn(f64) -> f64;

fn canonical(family: Family) -> Result<(Link, VarianceFunction)> {
    match family {
        Family::Gaussian => Ok((Link::Identity, |_| 1.0)),
        Family::Binomial => Ok((Link::Logit, |m| m * (1.0 - m))),
        Family::Poisson => Ok((Link::Log, |m| m)),
        _ => Err(Error::invalid(
            "family",
            "the outcome model is Gaussian, binomial or Poisson with its canonical link",
        )),
    }
}

/// The predictions of a strict fit of `glm` for every row of `x`, from its
/// coefficients.
pub(crate) fn fitted_mean(
    glm: &Glm,
    x: MatRef<'_, f64>,
    y: &[f64],
    weights: &[f64],
    model: Model,
) -> Result<Vec<f64>> {
    let (_, coefficients) = fit_glm(glm, &glm_data(x, y, Some(weights), None, model)?, model)?;
    Ok((0..x.nrows())
        .map(|i| {
            let eta: f64 = (0..x.ncols()).map(|j| x[(i, j)] * coefficients[j]).sum();
            glm.link.inverse(eta)
        })
        .collect())
}

/// For one model: with `G = X' diag(gram) X`, the contribution to the influence of
/// each person of the estimating equations of the model, `score_i (v_m' x_i)` with
/// `v_m = G^-1 X' rhs_m`, for the two means `m`.
fn block(
    x: MatRef<'_, f64>,
    gram: &[f64],
    rhs: [&[f64]; 2],
    score: &[f64],
    model: Model,
) -> Result<[Vec<f64>; 2]> {
    let (n, p) = (x.nrows(), x.ncols());
    let g = weighted_gram(x, gram).map_err(|e| Error::linalg(model, e))?;
    let factor =
        Factor::new(g.as_ref(), &Attempt::SYMMETRIC).map_err(|e| Error::linalg(model, e))?;
    let b = Mat::from_fn(p, 2, |j, m| {
        compensated((0..n).map(|i| x[(i, j)] * rhs[m][i]))
    });
    let v = factor
        .solve(b.as_ref())
        .map_err(|e| Error::linalg(model, e))?;
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
/// The errors of [`aipw`]; [`Error::Input`] for a design with another number
/// of rows than the data, without columns or with a value that is not finite;
/// [`Error::Invalid`] for an outcome family other than the three above and a
/// design without a constant (intercept) column;
/// [`Error::Aliased`] for a model that has an aliased column, [`Error::Fit`]
/// for a response outside the support of the family, [`Error::Separation`]
/// for a propensity model whose fitted probabilities are numerically 0 or 1
/// and [`Error::NotConverged`] for a model that does not converge.
pub fn aipw_glm(input: &GlmInput<'_>, family: Family, opts: &AipwOptions) -> Result<Aipw> {
    let n = input.treatment.len();
    check_estimand(opts.estimand)?;
    check_len("outcome", input.outcome.len(), n)?;
    check_design("outcome_design", input.outcome_design, n)?;
    check_design("propensity_design", input.propensity_design, n)?;
    check_intercept("outcome_design", input.outcome_design)?;
    check_intercept("propensity_design", input.propensity_design)?;
    let (link, variance_function) = canonical(family)?;
    let omega = case_weights("weight", input.weight, n)?;
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
    let propensity = fitted_mean(
        &Glm::new(Family::Binomial, Link::Logit),
        z,
        &a,
        &omega,
        Model::Propensity,
    )?;
    check_separation(&propensity)?;
    let glm = Glm::new(family, link);
    let m1 = fitted_mean(&glm, x, input.outcome, &arm(true), Model::Outcome)?;
    let m0 = fitted_mean(&glm, x, input.outcome, &arm(false), Model::Outcome)?;
    let data = AipwInput {
        outcome: input.outcome,
        treatment: input.treatment,
        mu1: &m1,
        mu0: &m0,
        propensity: &propensity,
        weight: input.weight,
        cluster: input.cluster,
    };
    let c = core(&data, opts)?;
    let estimand = opts.estimand;

    // Derivatives of U_1 and U_0 per person with respect to the outcome models
    // (a scalar times the design row) and to the propensity model, divided by the
    // size of the target.
    let (mut d1b, mut d0b) = (vec![0.0; n], vec![0.0; n]);
    let (mut d1e, mut d0e) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let (y, p) = (input.outcome[i], c.e[i]);
        let (v1, v0) = (variance_function(m1[i]), variance_function(m0[i]));
        let (ai, wi) = (a[i], c.omega[i] / c.size);
        let odds = p / (1.0 - p);
        let inverse_odds = (1.0 - p) / p;
        d1b[i] = wi
            * v1
            * match estimand {
                Estimand::Atc => (1.0 - ai) - ai * inverse_odds,
                Estimand::Att => 0.0,
                _ => 1.0 - ai / p,
            };
        d0b[i] = wi
            * v0
            * match estimand {
                Estimand::Att => ai - (1.0 - ai) * odds,
                Estimand::Atc => 0.0,
                _ => 1.0 - (1.0 - ai) / (1.0 - p),
            };
        if !c.clipped[i] {
            if estimand != Estimand::Att {
                d1e[i] = -wi * ai * (y - m1[i]) * inverse_odds;
            }
            if estimand != Estimand::Atc {
                d0e[i] = wi * (1.0 - ai) * (y - m0[i]) * odds;
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
        .map(|i| c.omega[i] * (a[i] - propensity[i]))
        .collect();
    let ps_gram: Vec<f64> = (0..n)
        .map(|i| c.omega[i] * propensity[i] * (1.0 - propensity[i]))
        .collect();
    let b1 = block(
        x,
        &gram(true, &m1),
        [&d1b, &zeros],
        &score(true, &m1),
        Model::Outcome,
    )?;
    let b0 = block(
        x,
        &gram(false, &m0),
        [&zeros, &d0b],
        &score(false, &m0),
        Model::Outcome,
    )?;
    let be = block(z, &ps_gram, [&d1e, &d0e], &ps_score, Model::Propensity)?;
    let [e1, e0] = be;
    let mut influence = c.influence.clone();
    for i in 0..n {
        influence[0][i] += b1[0][i] + e1[i];
        influence[1][i] += b0[1][i] + e0[i];
    }
    finish(&c, &influence, input.cluster, Variance::Stacked, opts)
}

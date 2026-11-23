//! Targeted maximum likelihood estimation for binary and bounded continuous
//! outcomes, as the R package `tmle` does it.
//!
//! # Algorithm
//!
//! With treatment `A`, outcome `Y`, the outcome regressions `Q_a(x) = E[Y | A =
//! a, X = x]` and the propensity score `g(x) = P(A = 1 | X = x)`, the estimator
//! starts from the predictions of the two models and moves the outcome
//! predictions along a path chosen so that the efficient influence function of
//! the target has mean 0 at the end. The estimate is then the mean of the moved
//! predictions, a plug-in, and stays inside the range of the outcome.
//!
//! Scale. The outcome is mapped to `[0, 1]` by `Y* = (Y - l) / (u - l)`, with
//! `[l, u]` the observed range of `Y` ([`OutcomeRange::Observed`], as `tmle`)
//! or a known one ([`OutcomeRange::Known`]), and so are the predictions, which
//! are then bounded to `[delta, 1 - delta]` with `delta` =
//! [`TmleOptions::outcome_bound`] (`1 - alpha` of `tmle`, by default 0.0005).
//! A binary outcome has `l = 0` and `u = 1`. Estimates and influences are mapped
//! back by `u - l`.
//!
//! Propensity bound. With `b` the bound of [`TmleOptions::propensity_bound`],
//! `g1 = max(g, b)` and `g0 = max(1 - g, b)`, as `tmle`'s `gbound`; its default
//! is `b = 5 / (sqrt(n) ln n)`, here [`PropensityBound::Tmle`].
//!
//! The effect in the population ([`Target::Ate`]). Logistic fluctuation with
//! the two clever covariates `H0 = (1 - A) / g0` and `H1 = A / g1`:
//!
//! ```text
//! logit Q*(A, x) = logit Q(A, x) + e0 H0 + e1 H1      (logistic fit with offset, case weights)
//! logit Q*_1(x)  = logit Q_1(x) + e1 / g1
//! logit Q*_0(x)  = logit Q_0(x) + e0 / g0
//! mean_a = sum w Q*_a / sum w
//! IF_1,i = w_i [ A_i / g1_i (Y_i - Q*_1,i) + Q*_1,i - mean_1 ] / sum w
//! IF_0,i = w_i [ (1 - A_i) / g0_i (Y_i - Q*_0,i) + Q*_0,i - mean_0 ] / sum w
//! ```
//!
//! The fit of `(e0, e1)` solves the score equations `sum w H_a (Y* - Q*) = 0`,
//! which are the means of the augmentation terms, so the plug-in equals the
//! augmented estimator of [`crate::aipw`] at `Q*`. The two covariates target each
//! mean separately; the single covariate `H1 - H0` would target the difference
//! only. The fit stops at R's `glm` rule, so the equations hold to its
//! tolerance, which [`Tmle::score`] reports.
//!
//! The effect on the treated ([`Target::Att`]). The one-step estimator of van der
//! Laan and Gruber (2016) along the universal least favourable path, which moves
//! both `Q` and `g` in steps of size `h` ([`TmleOptions::step`]), as `tmle`'s
//! `oneStepATT`. With `q = mean(w A)`, the target is `psi = mean(w (Q_1 - Q_0) g1)
//! / q` and the efficient influence function `IC = w [ (A - (1 - A) g1 / g0)(Y* -
//! Q(A)) + A (Q_1 - Q_0 - psi) ] / q`. The sign of `h` is that of `-mean(q IC)`.
//! Each step is
//!
//! ```text
//! logit g   <- logit g - h (Q_1 - Q_0 - psi')
//! logit Q(A) <- logit Q(A) - h (A - (1 - A) g / (1 - g))
//! logit Q_0 <- logit Q_0 + h g / (1 - g)
//! logit Q_1 <- logit Q_1 - h
//! ```
//!
//! with `Q` bounded to `[delta, 1 - delta]` and `g1`, `g0` bounded below by `b`
//! after the step, and `psi'` the target two steps back (one step back at the
//! first step), as `tmle` computes it. The steps continue while the loss `-mean(w
//! [Y* ln Q(A) + (1 - Y*) ln(1 - Q(A)) + A ln g1 + (1 - A) ln g0])` falls; the
//! estimate is the one before the first step that does not lower it, so `|P_n
//! IC|` is of the order of `h` ([`Tmle::score`]). At most `max(1000, 2 / h)`
//! steps are taken. The effect on the controls ([`Target::Atc`]) is minus the
//! effect on the treated with the roles of the arms swapped. The influence of the
//! difference is `IC (u - l) / m` over the `m` persons used. The mean of the
//! treated arm is the weighted mean of `Y` among the treated, and that of the
//! other arm is it minus the effect, with the influences that follow (for the
//! controls the other way round).
//!
//! Support of the effect on the treated. With
//! [`TmleOptions::restrict_support`], as `tmle`: the controls whose propensity is
//! below the smallest of the treated are left out of the effect on the treated,
//! and the treated above the largest of the controls out of the effect on the
//! controls, when more than 10 controls and more than 10 treated remain in those
//! two sets. The propensity of the persons used is then recalibrated: when at
//! least 90 percent of the arm that loses persons remain, by the intercept `c` of
//! an unweighted logistic regression of `A` with offset `logit g` among them,
//! `logit g <- c + logit g`. Otherwise `tmle` refits its learners on the
//! covariates and `g` among them, which [`tmle_glm`] does with a logistic model
//! on the propensity design and `g`, and [`tmle`], which has no covariates, with
//! the intercept as above. Case weights are rescaled to mean 1 among the persons
//! used.
//!
//! # Variance
//!
//! The variance of the efficient influence function with the fitted models taken
//! as given: `sum IF_i^2`, with the influences of the rows of a cluster summed
//! first and the factor `G / (G - 1)` for `G` clusters, as in [`crate::aipw`].
//! `tmle` reports `var(n IF) / n` over persons, and with clusters the variance of
//! the cluster means of `n IF` over `G`: the same up to `n / (n - 1)`, the
//! centring and, for clusters of unequal size, the weighting of the clusters.
//! The ratio has the delta-method standard error `(IF_1 - ratio IF_0) / mean_0`.
//!
//! # Edge cases
//!
//! A propensity of 0 or 1 or outside the open interval, an outcome outside a
//! known range, an outcome with one value under the observed range, a group
//! without persons of positive weight, a fluctuation that does not converge and a
//! one-step path that reaches its step limit while the loss still falls are
//! errors. The ratio is `None` when the mean of the controls is 0. The targeted
//! predictions of persons left out by the support rule are `None`.
//!
//! # References
//!
//! 1. van der Laan MJ, Rubin D. Targeted maximum likelihood learning. Int J
//!    Biostat. 2006;2(1):Article 11.
//! 2. Gruber S, van der Laan MJ. A targeted maximum likelihood estimator of a
//!    causal effect on a bounded continuous outcome. Int J Biostat.
//!    2010;6(1):Article 26.
//! 3. Gruber S, van der Laan MJ. tmle: An R package for targeted maximum
//!    likelihood estimation. J Stat Softw. 2012;51(13):1-35.
//! 4. van der Laan MJ, Gruber S. One-step targeted minimum loss-based estimation
//!    based on universal least favorable one-dimensional submodels. Int J
//!    Biostat. 2016;12(1):351-78.
//! 5. Schuler MS, Rose S. Targeted maximum likelihood estimation for causal
//!    inference in observational studies. Am J Epidemiol. 2017;185(1):65-73.

use epistat_glm::{Family, Glm, Link};
use epistat_linalg::{Mat, MatRef};

use crate::aipw::{
    AipwInput, Estimate, GlmInput, Target, check_alpha, check_len, compensated, estimate,
    fit_model, variance_of,
};
use crate::error::{Error, Result};

/// The predictions of the models and the data of [`tmle`], the same as those of
/// [`crate::aipw::aipw`]. The predictions `mu1` and `mu0` are on the scale of the
/// outcome.
pub type TmleInput<'a> = AipwInput<'a>;

/// The range `[l, u]` that maps the outcome to `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum OutcomeRange {
    /// The smallest and largest outcome, as `tmle`.
    #[default]
    Observed,
    /// Limits known in advance, which every outcome must lie within.
    Known {
        /// The lower limit `l`.
        lower: f64,
        /// The upper limit `u`, above `l`.
        upper: f64,
    },
}

/// The lower bound `b` of `g` and `1 - g` in the clever covariates.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PropensityBound {
    /// No bound.
    #[default]
    None,
    /// `5 / (sqrt(n) ln n)` for `n` persons, the default of `tmle`.
    Tmle,
    /// A bound `0 < b < 0.5`.
    At(f64),
}

/// The outcome model of [`tmle_glm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutcomeModel {
    /// One model with the treatment as a column after the design, the main-terms
    /// model of `tmle` with `Q.SL.library = "SL.glm"`.
    #[default]
    Pooled,
    /// One model in each arm, as [`crate::aipw::aipw_glm`].
    ByArm,
}

/// What the estimators do besides the data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TmleOptions {
    /// One minus the level of the confidence limits. The default is 0.05.
    pub alpha: f64,
    /// The range that maps the outcome to `[0, 1]`. The default is the observed
    /// range.
    pub range: OutcomeRange,
    /// The bound of the propensity. The default is none.
    pub propensity_bound: PropensityBound,
    /// The bound `delta` of the scaled outcome predictions, which are kept in
    /// `[delta, 1 - delta]`, with `0 < delta < 0.5`. The default is 0.0005, as
    /// `tmle`.
    pub outcome_bound: f64,
    /// The step `h` of the one-step estimator of the effect on the treated or on
    /// the controls. The default is 0.001, as `tmle`.
    pub step: f64,
    /// Whether the effect on the treated or on the controls leaves out persons
    /// outside the support of the other arm, as `tmle`. The default is true.
    pub restrict_support: bool,
}

impl Default for TmleOptions {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            range: OutcomeRange::Observed,
            propensity_bound: PropensityBound::None,
            outcome_bound: 0.0005,
            step: 0.001,
            restrict_support: true,
        }
    }
}

/// How the outcome predictions were moved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Targeting {
    /// The logistic fluctuation of the effect in the population.
    Fluctuation {
        /// The coefficients `(e0, e1)` of the clever covariates of the controls
        /// and of the treated.
        epsilon: [f64; 2],
        /// The iterations of the logistic fit.
        iterations: usize,
    },
    /// The one-step path of the effect on the treated or on the controls.
    OneStep {
        /// The number of steps to the estimate.
        steps: usize,
        /// Whether the propensity was recalibrated by the support rule.
        recalibrated: bool,
    },
}

/// The result of a targeted estimator.
#[derive(Debug, Clone)]
pub struct Tmle {
    /// The population of the estimate.
    pub target: Target,
    /// The mean outcome had everyone been treated.
    pub mean1: Estimate,
    /// The mean outcome had no one been treated.
    pub mean0: Estimate,
    /// The difference `mean1 - mean0`.
    pub difference: Estimate,
    /// The ratio `mean1 / mean0`, `None` when `mean0` is 0.
    pub ratio: Option<Estimate>,
    /// The influence function of the difference, one value per person and 0 for
    /// persons not used, so that the variance is the sum of squares of the sums
    /// over clusters.
    pub influence: Vec<f64>,
    /// The targeted prediction `Q*_1` of each person on the scale of the
    /// outcome, `None` for persons not used.
    pub targeted1: Vec<Option<f64>>,
    /// The targeted prediction `Q*_0` of each person.
    pub targeted0: Vec<Option<f64>>,
    /// How the predictions were moved.
    pub targeting: Targeting,
    /// The sum of [`Tmle::influence`], the mean of the efficient influence
    /// function at the targeted predictions, which targeting brings to 0.
    pub score: f64,
    /// The range `(l, u)` that mapped the outcome to `[0, 1]`.
    pub range: (f64, f64),
    /// The bound `b` of the propensity, 0 when there is none.
    pub propensity_bound: f64,
    /// The number of persons used whose `g` or `1 - g` was below the bound.
    pub n_trimmed: usize,
    /// The number of persons used, fewer than all for the effect on the treated
    /// or on the controls under the support rule.
    pub n_used: usize,
}

/// The data after the checks, on the scale of the targeting.
struct Prepared<'a> {
    treatment: &'a [bool],
    outcome: &'a [f64],
    cluster: Option<&'a [u32]>,
    /// The case weights rescaled to mean 1.
    w: Vec<f64>,
    ystar: Vec<f64>,
    lower: f64,
    width: f64,
    bound: f64,
    delta: f64,
}

/// R's `qlogis`.
fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

/// R's `plogis`.
fn plogis(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn flag(t: bool) -> f64 {
    f64::from(u8::from(t))
}

fn prepare<'a>(
    outcome: &'a [f64],
    treatment: &'a [bool],
    weight: Option<&[f64]>,
    cluster: Option<&'a [u32]>,
    opts: &TmleOptions,
) -> Result<Prepared<'a>> {
    let n = treatment.len();
    if n == 0 {
        return Err(Error::input("treatment: no persons"));
    }
    check_alpha(opts.alpha)?;
    check_len("outcome", outcome.len(), n)?;
    if let Some(w) = weight {
        check_len("weight", w.len(), n)?;
    }
    if let Some(c) = cluster {
        check_len("cluster", c.len(), n)?;
    }
    if !(opts.outcome_bound > 0.0 && opts.outcome_bound < 0.5) {
        return Err(Error::input(format!(
            "outcome_bound: {} is not in (0, 0.5)",
            opts.outcome_bound
        )));
    }
    if !(opts.step.is_finite() && opts.step > 0.0) {
        return Err(Error::input(format!(
            "step: {} is not positive and finite",
            opts.step
        )));
    }
    if let Some(i) = outcome.iter().position(|y| !y.is_finite()) {
        return Err(Error::input(format!("outcome: value {i} is not finite")));
    }
    let raw: Vec<f64> = weight.map_or_else(|| vec![1.0; n], <[f64]>::to_vec);
    if let Some(i) = raw.iter().position(|&w| !(w.is_finite() && w >= 0.0)) {
        return Err(Error::input(format!(
            "weight: value {i} is negative or not finite"
        )));
    }
    let (mut treated, mut controls) = (false, false);
    for (&w, &t) in raw.iter().zip(treatment) {
        if w > 0.0 {
            treated |= t;
            controls |= !t;
        }
    }
    if !(treated && controls) {
        return Err(Error::input("treatment: both groups need persons"));
    }
    let total = compensated(raw.iter().copied());
    let w: Vec<f64> = raw.iter().map(|&x| x / total * n as f64).collect();
    let (lower, upper) = match opts.range {
        OutcomeRange::Observed => {
            let lo = outcome.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = outcome.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if lo >= hi {
                return Err(Error::input(
                    "outcome: one value, so the observed range is empty",
                ));
            }
            (lo, hi)
        }
        OutcomeRange::Known { lower, upper } => {
            if !(lower.is_finite() && upper.is_finite() && lower < upper) {
                return Err(Error::input(format!(
                    "range: ({lower}, {upper}) is not a range with lower < upper"
                )));
            }
            if let Some(i) = outcome.iter().position(|&y| y < lower || y > upper) {
                return Err(Error::input(format!(
                    "outcome: value {i} is {}, outside the range ({lower}, {upper})",
                    outcome[i]
                )));
            }
            (lower, upper)
        }
    };
    let width = upper - lower;
    let ystar = outcome.iter().map(|&y| (y - lower) / width).collect();
    let bound = match opts.propensity_bound {
        PropensityBound::None => 0.0,
        PropensityBound::Tmle => {
            let m = n as f64;
            let b = 5.0 / m.sqrt() / m.ln();
            if !(b > 0.0 && b < 0.5) {
                return Err(Error::input(format!(
                    "propensity_bound: tmle's bound 5 / (sqrt(n) ln n) is {b} for {n} persons, not in (0, 0.5)"
                )));
            }
            b
        }
        PropensityBound::At(b) => {
            if !(b > 0.0 && b < 0.5) {
                return Err(Error::input(format!(
                    "propensity_bound: {b} is not in (0, 0.5)"
                )));
            }
            b
        }
    };
    Ok(Prepared {
        treatment,
        outcome,
        cluster,
        w,
        ystar,
        lower,
        width,
        bound,
        delta: opts.outcome_bound,
    })
}

fn check_propensity(g: &[f64]) -> Result<()> {
    match g.iter().position(|&p| !(p > 0.0 && p < 1.0)) {
        Some(i) => Err(Error::input(format!(
            "propensity: person {i} has {}, outside (0, 1)",
            g[i]
        ))),
        None => Ok(()),
    }
}

/// How the propensity is refitted when the support rule leaves out more than a
/// tenth of an arm.
#[derive(Clone, Copy)]
enum Refit<'a> {
    /// The intercept recalibration of the other case.
    Intercept,
    /// A logistic model on these columns and `g`, with the case weights.
    Design(MatRef<'a, f64>),
}

/// The targeted estimator on scaled and bounded predictions `q1`, `q0` and the
/// propensity `g`.
fn targeted(
    p: &Prepared<'_>,
    q1: &[f64],
    q0: &[f64],
    g: &[f64],
    target: Target,
    refit: Refit<'_>,
    opts: &TmleOptions,
) -> Result<Tmle> {
    match target {
        Target::Ate => population(p, q1, q0, g, opts),
        Target::Att | Target::Atc => on_arm(p, q1, q0, g, target, refit, opts),
    }
}

fn population(
    p: &Prepared<'_>,
    q1: &[f64],
    q0: &[f64],
    g: &[f64],
    opts: &TmleOptions,
) -> Result<Tmle> {
    let n = p.w.len();
    let a: Vec<f64> = p.treatment.iter().map(|&t| flag(t)).collect();
    let g1: Vec<f64> = g.iter().map(|&x| x.max(p.bound)).collect();
    let g0: Vec<f64> = g.iter().map(|&x| (1.0 - x).max(p.bound)).collect();
    let l1: Vec<f64> = q1.iter().map(|&q| logit(q)).collect();
    let l0: Vec<f64> = q0.iter().map(|&q| logit(q)).collect();
    let offset: Vec<f64> = (0..n)
        .map(|i| if p.treatment[i] { l1[i] } else { l0[i] })
        .collect();
    let design = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            (1.0 - a[i]) / g0[i]
        } else {
            a[i] / g1[i]
        }
    });
    let fit = Glm::new(Family::Binomial, Link::Logit)
        .fit(design.as_ref(), &p.ystar, Some(&p.w), Some(&offset), None)
        .map_err(|e| Error::input(format!("fluctuation: {e}")))?;
    if !fit.converged {
        return Err(Error::NoConvergence(
            "fluctuation: the logistic fit did not converge".into(),
        ));
    }
    let e0 = fit.coefficients[0].unwrap_or(0.0);
    let e1 = fit.coefficients[1].unwrap_or(0.0);
    let back = |x: f64| x * p.width + p.lower;
    let s1: Vec<f64> = (0..n).map(|i| back(plogis(l1[i] + e1 / g1[i]))).collect();
    let s0: Vec<f64> = (0..n).map(|i| back(plogis(l0[i] + e0 / g0[i]))).collect();
    let total = compensated(p.w.iter().copied());
    let mean1 = compensated((0..n).map(|i| p.w[i] * s1[i])) / total;
    let mean0 = compensated((0..n).map(|i| p.w[i] * s0[i])) / total;
    let y = p.outcome;
    let if1: Vec<f64> = (0..n)
        .map(|i| p.w[i] * (a[i] / g1[i] * (y[i] - s1[i]) + s1[i] - mean1) / total)
        .collect();
    let if0: Vec<f64> = (0..n)
        .map(|i| p.w[i] * ((1.0 - a[i]) / g0[i] * (y[i] - s0[i]) + s0[i] - mean0) / total)
        .collect();
    let n_trimmed = g
        .iter()
        .filter(|&&x| x < p.bound || 1.0 - x < p.bound)
        .count();
    finish(
        p,
        Target::Ate,
        [mean1, mean0],
        [&if1, &if0],
        None,
        (
            s1.into_iter().map(Some).collect(),
            s0.into_iter().map(Some).collect(),
        ),
        Targeting::Fluctuation {
            epsilon: [e0, e1],
            iterations: fit.iterations,
        },
        n_trimmed,
        n,
        opts,
    )
}

/// The rows used for the effect on the treated (or on the controls) and their
/// propensity, after the support rule of `tmle`.
fn support(
    p: &Prepared<'_>,
    g: &[f64],
    target: Target,
    refit: Refit<'_>,
    opts: &TmleOptions,
) -> Result<(Vec<usize>, Vec<f64>, bool)> {
    let n = g.len();
    let all = || (0..n).collect::<Vec<usize>>();
    if !opts.restrict_support {
        return Ok((all(), g.to_vec(), false));
    }
    let a = p.treatment;
    let min_treated = (0..n)
        .filter(|&i| a[i])
        .map(|i| g[i])
        .fold(f64::INFINITY, f64::min);
    let max_control = (0..n)
        .filter(|&i| !a[i])
        .map(|i| g[i])
        .fold(f64::NEG_INFINITY, f64::max);
    let att: Vec<usize> = (0..n).filter(|&i| g[i] >= min_treated).collect();
    let atc: Vec<usize> = (0..n).filter(|&i| g[i] <= max_control).collect();
    let controls_kept = att.iter().filter(|&&i| !a[i]).count();
    let treated_kept = atc.iter().filter(|&&i| a[i]).count();
    if !(controls_kept > 10 && treated_kept > 10) {
        return Ok((all(), g.to_vec(), false));
    }
    let (rows, kept, total) = if target == Target::Att {
        (att, controls_kept, a.iter().filter(|&&t| !t).count())
    } else {
        (atc, treated_kept, a.iter().filter(|&&t| t).count())
    };
    let ar: Vec<f64> = rows.iter().map(|&i| flag(a[i])).collect();
    let calibrated = if kept as f64 >= 0.9 * total as f64 {
        intercept(&rows, &ar, g)?
    } else {
        match refit {
            Refit::Intercept => intercept(&rows, &ar, g)?,
            Refit::Design(z) => {
                let x = Mat::from_fn(rows.len(), z.ncols() + 1, |r, j| {
                    if j < z.ncols() {
                        z[(rows[r], j)]
                    } else {
                        g[rows[r]]
                    }
                });
                let wr: Vec<f64> = rows.iter().map(|&i| p.w[i]).collect();
                let fitted = fit_model(
                    &Glm::new(Family::Binomial, Link::Logit),
                    x.as_ref(),
                    &ar,
                    &wr,
                    "propensity (support)",
                )?;
                fitted.mean
            }
        }
    };
    Ok((rows, calibrated, true))
}

/// `plogis(c + qlogis(g))` on `rows`, with `c` the intercept of an unweighted
/// logistic regression of the treatment on `rows` with offset `qlogis(g)`.
fn intercept(rows: &[usize], a: &[f64], g: &[f64]) -> Result<Vec<f64>> {
    let offset: Vec<f64> = rows.iter().map(|&i| logit(g[i])).collect();
    let ones = Mat::from_fn(rows.len(), 1, |_, _| 1.0);
    let fit = Glm::new(Family::Binomial, Link::Logit)
        .fit(ones.as_ref(), a, None, Some(&offset), None)
        .map_err(|e| Error::input(format!("propensity recalibration: {e}")))?;
    if !fit.converged {
        return Err(Error::NoConvergence(
            "propensity recalibration: the logistic fit did not converge".into(),
        ));
    }
    let c = fit.coefficients[0].unwrap_or(0.0);
    Ok(offset.iter().map(|&l| plogis(c + l)).collect())
}

/// The state of the one-step path.
#[derive(Clone)]
struct State {
    /// `Q(A)`, `Q_0`, `Q_1`.
    q: [Vec<f64>; 3],
    g1: Vec<f64>,
    g0: Vec<f64>,
    psi: f64,
}

struct OneStep {
    state: State,
    ic: Vec<f64>,
    steps: usize,
}

/// The one-step estimator of the effect on the arm `a = 1`, on scaled data.
fn one_step(
    y: &[f64],
    a: &[f64],
    q: [Vec<f64>; 3],
    g: &[f64],
    w: &[f64],
    b: f64,
    delta: f64,
    step: f64,
) -> Result<OneStep> {
    let m = y.len();
    let mean = |v: &mut dyn Iterator<Item = f64>| compensated(v) / m as f64;
    let qq = mean(&mut (0..m).map(|i| a[i] * w[i]));
    let g1: Vec<f64> = g.iter().map(|&x| x.clamp(b, 1.0)).collect();
    let g0: Vec<f64> = g1.iter().map(|&x| (1.0 - x).clamp(b, 1.0)).collect();
    let psi_of = |q: &[Vec<f64>; 3], g1: &[f64]| {
        mean(&mut (0..m).map(|i| w[i] * (q[2][i] - q[1][i]) * g1[i] / qq))
    };
    let loss = |q: &[Vec<f64>; 3], g1: &[f64], g0: &[f64]| {
        let l = -mean(&mut (0..m).map(|i| {
            w[i] * (y[i] * q[0][i].ln()
                + (1.0 - y[i]) * (1.0 - q[0][i]).ln()
                + a[i] * g1[i].ln()
                + (1.0 - a[i]) * g0[i].ln())
        }));
        if l.is_finite() { l } else { f64::INFINITY }
    };
    let ic = |s: &State| -> Vec<f64> {
        (0..m)
            .map(|i| {
                w[i] * ((a[i] - (1.0 - a[i]) * s.g1[i] / s.g0[i]) * (y[i] - s.q[0][i])
                    + a[i] * (s.q[2][i] - s.q[1][i] - s.psi))
                    / qq
            })
            .collect()
    };
    let psi = psi_of(&q, &g1);
    let mut current = State { q, g1, g0, psi };
    let mut current_ic = ic(&current);
    let derivative = mean(&mut current_ic.iter().map(|&x| qq * x));
    let h = if derivative > 0.0 { -step } else { step };
    let mut loss_current = loss(&current.q, &current.g1, &current.g0);
    if !loss_current.is_finite() {
        return Err(Error::input(
            "one-step: the loss at the start is not finite",
        ));
    }
    let mut loss_previous = f64::INFINITY;
    let limit = (2.0 / step).max(1000.0);
    // The target that the next update of `g` uses, which lags one step.
    let mut lag = psi;
    let mut k = 0usize;
    let mut previous = current.clone();
    let mut previous_ic = current_ic.clone();
    while loss_previous > loss_current && (k as f64) < limit {
        previous.clone_from(&current);
        previous_ic.clone_from(&current_ic);
        let s = &previous;
        let mut next_q = [vec![0.0; m], vec![0.0; m], vec![0.0; m]];
        let mut g1 = vec![0.0; m];
        let mut g0 = vec![0.0; m];
        for i in 0..m {
            let gn = plogis(logit(s.g1[i]) - h * (s.q[2][i] - s.q[1][i] - lag));
            let odds = s.g1[i] / (1.0 - s.g1[i]);
            let hh = [a[i] - (1.0 - a[i]) * odds, -odds, 1.0];
            for c in 0..3 {
                next_q[c][i] = plogis(logit(s.q[c][i]) - h * hh[c]).clamp(delta, 1.0 - delta);
            }
            g0[i] = (1.0 - gn).clamp(b, 1.0);
            g1[i] = gn.clamp(b, 1.0);
        }
        lag = current.psi;
        let psi = psi_of(&next_q, &g1);
        current = State {
            q: next_q,
            g1,
            g0,
            psi,
        };
        loss_previous = loss_current;
        loss_current = loss(&current.q, &current.g1, &current.g0);
        current_ic = ic(&current);
        k += 1;
    }
    if loss_previous > loss_current {
        return Err(Error::NoConvergence(format!(
            "one-step: the loss still falls after {k} steps of {step}"
        )));
    }
    Ok(OneStep {
        state: previous,
        ic: previous_ic,
        steps: k - 1,
    })
}

fn on_arm(
    p: &Prepared<'_>,
    q1: &[f64],
    q0: &[f64],
    g: &[f64],
    target: Target,
    refit: Refit<'_>,
    opts: &TmleOptions,
) -> Result<Tmle> {
    let n = g.len();
    let (rows, calibrated, recalibrated) = support(p, g, target, refit, opts)?;
    let m = rows.len();
    let treated_arm = target == Target::Att;
    // The arm whose effect is estimated is the arm `a = 1` of the one-step path.
    let arm = |i: usize| p.treatment[i] == treated_arm;
    let a: Vec<f64> = rows.iter().map(|&i| flag(arm(i))).collect();
    let y: Vec<f64> = rows.iter().map(|&i| p.ystar[i]).collect();
    let sum_rows = compensated(rows.iter().map(|&i| p.w[i]));
    let w: Vec<f64> = rows.iter().map(|&i| p.w[i] / sum_rows * m as f64).collect();
    let pick = |v: &[f64], i: usize| plogis(logit(v[i]));
    let (own, other) = if treated_arm { (q1, q0) } else { (q0, q1) };
    let q = [
        rows.iter()
            .map(|&i| pick(if p.treatment[i] { q1 } else { q0 }, i))
            .collect(),
        rows.iter().map(|&i| pick(other, i)).collect(),
        rows.iter().map(|&i| pick(own, i)).collect(),
    ];
    let ga: Vec<f64> = (0..m)
        .map(|r| {
            if treated_arm {
                calibrated[r]
            } else {
                1.0 - calibrated[r]
            }
        })
        .collect();
    let n_trimmed = ga
        .iter()
        .filter(|&&x| x < p.bound || 1.0 - x.clamp(p.bound, 1.0) < p.bound)
        .count();
    let path = one_step(&y, &a, q, &ga, &w, p.bound, p.delta, opts.step)?;
    let sign = if treated_arm { 1.0 } else { -1.0 };
    let effect = sign * path.state.psi * p.width;
    let mut influence = vec![0.0; n];
    for (r, &i) in rows.iter().enumerate() {
        influence[i] = sign * path.ic[r] * p.width / m as f64;
    }
    // The mean of the arm of the estimand is the weighted mean of its outcomes.
    let size = compensated((0..m).map(|r| w[r] * a[r]));
    let own_mean = compensated((0..m).map(|r| w[r] * a[r] * p.outcome[rows[r]])) / size;
    let mut own_if = vec![0.0; n];
    for (r, &i) in rows.iter().enumerate() {
        own_if[i] = w[r] * a[r] * (p.outcome[i] - own_mean) / size;
    }
    let (means, if1, if0) = if treated_arm {
        let if0: Vec<f64> = (0..n).map(|i| own_if[i] - influence[i]).collect();
        ([own_mean, own_mean - effect], own_if, if0)
    } else {
        let if1: Vec<f64> = (0..n).map(|i| own_if[i] + influence[i]).collect();
        ([own_mean + effect, own_mean], if1, own_if)
    };
    let back = |x: f64| x * p.width + p.lower;
    let (mut t1, mut t0) = (vec![None; n], vec![None; n]);
    for (r, &i) in rows.iter().enumerate() {
        let (s_own, s_other) = (back(path.state.q[2][r]), back(path.state.q[1][r]));
        if treated_arm {
            (t1[i], t0[i]) = (Some(s_own), Some(s_other));
        } else {
            (t1[i], t0[i]) = (Some(s_other), Some(s_own));
        }
    }
    let used = Some(rows.as_slice());
    finish(
        p,
        target,
        means,
        [&if1, &if0],
        used,
        (t1, t0),
        Targeting::OneStep {
            steps: path.steps,
            recalibrated,
        },
        n_trimmed,
        m,
        opts,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "the parts of the result that the two targetings compute differently"
)]
fn finish(
    p: &Prepared<'_>,
    target: Target,
    values: [f64; 2],
    influence: [&[f64]; 2],
    rows: Option<&[usize]>,
    targeted: (Vec<Option<f64>>, Vec<Option<f64>>),
    targeting: Targeting,
    n_trimmed: usize,
    n_used: usize,
    opts: &TmleOptions,
) -> Result<Tmle> {
    let n = p.w.len();
    let diff: Vec<f64> = (0..n).map(|i| influence[0][i] - influence[1][i]).collect();
    // The variance over the persons used, so that clusters without them do not
    // count.
    let variance = |v: &[f64]| -> Result<f64> {
        match rows {
            None => variance_of(v, p.cluster),
            Some(rows) => {
                let sub: Vec<f64> = rows.iter().map(|&i| v[i]).collect();
                let cl: Option<Vec<u32>> = p.cluster.map(|c| rows.iter().map(|&i| c[i]).collect());
                variance_of(&sub, cl.as_deref())
            }
        }
    };
    let mean1 = estimate(values[0], variance(influence[0])?, 0.0, opts.alpha)?;
    let mean0 = estimate(values[1], variance(influence[1])?, 0.0, opts.alpha)?;
    let difference = estimate(values[0] - values[1], variance(&diff)?, 0.0, opts.alpha)?;
    let ratio = if values[1] == 0.0 {
        None
    } else {
        let r = values[0] / values[1];
        let if_r: Vec<f64> = (0..n)
            .map(|i| (influence[0][i] - r * influence[1][i]) / values[1])
            .collect();
        Some(estimate(r, variance(&if_r)?, 1.0, opts.alpha)?)
    };
    let score = compensated(diff.iter().copied());
    Ok(Tmle {
        target,
        mean1,
        mean0,
        difference,
        ratio,
        influence: diff,
        targeted1: targeted.0,
        targeted0: targeted.1,
        targeting,
        score,
        range: (p.lower, p.lower + p.width),
        propensity_bound: p.bound,
        n_trimmed,
        n_used,
    })
}

/// The targeted estimator on given predictions of the outcome and propensity
/// models, with the variance of the efficient influence function.
///
/// The predictions are mapped to `[0, 1]` by the range of
/// [`TmleOptions::range`] and bounded by [`TmleOptions::outcome_bound`], as
/// `tmle` does with predictions it is given. For the effect on the treated or on
/// the controls the support rule recalibrates the propensity by an intercept in
/// both of its cases.
///
/// # Errors
///
/// [`Error::Input`] for inputs of different lengths, no persons, a value that is
/// not finite, a negative weight, a propensity outside `(0, 1)`, an outcome
/// outside the known range or with one value, options outside their ranges, a
/// group without persons of positive weight and fewer than two clusters;
/// [`Error::NoConvergence`] for a fluctuation, a recalibration or a one-step path
/// that does not converge.
pub fn tmle(input: &TmleInput<'_>, target: Target, opts: &TmleOptions) -> Result<Tmle> {
    let p = prepare(
        input.outcome,
        input.treatment,
        input.weight,
        input.cluster,
        opts,
    )?;
    let n = p.w.len();
    check_len("mu1", input.mu1.len(), n)?;
    check_len("mu0", input.mu0.len(), n)?;
    check_len("propensity", input.propensity.len(), n)?;
    for (name, v) in [("mu1", input.mu1), ("mu0", input.mu0)] {
        if let Some(i) = v.iter().position(|x| !x.is_finite()) {
            return Err(Error::input(format!("{name}: value {i} is not finite")));
        }
    }
    check_propensity(input.propensity)?;
    let scale = |v: &[f64]| -> Vec<f64> {
        v.iter()
            .map(|&x| ((x - p.lower) / p.width).clamp(p.delta, 1.0 - p.delta))
            .collect()
    };
    let (q1, q0) = (scale(input.mu1), scale(input.mu0));
    targeted(
        &p,
        &q1,
        &q0,
        input.propensity,
        target,
        Refit::Intercept,
        opts,
    )
}

/// The targeted estimator with generalised linear models for the outcome and a
/// logistic model for the treatment, with the variance of the efficient influence
/// function.
///
/// The outcome model is fitted to the scaled outcome `Y*`, logistic for
/// [`Family::Binomial`] and linear for [`Family::Gaussian`], as `tmle` with
/// `Q.SL.library = "SL.glm"` and `cvQinit = FALSE`; [`OutcomeModel::Pooled`] is
/// `tmle`'s main-terms model. The propensity model is a logistic regression of
/// the treatment on `propensity_design`. Both get the case weights rescaled to
/// mean 1.
///
/// # Errors
///
/// The errors of [`tmle`], and [`Error::Input`] for a design with another number
/// of rows than the data or without columns, a family other than the two above,
/// a model that does not converge or has an aliased column, and a propensity
/// model whose fitted probabilities are numerically 0 or 1.
pub fn tmle_glm(
    input: &GlmInput<'_>,
    family: Family,
    model: OutcomeModel,
    target: Target,
    opts: &TmleOptions,
) -> Result<Tmle> {
    let p = prepare(
        input.outcome,
        input.treatment,
        input.weight,
        input.cluster,
        opts,
    )?;
    let n = p.w.len();
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
    let link = match family {
        Family::Binomial => Link::Logit,
        Family::Gaussian => Link::Identity,
        _ => {
            return Err(Error::input(
                "family: the outcome model is binomial or Gaussian",
            ));
        }
    };
    let a: Vec<f64> = p.treatment.iter().map(|&t| flag(t)).collect();
    let z = input.propensity_design;
    let propensity = fit_model(
        &Glm::new(Family::Binomial, Link::Logit),
        z,
        &a,
        &p.w,
        "propensity",
    )?;
    let eps10 = 10.0 * f64::EPSILON;
    if let Some(i) = propensity
        .mean
        .iter()
        .position(|&g| g < eps10 || g > 1.0 - eps10)
    {
        return Err(Error::input(format!(
            "propensity model: person {i} has fitted probability {} (separation)",
            propensity.mean[i]
        )));
    }
    let glm = Glm::new(family, link);
    let x = input.outcome_design;
    let (q1, q0) = match model {
        OutcomeModel::Pooled => {
            let k = x.ncols();
            let design = Mat::from_fn(n, k + 1, |i, j| if j < k { x[(i, j)] } else { a[i] });
            let fit = glm
                .fit(design.as_ref(), &p.ystar, Some(&p.w), None, None)
                .map_err(|e| Error::input(format!("outcome model: {e}")))?;
            if !fit.converged {
                return Err(Error::input("outcome model: no convergence"));
            }
            let mut beta = Vec::with_capacity(k + 1);
            for (j, c) in fit.coefficients.iter().enumerate() {
                beta.push(c.ok_or_else(|| {
                    Error::input(format!("outcome model: design column {j} is aliased"))
                })?);
            }
            let eta = |i: usize| -> f64 { (0..k).map(|j| x[(i, j)] * beta[j]).sum() };
            let q1: Vec<f64> = (0..n).map(|i| link.inverse(eta(i) + beta[k])).collect();
            let q0: Vec<f64> = (0..n).map(|i| link.inverse(eta(i))).collect();
            (q1, q0)
        }
        OutcomeModel::ByArm => {
            let arm = |treated: bool| -> Vec<f64> {
                (0..n)
                    .map(|i| {
                        if p.treatment[i] == treated {
                            p.w[i]
                        } else {
                            0.0
                        }
                    })
                    .collect()
            };
            let m1 = fit_model(&glm, x, &p.ystar, &arm(true), "outcome (treated)")?;
            let m0 = fit_model(&glm, x, &p.ystar, &arm(false), "outcome (controls)")?;
            (m1.mean, m0.mean)
        }
    };
    let bound = |v: Vec<f64>| -> Vec<f64> {
        v.into_iter()
            .map(|q| q.clamp(p.delta, 1.0 - p.delta))
            .collect()
    };
    let (q1, q0) = (bound(q1), bound(q0));
    targeted(
        &p,
        &q1,
        &q0,
        &propensity.mean,
        target,
        Refit::Design(z),
        opts,
    )
}

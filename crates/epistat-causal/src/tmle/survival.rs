//! Targeted maximum likelihood estimation of the risk by a time with right
//! censoring, in discrete time.
//!
//! # Algorithm
//!
//! Follow-up is in intervals `1, 2, ...`. A person is observed through interval
//! `T` and has the event in it or is censored after it. With treatment `a`,
//! covariates `x`, the event hazard `lambda_k(a, x)` (the probability of the
//! event in interval `k` given no event before it and no censoring before it),
//! the censoring hazard `c_k(a, x)` (the probability of censoring after interval
//! `k` given no event up to and in it), the propensity `g(x)` with `g_1 = g`
//! and `g_0 = 1 - g`, and the horizon `t`:
//!
//! ```text
//! S(k | a, x) = prod_{j <= k} (1 - lambda_j(a, x))
//! G(k | a, x) = prod_{j <= k} (1 - c_j(a, x)),        G(0) = 1
//! H_k(a, x)   = -S(t | a, x) / ( S(k | a, x) max(g_a(x) G(k - 1 | a, x), b) )
//! ```
//!
//! for `k <= t`, with `b` the bound of [`SurvivalOptions::propensity_bound`]. The
//! efficient influence function of the survival `S_a(t) = E[S(t | a, X)]` is
//!
//! ```text
//! D_a = 1(A = a) sum_{k <= min(T, t)} H_k(a, X) (dN_k - lambda_k(a, X)) + S(t | a, X) - S_a(t)
//! ```
//!
//! with `dN_k` the event indicator of interval `k` (Moore and van der Laan
//! 2009). For each arm, the hazards up to `t` are moved by
//!
//! ```text
//! logit lambda*_k(a, x) = logit lambda_k(a, x) + e H_k(a, x)
//! ```
//!
//! with `e` the coefficient of a logistic regression of `dN_k` on `H_k` with
//! offset `logit lambda_k`, without intercept and with the case weights, over the
//! person-intervals at risk (`A = a`, `k <= min(T, t)`). Since `H` depends on the
//! hazards after `k`, this is repeated, each time with `H` from the moved
//! hazards, until the mean of the first term of `D_a` is at most `se_a / (sqrt(n)
//! ln n)`, with `se_a` its standard error below. The risk is the plug-in `F_a(t)
//! = 1 - sum w S*(t | a, x) / sum w`, which lies in `[0, 1]`, with the influence
//! `-w D_a / sum w` per person.
//!
//! The arms are targeted separately, each with its own `e`. Without censoring and
//! with `t = 1` the estimator is the targeted estimator of the effect in the
//! population of [`crate::tmle`] on a binary outcome, when no prediction is at
//! its outcome bound.
//!
//! # Variance
//!
//! As in [`crate::tmle`]: the sum of squared influences, with clusters summed
//! first and the factor `G / (G - 1)`. The difference and the ratio of the risks
//! have the influences `IF_1 - IF_0` and `(IF_1 - ratio IF_0) / F_0(t)`.
//!
//! # Edge cases
//!
//! An event hazard of 0 or 1 or outside the open interval up to the horizon, a
//! censoring hazard outside `[0, 1)` before it, a propensity outside `(0, 1)`, a
//! follow-up of 0 intervals, an arm without persons of positive weight, a
//! fluctuation that does not converge and targeting that has not met its rule
//! after [`SurvivalOptions::max_iterations`] rounds are errors. The ratio is
//! `None` when the risk of the controls is 0.
//!
//! # Fitting the models
//!
//! [`tmle_survival_glm`] fits the three models and passes their predictions to
//! [`tmle_survival`]. With `T_i` the last interval of person `i` and `K` the
//! largest `T_i`:
//!
//! ```text
//! event:      logit lambda_k = x_e(i, k, A_i)' b    on the intervals k <= T_i,
//!             with the response 1 in k = T_i when the person has the event;
//! censoring:  logit c_k = x_c(i, k, A_i)' d         on the intervals k <= T_i
//!             without the event and k < K, with the response 1 in k = T_i;
//! treatment:  logit g = z_i' a                       on the persons,
//! ```
//!
//! all with the case weights rescaled to mean 1, as `glm` fits them. Censoring after the last
//! interval `K` is the end of follow-up and is not modelled. The predictions
//! are `lambda_k(a, x)` and `c_k(a, x)` from the designs with the treatment set
//! to `a`, for `k` up to the horizon. The variance treats the fitted models as
//! the predictions of [`tmle_survival`] do, through the efficient influence
//! function, which is valid when the models are fitted on the same data and
//! the hazard model is correct or the propensity and censoring models are.
//!
//! # References
//!
//! 1. Moore KL, van der Laan MJ. Application of time-to-event methods in the
//!    assessment of safety in clinical trials. In: Peace KE, editor. Design and
//!    analysis of clinical trials with time-to-event endpoints. Boca Raton:
//!    Chapman and Hall/CRC; 2009. p. 455-82.
//! 2. Stitelman OM, De Gruttola V, van der Laan MJ. A general implementation of
//!    TMLE for longitudinal data applied to causal inference in survival analysis.
//!    Int J Biostat. 2012;8(1):Article 1.
//! 3. Benkeser D, Carone M, Gilbert PB. Improved estimation of the cumulative
//!    incidence of rare outcomes. Stat Med. 2018;37(2):280-93.

use epistat_glm::{Family, Glm, Link};
use epistat_linalg::{Mat, MatRef};

use super::{PropensityBound, logit, plogis};
use crate::aipw::{Estimate, check_alpha, check_len, compensated, estimate, variance_of};
use crate::error::{Error, Result};

/// The data and the predictions of [`tmle_survival`]. Column `k - 1` of a
/// matrix of predictions is interval `k`; columns after those used are ignored.
#[derive(Debug, Clone, Copy)]
pub struct SurvivalInput<'a> {
    /// The last interval in which each person was observed, at least 1.
    pub time: &'a [usize],
    /// Whether the person had the event in that interval; otherwise the person
    /// was censored after it.
    pub event: &'a [bool],
    /// Whether each person is treated.
    pub treatment: &'a [bool],
    /// The event hazard of each person and interval had the person been treated,
    /// with at least `horizon` columns.
    pub hazard1: MatRef<'a, f64>,
    /// The event hazard had the person not been treated.
    pub hazard0: MatRef<'a, f64>,
    /// The censoring hazard of each person and interval had the person been
    /// treated, with at least `horizon - 1` columns.
    pub censoring1: MatRef<'a, f64>,
    /// The censoring hazard had the person not been treated.
    pub censoring0: MatRef<'a, f64>,
    /// `P(A = 1 | X)` of each person.
    pub propensity: &'a [f64],
    /// Case weights, 1 for each person when `None`.
    pub weight: Option<&'a [f64]>,
    /// The cluster of each person, each person a cluster when `None`.
    pub cluster: Option<&'a [u32]>,
}

/// What [`tmle_survival`] does besides the data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurvivalOptions {
    /// The interval `t` by whose end the risk is estimated, at least 1.
    pub horizon: usize,
    /// One minus the level of the confidence limits. The default is 0.05.
    pub alpha: f64,
    /// The lower bound `b` of `g_a G(k - 1)`. The default is none.
    pub propensity_bound: PropensityBound,
    /// The largest number of targeting rounds of each arm. The default is 50.
    pub max_iterations: usize,
}

impl SurvivalOptions {
    /// The options for the risk by the end of interval `horizon`, with the
    /// defaults otherwise.
    #[must_use]
    pub const fn new(horizon: usize) -> Self {
        Self {
            horizon,
            alpha: 0.05,
            propensity_bound: PropensityBound::None,
            max_iterations: 50,
        }
    }
}

/// The result of [`tmle_survival`].
#[derive(Debug, Clone)]
pub struct SurvivalTmle {
    /// The risk by the horizon had everyone been treated.
    pub risk1: Estimate,
    /// The risk had no one been treated.
    pub risk0: Estimate,
    /// The difference `risk1 - risk0`.
    pub difference: Estimate,
    /// The ratio `risk1 / risk0`, `None` when `risk0` is 0.
    pub ratio: Option<Estimate>,
    /// The influence function of the difference, one value per person.
    pub influence: Vec<f64>,
    /// The targeting rounds of the treated and of the control arm.
    pub iterations: [usize; 2],
    /// The fluctuation coefficients of the rounds of the two arms.
    pub epsilon: [Vec<f64>; 2],
    /// The mean of the first term of the influence function of each risk at the
    /// end, which targeting brings below `se / (sqrt(n) ln n)`.
    pub score: [f64; 2],
    /// The bound `b`, 0 when there is none.
    pub propensity_bound: f64,
    /// The number of persons whose `g_a G(k - 1)` was below the bound for an arm
    /// and an interval up to the horizon.
    pub n_trimmed: usize,
}

struct Arm {
    risk: f64,
    influence: Vec<f64>,
    iterations: usize,
    epsilon: Vec<f64>,
    score: f64,
}

struct Common<'a> {
    input: &'a SurvivalInput<'a>,
    w: Vec<f64>,
    total: f64,
    horizon: usize,
    bound: f64,
    max_iterations: usize,
}

fn target_arm(c: &Common<'_>, treated: bool) -> Result<Arm> {
    let input = c.input;
    let n = c.w.len();
    let t = c.horizon;
    let (hazard, censoring) = if treated {
        (input.hazard1, input.censoring1)
    } else {
        (input.hazard0, input.censoring0)
    };
    let mut lam: Vec<f64> = (0..n * t).map(|ik| hazard[(ik / t, ik % t)]).collect();
    let mut den = vec![0.0; n * t];
    for i in 0..n {
        let ga = if treated {
            input.propensity[i]
        } else {
            1.0 - input.propensity[i]
        };
        let mut g = 1.0;
        for k in 0..t {
            if k > 0 {
                g *= 1.0 - censoring[(i, k - 1)];
            }
            den[i * t + k] = (ga * g).max(c.bound);
        }
    }
    // The person-intervals at risk in this arm, and their event indicators.
    let mut rows = Vec::new();
    let mut dn = Vec::new();
    for i in 0..n {
        if input.treatment[i] == treated && c.w[i] > 0.0 {
            for k in 0..t.min(input.time[i]) {
                rows.push(i * t + k);
                dn.push(f64::from(u8::from(
                    input.event[i] && input.time[i] == k + 1,
                )));
            }
        }
    }
    let rule = (n as f64).sqrt() * (n as f64).ln();
    let mut epsilon = Vec::new();
    loop {
        let mut s = vec![0.0; n * t];
        for i in 0..n {
            let mut acc = 1.0;
            for k in 0..t {
                acc *= 1.0 - lam[i * t + k];
                s[i * t + k] = acc;
            }
        }
        let h: Vec<f64> = (0..n * t)
            .map(|ik| -s[(ik / t) * t + t - 1] / (s[ik] * den[ik]))
            .collect();
        let mut first = vec![0.0; n];
        for (r, &ik) in rows.iter().enumerate() {
            first[ik / t] += h[ik] * (dn[r] - lam[ik]);
        }
        let psi = compensated((0..n).map(|i| c.w[i] * s[i * t + t - 1])) / c.total;
        let influence: Vec<f64> = (0..n)
            .map(|i| c.w[i] * (first[i] + s[i * t + t - 1] - psi) / c.total)
            .collect();
        let se = variance_of(&influence, input.cluster)?.sqrt();
        let score = compensated((0..n).map(|i| c.w[i] * first[i])) / c.total;
        if score.abs() <= se / rule {
            return Ok(Arm {
                risk: 1.0 - psi,
                influence: influence.iter().map(|x| -x).collect(),
                iterations: epsilon.len(),
                epsilon,
                score: -score,
            });
        }
        if epsilon.len() == c.max_iterations {
            return Err(Error::NoConvergence(format!(
                "targeting: the mean of the influence function is {score:e} after {} rounds",
                c.max_iterations
            )));
        }
        let design = Mat::from_fn(rows.len(), 1, |r, _| h[rows[r]]);
        let offset: Vec<f64> = rows.iter().map(|&ik| logit(lam[ik])).collect();
        let weights: Vec<f64> = rows.iter().map(|&ik| c.w[ik / t]).collect();
        let fit = Glm::new(Family::Binomial, Link::Logit)
            .fit(design.as_ref(), &dn, Some(&weights), Some(&offset), None)
            .map_err(|e| Error::input(format!("fluctuation: {e}")))?;
        if !fit.converged {
            return Err(Error::NoConvergence(
                "fluctuation: the logistic fit did not converge".into(),
            ));
        }
        let e = fit.coefficients[0].unwrap_or(0.0);
        for (l, &hk) in lam.iter_mut().zip(&h) {
            *l = plogis(logit(*l) + e * hk);
        }
        epsilon.push(e);
    }
}

fn check_matrix(
    name: &str,
    m: MatRef<'_, f64>,
    n: usize,
    columns: usize,
    valid: impl Fn(f64) -> bool,
) -> Result<()> {
    if m.nrows() != n || m.ncols() < columns {
        return Err(Error::input(format!(
            "{name}: {} by {} for {n} persons and {columns} intervals",
            m.nrows(),
            m.ncols()
        )));
    }
    for i in 0..n {
        for k in 0..columns {
            if !valid(m[(i, k)]) {
                return Err(Error::input(format!(
                    "{name}: person {i}, interval {} has {}",
                    k + 1,
                    m[(i, k)]
                )));
            }
        }
    }
    Ok(())
}

/// The targeted estimator of the risk by the end of interval
/// [`SurvivalOptions::horizon`] in each arm, on given predictions of the event
/// and censoring hazards and of the propensity, with the variance of the
/// efficient influence function.
///
/// # Errors
///
/// [`Error::Input`] for inputs of different lengths, no persons, a horizon of 0,
/// a follow-up of 0 intervals, predictions outside their ranges (see the module
/// documentation), a negative or non-finite weight, an arm without persons of
/// positive weight, options outside their ranges and fewer than two clusters;
/// [`Error::NoConvergence`] for a fluctuation that does not converge or
/// targeting that does not meet its rule.
pub fn tmle_survival(input: &SurvivalInput<'_>, opts: &SurvivalOptions) -> Result<SurvivalTmle> {
    let n = input.treatment.len();
    if n == 0 {
        return Err(Error::input("treatment: no persons"));
    }
    check_alpha(opts.alpha)?;
    let t = opts.horizon;
    if t == 0 {
        return Err(Error::input("horizon: 0 intervals"));
    }
    check_len("time", input.time.len(), n)?;
    check_len("event", input.event.len(), n)?;
    check_len("propensity", input.propensity.len(), n)?;
    if let Some(w) = input.weight {
        check_len("weight", w.len(), n)?;
    }
    if let Some(c) = input.cluster {
        check_len("cluster", c.len(), n)?;
    }
    if let Some(i) = input.time.iter().position(|&k| k == 0) {
        return Err(Error::input(format!("time: person {i} has 0 intervals")));
    }
    let open = |x: f64| x > 0.0 && x < 1.0;
    check_matrix("hazard1", input.hazard1, n, t, open)?;
    check_matrix("hazard0", input.hazard0, n, t, open)?;
    let censoring = |x: f64| (0.0..1.0).contains(&x);
    check_matrix("censoring1", input.censoring1, n, t - 1, censoring)?;
    check_matrix("censoring0", input.censoring0, n, t - 1, censoring)?;
    if let Some(i) = input.propensity.iter().position(|&p| !open(p)) {
        return Err(Error::input(format!(
            "propensity: person {i} has {}, outside (0, 1)",
            input.propensity[i]
        )));
    }
    let raw: Vec<f64> = input.weight.map_or_else(|| vec![1.0; n], <[f64]>::to_vec);
    if let Some(i) = raw.iter().position(|&w| !(w.is_finite() && w >= 0.0)) {
        return Err(Error::input(format!(
            "weight: value {i} is negative or not finite"
        )));
    }
    let (mut treated, mut controls) = (false, false);
    for (&w, &a) in raw.iter().zip(input.treatment) {
        if w > 0.0 {
            treated |= a;
            controls |= !a;
        }
    }
    if !(treated && controls) {
        return Err(Error::input("treatment: both groups need persons"));
    }
    let bound = match opts.propensity_bound {
        PropensityBound::None => 0.0,
        PropensityBound::Tmle => {
            let m = n as f64;
            5.0 / m.sqrt() / m.ln()
        }
        PropensityBound::At(b) => b,
    };
    if !(0.0..0.5).contains(&bound) {
        return Err(Error::input(format!(
            "propensity_bound: {bound} is not in (0, 0.5)"
        )));
    }
    let sum = compensated(raw.iter().copied());
    let w: Vec<f64> = raw.iter().map(|&x| x / sum * n as f64).collect();
    let common = Common {
        input,
        total: compensated(w.iter().copied()),
        w,
        horizon: t,
        bound,
        max_iterations: opts.max_iterations,
    };
    let n_trimmed = (0..n)
        .filter(|&i| {
            [
                (input.propensity[i], input.censoring1),
                (1.0 - input.propensity[i], input.censoring0),
            ]
            .iter()
            .any(|&(ga, cens)| {
                let mut g = ga;
                (0..t).any(|k| {
                    if k > 0 {
                        g *= 1.0 - cens[(i, k - 1)];
                    }
                    g < bound
                })
            })
        })
        .count();
    let one = target_arm(&common, true)?;
    let zero = target_arm(&common, false)?;
    let diff: Vec<f64> = (0..n)
        .map(|i| one.influence[i] - zero.influence[i])
        .collect();
    let cluster = input.cluster;
    let risk1 = estimate(
        one.risk,
        variance_of(&one.influence, cluster)?,
        0.0,
        opts.alpha,
    )?;
    let risk0 = estimate(
        zero.risk,
        variance_of(&zero.influence, cluster)?,
        0.0,
        opts.alpha,
    )?;
    let difference = estimate(
        one.risk - zero.risk,
        variance_of(&diff, cluster)?,
        0.0,
        opts.alpha,
    )?;
    let ratio = if zero.risk == 0.0 {
        None
    } else {
        let r = one.risk / zero.risk;
        let inf: Vec<f64> = (0..n)
            .map(|i| (one.influence[i] - r * zero.influence[i]) / zero.risk)
            .collect();
        Some(estimate(r, variance_of(&inf, cluster)?, 1.0, opts.alpha)?)
    };
    Ok(SurvivalTmle {
        risk1,
        risk0,
        difference,
        ratio,
        influence: diff,
        iterations: [one.iterations, zero.iterations],
        epsilon: [one.epsilon, zero.epsilon],
        score: [one.score, zero.score],
        propensity_bound: bound,
        n_trimmed,
    })
}

/// The design of a model of a hazard in discrete time, formed for a person and
/// an interval when it is needed.
#[derive(Clone, Copy)]
pub struct IntervalDesign<'a> {
    /// The number of columns, with the intercept.
    pub columns: usize,
    /// Writes into `out` the design of person `i` in interval `k` (from 1)
    /// had the person been treated (`treated`) or not.
    pub fill: &'a IntervalFill<'a>,
}

/// The function of an [`IntervalDesign`]: person, interval, treated, output.
pub type IntervalFill<'a> = dyn Fn(usize, usize, bool, &mut [f64]) + Sync + 'a;

impl std::fmt::Debug for IntervalDesign<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IntervalDesign")
            .field("columns", &self.columns)
            .finish_non_exhaustive()
    }
}

/// The data and the models of [`tmle_survival_glm`].
#[derive(Debug, Clone, Copy)]
pub struct SurvivalGlmInput<'a> {
    /// The last interval in which each person was observed, at least 1.
    pub time: &'a [usize],
    /// Whether the person had the event in that interval; otherwise the person
    /// was censored after it.
    pub event: &'a [bool],
    /// Whether each person is treated.
    pub treatment: &'a [bool],
    /// The design of the logistic model of the event hazard.
    pub hazard: IntervalDesign<'a>,
    /// The design of the logistic model of the censoring hazard, `None` when
    /// no one is censored before the last interval, which makes every
    /// censoring hazard 0.
    pub censoring: Option<IntervalDesign<'a>>,
    /// The design of the logistic model of the treatment, one row per person,
    /// with an intercept.
    pub propensity: MatRef<'a, f64>,
    /// Case weights, 1 for each person when `None`.
    pub weight: Option<&'a [f64]>,
    /// The cluster of each person, each person a cluster when `None`.
    pub cluster: Option<&'a [u32]>,
}

/// The result of [`tmle_survival_glm`]: the estimator and the fitted models
/// with their predictions.
#[derive(Debug, Clone)]
pub struct SurvivalTmleGlm {
    /// The targeted estimator on the predictions.
    pub tmle: SurvivalTmle,
    /// The coefficients of the event hazard model.
    pub hazard_coefficients: Vec<f64>,
    /// The coefficients of the censoring hazard model, empty without one.
    pub censoring_coefficients: Vec<f64>,
    /// The coefficients of the propensity model.
    pub propensity_coefficients: Vec<f64>,
    /// The event hazard of each person (rows) and interval up to the horizon
    /// (columns) had the person been treated.
    pub hazard1: Mat<f64>,
    /// The same had the person not been treated.
    pub hazard0: Mat<f64>,
    /// The censoring hazard of each person and interval before the horizon had
    /// the person been treated, at least one column.
    pub censoring1: Mat<f64>,
    /// The same had the person not been treated.
    pub censoring0: Mat<f64>,
    /// The fitted propensity of each person.
    pub propensity: Vec<f64>,
}

fn fit_logistic(x: MatRef<'_, f64>, y: &[f64], w: &[f64], what: &str) -> Result<Vec<f64>> {
    let fit = Glm::new(Family::Binomial, Link::Logit)
        .fit(x, y, Some(w), None, None)
        .map_err(|e| Error::input(format!("{what} model: {e}")))?;
    if !fit.converged {
        return Err(Error::NoConvergence(format!("{what} model")));
    }
    fit.coefficients
        .iter()
        .enumerate()
        .map(|(j, c)| {
            c.ok_or_else(|| Error::input(format!("{what} model: design column {j} is aliased")))
        })
        .collect()
}

/// The person-intervals of a hazard model as a design, a response and weights.
fn intervals(
    design: IntervalDesign<'_>,
    rows: &[(usize, usize, bool)],
    treatment: &[bool],
    w: &[f64],
    what: &str,
) -> Result<(Mat<f64>, Vec<f64>, Vec<f64>)> {
    if design.columns == 0 {
        return Err(Error::input(format!("{what} design: no columns")));
    }
    let mut x = Mat::<f64>::zeros(rows.len(), design.columns);
    let mut values = vec![0.0; design.columns];
    for (r, &(i, k, _)) in rows.iter().enumerate() {
        (design.fill)(i, k, treatment[i], &mut values);
        for (j, &v) in values.iter().enumerate() {
            if !v.is_finite() {
                return Err(Error::input(format!(
                    "{what} design: person {i}, interval {k} has {v}"
                )));
            }
            x[(r, j)] = v;
        }
    }
    let y = rows
        .iter()
        .map(|&(_, _, y)| f64::from(u8::from(y)))
        .collect();
    let weight = rows.iter().map(|&(i, _, _)| w[i]).collect();
    Ok((x, y, weight))
}

/// The predictions of a hazard model with coefficients `b` for every person
/// and the intervals `1..=columns` under treatment `treated`.
fn predictions(
    design: IntervalDesign<'_>,
    b: &[f64],
    n: usize,
    columns: usize,
    treated: bool,
) -> Mat<f64> {
    let mut values = vec![0.0; design.columns];
    let mut m = Mat::<f64>::zeros(n, columns);
    for i in 0..n {
        for k in 0..columns {
            (design.fill)(i, k + 1, treated, &mut values);
            let eta: f64 = values.iter().zip(b).map(|(x, c)| x * c).sum();
            m[(i, k)] = plogis(eta);
        }
    }
    m
}

/// [`tmle_survival`] with the event and censoring hazards and the propensity
/// from logistic models fitted to the data (see the module documentation), as
/// [`crate::tmle::tmle_glm`] does for a binary outcome.
///
/// # Errors
///
/// The errors of [`tmle_survival`], and [`Error::Input`] for inputs of
/// different lengths, a design without columns or with a value that is not
/// finite, a person censored before the last interval without a censoring
/// model, and a model that has an aliased column; [`Error::NoConvergence`] for
/// a model that does not converge.
pub fn tmle_survival_glm(
    input: &SurvivalGlmInput<'_>,
    opts: &SurvivalOptions,
) -> Result<SurvivalTmleGlm> {
    let n = input.treatment.len();
    check_len("time", input.time.len(), n)?;
    check_len("event", input.event.len(), n)?;
    if let Some(w) = input.weight {
        check_len("weight", w.len(), n)?;
    }
    if input.propensity.nrows() != n || input.propensity.ncols() == 0 {
        return Err(Error::input(format!(
            "propensity design: {} rows and {} columns for {n} persons",
            input.propensity.nrows(),
            input.propensity.ncols()
        )));
    }
    if !epistat_linalg::is_finite(input.propensity) {
        return Err(Error::input("propensity design: a value is not finite"));
    }
    if let Some(i) = input.time.iter().position(|&k| k == 0) {
        return Err(Error::input(format!("time: person {i} has 0 intervals")));
    }
    let raw: Vec<f64> = input.weight.map_or_else(|| vec![1.0; n], <[f64]>::to_vec);
    if let Some(i) = raw.iter().position(|&v| !(v.is_finite() && v >= 0.0)) {
        return Err(Error::input(format!(
            "weight: value {i} is negative or not finite"
        )));
    }
    // The fits take the weights rescaled to mean 1, as the estimator does:
    // the coefficients do not depend on the scale, but the stopping rule of
    // IRLS, a relative change of the deviance with 0.1 added, does.
    let sum = compensated(raw.iter().copied());
    let w: Vec<f64> = raw.iter().map(|&x| x / sum * n as f64).collect();
    let last = input.time.iter().copied().max().unwrap_or(0);
    let t = opts.horizon;
    let mut event_rows = Vec::new();
    let mut censoring_rows = Vec::new();
    for i in 0..n {
        let end = input.time[i];
        for k in 1..=end {
            let at_end = k == end;
            event_rows.push((i, k, at_end && input.event[i]));
            if !(at_end && input.event[i]) && k < last {
                censoring_rows.push((i, k, at_end));
            }
        }
    }
    let (x, y, wr) = intervals(input.hazard, &event_rows, input.treatment, &w, "event")?;
    let hazard_coefficients = fit_logistic(x.as_ref(), &y, &wr, "event")?;
    drop(x);
    let cens_columns = t.saturating_sub(1).max(1);
    let (censoring_coefficients, censoring1, censoring0) = if let Some(design) = input.censoring {
        let (x, y, wr) = intervals(design, &censoring_rows, input.treatment, &w, "censoring")?;
        let d = fit_logistic(x.as_ref(), &y, &wr, "censoring")?;
        let c1 = predictions(design, &d, n, cens_columns, true);
        let c0 = predictions(design, &d, n, cens_columns, false);
        (d, c1, c0)
    } else {
        if let Some(&(i, k, _)) = censoring_rows.iter().find(|r| r.2) {
            return Err(Error::input(format!(
                "censoring: person {i} is censored after interval {k}, before the last; give a censoring model"
            )));
        }
        let zero = Mat::<f64>::zeros(n, cens_columns);
        (Vec::new(), zero.clone(), zero)
    };
    let a: Vec<f64> = input
        .treatment
        .iter()
        .map(|&b| f64::from(u8::from(b)))
        .collect();
    let propensity_coefficients = fit_logistic(input.propensity, &a, &w, "propensity")?;
    let z = input.propensity;
    let propensity: Vec<f64> = (0..n)
        .map(|i| {
            let eta: f64 = (0..z.ncols())
                .map(|j| z[(i, j)] * propensity_coefficients[j])
                .sum();
            plogis(eta)
        })
        .collect();
    let columns = t.max(1);
    let hazard1 = predictions(input.hazard, &hazard_coefficients, n, columns, true);
    let hazard0 = predictions(input.hazard, &hazard_coefficients, n, columns, false);
    let tmle = tmle_survival(
        &SurvivalInput {
            time: input.time,
            event: input.event,
            treatment: input.treatment,
            hazard1: hazard1.as_ref(),
            hazard0: hazard0.as_ref(),
            censoring1: censoring1.as_ref(),
            censoring0: censoring0.as_ref(),
            propensity: &propensity,
            weight: input.weight,
            cluster: input.cluster,
        },
        opts,
    )?;
    Ok(SurvivalTmleGlm {
        tmle,
        hazard_coefficients,
        censoring_coefficients,
        propensity_coefficients,
        hazard1,
        hazard0,
        censoring1,
        censoring0,
        propensity,
    })
}

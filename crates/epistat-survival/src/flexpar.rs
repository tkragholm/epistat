//! Flexible parametric survival models of Royston and Parmar: a model for the
//! survivor function whose baseline is a restricted cubic spline in log time on
//! the scale of the log cumulative hazard, the log cumulative odds or the
//! probit, as `flexsurv::flexsurvspline` and `rstpm2::stpm2`.
//!
//! # The model
//!
//! With `u = ln t`, a spline `s(u)` and a linear predictor `x'beta`, the model
//! is `g(S(t | x)) = eta(t | x)` with
//!
//! ```text
//! eta(t | x) = s(u) + x'beta + sum_j x_j s_j(u),
//! s(u)       = gamma_0 + gamma_1 u + sum_{m=1}^{k-2} gamma_{m+1} v_m(u),
//!
//! Scale::Hazard   S = exp(-exp(eta))      g(S) = ln(-ln S)        proportional hazards
//! Scale::Odds     S = 1 / (1 + exp(eta))  g(S) = ln((1 - S) / S)  proportional odds
//! Scale::Probit   S = Phi(-eta)           g(S) = -Phi^-1(S)
//! ```
//!
//! `v_m` are the nonlinear functions of Harrell's restricted cubic spline on `k`
//! knots, so `s` is linear beyond the boundary knots. The baseline has `df =
//! k - 1` degrees of freedom besides its intercept. The knots are, as in
//! `stpm2` and `flexsurvspline`, the smallest and largest log event time and the
//! `df - 1` type 7 quantiles of the log event times between them, at
//! `j / df`. Time-varying effects are the terms `x_j s_j(u)`, where `s_j` is a
//! spline without an intercept of its own degrees of freedom, with knots placed
//! in the same way, so that `beta_j + s_j(u)` is the effect of `x_j` at time `t`.
//! Under `Scale::Hazard` the ratio of hazards for a unit change of `x_j` is
//! `exp(beta_j + s_j(u))`. The functions of the baseline and of the
//! time-varying splines differ between packages, and so do their coefficients.
//! The spaces they span are the same for the same knots, so the coefficients
//! of the covariates, the curves and the log likelihood do not depend on the
//! basis.
//!
//! # Likelihood
//!
//! The hazard is `h(t | x) = m(eta) eta'(u) / t`, where `eta'(u) = w'theta` is
//! the derivative of the linear predictor in `u` and `m = H'` for the
//! cumulative hazard `H(eta) = -ln S` (see the table in the private module
//! `link`): `e^eta`, `1 / (1 + e^-eta)` and the inverse Mills ratio
//! `phi(eta) / Phi(-eta)`. A row with entry `t_0`, exit `t`, status `d` and
//! weight `c` contributes
//!
//! ```text
//! c [ d ( ln m(eta) + ln eta'(u) - ln t ) - H(eta) + H(eta_0) ],
//! ```
//!
//! with the last term only for `t_0 > 0`, where `eta_0` is the linear predictor
//! at `t_0`. The term `-ln t` is the Jacobian of the log time, so this is the
//! likelihood of the density of `t`, as `survreg`, `flexsurv` and `stpm2` give
//! it. For `df = 1` the three scales are the Weibull, the log-logistic and the
//! log-normal distributions with `gamma_1 = 1 / sigma`, and the fit is `survreg`'s. Without delayed entry
//! each term is concave in `theta` where `eta'(u) > 0`. Delayed entry adds the
//! convex term `H(eta_0)`, so the information may fail to be positive definite
//! away from the maximum. A row of weight 0 contributes nothing and does not
//! count.
//!
//! The model is defined where `eta'(u) > 0` at every event time. The fit never
//! leaves that set, but the fitted curve may still decrease between event
//! times or beyond the data. [`FlexParFit::monotone_on_data`] says whether
//! `eta'(u) > 0` at every event time and on a grid in time of 101 points
//! between the boundary knots for up to 200 rows spread through the data, and
//! [`predict`] refuses to give survival curves of a fit where it is false
//! unless [`PredictOptions::allow_non_monotone`] is set.
//!
//! # Algorithm
//!
//! Newton's method with the analytic score and observed information
//! ([`fit`]), starting from the baseline spline fitted by weighted least squares
//! to the transformed Nelson-Aalen estimate (a straight line in `ln t`
//! where that fit has `eta'(u) <= 0` at an event). A step that does not raise the
//! log likelihood or leaves the set where `eta'(u) > 0` at every event is
//! halved up to 40 times. Where the information is not positive definite the
//! step solves the information plus a ridge from the ladder `1e-8, 1e-7, ...`
//! times its largest diagonal element, and a fit that has not converged in
//! `max_iterations` goes to BFGS of [`epistat_optim`] from its last point and
//! then to Newton again. Convergence is declared when the Newton decrement
//! `score' I^-1 score` is at most `tolerance (1 + |l|)`, with no ridge, and the
//! decrement is twice the distance of the log likelihood from its maximum to
//! second order. The covariance is the inverse of the observed information at the
//! estimate. With a cluster in the data the robust covariance is `V M V` with
//! the meat `M` the sum over clusters of the outer product of the cluster's
//! summed scores, without a small-sample factor.
//!
//! A covariate that is constant or a linear combination of the columns before
//! it is aliased: its coefficient is `None`, and it is left out of the fit.
//! The aliased columns are found from the weighted correlation matrix of the
//! varying columns by the `L D L'` factorisation with a tolerance of `1e-10`.
//!
//! # Predictions
//!
//! [`predict`] gives, for each row of covariates and each time, the survivor
//! function, the cumulative hazard or the hazard, and [`contrast`] gives the
//! hazard ratio and the difference of two rows of the survivor function or of the
//! hazard. Variances are by the delta method from the model-based or
//! the robust covariance. Limits are normal on the log scale for the cumulative
//! hazard, the hazard and the hazard ratio, and on the scale `ln(-ln S)` for the
//! survivor function, which keeps them in `[0, 1]` and is the interval of
//! `rstpm2`. On the hazard scale `ln H` is the linear predictor itself. The limits
//! of a difference are normal on its own scale. Times are those of the follow-up, so that a time below the
//! first knot or above the last is extrapolated linearly in `ln t` on the
//! scale of `eta`. The covariates of a prediction have the columns of `x`
//! of the fit. An offset is not part of a prediction.
//!
//! # Scale
//!
//! Each iteration is one pass over the rows in fixed blocks of 1024, each
//! block summed on its own and the blocks added in order, so that the result does
//! not depend on the number of threads. The work is `O(n q^2)` with `q` the
//! number of parameters, and the design rows are recomputed in every pass, so
//! the memory is `O(q^2)` besides the data.
//!
//! # References
//!
//! 1. Royston P, Parmar MKB. Flexible parametric proportional-hazards and
//!    proportional-odds models for censored survival data, with application to
//!    prognostic modelling and estimation of treatment effects. Stat Med.
//!    2002;21(15):2175-97.
//! 2. Royston P, Lambert PC. Flexible Parametric Survival Analysis Using Stata:
//!    Beyond the Cox Model. College Station: Stata Press; 2011.
//! 3. Lambert PC, Royston P. Further development of flexible parametric models
//!    for survival analysis. Stata J. 2009;9(2):265-90.
//! 4. Jackson CH. flexsurv: a platform for parametric survival modeling in R.
//!    J Stat Softw. 2016;70(8):1-33.
//! 5. Harrell FE Jr. Regression Modeling Strategies. 2nd ed. Cham: Springer;
//!    2015. Section 2.4.4.

mod design;
mod link;
mod model;
mod predict;
mod start;

use std::collections::HashSet;

use epistat_inference::sandwich::sandwich_with_inverse;
use epistat_linalg::{Mat, MatRef, row_scaled_gram};

use self::design::{Design, Spline, TvcTerm};
use self::model::{Model, Work, maximise};
use crate::error::{Error, Result};
use crate::ldl;
use crate::surv::SurvData;

pub use self::predict::{Contrast, Predict, PredictOptions, Prediction, contrast, predict};

/// The scale on which the survivor function is linear in the covariates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scale {
    /// The log cumulative hazard `ln(-ln S)`: proportional hazards.
    Hazard,
    /// The log cumulative odds `ln((1 - S) / S)`: proportional odds.
    Odds,
    /// The probit `-Phi^-1(S)`.
    Probit,
}

/// A covariate whose effect changes with time: the column of `x` times a spline
/// in `ln t` of `df` degrees of freedom without an intercept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeVarying {
    /// The column of the covariate matrix.
    pub column: usize,
    /// The degrees of freedom of the spline, at least 1. One is a linear
    /// function of `ln t`.
    pub df: usize,
}

/// The settings of a fit.
#[derive(Clone, Copy, Debug)]
pub struct FlexParOptions<'a> {
    /// The scale of the model.
    pub scale: Scale,
    /// The degrees of freedom of the baseline spline besides its intercept, at least 1.
    pub df: usize,
    /// The knots of the baseline on the `ln t` scale, boundary knots included
    /// and `df + 1` in all, or `None` for the smallest and largest log event
    /// time and the quantiles between them.
    pub knots: Option<&'a [f64]>,
    /// The time-varying effects.
    pub tvc: &'a [TimeVarying],
    /// A term added to the linear predictor of each row, such as a log offset.
    pub offset: Option<&'a [f64]>,
    /// Convergence is declared when the Newton decrement is at most
    /// `tolerance (1 + |log likelihood|)`.
    pub tolerance: f64,
    /// The most Newton iterations in one run of the method.
    pub max_iterations: usize,
}

impl Default for FlexParOptions<'_> {
    fn default() -> Self {
        Self {
            scale: Scale::Hazard,
            df: 3,
            knots: None,
            tvc: &[],
            offset: None,
            tolerance: 1e-12,
            max_iterations: 50,
        }
    }
}

impl FlexParOptions<'_> {
    /// The defaults with the scale and degrees of freedom given.
    #[must_use]
    pub fn new(scale: Scale, df: usize) -> Self {
        Self {
            scale,
            df,
            ..Self::default()
        }
    }
}

/// A time-varying term of a fit with the knots of its spline.
#[derive(Clone, Debug, PartialEq)]
pub struct TimeVaryingFit {
    /// The column of the covariate matrix.
    pub column: usize,
    /// The knots of the spline on the `ln t` scale, boundary knots included.
    pub knots: Vec<f64>,
    /// The coefficients of the spline functions.
    pub coefficients: Vec<f64>,
}

/// A fitted model.
///
/// The parameters are ordered as `gamma` (the intercept and the `df` baseline
/// coefficients), then `beta`, then the coefficients of each time-varying term
/// in turn. The covariances are square matrices of that dimension in row-major
/// order, with `None` in the rows and columns of aliased covariates.
#[derive(Clone, Debug)]
pub struct FlexParFit {
    /// The intercept and the baseline spline coefficients, `df + 1` values.
    pub gamma: Vec<f64>,
    /// The coefficient of each column of `x`, `None` for an aliased column.
    pub beta: Vec<Option<f64>>,
    /// The coefficients of each time-varying term.
    pub tvc_coefficients: Vec<Vec<f64>>,
    /// The time-varying terms with their knots.
    pub tvc: Vec<TimeVaryingFit>,
    /// The inverse of the observed information, or `None` where it is not
    /// positive definite.
    pub covariance: Option<Vec<Option<f64>>>,
    /// The cluster-robust covariance, with clusters in the data and a
    /// positive definite information.
    pub robust_covariance: Option<Vec<Option<f64>>>,
    /// The log likelihood at the estimate.
    pub log_likelihood: f64,
    /// The log likelihood of the model without covariates and time-varying
    /// terms, `None` with an offset.
    pub null_log_likelihood: Option<f64>,
    /// The knots of the baseline spline on the `ln t` scale.
    pub knots: Vec<f64>,
    /// The scale of the model.
    pub scale: Scale,
    /// The Newton iterations used, with those after a BFGS run.
    pub iterations: u32,
    /// Whether the Newton decrement fell below the tolerance.
    pub converged: bool,
    /// Not converged with a coefficient above 30 in absolute value, which is
    /// what a covariate that separates the outcomes does to a fit.
    pub diverging: bool,
    /// Whether the derivative of the linear predictor in log time is positive
    /// at every event time and on the grid described in the module
    /// documentation.
    pub monotone_on_data: bool,
    /// The number of rows of positive weight.
    pub n_rows: usize,
    /// The sum of the weights of the events.
    pub n_events: f64,
    design: Design,
    theta: Vec<f64>,
}

impl FlexParFit {
    /// The number of columns of the covariate matrix.
    #[must_use]
    pub fn n_covariates(&self) -> usize {
        self.design.n_columns
    }

    /// The side of the covariance matrices.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.gamma.len()
            + self.design.n_columns
            + self.tvc_coefficients.iter().map(Vec::len).sum::<usize>()
    }

    /// The position in the full parameter vector, aliased ones included, of
    /// the estimated parameter `k`.
    /// The position of the coefficient of covariate `column` in the covariance
    /// matrices, `None` if it is aliased.
    #[must_use]
    pub fn beta_index(&self, column: usize) -> Option<usize> {
        self.beta[column]?;
        Some(self.gamma.len() + column)
    }

    fn full_index(&self, k: usize) -> usize {
        let xs = self.design.x_start();
        let kept = self.design.kept.len();
        if k < xs {
            k
        } else if k < xs + kept {
            xs + self.design.kept[k - xs]
        } else {
            xs + self.design.n_columns + (k - xs - kept)
        }
    }
}

fn check_inputs(data: &SurvData<'_>, x: MatRef<'_, f64>, opts: &FlexParOptions<'_>) -> Result<()> {
    let n = data.n_rows();
    if x.nrows() != n {
        return Err(Error::input(format!(
            "the covariate matrix has {} rows and the follow-up {n}",
            x.nrows()
        )));
    }
    if opts.df == 0 {
        return Err(Error::input(
            "the baseline needs at least one degree of freedom",
        ));
    }
    if opts.tolerance.is_nan() || opts.tolerance <= 0.0 || opts.max_iterations == 0 {
        return Err(Error::input(
            "tolerance and max_iterations must be positive",
        ));
    }
    if let Some(o) = opts.offset
        && (o.len() != n || o.iter().any(|v| !v.is_finite()))
    {
        return Err(Error::input(
            "the offset must be finite and have one value per row",
        ));
    }
    if data.n_strata() > 0 {
        return Err(Error::input("strata are not supported by flexpar"));
    }
    if (0..x.nrows()).any(|i| (0..x.ncols()).any(|j| !x[(i, j)].is_finite())) {
        return Err(Error::input("the covariates must be finite"));
    }
    Ok(())
}

/// Fits the model by maximum likelihood.
///
/// `x` has one column for each covariate and no intercept, since the baseline
/// has its own. Entry times (the start of a counting-process row) give left
/// truncation. For a scale other than [`Scale::Hazard`] a subject must have one
/// row, because the survivor function of the other scales is not
/// additive over the pieces of follow-up.
///
/// # Errors
///
/// [`Error::Input`] for inputs of other lengths, a covariate that is not
/// finite, an event code above 1, a time of a used row that is not positive, a
/// tvc column out of range or aliased, knots that are not increasing (which
/// is what ties among the event times do to the quantiles), no events or fewer
/// events than parameters, a start value that makes the model undefined, and a
/// subject with several rows under a scale other than the hazard.
///
/// # Panics
///
/// Does not panic on valid input.
pub fn fit(
    data: &SurvData<'_>,
    x: MatRef<'_, f64>,
    opts: &FlexParOptions<'_>,
) -> Result<FlexParFit> {
    check_inputs(data, x, opts)?;
    let surv = data.followup()?;
    if surv.n_codes() != 1 {
        return Err(Error::input(
            "flexpar needs a binary status: use SurvData::cause for competing events",
        ));
    }
    let (stop, event) = (surv.stop(), surv.status());
    let start = surv.start();
    let weight = data.weight();
    let n = stop.len();
    let used = |i: usize| weight.is_none_or(|w| w[i] > 0.0);
    for i in (0..n).filter(|&i| used(i)) {
        if stop[i] <= 0.0 {
            return Err(Error::input(format!(
                "row {i} has a time that is not positive, and ln t is needed"
            )));
        }
        if start.is_some_and(|s| s[i] < 0.0) {
            return Err(Error::input(format!("row {i} has a negative entry time")));
        }
    }
    if opts.scale != Scale::Hazard
        && let Some(id) = data.id()
    {
        let mut seen = HashSet::new();
        if (0..n).filter(|&i| used(i)).any(|i| !seen.insert(id[i])) {
            return Err(Error::input(
                "a subject has several rows: only the hazard scale accepts split follow-up",
            ));
        }
    }
    let mut events: Vec<f64> = (0..n)
        .filter(|&i| used(i) && event[i] != 0)
        .map(|i| stop[i].ln())
        .collect();
    if events.is_empty() {
        return Err(Error::input("there are no events with positive weight"));
    }
    events.sort_by(f64::total_cmp);

    let baseline_knots = match opts.knots {
        Some(k) => {
            if k.len() != opts.df + 1 {
                return Err(Error::input(format!(
                    "df = {} needs {} knots, boundary knots included, and {} were given",
                    opts.df,
                    opts.df + 1,
                    k.len()
                )));
            }
            k.to_vec()
        }
        None => start::quantile_knots(&events, opts.df)?,
    };
    let p = x.ncols();
    let aliased = start::aliased_columns(x, weight);
    let kept: Vec<usize> = (0..p).filter(|&j| !aliased[j]).collect();
    let mut tvc = Vec::with_capacity(opts.tvc.len());
    for (j, t) in opts.tvc.iter().enumerate() {
        if t.column >= p {
            return Err(Error::input(format!(
                "the time-varying column {} is out of range",
                t.column
            )));
        }
        if t.df == 0 {
            return Err(Error::input(
                "a time-varying effect needs at least one degree of freedom",
            ));
        }
        if aliased[t.column] {
            return Err(Error::input(format!(
                "the time-varying column {} is aliased with the intercept or another column",
                t.column
            )));
        }
        if opts.tvc[..j].iter().any(|u| u.column == t.column) {
            return Err(Error::input(format!(
                "column {} has two time-varying terms",
                t.column
            )));
        }
        tvc.push(TvcTerm {
            column: t.column,
            spline: Spline::new(&start::quantile_knots(&events, t.df)?)?,
        });
    }
    let design = Design {
        baseline: Spline::new(&baseline_knots)?,
        kept,
        tvc,
        n_columns: p,
    };
    let q = design.dim();
    let n_used = (0..n).filter(|&i| used(i)).count();
    let n_event_rows = (0..n).filter(|&i| used(i) && event[i] != 0).count();
    if n_event_rows < q {
        return Err(Error::input(format!(
            "{n_event_rows} events cannot fit {q} parameters"
        )));
    }
    let model = Model {
        design: &design,
        scale: opts.scale,
        x,
        stop,
        start,
        event,
        weight,
        offset: opts.offset,
    };
    let theta0 = start::starting_values(&design, opts.scale, stop, start, event, weight);
    let best = maximise(&model, &theta0, opts.tolerance, opts.max_iterations).ok_or_else(|| {
        Error::input("the model is undefined at the starting values: the derivative of the spline is not positive at every event")
    })?;

    let null_log_likelihood = if opts.offset.is_some() {
        None
    } else if design.kept.is_empty() && design.tvc.is_empty() {
        Some(best.totals.ll)
    } else {
        let null_design = Design {
            baseline: design.baseline.clone(),
            kept: Vec::new(),
            tvc: Vec::new(),
            n_columns: p,
        };
        let null_model = Model {
            design: &null_design,
            ..Model {
                design: &design,
                scale: opts.scale,
                x,
                stop,
                start,
                event,
                weight,
                offset: None,
            }
        };
        let s0 = start::starting_values(&null_design, opts.scale, stop, start, event, weight);
        maximise(&null_model, &s0, opts.tolerance, opts.max_iterations).map(|m| m.totals.ll)
    };

    let theta = best.theta.clone();
    let mut info = best.totals.info.clone();
    let pivots = ldl::factorise(&mut info, q, 1e-13);
    let inverse = (pivots.clean && pivots.rank == q).then(|| ldl::inverse(&info, q));

    let robust = match (&inverse, data.cluster()) {
        (Some(v), Some(cluster)) => Some(robust_variance(
            &model,
            &theta,
            cluster,
            data.n_clusters(),
            v,
        )),
        _ => None,
    };

    let gamma = theta[..=opts.df].to_vec();
    let xs = design.x_start();
    let mut beta: Vec<Option<f64>> = vec![None; p];
    for (slot, &col) in design.kept.iter().enumerate() {
        beta[col] = Some(theta[xs + slot]);
    }
    let mut tvc_fit = Vec::new();
    let mut tvc_coefficients = Vec::new();
    for (j, t) in design.tvc.iter().enumerate() {
        let at = design.tvc_start(j);
        let c = theta[at..at + t.spline.len()].to_vec();
        tvc_coefficients.push(c.clone());
        tvc_fit.push(TimeVaryingFit {
            column: t.column,
            knots: t.spline.knots().to_vec(),
            coefficients: c,
        });
    }
    let monotone = monotone_on_data(&design, &theta, x, stop, event, weight);
    let diverging = !best.converged && theta.iter().any(|v| v.abs() > 30.0);
    let n_events = (0..n)
        .filter(|&i| event[i] != 0)
        .map(|i| weight.map_or(1.0, |w| w[i]))
        .sum();
    let mut result = FlexParFit {
        gamma,
        beta,
        tvc_coefficients,
        tvc: tvc_fit,
        covariance: None,
        robust_covariance: None,
        log_likelihood: best.totals.ll,
        null_log_likelihood,
        knots: baseline_knots,
        scale: opts.scale,
        iterations: best.iterations,
        converged: best.converged,
        diverging,
        monotone_on_data: monotone,
        n_rows: n_used,
        n_events,
        design,
        theta,
    };
    result.covariance = inverse.map(|v| expand(&result, &v));
    result.robust_covariance = robust.map(|v| expand(&result, &v));
    Ok(result)
}

/// A matrix over the estimated parameters as one over all of them, with
/// `None` in the rows and columns of aliased covariates.
fn expand(fit: &FlexParFit, v: &[f64]) -> Vec<Option<f64>> {
    let q = fit.design.dim();
    let d = fit.dimension();
    let mut out = vec![None; d * d];
    for a in 0..q {
        for b in 0..q {
            out[fit.full_index(a) * d + fit.full_index(b)] = Some(v[a * q + b]);
        }
    }
    out
}

fn robust_variance(
    model: &Model<'_>,
    theta: &[f64],
    cluster: &[u32],
    n_clusters: usize,
    v: &[f64],
) -> Vec<f64> {
    let q = theta.len();
    let mut sums = Mat::<f64>::zeros(n_clusters, q);
    let mut work = Work::new(q);
    for (i, &group) in cluster.iter().enumerate() {
        if let Ok(Some(r)) = model.row(i, theta, &mut work) {
            let g = group as usize;
            for k in 0..q {
                sums[(g, k)] += r.sz * work.z[k]
                    + r.sw * work.w[k]
                    + if r.has_entry { r.s0 * work.z0[k] } else { 0.0 };
            }
        }
    }
    let meat = row_scaled_gram(sums.as_ref(), |_| 1.0);
    let inverse = Mat::from_fn(q, q, |i, j| v[i * q + j]);
    let s = sandwich_with_inverse(inverse.as_ref(), meat.as_ref());
    (0..q * q).map(|k| s[(k / q, k % q)]).collect()
}

/// Whether `eta'(u) > 0` at every event time and on a grid of times for rows
/// spread through the data.
fn monotone_on_data(
    design: &Design,
    theta: &[f64],
    x: MatRef<'_, f64>,
    stop: &[f64],
    event: &[i32],
    weight: Option<&[f64]>,
) -> bool {
    let q = theta.len();
    let (mut z, mut w) = (vec![0.0; q], vec![0.0; q]);
    let mut positive = |row: usize, u: f64| {
        design.fill(x, row, u, &mut z, Some(&mut w));
        w.iter().zip(theta).map(|(a, b)| a * b).sum::<f64>() > 0.0
    };
    let n = stop.len();
    let used = |i: usize| weight.is_none_or(|c| c[i] > 0.0);
    if !(0..n)
        .filter(|&i| used(i) && event[i] != 0)
        .all(|i| positive(i, stop[i].ln()))
    {
        return false;
    }
    let knots = design.baseline.knots();
    let (lo, hi) = (knots[0], knots[knots.len() - 1]);
    let rows: Vec<usize> = (0..n).filter(|&i| used(i)).collect();
    let stride = rows.len().div_ceil(200).max(1);
    rows.iter().step_by(stride).all(|&i| {
        (0..=100).all(|g| {
            let u = lo + (hi - lo) * f64::from(g) / 100.0;
            positive(i, u)
        })
    })
}

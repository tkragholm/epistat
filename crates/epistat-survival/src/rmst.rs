//! Restricted mean survival time and restricted mean time lost in two groups,
//! with their difference and ratios, as `survRM2::rmst2` gives them without
//! covariates.
//!
//! # Algorithm
//!
//! Each group has a Kaplan-Meier curve `S`. Its restricted mean survival time
//! up to the horizon `tau` is the area under the curve, and the restricted
//! mean time lost is the area above it,
//!
//! ```text
//! RMST = integral of S(t) dt from the origin to tau,     RMTL = tau - RMST.
//! ```
//!
//! The variance of the area is the one of [`GroupCurve::rmean`] \[1\]\[2\]. It is
//! the same for the time lost, whose confidence limits are those of the mean
//! turned around `tau`. The two groups are independent, so for group 1 against
//! group 0, with the variances `v1` and `v0`,
//!
//! ```text
//! difference     RMST1 - RMST0,        standard error sqrt(v1 + v0)
//! ratio          RMST1 / RMST0,        standard error of its log
//!                                      sqrt(v1 / RMST1^2 + v0 / RMST0^2)
//! loss ratio     RMTL1 / RMTL0,        standard error of its log
//!                                      sqrt(v1 / RMTL1^2 + v0 / RMTL0^2)
//! ```
//!
//! The limits of the difference are `est -+ z se`, those of a ratio are
//! `exp(log est -+ z se)`, and the p-value of each is the two-sided normal one
//! for the estimate over its standard error, on the log scale for a ratio \[3\].
//! An undefined value is `None`: a ratio with a restricted mean or time lost of
//! zero in either group, and a p-value of `0 / 0`.
//!
//! # The horizon
//!
//! The horizon has no default. [`Tau::Fixed`] is a horizon the caller
//! chooses, which is the question the analysis asks. [`Tau::Largest`] is the
//! rule of `rmst2` when it is given no `tau`: the largest horizon up to which
//! both curves are determined by the data. A curve is determined to its last
//! time, and beyond it only when every row at the last time has the event, so
//! that the curve is zero from there on. With `U` that limit of each group and
//! `L` the last time of each, the largest horizon is the smaller of `U` of the
//! two groups, and at most the larger of the two last times. A horizon above
//! the largest is an error, as in `rmst2`, since the curve beyond it would
//! be carried forward by assumption.
//!
//! # Input
//!
//! The data are follow-up data with one event code and a stratum column with
//! two levels, the groups. The stratum with the smaller code is group 0, the
//! reference, and the other is group 1, as `arm == 0` and `arm == 1` in
//! `rmst2`. The tie rule of [`TimeFix`] is applied first, as `survfit` does.
//! Rows of weight 0 are not used, and a group with no row of positive weight
//! is an error. Case weights, delayed entry and the other things
//! [`survfit`](crate::survfit) accepts are accepted here, which `rmst2` does
//! not take.
//!
//! # Differences from survRM2
//!
//! - `rmst2` calls the origin 0 whatever the data. The origin here is that of
//!   the curves, the smaller of 0 and every time, which is the same for times
//!   that are not negative.
//! - `rmst2` with `covariates` is [`rmst2_regression`], see below. It takes the
//!   arm from the stratum column and the covariates as a matrix, and refuses
//!   case weights and delayed entry, which `rmst2` does not take.
//!
//! # Covariates
//!
//! [`rmst2_regression`] fits the regression of the restricted mean on the arm
//! and covariates by inverse probability of censoring weighting \[4\]. With
//! `Y = min(T, tau)`, `D = 1` for an event or when `Y = tau` and `0` otherwise,
//! and `G` the Kaplan-Meier estimate of the censoring distribution of the
//! subject's own arm, with the censorings at a time counted as if they came
//! after the events at it, so that `G(Y)` includes them, the coefficients
//! solve
//!
//! ```text
//! sum over subjects of  D / G(Y)  x  (Y - g^-1(x' beta)) = 0,
//! ```
//!
//! with `x` the intercept, the arm and the covariates. The link is the identity
//! for a difference, the log for a ratio of restricted means, where the
//! equation is that of the quasi-Poisson model, and the log again for a ratio of
//! restricted times lost, with `tau - Y` for `Y`. The covariance is `A^-1 S A^-1`,
//! with `A` the sum of `x x'` over all subjects, times `exp(x' beta)` for a log
//! link, and `S` the sum over subjects of the products of
//!
//! ```text
//! kappa_i = s_i + (1 - D_i) q(Y_i) / R(Y_i) - sum over k with Y_k <= Y_i of (1 - D_k) q(Y_k) / R(Y_k)^2,
//! s_i = D_i / G(Y_i) x_i e_i,     q(u) = sum of s_j over subjects with Y_j >= u,
//! ```
//!
//! `e_i` the residual and `R(u)` the number of subjects of the arm with
//! `Y >= u`. The third term is the effect of estimating `G`. The limits and the
//! p-value are Wald, `exp` of the limits for a ratio, and the p-value is the
//! normal one, `2 Phi(-|z|)`, where `rmst2` forms `1 - pchisq(z^2, 1)`, which
//! is 0 below about `1e-16` and cannot show a smaller one.
//!
//! # References
//!
//! 1. Miller R. G. (1981) Survival Analysis, Wiley, pp. 195-6.
//! 2. Royston P., Parmar M. K. B. (2013) Restricted mean survival time: an
//!    alternative to the hazard ratio for the design and analysis of randomized
//!    trials with a time-to-event outcome, BMC Med Res Methodol 13: 152.
//! 3. Uno H., Claggett B., Tian L. et al. (2014) Moving beyond the hazard
//!    ratio in quantifying the between-group difference in survival analysis,
//!    J Clin Oncol 32: 2380-5.
//! 4. Tian L., Zhao L., Wei L. J. (2014) Predicting the restricted mean event
//!    time with the subject's baseline covariates in survival analysis,
//!    Biostatistics 15: 222-233.

use epistat_core::{Covariance, InputError, Level};
use epistat_glm::{Family, Glm, Link};
use epistat_inference::sandwich::sandwich;
use epistat_linalg::{Attempt, Mat, MatRef, weighted_gram};

use crate::check::refuse_non_finite_matrix;
use crate::covariance;
use crate::error::{Error, Result};
use crate::metrics::censoring::CensoringKm;
use crate::surv::SurvData;
use crate::survfit::{ConfType, GroupCurve, SurvfitOptions, estimate};
use crate::timefix::TimeFix;

/// The horizon of the restricted means.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Tau {
    /// A horizon chosen by the caller.
    Fixed(f64),
    /// The largest horizon up to which both curves are determined, the rule
    /// `rmst2` follows without a `tau`. See the [module documentation](self).
    Largest,
}

/// What [`rmst2`] does besides the data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RmstOptions {
    /// The horizon. It has no default.
    pub tau: Tau,
    /// How near-equal times are treated. The default is `survival`'s rule,
    /// see [`TimeFix`].
    pub timefix: TimeFix,
    /// The confidence level, `1 - alpha` of `rmst2`.
    pub level: Level,
}

impl RmstOptions {
    /// Options with the horizon given, the tie rule of `survfit` and 95 per
    /// cent limits.
    #[must_use]
    pub fn new(tau: Tau) -> Self {
        Self {
            tau,
            timefix: TimeFix::default(),
            level: Level::default(),
        }
    }
}

/// The restricted mean survival time and time lost of one group.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct GroupRmst {
    /// The caller's stratum code of the group.
    pub group: u32,
    /// The number of rows used in the group.
    pub n: usize,
    /// The restricted mean survival time.
    pub rmst: f64,
    /// Its standard error, which is also that of the time lost.
    pub se: f64,
    /// The lower confidence limit of the restricted mean survival time.
    pub rmst_lower: f64,
    /// The upper confidence limit of the restricted mean survival time.
    pub rmst_upper: f64,
    /// The restricted mean time lost, `tau - rmst`.
    pub rmtl: f64,
    /// The lower confidence limit of the time lost.
    pub rmtl_lower: f64,
    /// The upper confidence limit of the time lost.
    pub rmtl_upper: f64,
    /// The last time of the group's curve.
    pub last_time: f64,
    /// Whether every row at the last time has the event, so that the curve is
    /// determined beyond it.
    pub ends_in_event: bool,
}

/// A contrast between the groups with its limits and p-value.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct RmstContrast {
    /// The difference or ratio, group 1 against group 0.
    pub estimate: Option<f64>,
    /// The standard error of the difference, or of the logarithm of a ratio.
    pub se: Option<f64>,
    /// The lower confidence limit.
    pub lower: Option<f64>,
    /// The upper confidence limit.
    pub upper: Option<f64>,
    /// The two-sided p-value of no difference, or of a ratio of 1.
    pub p_value: Option<f64>,
}

/// `RmstContrast`, under its name in 0.3.
#[deprecated(since = "0.4.0", note = "renamed `RmstContrast`")]
pub type Contrast = RmstContrast;

/// The restricted means of two groups and their contrasts.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Rmst2 {
    /// The horizon used.
    pub tau: f64,
    /// The largest horizon the data determine, see [`Tau::Largest`].
    pub tau_largest: f64,
    /// The reference group, the stratum with the smaller code.
    pub group0: GroupRmst,
    /// The other group.
    pub group1: GroupRmst,
    /// `RMST1 - RMST0`.
    pub difference: RmstContrast,
    /// `RMST1 / RMST0`.
    pub ratio: RmstContrast,
    /// `RMTL1 / RMTL0`.
    pub loss_ratio: RmstContrast,
    /// The confidence level of the limits.
    pub level: Level,
    /// The number of rows given.
    pub n_rows: usize,
    /// The number of rows of positive weight, which are the rows used.
    pub n_used: usize,
}

/// The restricted mean survival time and time lost of two groups, and their
/// difference and ratios, as `survRM2::rmst2` without covariates.
///
/// See the [module documentation](self) for the formulas, the groups and the
/// horizon.
///
/// # Errors
///
/// An input error when `data` has more than one event code or no row of
/// positive weight, a stratum column with other than two levels used, or a
/// group with no row used, the tie rule fails (see
/// [`SurvData::with_timefix`]), or the horizon is not finite, is before the
/// origin of the curves or is above the largest the data determine.
pub fn rmst2(data: &SurvData<'_>, opts: &RmstOptions) -> Result<Rmst2> {
    if data.n_strata() != 2 {
        return Err(Error::not_applicable(format!(
            "rmst2 compares two groups, and the stratum column has {} levels",
            data.n_strata()
        )));
    }
    let (curve0, curve1, tau, tau_largest) = horizon(data, opts)?;
    let (curve0, curve1) = (&curve0, &curve1);
    let z = epistat_stats::dist::normal::quantile(opts.level.upper_probability());
    let group0 = group_rmst(curve0, tau, z)?;
    let group1 = group_rmst(curve1, tau, z)?;
    let (v0, v1) = (group0.se * group0.se, group1.se * group1.se);

    let difference = {
        let estimate = group1.rmst - group0.rmst;
        let se = (v1 + v0).sqrt();
        contrast(estimate, se, z, false)
    };
    let log_ratio = |a: f64, b: f64| {
        if a > 0.0 && b > 0.0 {
            let se = (v1 / (a * a) + v0 / (b * b)).sqrt();
            contrast((a / b).ln(), se, z, true)
        } else {
            RmstContrast {
                estimate: None,
                se: None,
                lower: None,
                upper: None,
                p_value: None,
            }
        }
    };
    let ratio = log_ratio(group1.rmst, group0.rmst);
    let loss_ratio = log_ratio(group1.rmtl, group0.rmtl);
    Ok(Rmst2 {
        tau,
        tau_largest,
        group0,
        group1,
        difference,
        ratio,
        loss_ratio,
        level: opts.level,
        n_rows: data.n_rows(),
        n_used: data.n_used(),
    })
}

/// The two curves and the horizon: the one asked for, checked against the
/// largest the data determine, and that largest.
fn horizon(data: &SurvData<'_>, opts: &RmstOptions) -> Result<(GroupCurve, GroupCurve, f64, f64)> {
    let fit = estimate(
        data,
        &SurvfitOptions {
            timefix: opts.timefix,
            conf_type: ConfType::None,
            ..SurvfitOptions::default()
        },
    )?;
    let [curve0, curve1] = fit.groups.as_slice() else {
        return Err(Error::no_information(
            "rmst2 needs rows of positive weight in both groups",
        ));
    };
    let (limit0, limit1) = (determined_to(curve0), determined_to(curve1));
    let last = curve0.time[curve0.time.len() - 1].max(curve1.time[curve1.time.len() - 1]);
    let tau_largest = limit0.min(limit1).min(last);
    let tau = match opts.tau {
        Tau::Largest => tau_largest,
        Tau::Fixed(tau) => {
            if !tau.is_finite() {
                return Err(Error::options("the horizon must be finite"));
            }
            if tau > tau_largest {
                return Err(Error::options(format!(
                    "the horizon {tau} is above {tau_largest}, the largest up to which \
                     both curves are determined by the data"
                )));
            }
            tau
        }
    };
    Ok((curve0.clone(), curve1.clone(), tau, tau_largest))
}

/// Up to which horizon a curve is determined: its last time, or without limit
/// when every row at the last time has the event.
fn determined_to(curve: &GroupCurve) -> f64 {
    if ends_in_event(curve) {
        f64::INFINITY
    } else {
        curve.time[curve.time.len() - 1]
    }
}

fn ends_in_event(curve: &GroupCurve) -> bool {
    let last = curve.time.len() - 1;
    curve.count_censor[last] == 0 && curve.count_event[last] > 0
}

fn group_rmst(curve: &GroupCurve, tau: f64, z: f64) -> Result<GroupRmst> {
    let r = curve.rmean(tau)?;
    let half = z * r.se;
    let rmtl = tau - r.mean;
    Ok(GroupRmst {
        group: curve.group,
        n: curve.n,
        rmst: r.mean,
        se: r.se,
        rmst_lower: r.mean - half,
        rmst_upper: r.mean + half,
        rmtl,
        rmtl_lower: tau - (r.mean + half),
        rmtl_upper: tau - (r.mean - half),
        last_time: curve.time[curve.time.len() - 1],
        ends_in_event: ends_in_event(curve),
    })
}

/// A contrast of `centre`, on the scale the limits are formed on, with
/// standard error `se`. What is reported is `exp(centre)` when `exponentiate`
/// is set.
fn contrast(centre: f64, se: f64, z: f64, exponentiate: bool) -> RmstContrast {
    let map = |v: f64| if exponentiate { v.exp() } else { v };
    let statistic = centre / se;
    let p_value = if statistic.is_nan() {
        None
    } else {
        Some(2.0 * epistat_stats::dist::normal::cdf(-statistic.abs()))
    };
    RmstContrast {
        estimate: Some(map(centre)),
        se: Some(se),
        lower: Some(map(centre - z * se)),
        upper: Some(map(centre + z * se)),
        p_value,
    }
}

/// The effect of an arm or covariate on the restricted mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RmstEffect {
    /// A difference of restricted mean survival times, the identity link.
    Difference,
    /// A ratio of restricted mean survival times, the log link.
    Ratio,
    /// A ratio of restricted mean times lost, the log link on `tau - Y`.
    LossRatio,
}

/// The coefficients of a regression of the restricted mean.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RmstRegression {
    /// The effect modelled.
    pub effect: RmstEffect,
    /// The horizon used.
    pub tau: f64,
    /// The largest horizon the data determine, see [`Tau::Largest`].
    pub tau_largest: f64,
    /// The coefficients: the intercept, the arm and then the covariates.
    pub coefficients: Vec<f64>,
    /// Their standard errors.
    pub se: Vec<f64>,
    /// The covariance matrix, `p` by `p`.
    pub covariance: Covariance,
    /// `coefficient / se`.
    pub z: Vec<f64>,
    /// The two-sided normal p-value of each coefficient.
    pub p_value: Vec<f64>,
    /// `exp(coefficient)` for a ratio, `None` for a difference.
    pub ratio: Option<Vec<f64>>,
    /// The lower limit, on the scale of the ratio for a ratio and of the
    /// coefficient for a difference.
    pub lower: Vec<f64>,
    /// The upper limit, on the same scale.
    pub upper: Vec<f64>,
    /// The confidence level of the limits.
    pub level: Level,
    /// The number of rows used.
    pub n_used: usize,
    /// The number of rows given.
    pub n_rows: usize,
}

/// The three effects of `rmst2` with covariates.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AdjustedRmst {
    /// The difference of restricted mean survival times.
    pub difference: RmstRegression,
    /// The ratio of restricted mean survival times.
    pub ratio: RmstRegression,
    /// The ratio of restricted mean times lost.
    pub loss_ratio: RmstRegression,
}

/// The three regressions of `rmst2` with `covariates`.
///
/// # Errors
///
/// As [`rmst2_regression`].
pub fn rmst2_adjusted(
    data: &SurvData<'_>,
    covariates: Option<MatRef<'_, f64>>,
    opts: &RmstOptions,
) -> Result<AdjustedRmst> {
    Ok(AdjustedRmst {
        difference: rmst2_regression(data, covariates, RmstEffect::Difference, opts)?,
        ratio: rmst2_regression(data, covariates, RmstEffect::Ratio, opts)?,
        loss_ratio: rmst2_regression(data, covariates, RmstEffect::LossRatio, opts)?,
    })
}

/// The regression of the restricted mean on the arm and `covariates`, by
/// inverse probability of censoring weighting, as `survRM2::rmst2` with
/// `covariates`.
///
/// The arm is the stratum column, which must have two levels, the one with the
/// smaller code being 0. `covariates` has one row per row of the data and is
/// given without an intercept. The horizon, the tie rule and the confidence
/// level are those of [`RmstOptions`]. The weighted fit uses `glm.control`'s
/// defaults, 1e-8 and 25 iterations, as `rmst2` does. See the [module
/// documentation](self) for the equations and the covariance.
///
/// # Errors
///
/// An input error as for [`rmst2`], and when the data have case weights or
/// delayed entry, `covariates` has the wrong number of rows or a value that is
/// not finite, the design is rank deficient, or the fit does not converge.
pub fn rmst2_regression(
    data: &SurvData<'_>,
    covariates: Option<MatRef<'_, f64>>,
    effect: RmstEffect,
    opts: &RmstOptions,
) -> Result<RmstRegression> {
    if data.n_strata() != 2 {
        return Err(Error::not_applicable(format!(
            "the arm is the stratum column, which must have two levels, and it has {}",
            data.n_strata()
        )));
    }
    if data.weight().is_some() {
        return Err(Error::not_applicable(
            "the regression of the restricted mean takes no case weights",
        ));
    }
    if data.followup().start().is_some() {
        return Err(Error::not_applicable(
            "the regression of the restricted mean takes no delayed entry",
        ));
    }
    let n = data.n_rows();
    let q = covariates.map_or(0, |c| c.ncols());
    if let Some(c) = covariates {
        if c.nrows() != n {
            return Err(InputError::Length {
                what: "covariates",
                expected: n,
                found: c.nrows(),
            }
            .into());
        }
        refuse_non_finite_matrix(c, "covariates")?;
    }
    let (_, _, tau, tau_largest) = horizon(data, opts)?;
    let data = data.clone().with_timefix(opts.timefix)?;
    let surv = data.followup();
    let arm_of = data
        .stratum()
        .ok_or_else(|| Error::missing("the arm is the stratum column"))?;
    let p = 2 + q;
    let x = Mat::from_fn(n, p, |i, j| match j {
        0 => 1.0,
        1 => f64::from(arm_of[i]),
        _ => covariates.map_or(0.0, |c| c[(i, j - 2)]),
    });
    let y: Vec<f64> = surv.stop().iter().map(|&t| t.min(tau)).collect();
    let d: Vec<bool> = (0..n)
        .map(|i| surv.status()[i] != 0 || y[i] >= tau)
        .collect();

    // The weight of each subject, from the Kaplan-Meier curve of the censoring
    // of its own arm.
    let mut weight = vec![0.0f64; n];
    let mut order: Vec<Vec<usize>> = vec![Vec::new(), Vec::new()];
    for i in 0..n {
        order[arm_of[i] as usize].push(i);
    }
    for rows in &mut order {
        rows.sort_by(|&a, &b| y[a].total_cmp(&y[b]).then(a.cmp(&b)));
        let time: Vec<f64> = rows.iter().map(|&i| y[i]).collect();
        let censored: Vec<bool> = rows.iter().map(|&i| !d[i]).collect();
        let unit = vec![1.0; rows.len()];
        let g = CensoringKm::new(&time, &censored, &unit, false);
        for &i in rows.iter() {
            if d[i] {
                weight[i] = 1.0 / g.at(y[i]);
            }
        }
    }

    let (response, family, link) = match effect {
        RmstEffect::Difference => (y.clone(), Family::Gaussian, Link::Identity),
        RmstEffect::Ratio => (y.clone(), Family::Poisson, Link::Log),
        RmstEffect::LossRatio => (
            y.iter().map(|v| tau - v).collect(),
            Family::Poisson,
            Link::Log,
        ),
    };
    let fit = Glm::new(family, link)
        .fit(&crate::logistic::data(
            x.as_ref(),
            &response,
            Some(&weight),
            None,
            None,
        )?)
        .map_err(Error::model("the regression"))?;
    let beta = fit.full_rank().map_err(Error::model("the regression"))?;
    let log_link = link == Link::Log;
    let mean: Vec<f64> = (0..n)
        .map(|i| {
            let eta: f64 = (0..p).map(|j| x[(i, j)] * beta[j]).sum();
            if log_link { eta.exp() } else { eta }
        })
        .collect();
    let score = Mat::from_fn(n, p, |i, j| weight[i] * x[(i, j)] * (response[i] - mean[i]));

    // The score with the effect of estimating the censoring distribution.
    let mut kappa = Mat::<f64>::zeros(n, p);
    for rows in &order {
        let m = rows.len();
        let mut suffix = Mat::<f64>::zeros(m + 1, p);
        for k in (0..m).rev() {
            for j in 0..p {
                suffix[(k, j)] = suffix[(k + 1, j)] + score[(rows[k], j)];
            }
        }
        let mut start = vec![0usize; m];
        let mut last = vec![0usize; m];
        let mut k = 0;
        while k < m {
            let mut end = k;
            while end < m && y[rows[end]].total_cmp(&y[rows[k]]).is_eq() {
                end += 1;
            }
            for t in k..end {
                start[t] = k;
                last[t] = end - 1;
            }
            k = end;
        }
        let mut cumulative = Mat::<f64>::zeros(m, p);
        for k in 0..m {
            let at_risk = (m - start[k]) as f64;
            for j in 0..p {
                let own = if d[rows[k]] {
                    0.0
                } else {
                    suffix[(start[k], j)] / at_risk / at_risk
                };
                cumulative[(k, j)] = if k == 0 { 0.0 } else { cumulative[(k - 1, j)] } + own;
            }
        }
        for k in 0..m {
            let at_risk = (m - start[k]) as f64;
            for j in 0..p {
                let second = if d[rows[k]] {
                    0.0
                } else {
                    suffix[(start[k], j)] / at_risk
                };
                kappa[(rows[k], j)] = score[(rows[k], j)] + second - cumulative[(last[k], j)];
            }
        }
    }
    let ones = vec![1.0f64; n];
    let gamma = weighted_gram(kappa.as_ref(), &ones)?;
    let a_weight: Vec<f64> = if log_link { mean.clone() } else { ones };
    let a = weighted_gram(x.as_ref(), &a_weight)?;
    let covariance = sandwich(a.as_ref(), gamma.as_ref(), &Attempt::SYMMETRIC)?;
    let std_err: Vec<f64> = (0..p).map(|j| covariance[(j, j)].max(0.0).sqrt()).collect();
    let z_value = epistat_stats::dist::normal::quantile(opts.level.upper_probability());
    let z: Vec<f64> = (0..p).map(|j| beta[j] / std_err[j]).collect();
    let p_value: Vec<f64> = z
        .iter()
        .map(|&v| 2.0 * epistat_stats::dist::normal::cdf(-v.abs()))
        .collect();
    let scale = |v: f64| if log_link { v.exp() } else { v };
    Ok(RmstRegression {
        effect,
        tau,
        tau_largest,
        lower: (0..p)
            .map(|j| scale(beta[j] - z_value * std_err[j]))
            .collect(),
        upper: (0..p)
            .map(|j| scale(beta[j] + z_value * std_err[j]))
            .collect(),
        ratio: log_link.then(|| beta.iter().map(|b| b.exp()).collect()),
        covariance: covariance::from_values(
            p,
            &(0..p)
                .flat_map(|i| (0..p).map(move |j| (i, j)))
                .map(|(i, j)| covariance[(i, j)])
                .collect::<Vec<_>>(),
        )?,
        coefficients: beta,
        se: std_err,
        z,
        p_value,
        level: opts.level,
        n_used: data.n_used(),
        n_rows: n,
    })
}

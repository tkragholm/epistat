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
//! - Covariate-adjusted analysis (`rmst2` with `covariates`, regression of the
//!   restricted mean on covariates by inverse-probability weighting) is not
//!   done. [`pseudo`](crate::pseudo) fits the same kind of regression on
//!   pseudo-observations.
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

use crate::error::{Error, Result};
use crate::surv::SurvData;
use crate::survfit::{ConfType, GroupCurve, SurvfitOptions, estimate};
use crate::timefix::TimeFix;

/// The horizon of the restricted means.
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// The confidence level, in `(0, 1)`, `1 - alpha` of `rmst2`.
    pub conf_level: f64,
}

impl RmstOptions {
    /// Options with the horizon given, the tie rule of `survfit` and 95 per
    /// cent limits.
    #[must_use]
    pub fn new(tau: Tau) -> Self {
        Self {
            tau,
            timefix: TimeFix::default(),
            conf_level: 0.95,
        }
    }
}

/// The restricted mean survival time and time lost of one group.
#[derive(Debug, Clone, Copy, PartialEq)]
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
pub struct Contrast {
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

/// The restricted means of two groups and their contrasts.
#[derive(Debug, Clone, PartialEq)]
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
    pub difference: Contrast,
    /// `RMST1 / RMST0`.
    pub ratio: Contrast,
    /// `RMTL1 / RMTL0`.
    pub loss_ratio: Contrast,
    /// The confidence level of the limits.
    pub conf_level: f64,
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
/// [`Error::Input`] when `data` has more than one event code or no row of
/// positive weight, a stratum column with other than two levels used, or a
/// group with no row used, the tie rule fails (see
/// [`SurvData::with_timefix`]), the confidence level is not in `(0, 1)`, or
/// the horizon is not finite, is before the origin of the curves or is above
/// the largest the data determine.
pub fn rmst2(data: &SurvData<'_>, opts: &RmstOptions) -> Result<Rmst2> {
    if !(opts.conf_level > 0.0 && opts.conf_level < 1.0) {
        return Err(Error::input(format!(
            "the confidence level {} is not in (0, 1)",
            opts.conf_level
        )));
    }
    if data.n_strata() != 2 {
        return Err(Error::input(format!(
            "rmst2 compares two groups, and the stratum column has {} levels",
            data.n_strata()
        )));
    }
    let fit = estimate(
        data,
        &SurvfitOptions {
            timefix: opts.timefix,
            conf_type: ConfType::None,
            ..SurvfitOptions::default()
        },
    )?;
    let [curve0, curve1] = fit.groups.as_slice() else {
        return Err(Error::input(
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
                return Err(Error::input("the horizon must be finite"));
            }
            if tau > tau_largest {
                return Err(Error::input(format!(
                    "the horizon {tau} is above {tau_largest}, the largest up to which \
                     both curves are determined by the data"
                )));
            }
            tau
        }
    };
    let z = epistat_stats::normal::two_sided_z(opts.conf_level);
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
            Contrast {
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
        conf_level: opts.conf_level,
        n_rows: data.n_rows(),
        n_used: data.n_used(),
    })
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
fn contrast(centre: f64, se: f64, z: f64, exponentiate: bool) -> Contrast {
    let map = |v: f64| if exponentiate { v.exp() } else { v };
    let statistic = centre / se;
    let p_value = if statistic.is_nan() {
        None
    } else {
        Some(2.0 * epistat_stats::normal::cdf(-statistic.abs()))
    };
    Contrast {
        estimate: Some(map(centre)),
        se: Some(se),
        lower: Some(map(centre - z * se)),
        upper: Some(map(centre + z * se)),
        p_value,
    }
}

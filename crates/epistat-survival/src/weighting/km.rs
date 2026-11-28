//! Weighted Kaplan-Meier curves with a variance that accounts for estimated
//! propensity weights.
//!
//! # Algorithm
//!
//! With case weights `w`, the weight at risk at the event time `t_l` of a
//! group is `Y_l`, the sum of `w` over the rows with `start < t_l <= stop`, and
//! the weight of events is `d_l`. The curve is `S(t) = prod_{t_l <= t} (1 -
//! d_l / Y_l)`, which is `survfit` with weights. Then
//!
//! ```text
//! log S(t) = sum_{t_l <= t} log(Y_l - d_l) - log Y_l.
//! ```
//!
//! Its derivative with respect to the weight of one row (start `s`, stop `u`,
//! event indicator `delta`) is
//!
//! ```text
//! phi(t) = delta p(u) 1(u <= t) + 1(s < t) (G(min(u, t)) - G(s)),
//! p_l = -1 / (Y_l - d_l),   g_l = d_l / (Y_l (Y_l - d_l)),   G(x) = sum_{t_l <= x} g_l.
//! ```
//!
//! This is the infinitesimal jackknife of Reid (1981) and Therneau and
//! Grambsch (2000, section 2.9), which `survfit(robust = TRUE)` uses. The
//! influence of a unit is `sum of w phi` over its rows, and the variance of
//! the curve with the weights taken as known is `S(t)^2` times the sum over
//! units of the square of the influence of `log S`. A unit is a cluster, a
//! subject or a row, in that order of preference.
//!
//! [`estimate`] adds the estimation of the propensity weights. The curve
//! depends on the parameters `theta` of the weights through
//! `d log S / d theta = sum over rows of phi (d w / d theta)`, and the
//! parameters have the influence `-A^-1 psi_u` of the unit (see the
//! [module documentation of `weighting`](super)). The influence of `log S` is
//! the sum of the two, and the variance is formed from it. Covariances between
//! the curves of different groups are formed from the same influences, so a
//! contrast between arms has its own standard error.
//!
//! The curves are evaluated at the times asked for, which keeps the work at
//! `O(n log m)` per time and the memory at one value per unit.
//!
//! # Edge cases
//!
//! A curve that has reached 0, that is a risk set emptied by its own events,
//! has no variance on the log scale: its covariances are `None`. A time before
//! the first event gives a curve of 1 and a variance of 0. A group without rows
//! is not reported.
//!
//! # References
//!
//! Reid N. Influence functions for censored data. Ann Stat 1981;9(1):78-92.
//! Therneau TM, Grambsch PM. Modeling Survival Data. Springer 2000, section 2.9.
//! Cole SR, Hernan MA. Adjusted survival curves with inverse probability
//! weights. Comput Methods Programs Biomed 2004;75(1):45-49.

use epistat_core::{Contrast, Covariance, Estimate, Level, Scale};
use epistat_linalg::{Factor, Mat};
use epistat_stats::sum::Compensated;
use epistat_stats::wald;

use super::adjust::Nuisance;
use super::estimated::{Weights, prepare};
use super::iptw::Iptw;
use super::stack::{GENERAL, check_subjects, units, with_weights};
use crate::check::refuse_non_finite;
use crate::covariance;
use crate::error::{Error, Result};
use crate::surv::SurvData;
use crate::timefix::TimeFix;

/// What [`estimate`] does besides the data.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct KmOptions {
    /// The rule that makes times equal when they differ by rounding error,
    /// applied before the estimate. The default is that of `survfit`.
    pub timefix: TimeFix,
    /// The event code when the status has several codes, which are then
    /// censoring for this estimate. It is `None` or `Some(1)` for one code.
    pub cause: Option<u32>,
}

/// The weighted curves at the chosen times.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct WeightedKm {
    /// The times, as given.
    pub times: Vec<f64>,
    /// The caller's stratum code of each group, 0 without strata.
    pub groups: Vec<u32>,
    /// The survival probability, `surv[g * times.len() + t]`. The curve is 1
    /// before the first event of its group.
    pub surv: Vec<f64>,
    /// The covariance between the curves of the groups at each time that
    /// accounts for the estimation of the weights, one matrix over the groups
    /// for each entry of `times`. `None` in the row and column of a group
    /// whose curve is 0.
    pub covariance: Vec<Covariance>,
    /// The same with the weights taken as known.
    pub covariance_known: Vec<Covariance>,
}

impl WeightedKm {
    /// The standard error of the curve of `group` (a position in `groups`) at
    /// `times[time]`, accounting for the estimation of the weights. `None`
    /// where the variance is undefined and for a group or time out of range.
    #[must_use]
    pub fn se(&self, group: usize, time: usize) -> Option<f64> {
        self.covariance
            .get(time)?
            .get(group, group)
            .map(|v| v.max(0.0).sqrt())
    }

    /// The standard error with the weights taken as known, as
    /// `survfit(robust = TRUE)`.
    #[must_use]
    pub fn se_known(&self, group: usize, time: usize) -> Option<f64> {
        self.covariance_known
            .get(time)?
            .get(group, group)
            .map(|v| v.max(0.0).sqrt())
    }

    /// The standard error of the curve of `group` at `times[time]`.
    #[must_use]
    #[deprecated(since = "0.4.0", note = "use `se`")]
    pub fn standard_error(&self, group: usize, time: usize) -> Option<f64> {
        self.se(group, time)
    }

    /// The standard error with the weights taken as known.
    #[must_use]
    #[deprecated(since = "0.4.0", note = "use `se_known`")]
    pub fn standard_error_known(&self, group: usize, time: usize) -> Option<f64> {
        self.se_known(group, time)
    }

    /// The covariance of the curves of two groups at `times[time]`, `None`
    /// where it is undefined and for a group or time out of range.
    #[must_use]
    pub fn covariance_between(&self, group: usize, other: usize, time: usize) -> Option<f64> {
        self.covariance.get(time)?.get(group, other)
    }

    /// The survival of `group` (a position in `groups`) at `times[time]`,
    /// `None` for a position beyond the groups or times.
    #[must_use]
    pub fn survival(&self, group: usize, time: usize) -> Option<f64> {
        let m = self.times.len();
        if group < self.groups.len() && time < m {
            self.surv.get(group * m + time).copied()
        } else {
            None
        }
    }

    /// The position in [`Self::times`] of the last chosen time at or before
    /// `t`, `None` when every chosen time is after `t`.
    #[must_use]
    pub fn position_at(&self, t: f64) -> Option<usize> {
        self.times.iter().rposition(|&x| x <= t)
    }

    /// The risk `1 - S` of `group` (a position in `groups`) at `times[time]`,
    /// with its standard error accounting for the estimation of the weights
    /// and its normal interval at `level`.
    ///
    /// # Errors
    ///
    /// An input error when `group` or `time` is beyond the groups or times.
    pub fn risk(&self, group: usize, time: usize, level: Level) -> Result<Estimate> {
        self.check_positions(&[group], time)?;
        let m = self.times.len();
        let risk = 1.0 - self.surv[group * m + time];
        Ok(wald(risk, self.se(group, time), level, Scale::Identity))
    }

    /// The difference of the risks `(1 - S_group) - (1 - S_other)` at
    /// `times[time]`, with the standard errors of the difference that account
    /// for the estimation of the weights and that take them as known,
    /// `sqrt(V_gg + V_hh - 2 V_gh)` from the covariances of the two curves,
    /// and normal intervals at `level` with the p-values of no difference.
    /// `group` and `other` are positions in `groups`, such as the treated and
    /// the control arm.
    ///
    /// A standard error, and with it the interval, is `None` when a
    /// covariance it needs is (a curve that has reached 0).
    ///
    /// # Errors
    ///
    /// An input error when `group`, `other` or `time` is beyond the groups or
    /// times.
    pub fn risk_difference(
        &self,
        group: usize,
        other: usize,
        time: usize,
        level: Level,
    ) -> Result<KmDifference> {
        self.check_positions(&[group, other], time)?;
        let m = self.times.len();
        let se = |known: bool| -> Option<f64> {
            let var = |a: usize, b: usize| {
                if known {
                    self.covariance_known_between(a, b, time)
                } else {
                    self.covariance_between(a, b, time)
                }
            };
            let (gg, hh, gh) = (var(group, group)?, var(other, other)?, var(group, other)?);
            Some((gg + hh - 2.0 * gh).max(0.0).sqrt())
        };
        let difference = self.surv[other * m + time] - self.surv[group * m + time];
        Ok(KmDifference {
            difference: wald(difference, se(false), level, Scale::Identity),
            known: wald(difference, se(true), level, Scale::Identity),
        })
    }

    /// The curve of `group` (a position in `groups`) at `t`, read at the last
    /// chosen time at or before `t`; `None` before the first chosen time and
    /// for a group out of range.
    #[must_use]
    pub fn at(&self, group: usize, t: f64) -> Option<WeightedKmAt> {
        let time = self.position_at(t)?;
        Some(WeightedKmAt {
            time: self.times[time],
            survival: self.survival(group, time)?,
            se: self.se(group, time),
            se_known: self.se_known(group, time),
        })
    }

    /// The risks `1 - S` of `group` and `other` (positions in `groups`, such
    /// as the treated and the control arm) at `times[time]`, their difference
    /// `risk_group - risk_other` and their ratio `risk_group / risk_other`,
    /// with standard errors that account for the estimation of the weights
    /// and normal intervals at `level`.
    ///
    /// The difference has the variance `w' V w` of the covariance `V` of the
    /// two curves at the time, `w = (1, -1)` on the two groups
    /// ([`Covariance::quadratic`]). The ratio has the delta-method variance of
    /// its logarithm, `V_gg / r_g^2 + V_hh / r_h^2 - 2 V_gh / (r_g r_h)`, and
    /// its interval and p-value are formed on the log scale
    /// ([`Scale::Log`]). The ratio is `None` when the risk of `other` is 0; a
    /// standard error is `None` where a covariance it needs is.
    ///
    /// # Errors
    ///
    /// An input error when `group`, `other` or `time` is beyond the groups or
    /// times.
    pub fn contrast(
        &self,
        group: usize,
        other: usize,
        time: usize,
        level: Level,
    ) -> Result<Contrast> {
        self.check_positions(&[group, other], time)?;
        let m = self.times.len();
        let (r1, r0) = (
            1.0 - self.surv[group * m + time],
            1.0 - self.surv[other * m + time],
        );
        let covariance = &self.covariance[time];
        let mut w = vec![0.0; covariance.dim()];
        w[group] = 1.0;
        w[other] -= 1.0;
        let difference_se = covariance.quadratic(&w).map(|v| v.max(0.0).sqrt());
        let ratio = (r0 != 0.0).then(|| {
            let se = (|| {
                let (gg, hh, gh) = (
                    covariance.get(group, group)?,
                    covariance.get(other, other)?,
                    covariance.get(group, other)?,
                );
                let v = gg / (r1 * r1) + hh / (r0 * r0) - 2.0 * gh / (r1 * r0);
                v.is_finite().then(|| v.max(0.0).sqrt())
            })();
            wald(r1 / r0, se, level, Scale::Log)
        });
        Ok(Contrast::new(
            wald(r1, self.se(group, time), level, Scale::Identity),
            wald(r0, self.se(other, time), level, Scale::Identity),
            wald(r1 - r0, difference_se, level, Scale::Identity),
            ratio,
        ))
    }

    /// The covariance of the curves of two groups at `times[time]` with the
    /// weights taken as known.
    #[must_use]
    pub fn covariance_known_between(&self, group: usize, other: usize, time: usize) -> Option<f64> {
        self.covariance_known.get(time)?.get(group, other)
    }

    fn check_positions(&self, groups: &[usize], time: usize) -> Result<()> {
        let n = self.groups.len();
        if let Some(&g) = groups.iter().find(|&&g| g >= n) {
            return Err(Error::out_of_range(format!(
                "group: position {g} is beyond the {n} groups"
            )));
        }
        if time >= self.times.len() {
            return Err(Error::out_of_range(format!(
                "time: position {time} is beyond the {} times",
                self.times.len()
            )));
        }
        if self.surv.len() != n * self.times.len() {
            return Err(Error::mismatch(format!(
                "surv: {} values for {n} groups by {} times",
                self.surv.len(),
                self.times.len()
            )));
        }
        Ok(())
    }
}

/// One weighted curve read at a time, from [`WeightedKm::at`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightedKmAt {
    /// The chosen time read: the last at or before the time asked for.
    pub time: f64,
    /// The survival probability.
    pub survival: f64,
    /// Its standard error accounting for the estimation of the weights.
    pub se: Option<f64>,
    /// Its standard error with the weights taken as known.
    pub se_known: Option<f64>,
}

/// The difference of the risks of two weighted curves at one time.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KmDifference {
    /// `(1 - S_group) - (1 - S_other)`, with the standard error and interval
    /// that account for the estimation of the weights.
    pub difference: Estimate,
    /// The same difference with the weights taken as known.
    pub known: Estimate,
}

/// What the variance needs of the estimated weights.
struct Correction<'a> {
    iptw: &'a Iptw,
    /// The influence of the parameters of the weights, one row per unit.
    influence: Mat<f64>,
    /// The subject of each row.
    subject: Vec<u32>,
}

/// The weighted curves of `data` at `times`, with the variance that accounts for
/// each model of the weights being estimated.
///
/// The groups are the strata of `data`, normally the treatment arm. With
/// [`Weights::known`], the case weights of `data`, or none, are taken as known
/// and the variance is the infinitesimal jackknife of `survfit(robust = TRUE)`.
/// With propensity weights alone, the variance stacks the propensity model
/// with the curves. Otherwise each model of the weights adds its term to the
/// influence of a unit, as for [`cox::fit`](super::cox::fit); for propensity
/// weights the two agree up to rounding. The units are those of
/// [`cox::fit`](super::cox::fit). With estimated weights `data` must carry no
/// case weights.
///
/// # Errors
///
/// An input error for interval-censored data, for data with case weights and
/// estimated weights, for `subject` of another length than the data or naming
/// a subject that does not exist, for a status with several codes and no
/// `cause`, for a time that is not finite, for the invalid input of the models
/// of the weights, and the errors of a solve that fails.
pub fn estimate(
    data: &SurvData<'_>,
    times: &[f64],
    opts: &KmOptions,
    weights: &Weights<'_>,
) -> Result<WeightedKm> {
    if weights.is_known() {
        return estimate_known_core(data, times, opts);
    }
    match (weights.iptw, weights.censoring) {
        (Some(iptw), None) => estimate_iptw(data, times, opts, iptw, weights.subject),
        _ => estimate_general(data, times, opts, weights),
    }
}

/// The curves weighted by `iptw`, with the stacked variance.
fn estimate_iptw(
    data: &SurvData<'_>,
    times: &[f64],
    opts: &KmOptions,
    iptw: &Iptw,
    subject: Option<&[u32]>,
) -> Result<WeightedKm> {
    let n = data.n_rows();
    let n_subjects = iptw.len();
    check_subjects(n, n_subjects, subject)?;
    let subj = |row: usize| subject.map_or(row, |s| s[row] as usize);
    let weight: Vec<f64> = (0..n).map(|row| iptw.weights[subj(row)]).collect();
    let buf = with_weights(data, weight)?;
    let weighted = buf.as_data().with_timefix(opts.timefix)?;
    let (unit_of_subject, n_units) = units(n_subjects, &subj, data.cluster(), data.n_clusters())?;
    let k = iptw.n_params();
    let mut psi = Mat::<f64>::zeros(k, n_units);
    let mut buffer = vec![0.0; k];
    for i in 0..n_subjects {
        iptw.score_into(i, &mut buffer);
        for (m, &b) in buffer.iter().enumerate() {
            psi[(m, unit_of_subject[i])] += b;
        }
    }
    let bread = iptw.bread()?;
    let solved = Factor::new(bread.as_ref(), &GENERAL).and_then(|f| f.solve(psi.as_ref()))?;
    // The influence of the parameters is minus the solution, one row per unit.
    let influence = Mat::from_fn(n_units, k, |u, m| -solved[(m, u)]);
    let unit_of_row: Vec<usize> = (0..n).map(|row| unit_of_subject[subj(row)]).collect();
    let correction = Correction {
        iptw,
        influence,
        subject: (0..n).map(|row| subj(row) as u32).collect(),
    };
    curves(
        &weighted,
        times,
        opts,
        &unit_of_row,
        n_units,
        Some(&correction),
        None,
    )
}

/// The curves of [`estimate`] for `weights`, through the terms of every model
/// of the weights also for propensity weights alone, as 0.3 gave them.
///
/// # Errors
///
/// As [`estimate`].
#[deprecated(since = "0.4.0", note = "use `estimate`")]
pub fn estimate_estimated(
    data: &SurvData<'_>,
    times: &[f64],
    opts: &KmOptions,
    weights: &Weights<'_>,
) -> Result<WeightedKm> {
    estimate_general(data, times, opts, weights)
}

/// The curves weighted by the weights of `estimated`, with the terms of every
/// model of the weights.
fn estimate_general(
    data: &SurvData<'_>,
    times: &[f64],
    opts: &KmOptions,
    estimated: &Weights<'_>,
) -> Result<WeightedKm> {
    let prepared = prepare(data, estimated)?;
    let buf = with_weights(data, prepared.weight.clone())?;
    let weighted = buf.as_data().with_timefix(opts.timefix)?;
    curves(
        &weighted,
        times,
        opts,
        &prepared.unit_of_row,
        prepared.n_units,
        None,
        Some(&prepared.nuisances),
    )
}

/// The curves of `data` with its case weights taken as known, as
/// [`estimate`] gives them for [`Weights::known`].
///
/// # Errors
///
/// As [`estimate`].
#[deprecated(since = "0.4.0", note = "use `estimate` with `Weights::known()`")]
pub fn estimate_known(data: &SurvData<'_>, times: &[f64], opts: &KmOptions) -> Result<WeightedKm> {
    estimate_known_core(data, times, opts)
}

/// The curves with the case weights of `data` taken as known, with the
/// infinitesimal jackknife variance of `survfit(robust = TRUE)`.
fn estimate_known_core(data: &SurvData<'_>, times: &[f64], opts: &KmOptions) -> Result<WeightedKm> {
    let weighted = data.clone().with_timefix(opts.timefix)?;
    let n = weighted.n_rows();
    let (unit_of_row, n_units): (Vec<usize>, usize) = match (weighted.cluster(), weighted.id()) {
        (Some(c), _) => (
            c.iter().map(|&u| u as usize).collect(),
            weighted.n_clusters(),
        ),
        (None, Some(i)) => (i.iter().map(|&u| u as usize).collect(), weighted.n_ids()),
        (None, None) => ((0..n).collect(), n),
    };
    curves(&weighted, times, opts, &unit_of_row, n_units, None, None)
}

/// The rows of one group and what the influence of each needs.
struct Group {
    label: u32,
    events: Vec<f64>,
    /// `p_l` and the prefix sums of `g_l`, one more entry than events.
    p: Vec<f64>,
    gsum: Vec<f64>,
    /// First event index at which the risk set is exhausted.
    dead: usize,
    s_at: Vec<f64>,
    rows: Vec<u32>,
    /// Per row of the group: event index of its stop, events at or before
    /// its stop and at or before its start.
    pos: Vec<u32>,
    at_stop: Vec<u32>,
    at_start: Vec<u32>,
}

fn curves(
    data: &SurvData<'_>,
    times: &[f64],
    opts: &KmOptions,
    unit_of_row: &[usize],
    n_units: usize,
    correction: Option<&Correction<'_>>,
    general: Option<&[Box<dyn Nuisance + '_>]>,
) -> Result<WeightedKm> {
    refuse_non_finite(times, "times")?;
    let surv = data.followup();
    let code = match (surv.n_codes(), opts.cause) {
        (1, None | Some(1)) => 1,
        (1, Some(c)) => {
            return Err(Error::out_of_range(format!(
                "cause: {c} is outside the single code 1"
            )));
        }
        (_, Some(c)) if (1..=surv.n_codes()).contains(&c) => i32::try_from(c).unwrap_or(i32::MAX),
        (_, Some(c)) => {
            return Err(Error::out_of_range(format!(
                "cause: {c} is outside 1..={}",
                surv.n_codes()
            )));
        }
        (_, None) => {
            return Err(Error::missing(
                "cause: needed when the status has several codes",
            ));
        }
    };
    let n = data.n_rows();
    let weight = data.weight().map_or_else(|| vec![1.0; n], <[f64]>::to_vec);
    let (start, stop, status) = (surv.start(), surv.stop(), surv.status());
    let n_groups = data.n_strata().max(1);
    let group_of = |row: usize| data.stratum().map_or(0, |s| s[row] as usize);
    let k = correction.map_or(0, |c| c.influence.ncols());
    let n_times = times.len();

    let mut groups = Vec::new();
    let mut surv_out = Vec::new();
    let mut built: Vec<Group> = Vec::new();
    for g in 0..n_groups {
        let rows: Vec<u32> = (0..n)
            .filter(|&r| group_of(r) == g && weight[r] > 0.0)
            .map(|r| r as u32)
            .collect();
        if rows.is_empty() {
            continue;
        }
        let is_event = |r: usize| status[r] == code;
        let mut events: Vec<f64> = rows
            .iter()
            .map(|&r| r as usize)
            .filter(|&r| is_event(r))
            .map(|r| stop[r])
            .collect();
        events.sort_unstable_by(f64::total_cmp);
        events.dedup();
        let m = events.len();
        // Weight of events and at risk at each event time, by descending sums.
        let mut d = vec![0.0; m];
        let mut by_stop: Vec<u32> = rows.clone();
        by_stop.sort_unstable_by(|&a, &b| stop[a as usize].total_cmp(&stop[b as usize]));
        let mut above_stop = vec![0.0; m];
        let mut above_start = vec![0.0; m];
        {
            // Sum of weights with stop >= t_l, by sweeping rows from the back.
            let mut acc = Compensated::new();
            let mut j = by_stop.len();
            for l in (0..m).rev() {
                while j > 0 && stop[by_stop[j - 1] as usize] >= events[l] {
                    acc.add(weight[by_stop[j - 1] as usize]);
                    j -= 1;
                }
                above_stop[l] = acc.value();
            }
            if let Some(start) = start {
                let mut by_start: Vec<u32> = rows.clone();
                by_start.sort_unstable_by(|&a, &b| start[a as usize].total_cmp(&start[b as usize]));
                let mut acc = Compensated::new();
                let mut j = by_start.len();
                for l in (0..m).rev() {
                    while j > 0 && start[by_start[j - 1] as usize] >= events[l] {
                        acc.add(weight[by_start[j - 1] as usize]);
                        j -= 1;
                    }
                    above_start[l] = acc.value();
                }
            }
            let mut l = 0;
            let mut acc = Compensated::new();
            for &r in &by_stop {
                let r = r as usize;
                if !is_event(r) {
                    continue;
                }
                while events[l] < stop[r] {
                    d[l] = acc.value();
                    acc = Compensated::new();
                    l += 1;
                }
                acc.add(weight[r]);
            }
            if m > 0 {
                d[l] = acc.value();
            }
        }
        let y: Vec<f64> = (0..m).map(|l| above_stop[l] - above_start[l]).collect();
        let mut dead = m;
        let mut p = vec![0.0; m];
        let mut gsum = vec![0.0; m + 1];
        let mut s_at = vec![1.0; m + 1];
        for l in 0..m {
            let rest = y[l] - d[l];
            if dead == m && rest <= 8.0 * f64::EPSILON * y[l] {
                dead = l;
            }
            if dead == m {
                p[l] = -1.0 / rest;
                gsum[l + 1] = gsum[l] + d[l] / (y[l] * rest);
                s_at[l + 1] = s_at[l] * (rest / y[l]);
            } else {
                gsum[l + 1] = gsum[l];
                s_at[l + 1] = 0.0;
            }
        }
        let pos: Vec<u32> = rows
            .iter()
            .map(|&r| events.partition_point(|&e| e < stop[r as usize]) as u32)
            .collect();
        let at_stop: Vec<u32> = rows
            .iter()
            .map(|&r| events.partition_point(|&e| e <= stop[r as usize]) as u32)
            .collect();
        let at_start: Vec<u32> = rows
            .iter()
            .map(|&r| {
                start.map_or(0, |s| {
                    events.partition_point(|&e| e <= s[r as usize]) as u32
                })
            })
            .collect();
        let label = data
            .stratum_levels()
            .map_or(0, |lv| lv.code(g).unwrap_or(0));
        built.push(Group {
            label,
            events,
            p,
            gsum,
            dead,
            s_at,
            rows,
            pos,
            at_stop,
            at_start,
        });
    }
    let n_out = built.len();
    for b in &built {
        groups.push(b.label);
    }
    for b in &built {
        for &t in times {
            let ct = b.events.partition_point(|&e| e <= t);
            surv_out.push(b.s_at[ct]);
        }
    }
    let mut cov = vec![None; n_times * n_out * n_out];
    let mut cov_known = vec![None; n_times * n_out * n_out];
    let mut grad = vec![0.0; k];
    for (ti, &t) in times.iter().enumerate() {
        // Influence of every unit on log S of every group at t.
        let mut known: Vec<Vec<f64>> = vec![vec![0.0; n_units]; n_out];
        let mut adjusted: Vec<Vec<f64>> = Vec::new();
        let mut defined = vec![true; n_out];
        let mut c = vec![vec![0.0; k]; n_out];
        let mut a_mat = general.map(|_| Mat::<f64>::zeros(data.n_rows(), n_out));
        for (gi, b) in built.iter().enumerate() {
            let ct = b.events.partition_point(|&e| e <= t);
            if ct > b.dead || b.s_at[ct] <= 0.0 {
                defined[gi] = false;
                continue;
            }
            for (idx, &r) in b.rows.iter().enumerate() {
                let r = r as usize;
                let mut phi = 0.0;
                if status[r] == code && (b.pos[idx] as usize) < ct {
                    phi += b.p[b.pos[idx] as usize];
                }
                let entered = start.is_none_or(|s| s[r] < t);
                if entered {
                    let upper = (b.at_stop[idx] as usize).min(ct);
                    phi += b.gsum[upper] - b.gsum[b.at_start[idx] as usize];
                }
                known[gi][unit_of_row[r]] += weight[r] * phi;
                if let Some(am) = a_mat.as_mut() {
                    am[(r, gi)] = weight[r] * phi;
                }
                if let Some(cor) = correction {
                    cor.iptw
                        .weight_gradient_into(cor.subject[r] as usize, &mut grad);
                    for (m, &gm) in grad.iter().enumerate() {
                        c[gi][m] += phi * gm;
                    }
                }
            }
        }
        if let Some(cor) = correction {
            for gi in 0..n_out {
                let mut v = known[gi].clone();
                for (u, x) in v.iter_mut().enumerate() {
                    for (m, &cm) in c[gi].iter().enumerate() {
                        *x += cm * cor.influence[(u, m)];
                    }
                }
                adjusted.push(v);
            }
        }
        if let (Some(nuisances), Some(am)) = (general, &a_mat) {
            let mut extra = Mat::<f64>::zeros(n_units, n_out);
            for nuisance in nuisances {
                nuisance.add(am.as_ref(), &mut extra)?;
            }
            for gi in 0..n_out {
                adjusted.push(
                    (0..n_units)
                        .map(|u| known[gi][u] + extra[(u, gi)])
                        .collect(),
                );
            }
        }
        let s_of = |gi: usize| built[gi].s_at[built[gi].events.partition_point(|&e| e <= t)];
        for gi in 0..n_out {
            for hi in 0..n_out {
                if !(defined[gi] && defined[hi]) {
                    continue;
                }
                let scale = s_of(gi) * s_of(hi);
                let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
                cov_known[(ti * n_out + gi) * n_out + hi] =
                    Some(scale * dot(&known[gi], &known[hi]));
                cov[(ti * n_out + gi) * n_out + hi] = Some(
                    scale
                        * if correction.is_some() || general.is_some() {
                            dot(&adjusted[gi], &adjusted[hi])
                        } else {
                            dot(&known[gi], &known[hi])
                        },
                );
            }
        }
    }
    Ok(WeightedKm {
        times: times.to_vec(),
        groups,
        surv: surv_out,
        covariance: covariance::option_blocks(n_times, n_out, &cov)?,
        covariance_known: covariance::option_blocks(n_times, n_out, &cov_known)?,
    })
}

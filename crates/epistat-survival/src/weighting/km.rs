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

use epistat_linalg::{Factor, Mat};
use epistat_stats::sum::Compensated;

use super::iptw::Iptw;
use super::stack::{GENERAL, check_subjects, units, with_weights};
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
pub struct WeightedKm {
    /// The times, as given.
    pub times: Vec<f64>,
    /// The caller's stratum code of each group, 0 without strata.
    pub groups: Vec<u32>,
    /// The survival probability, `surv[g * times.len() + t]`. The curve is 1
    /// before the first event of its group.
    pub surv: Vec<f64>,
    /// The covariance between the curves of the groups at each time that
    /// accounts for the estimation of the weights, `[(t * G + g) * G + h]`.
    /// `None` for a group whose curve is 0.
    pub covariance: Vec<Option<f64>>,
    /// The same with the weights taken as known.
    pub covariance_known: Vec<Option<f64>>,
}

impl WeightedKm {
    fn index(&self, group: usize, time: usize, other: usize) -> usize {
        let g = self.groups.len();
        (time * g + group) * g + other
    }

    /// The standard error of the curve of `group` (a position in `groups`) at
    /// `times[time]`, accounting for the estimation of the weights.
    #[must_use]
    pub fn standard_error(&self, group: usize, time: usize) -> Option<f64> {
        self.covariance[self.index(group, time, group)].map(|v| v.max(0.0).sqrt())
    }

    /// The standard error with the weights taken as known, as
    /// `survfit(robust = TRUE)`.
    #[must_use]
    pub fn standard_error_known(&self, group: usize, time: usize) -> Option<f64> {
        self.covariance_known[self.index(group, time, group)].map(|v| v.max(0.0).sqrt())
    }

    /// The covariance of the curves of two groups at `times[time]`.
    #[must_use]
    pub fn covariance_between(&self, group: usize, other: usize, time: usize) -> Option<f64> {
        self.covariance[self.index(group, time, other)]
    }
}

/// What the variance needs of the estimated weights.
struct Correction<'a> {
    iptw: &'a Iptw<'a>,
    /// The influence of the parameters of the weights, one row per unit.
    influence: Mat<f64>,
    /// The subject of each row.
    subject: Vec<u32>,
}

/// The curves of `data` weighted by `iptw`, with the variance that accounts
/// for the weights being estimated.
///
/// `subject` and the groups are as for [`cox::fit`](super::cox::fit): the
/// stratum of `data` gives the groups, normally the treatment arm. `data` must
/// carry no case weights.
///
/// # Errors
///
/// [`Error::Input`] for data with case weights or interval-censored data, for
/// `subject` of another length than the data or naming a subject that does not
/// exist, for a status with several codes and no `cause`, for a time that is
/// not finite, and when the stacked system cannot be solved.
pub fn estimate(
    data: &SurvData<'_>,
    times: &[f64],
    opts: &KmOptions,
    iptw: &Iptw<'_>,
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
    let solved = Factor::new(bread.as_ref(), &GENERAL)
        .and_then(|f| f.solve(psi.as_ref()))
        .map_err(|e| Error::input(format!("the propensity model: {e}")))?;
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
    )
}

/// The curves of `data` with its case weights taken as known, with the
/// infinitesimal jackknife variance of `survfit(robust = TRUE)`.
///
/// Use it for weights that are not estimated here, or for the comparison with
/// [`estimate`].
///
/// # Errors
///
/// As for [`estimate`], except for the checks of the weights.
pub fn estimate_known(data: &SurvData<'_>, times: &[f64], opts: &KmOptions) -> Result<WeightedKm> {
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
    curves(&weighted, times, opts, &unit_of_row, n_units, None)
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
) -> Result<WeightedKm> {
    if let Some(t) = times.iter().find(|t| !t.is_finite()) {
        return Err(Error::input(format!("times: {t} is not finite")));
    }
    let surv = data.followup()?;
    let code = match (surv.n_codes(), opts.cause) {
        (1, None | Some(1)) => 1,
        (1, Some(c)) => {
            return Err(Error::input(format!(
                "cause: {c} is outside the single code 1"
            )));
        }
        (_, Some(c)) if (1..=surv.n_codes()).contains(&c) => i32::try_from(c).unwrap_or(i32::MAX),
        (_, Some(c)) => {
            return Err(Error::input(format!(
                "cause: {c} is outside 1..={}",
                surv.n_codes()
            )));
        }
        (_, None) => {
            return Err(Error::input(
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
                        * if correction.is_some() {
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
        covariance: cov,
        covariance_known: cov_known,
    })
}

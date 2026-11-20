//! The curve of one group: risk sets, steps, variance and limits.

use super::confint::{Scale, limits, lower_se};
use super::jackknife::{Rows, sum_of_squares};
use super::{ConfLower, ConfType, GroupCurve, HazardType, SurvivalType, Variance};

/// The rows of one group, with the columns the curve needs.
pub(super) struct GroupRows {
    pub(super) start: Option<Vec<f64>>,
    pub(super) stop: Vec<f64>,
    pub(super) event: Vec<bool>,
    pub(super) weight: Vec<f64>,
    /// The subject of each row, when there is an id column.
    pub(super) id: Option<Vec<u32>>,
    /// Dense cluster code of each row for the jackknife variance.
    pub(super) cluster: Vec<u32>,
    pub(super) n_clusters: usize,
}

/// What a group is fitted with.
pub(super) struct Settings {
    pub(super) survival: SurvivalType,
    pub(super) hazard: HazardType,
    pub(super) conf_type: ConfType,
    pub(super) conf_lower: ConfLower,
    pub(super) z: f64,
    pub(super) variance: Variance,
}

/// The counts at each time that enters the curve.
struct Counts {
    time: Vec<f64>,
    count_risk: Vec<u32>,
    count_event: Vec<u32>,
    count_censor: Vec<u32>,
    n_risk: Vec<f64>,
    n_event: Vec<f64>,
    n_censor: Vec<f64>,
}

/// Row indices ordered by `key`, ties by row index.
fn order_by(key: &[f64]) -> Vec<u32> {
    let mut index: Vec<u32> = (0..u32::try_from(key.len()).expect("rows fit u32")).collect();
    index.sort_unstable_by(|&a, &b| key[a as usize].total_cmp(&key[b as usize]).then(a.cmp(&b)));
    index
}

/// Whether each row is the last of its subject's chain of intervals. A chain
/// breaks at a new subject or at a gap between one row's stop and the next
/// row's start. Without an id every row stands alone.
fn last_of_chain(rows: &GroupRows, by_stop: &[u32]) -> Vec<bool> {
    let n = rows.stop.len();
    let (Some(start), Some(id)) = (&rows.start, &rows.id) else {
        return vec![true; n];
    };
    // Order by subject, then stop, ties by row.
    let mut index = by_stop.to_vec();
    index.sort_by_key(|&r| id[r as usize]);
    let mut last = vec![false; n];
    for pair in index.windows(2) {
        let (a, b) = (pair[0] as usize, pair[1] as usize);
        last[a] = id[a] != id[b] || rows.stop[a] < start[b];
    }
    if let Some(&r) = index.last() {
        last[r as usize] = true;
    }
    last
}

/// The risk sets and counts at every time the curve reports.
///
/// A time is reported when an event happens there or a chain of intervals ends
/// there, so a subject whose follow-up is cut into pieces is not counted as
/// censored at each cut.
#[expect(
    clippy::float_cmp,
    reason = "rows tie when their stop times are the same number"
)]
fn count(rows: &GroupRows, by_start: &[u32], by_stop: &[u32], last: &[bool]) -> Counts {
    let n = rows.stop.len();
    let mut time = Vec::new();
    let mut count_risk = Vec::new();
    let mut count_event = Vec::new();
    let mut count_censor = Vec::new();
    let mut n_risk = Vec::new();
    let mut n_event = Vec::new();
    let mut n_censor = Vec::new();
    let mut at_risk = 0u32;
    let mut weight_at_risk = 0.0;
    let mut removed = by_start.len();
    let mut hi = n;
    while hi > 0 {
        let u = rows.stop[by_stop[hi - 1] as usize];
        let mut lo = hi;
        while lo > 0 && rows.stop[by_stop[lo - 1] as usize] == u {
            lo -= 1;
        }
        let (mut events, mut censors) = (0u32, 0u32);
        let (mut weight_events, mut weight_censors) = (0.0, 0.0);
        let mut reported = false;
        for &r in by_stop[lo..hi].iter().rev() {
            let r = r as usize;
            at_risk += 1;
            weight_at_risk += rows.weight[r];
            if rows.event[r] {
                events += 1;
                weight_events += rows.weight[r];
                reported = true;
            } else if last[r] {
                censors += 1;
                weight_censors += rows.weight[r];
                reported = true;
            }
        }
        if let Some(start) = &rows.start {
            while removed > 0 && start[by_start[removed - 1] as usize] >= u {
                removed -= 1;
                at_risk -= 1;
                weight_at_risk -= rows.weight[by_start[removed] as usize];
            }
        }
        if reported {
            time.push(u);
            count_risk.push(at_risk);
            count_event.push(events);
            count_censor.push(censors);
            n_risk.push(weight_at_risk);
            n_event.push(weight_events);
            n_censor.push(weight_censors);
        }
        hi = lo;
    }
    time.reverse();
    count_risk.reverse();
    count_event.reverse();
    count_censor.reverse();
    n_risk.reverse();
    n_event.reverse();
    n_censor.reverse();
    Counts {
        time,
        count_risk,
        count_event,
        count_censor,
        n_risk,
        n_event,
        n_censor,
    }
}

/// The weight and the unweighted number of events of the correction for tied
/// events: `b_j = 1 / (Y - j a)` for `j` below the number of events `d`, with
/// `a = weight of events / d`.
fn fleming_terms(y: f64, d1: f64, d0: u32) -> (f64, f64, f64) {
    let (mut plain_sum, mut plain_sq, mut weighted_sq) = (0.0, 0.0, 0.0);
    let a = d1 / f64::from(d0);
    for j in 0..d0 {
        let b = 1.0 / (y - f64::from(j) * a);
        plain_sum += b;
        plain_sq += b * b;
        weighted_sq += f64::from(j) * b * b;
    }
    (plain_sum, plain_sq, weighted_sq)
}

/// The standard error of the log of the lower limit scale: Greenwood sums or
/// the jackknife, carried over the times without events.
fn carry(values: &[f64], steps: &[usize], len: usize) -> Vec<f64> {
    let mut out = vec![0.0; len];
    let mut next = 0;
    let mut current = 0.0;
    for (i, slot) in out.iter_mut().enumerate() {
        if next < steps.len() && steps[next] == i {
            current = values[next];
            next += 1;
        }
        *slot = current;
    }
    out
}

/// Fit one group.
#[expect(
    clippy::too_many_lines,
    reason = "one pass over the times, then the variance, then the limits"
)]
pub(super) fn fit(rows: &GroupRows, s: &Settings, origin: f64) -> GroupCurve {
    let n = rows.stop.len();
    let by_stop = order_by(&rows.stop);
    let by_start = rows.start.as_deref().map(order_by).unwrap_or_default();
    let last = last_of_chain(rows, &by_stop);
    let counts = count(rows, &by_start, &by_stop, &last);
    let m = counts.time.len();

    let mut surv = Vec::with_capacity(m);
    let mut cumhaz = Vec::with_capacity(m);
    let mut greenwood = Vec::with_capacity(m);
    let mut var_h = Vec::with_capacity(m);
    // Steps at which an event with positive weight occurs, and what the
    // jackknife needs there.
    let mut steps = Vec::new();
    let mut p_km = Vec::new();
    let mut q_km = Vec::new();
    let mut p_h = Vec::new();
    let mut q_h = Vec::new();
    let mut zero_step: Option<usize> = None;
    let (mut km, mut nelson, mut v1, mut v2) = (1.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..m {
        let y = counts.n_risk[i];
        let d1 = counts.n_event[i];
        let d0 = counts.count_event[i];
        if d0 > 0 && d1 > 0.0 {
            match s.hazard {
                HazardType::NelsonAalen => {
                    nelson += d1 / y;
                    v2 += d1 / (y * y);
                    p_h.push(1.0 / y);
                    q_h.push(d1 / (y * y));
                }
                HazardType::FlemingHarrington => {
                    let a = d1 / f64::from(d0);
                    for j in 0..d0 {
                        let denominator = y - f64::from(j) * a;
                        nelson += d1 / (f64::from(d0) * denominator);
                        v2 += d1 / (f64::from(d0) * denominator * denominator);
                    }
                    let (plain_sum, plain_sq, weighted_sq) = fleming_terms(y, d1, d0);
                    p_h.push(plain_sum / f64::from(d0) + a / f64::from(d0) * weighted_sq);
                    q_h.push(a * plain_sq);
                }
            }
            km *= (y - d1) / y;
            v1 += d1 / (y * (y - d1));
            steps.push(i);
            if y - d1 == 0.0 && zero_step.is_none() {
                zero_step = Some(steps.len() - 1);
            } else if zero_step.is_none() {
                p_km.push(1.0 / (y - d1));
                q_km.push(d1 / (y * (y - d1)));
            }
        }
        cumhaz.push(nelson);
        var_h.push(v2);
        greenwood.push(v1);
        surv.push(match s.survival {
            SurvivalType::KaplanMeier => km,
            SurvivalType::ExpCumulativeHazard => (-nelson).exp(),
        });
    }

    // Standard errors: of the cumulative hazard and of the survival, and the
    // one the limits are formed from with its scale.
    let (se_cumhaz, se_surv, se_raw, scale): (Vec<f64>, Vec<Option<f64>>, Vec<f64>, Scale);
    match s.variance {
        Variance::Greenwood => {
            se_cumhaz = var_h.iter().map(|v| v.sqrt()).collect();
            match s.survival {
                SurvivalType::KaplanMeier => {
                    se_raw = greenwood.iter().map(|v| v.sqrt()).collect();
                }
                SurvivalType::ExpCumulativeHazard => se_raw = se_cumhaz.clone(),
            }
            se_surv = surv
                .iter()
                .zip(&se_raw)
                .map(|(&p, &se)| (!se.is_infinite()).then_some(p * se))
                .collect();
            scale = Scale::Log;
        }
        Variance::InfinitesimalJackknife => {
            let jack = Rows {
                start: rows.start.as_deref(),
                stop: &rows.stop,
                event: &rows.event,
                weight: &rows.weight,
                cluster: &rows.cluster,
                n_clusters: rows.n_clusters,
                by_start: &by_start,
                by_stop: &by_stop,
            };
            let times: Vec<f64> = steps.iter().map(|&i| counts.time[i]).collect();
            let hazard_squares = sum_of_squares(&jack, &times, &p_h, &q_h);
            se_cumhaz = carry(&hazard_squares, &steps, m)
                .into_iter()
                .map(f64::sqrt)
                .collect();
            match s.survival {
                SurvivalType::KaplanMeier => {
                    // The curve is zero from the first time nobody is left
                    // at risk, and so is its influence.
                    let used = p_km.len();
                    let squares = sum_of_squares(&jack, &times[..used], &p_km, &q_km);
                    let mut all = squares;
                    all.resize(steps.len(), 0.0);
                    let direct = carry(&all, &steps, m);
                    se_surv = surv
                        .iter()
                        .zip(&direct)
                        .map(|(&p, &v)| Some(p * v.sqrt()))
                        .collect();
                    se_raw = se_surv.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
                    scale = Scale::Surv;
                }
                SurvivalType::ExpCumulativeHazard => {
                    se_surv = surv
                        .iter()
                        .zip(&se_cumhaz)
                        .map(|(&p, &se)| Some(p * se))
                        .collect();
                    se_raw = se_cumhaz.clone();
                    scale = Scale::Log;
                }
            }
        }
    }

    // Limits.
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    if s.conf_type != ConfType::None {
        // The weighted number at risk at the last time with an event, or at
        // the first time of the curve before any.
        let mut lag = Vec::with_capacity(m);
        let mut current = counts.n_risk.first().copied().unwrap_or(f64::NAN);
        for i in 0..m {
            if counts.n_event[i] > 0.0 {
                current = counts.n_risk[i];
            }
            lag.push(current);
        }
        for i in 0..m {
            let low_se = lower_se(s.conf_lower, surv[i], se_raw[i], counts.n_risk[i], lag[i]);
            let (lo, hi) = limits(s.conf_type, s.z, surv[i], se_raw[i], scale, low_se);
            lower.push((!lo.is_nan()).then_some(lo));
            upper.push((!hi.is_nan()).then_some(hi));
        }
    }

    let n_id = rows.id.as_ref().map(|id| {
        let mut ids = id.clone();
        ids.sort_unstable();
        ids.dedup();
        ids.len()
    });
    GroupCurve {
        group: 0,
        n,
        n_id,
        origin,
        time: counts.time,
        n_risk: counts.n_risk,
        n_event: counts.n_event,
        n_censor: counts.n_censor,
        count_risk: counts.count_risk,
        count_event: counts.count_event,
        count_censor: counts.count_censor,
        surv,
        se: se_surv,
        lower,
        upper,
        cumhaz,
        se_cumhaz,
    }
}

//! Drug use periods from purchases by the PRE2DUP method.
//!
//! PRE2DUP, from Tanskanen et al. \[1\], builds drug use periods from each
//! person's own pattern of purchases. Where [`crate::exposure`] gives every
//! dispensing a duration from a dose the caller assumes, PRE2DUP reads the
//! daily dose from a sliding average over the person's neighbouring
//! purchases, widens it by the regularity of the person's purchases, and
//! falls back on package and drug class parameters only where the person's
//! history says too little. This module implements the core of the method as
//! the paper describes it, with the parameters supplied by the caller.
//!
//! # Method
//!
//! Per person and drug group (the paper's ATC code), purchases on one day are
//! added into one purchase, and the purchases are taken in order of day.
//!
//! **Refill times.** `T_i` is the number of days from purchase `i` to purchase
//! `i + 1`, less the hospital days between them. For the last purchase the
//! dose of the one before is assumed, `T_i = DDD_i / (DDD_{i-1} / T_{i-1})`.
//!
//! **Temporal average** (equation 1). The history is cut where `T_i` exceeds
//! the maximum refill time of the averages (300 days in the paper), and in
//! each run of at least three purchases
//!
//! ```text
//! DDDAVG_i = (DDD_{i-1} + 4 DDD_i + DDD_{i+1}) / (T_{i-1} + 4 T_i + T_{i+1})
//! ```
//!
//! with the weights `(5, 1)` on `(i, i + 1)` for the first purchase of the run
//! and `(1, 5)` on `(i - 1, i)` for the last. Runs of one or two purchases
//! have no average. `DDDAVGcv` is the standard deviation of the person's
//! averages in the group over their mean.
//!
//! **Expected refill length** (equation 2). For a purchase with an average,
//! `d_i = min(DDDAVG_i, upper)` and
//!
//! ```text
//! ERFL_i = min(DDD_i (1 + DVAR * DDDAVGcv) / d_i, max_refill_days),   DVAR = 0.5
//! ```
//!
//! **Joining.** The period continues from purchase `i` to `i + 1` when
//! `ERFL_i >= T_i`. Otherwise the stockpiling test: when `DDDAVG_i` is below
//! both `DDDAVG_{i-1}` and `DDDAVG_{i+1}`, purchases `i - 1` and `i` are taken
//! together as if bought at `i - 1`, and the period continues when their
//! expected refill length reaches `T_{i-1} + T_i`. A purchase whose average is
//! below the group's lower limit of daily dose ends the period, with the
//! duration of a single purchase. A purchase with no average joins the next
//! when its single-purchase duration reaches `T_i`.
//!
//! **Single purchase.** The typical refill length of the package times the
//! number of packages when the caller knows it, else
//! `DDD (1 + 0.2) / common_ddd_per_day`, else the group's minimum period
//! length; at most the longest single purchase (150 days in the paper).
//!
//! **End of a period** (equation 3), from its last purchase `i` and its number
//! of purchases `k`:
//!
//! ```text
//! END = day_i + DDD_i (1 + 0.5 DDDAVGcv) / d_i * (1 + exp(-k)) + HD_i
//! ```
//!
//! with the single-purchase duration in place of the middle term when the
//! purchase has no average or ended the period by the lower limit. `HD_i` is
//! the hospital days of the stays that start before the supply ends, taken in
//! order, each extending the end; a stay longer than the longest stay (30 days
//! in the paper) is not added, and ends the extension. A period lasts at least
//! the group's minimum period length. As in [`crate::exposure`], the end is a
//! real day and a period covers day `t` when `start <= t < end`.
//!
//! The daily dose of a period is its DDD over its days less its hospital
//! days.
//!
//! # Readings of the paper
//!
//! The paper leaves some steps open, and these are the choices made here:
//!
//! - Purchases on one day are one purchase. The paper does not say; two rows
//!   on a day would give `T = 0` and averages that mean nothing.
//! - `DDDAVGcv` uses the sample standard deviation, with `n - 1`.
//! - The lower limit of the daily dose is compared with the average before the
//!   upper limit is applied, and the upper limit (10 DDD a day in the paper,
//!   or a lower one the caller gives for a group) caps `d_i`.
//! - The expected refill length of two purchases in the stockpiling test is
//!   capped by `max_refill_days` as one purchase is.
//! - The iteration of the paper, which reruns the core with refill lengths
//!   per package estimated from the population's periods, is left to the
//!   caller: [`Purchases::typical_refill_days`] is where a package's typical
//!   refill length goes, from the expert parameters or from an earlier run.
//!   The adoption of the dose of the nearest longer period for an isolated
//!   purchase is not implemented.
//!
//! # Scale
//!
//! One sort per person and one pass over the purchases and stays: `O(n log m)`
//! for `m` purchases of the largest person. Serial; the output is ordered by
//! person, group and start.
//!
//! # References
//!
//! 1. Tanskanen A, Taipale H, Koponen M, Tolppanen AM, Hartikainen S, Ahonen
//!    R, Tiihonen J. From prescription drug purchases to drug use periods: a
//!    second generation method (PRE2DUP). BMC Med Inform Decis Mak.
//!    2015;15:21.

use crate::RegisterError;
use crate::exposure::Pauses;

/// Purchases in CSR layout by person.
#[derive(Clone, Copy, Debug)]
pub struct Purchases<'a> {
    /// `person_offsets[p]..person_offsets[p + 1]` are the rows of person `p`.
    /// The rows of a person need not be sorted.
    pub person_offsets: &'a [usize],
    /// The group of each row, an index into the parameters, or [`None`] for a
    /// row that belongs to no group.
    pub group: &'a [Option<u32>],
    /// The day of each row.
    pub day: &'a [i64],
    /// The amount purchased in DDD, positive and finite, or [`None`] when the
    /// register has none.
    pub ddd: &'a [Option<f64>],
    /// The number of packages, positive and finite.
    pub packages: &'a [f64],
    /// The typical refill length of one package of the row, in days, when the
    /// caller has one: the expert-defined value or the most common refill
    /// length of the package in an earlier run.
    pub typical_refill_days: &'a [Option<f64>],
}

/// The parameters of a drug group, the paper's ATC class parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupParameters {
    /// The lower limit of the daily dose in DDD: a lower average ends a
    /// period.
    pub lower_ddd_per_day: f64,
    /// The common daily dose in DDD, for a purchase with no better estimate.
    pub common_ddd_per_day: f64,
    /// The upper limit of the daily dose in DDD for the group, at most the
    /// global one.
    pub upper_ddd_per_day: f64,
    /// The shortest drug use period, in days.
    pub min_period_days: f64,
    /// The longest refill length allowed in continuous use, in days.
    pub max_refill_days: f64,
}

/// The global parameters of the method.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalParameters {
    /// The longest refill time within which temporal averages are computed.
    pub max_refill_for_average_days: f64,
    /// The longest hospital stay added to the end of a period.
    pub max_hospital_stay_days: f64,
    /// The highest daily dose in DDD.
    pub max_ddd_per_day: f64,
    /// The longest duration of a single purchase.
    pub max_single_purchase_days: f64,
    /// The weight of the coefficient of variation in the refill length,
    /// `DVAR`.
    pub dvar: f64,
    /// The constant for imperfect adherence in a single purchase.
    pub adherence: f64,
}

impl GlobalParameters {
    /// The values of Tanskanen et al. 2015, section 2.5.1 and 2.8: 300 days,
    /// 30 days, 10 DDD a day, 150 days, `DVAR = 0.5` and 0.2.
    pub const TANSKANEN_2015: Self = Self {
        max_refill_for_average_days: 300.0,
        max_hospital_stay_days: 30.0,
        max_ddd_per_day: 10.0,
        max_single_purchase_days: 150.0,
        dvar: 0.5,
        adherence: 0.2,
    };
}

/// The drug use periods, ordered by person, group and start.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DrugUsePeriods {
    /// The person of each period.
    pub person: Vec<u32>,
    /// The group of each period.
    pub group: Vec<u32>,
    /// The day of the first purchase.
    pub start: Vec<i64>,
    /// The real day on which the period ends.
    pub end: Vec<f64>,
    /// The number of purchase days in the period.
    pub n_purchases: Vec<u32>,
    /// The DDD purchased in the period, [`None`] when a purchase has none.
    pub ddd: Vec<Option<f64>>,
    /// The hospital days inside the period.
    pub hospital_days: Vec<f64>,
    /// The DDD over the days of the period less its hospital days.
    pub ddd_per_day: Vec<Option<f64>>,
    /// The number of rows that belong to no group.
    pub unassigned: u64,
}

impl DrugUsePeriods {
    /// The number of periods.
    #[must_use]
    pub fn len(&self) -> usize {
        self.person.len()
    }

    /// Whether there are no periods.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.person.is_empty()
    }

    /// The first whole day of period `i` that it does not cover, `ceil(end)`.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "an end day is far inside the i64 range"
    )]
    pub fn end_day(&self, i: usize) -> i64 {
        self.end[i].ceil() as i64
    }
}

/// One purchase day of a person and group.
#[derive(Clone, Copy, Debug)]
struct Purchase {
    day: i64,
    ddd: Option<f64>,
    single: f64,
}

/// Hospital days of the stays inside `[from, to)`.
#[expect(clippy::cast_precision_loss, reason = "days of register data")]
fn hospital_days(stays: &[(i64, i64)], from: f64, to: f64) -> f64 {
    stays
        .iter()
        .map(|&(a, b)| ((b as f64).min(to) - (a as f64).max(from)).max(0.0))
        .sum()
}

/// The refill times `T_i` of a series.
#[expect(clippy::cast_precision_loss, reason = "days of register data")]
fn refill_times(p: &[Purchase], stays: &[(i64, i64)]) -> Vec<Option<f64>> {
    let m = p.len();
    let mut t: Vec<Option<f64>> = (0..m.saturating_sub(1))
        .map(|i| {
            let (a, b) = (p[i].day as f64, p[i + 1].day as f64);
            Some(b - a - hospital_days(stays, a, b))
        })
        .collect();
    if m >= 2 {
        let last = match (p[m - 1].ddd, p[m - 2].ddd, t[m - 2]) {
            (Some(d), Some(prev), Some(tp)) if prev > 0.0 && tp > 0.0 => Some(d / (prev / tp)),
            _ => None,
        };
        t.push(last);
    } else {
        t.push(None);
    }
    t
}

/// The temporal averages of equation 1 over the runs of a series, and the
/// coefficient of variation of the averages.
pub(crate) fn temporal_averages(
    ddd: &[Option<f64>],
    t: &[Option<f64>],
    max_refill: f64,
) -> (Vec<Option<f64>>, f64) {
    let m = ddd.len();
    let mut avg = vec![None; m];
    let mut run_start = 0;
    while run_start < m {
        let mut run_end = run_start + 1;
        while run_end < m && t[run_end - 1].is_some_and(|x| x <= max_refill) {
            run_end += 1;
        }
        let len = run_end - run_start;
        let ok = (run_start..run_end).all(|i| ddd[i].is_some());
        if len >= 3 && ok {
            // Within a run the last purchase's refill time is the assumed one
            // when it is the series' last, and the run's own otherwise.
            let tt = |i: usize| -> Option<f64> {
                if i + 1 == run_end && run_end < m {
                    match (ddd[i], ddd[i - 1], t[i - 1]) {
                        (Some(d), Some(prev), Some(tp)) if prev > 0.0 && tp > 0.0 => {
                            Some(d / (prev / tp))
                        }
                        _ => None,
                    }
                } else {
                    t[i]
                }
            };
            let dd = |i: usize| ddd[i].unwrap_or(f64::NAN);
            for (i, slot) in avg.iter_mut().enumerate().take(run_end).skip(run_start) {
                let (num, den) = if i == run_start {
                    (
                        tt(i).zip(tt(i + 1)).map(|_| 5.0 * dd(i) + dd(i + 1)),
                        tt(i).zip(tt(i + 1)).map(|(a, b)| 5.0 * a + b),
                    )
                } else if i + 1 == run_end {
                    (
                        tt(i - 1).zip(tt(i)).map(|_| dd(i - 1) + 5.0 * dd(i)),
                        tt(i - 1).zip(tt(i)).map(|(a, b)| a + 5.0 * b),
                    )
                } else {
                    let all = tt(i - 1).zip(tt(i)).zip(tt(i + 1));
                    (
                        all.map(|_| dd(i - 1) + 4.0 * dd(i) + dd(i + 1)),
                        all.map(|((a, b), c)| a + 4.0 * b + c),
                    )
                };
                *slot = match (num, den) {
                    (Some(n), Some(d)) if d > 0.0 => Some(n / d),
                    _ => None,
                };
            }
        }
        run_start = run_end;
    }
    let values: Vec<f64> = avg.iter().flatten().copied().collect();
    #[expect(clippy::cast_precision_loss, reason = "a count of purchases")]
    let cv = if values.len() >= 2 {
        let k = values.len() as f64;
        let mean = values.iter().sum::<f64>() / k;
        let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (k - 1.0);
        var.sqrt() / mean
    } else {
        0.0
    };
    (avg, cv)
}

/// Checks the columns and the parameters.
fn validate(
    p: &Purchases<'_>,
    groups: &[GroupParameters],
    globals: &GlobalParameters,
    stays: Option<&Pauses<'_>>,
) -> Result<usize, RegisterError> {
    if groups.is_empty() {
        return Err(RegisterError::input(
            "no group parameters given; none is built in",
        ));
    }
    let positive = |x: f64| x.is_finite() && x > 0.0;
    let g = globals;
    if !(positive(g.max_refill_for_average_days)
        && g.max_hospital_stay_days >= 0.0
        && positive(g.max_ddd_per_day)
        && positive(g.max_single_purchase_days)
        && g.dvar >= 0.0
        && g.adherence >= 0.0
        && g.max_hospital_stay_days.is_finite()
        && g.dvar.is_finite()
        && g.adherence.is_finite())
    {
        return Err(RegisterError::input("the global parameters are not valid"));
    }
    for (k, q) in groups.iter().enumerate() {
        if !(positive(q.lower_ddd_per_day)
            && positive(q.common_ddd_per_day)
            && positive(q.upper_ddd_per_day)
            && q.lower_ddd_per_day <= q.upper_ddd_per_day
            && positive(q.min_period_days)
            && positive(q.max_refill_days))
        {
            return Err(RegisterError::input(format!(
                "the parameters of group {k} are not valid: doses and days must be positive and finite, the lower dose at most the upper"
            )));
        }
    }
    let o = p.person_offsets;
    let n = p.day.len();
    if o.first() != Some(&0) || o.last() != Some(&n) || o.windows(2).any(|w| w[0] > w[1]) {
        return Err(RegisterError::input(
            "person_offsets must be a CSR layout over the rows",
        ));
    }
    if p.group.len() != n
        || p.ddd.len() != n
        || p.packages.len() != n
        || p.typical_refill_days.len() != n
    {
        return Err(RegisterError::input(
            "group, day, ddd, packages and typical_refill_days must have one value per row",
        ));
    }
    if u32::try_from(n).is_err() {
        return Err(RegisterError::input("more rows than a u32 holds"));
    }
    for row in 0..n {
        let bad_ddd = p.ddd[row].is_some_and(|d| !positive(d));
        let bad_refill = p.typical_refill_days[row].is_some_and(|d| !positive(d));
        if bad_ddd || !positive(p.packages[row]) || bad_refill {
            return Err(RegisterError::input(format!(
                "row {row}: ddd, packages and typical refill days must be positive and finite"
            )));
        }
        if p.group[row].is_some_and(|g| g as usize >= groups.len()) {
            return Err(RegisterError::input(format!(
                "row {row} names a group without parameters"
            )));
        }
    }
    let persons = o.len() - 1;
    if let Some(s) = stays {
        let so = s.person_offsets;
        if so.len() != o.len()
            || so.first() != Some(&0)
            || so.last() != Some(&s.start.len())
            || so.windows(2).any(|w| w[0] > w[1])
            || s.end.len() != s.start.len()
        {
            return Err(RegisterError::input(
                "hospital stays must be a CSR layout over the same persons as the purchases",
            ));
        }
        for q in 0..persons {
            let (a, b) = (so[q], so[q + 1]);
            if s.start[a..b].windows(2).any(|w| w[0] > w[1])
                || (a..b).any(|i| s.end[i] < s.start[i])
                || (a + 1..b).any(|i| s.start[i] < s.end[i - 1])
            {
                return Err(RegisterError::input(format!(
                    "the hospital stays of person {q} are not sorted, disjoint spans"
                )));
            }
        }
    }
    Ok(persons)
}

/// The drug use periods of every person and group.
///
/// See the [module](self) for the method. `stays` are hospital stays
/// `[admission, discharge)` per person, sorted and disjoint.
///
/// # Errors
///
/// [`RegisterError::Input`] when there are no group parameters or a
/// parameter is not valid, the columns differ in length or the offsets are
/// not a layout, a value is not positive and finite, a row names a group
/// without parameters, or the stays are not sorted, disjoint spans over the
/// same persons.
#[expect(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "days of register data; the core is one decision procedure"
)]
pub fn drug_use_periods(
    purchases: &Purchases<'_>,
    groups: &[GroupParameters],
    globals: &GlobalParameters,
    stays: Option<&Pauses<'_>>,
) -> Result<DrugUsePeriods, RegisterError> {
    let persons = validate(purchases, groups, globals, stays)?;
    let p = purchases;
    let mut out = DrugUsePeriods::default();
    let mut rows: Vec<(u32, i64, usize)> = Vec::new();
    let mut person_stays: Vec<(i64, i64)> = Vec::new();
    for person in 0..persons {
        rows.clear();
        for row in p.person_offsets[person]..p.person_offsets[person + 1] {
            match p.group[row] {
                Some(g) => rows.push((g, p.day[row], row)),
                None => out.unassigned += 1,
            }
        }
        rows.sort_unstable();
        person_stays.clear();
        if let Some(s) = stays {
            for i in s.person_offsets[person]..s.person_offsets[person + 1] {
                person_stays.push((s.start[i], s.end[i]));
            }
        }
        let mut i = 0;
        while i < rows.len() {
            let group = rows[i].0;
            let q = &groups[group as usize];
            let upper = q.upper_ddd_per_day.min(globals.max_ddd_per_day);
            // Same-day rows are one purchase.
            let mut series: Vec<Purchase> = Vec::new();
            let mut single_parts: Vec<(Option<f64>, Option<f64>, f64)> = Vec::new();
            while i < rows.len() && rows[i].0 == group {
                let (_, day, row) = rows[i];
                let refill = p.typical_refill_days[row].map(|t| t * p.packages[row]);
                match series.last_mut() {
                    Some(last) if last.day == day => {
                        last.ddd = last.ddd.zip(p.ddd[row]).map(|(a, b)| a + b);
                        let part = single_parts.last_mut().unwrap_or_else(|| unreachable!());
                        part.0 = part.0.zip(refill).map(|(a, b)| a + b);
                        part.1 = last.ddd;
                    }
                    _ => {
                        series.push(Purchase {
                            day,
                            ddd: p.ddd[row],
                            single: 0.0,
                        });
                        single_parts.push((refill, p.ddd[row], 0.0));
                    }
                }
                i += 1;
            }
            for (s, part) in series.iter_mut().zip(&single_parts) {
                let raw = match (part.0, part.1) {
                    (Some(refill), _) => refill,
                    (None, Some(ddd)) => ddd * (1.0 + globals.adherence) / q.common_ddd_per_day,
                    (None, None) => q.min_period_days,
                };
                s.single = raw.min(globals.max_single_purchase_days);
            }
            let t = refill_times(&series, &person_stays);
            let ddd: Vec<Option<f64>> = series.iter().map(|s| s.ddd).collect();
            let (avg, cv) = temporal_averages(&ddd, &t, globals.max_refill_for_average_days);
            let erfl = |i: usize, amount: f64| -> Option<f64> {
                avg[i].map(|a| {
                    (amount * (1.0 + globals.dvar * cv) / a.min(upper)).min(q.max_refill_days)
                })
            };
            let m = series.len();
            let mut first = 0;
            while first < m {
                // Walk until the period ends at `last`.
                let mut last = first;
                let mut by_single = false;
                loop {
                    if last + 1 == m {
                        break;
                    }
                    if let Some(a) = avg[last] {
                        if a < q.lower_ddd_per_day {
                            by_single = true;
                            break;
                        }
                        let tl = t[last].unwrap_or(f64::INFINITY);
                        let reach = erfl(last, series[last].ddd.unwrap_or(0.0)).unwrap_or(0.0);
                        if reach >= tl {
                            last += 1;
                            continue;
                        }
                        let stockpiled = last > first
                            && avg[last - 1].is_some_and(|b| a < b)
                            && avg[last + 1].is_some_and(|b| a < b);
                        if stockpiled {
                            let both = series[last - 1].ddd.unwrap_or(0.0)
                                + series[last].ddd.unwrap_or(0.0);
                            let two = erfl(last, both).unwrap_or(0.0);
                            let gap = t[last - 1].unwrap_or(f64::INFINITY) + tl;
                            if two >= gap {
                                last += 1;
                                continue;
                            }
                        }
                        break;
                    }
                    if series[last].single >= t[last].unwrap_or(f64::INFINITY) {
                        last += 1;
                        continue;
                    }
                    break;
                }
                let k = last - first + 1;
                let s = series[last];
                let base = match (avg[last], s.ddd, by_single) {
                    (Some(a), Some(amount), false) => {
                        amount * (1.0 + 0.5 * cv) / a.min(upper) * (1.0 + (-(k as f64)).exp())
                    }
                    _ => s.single,
                };
                let day = s.day as f64;
                let mut end = day + base;
                for &(a, b) in &person_stays {
                    let (a, b) = (a as f64, b as f64);
                    if b <= day {
                        continue;
                    }
                    if a >= end {
                        break;
                    }
                    if b - a > globals.max_hospital_stay_days {
                        break;
                    }
                    end += b - a.max(day);
                }
                let start = series[first].day;
                end = end.max(start as f64 + q.min_period_days);
                let total = (first..=last)
                    .map(|j| series[j].ddd)
                    .try_fold(0.0, |acc, d| d.map(|d| acc + d));
                let hd = hospital_days(&person_stays, start as f64, end);
                let days = end - start as f64 - hd;
                out.person.push(u32::try_from(person).unwrap_or(u32::MAX));
                out.group.push(group);
                out.start.push(start);
                out.end.push(end);
                out.n_purchases.push(u32::try_from(k).unwrap_or(u32::MAX));
                out.ddd.push(total);
                out.hospital_days.push(hd);
                out.ddd_per_day
                    .push(total.and_then(|d| (days > 0.0).then(|| d / days)));
                first = last + 1;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Table 2 of Tanskanen et al.: five purchases of 10 DDD, 10, 5, 15 and 10
    /// days apart. The paper prints the averages and the refill lengths
    /// `DDD / DDDAVG` to two decimals.
    #[test]
    fn the_averages_of_table_2() {
        let ddd = [Some(10.0); 5];
        let t = [Some(10.0), Some(5.0), Some(15.0), Some(10.0), Some(10.0)];
        let (avg, _) = temporal_averages(&ddd, &t, 300.0);
        let avg: Vec<f64> = avg.into_iter().map(Option::unwrap).collect();
        let printed = [1.09, 1.33, 0.80, 0.92, 1.00];
        let refill = [9.17, 7.50, 12.50, 10.83, 10.00];
        for i in 0..5 {
            assert!((avg[i] - printed[i]).abs() < 0.005, "{i}: {}", avg[i]);
            assert!((10.0 / avg[i] - refill[i]).abs() < 0.005, "{i}");
        }
        // The third purchase is the stockpiling case: its average is below
        // both neighbours'.
        assert!(avg[2] < avg[1] && avg[2] < avg[3]);
    }

    fn group() -> GroupParameters {
        GroupParameters {
            lower_ddd_per_day: 0.3,
            common_ddd_per_day: 1.0,
            upper_ddd_per_day: 10.0,
            min_period_days: 15.0,
            max_refill_days: 120.0,
        }
    }

    fn run(days: &[i64], ddd: &[f64]) -> DrugUsePeriods {
        let n = days.len();
        let ddd: Vec<Option<f64>> = ddd.iter().map(|&d| Some(d)).collect();
        let p = Purchases {
            person_offsets: &[0, n],
            group: &vec![Some(0); n],
            day: days,
            ddd: &ddd,
            packages: &vec![1.0; n],
            typical_refill_days: &vec![None; n],
        };
        drug_use_periods(&p, &[group()], &GlobalParameters::TANSKANEN_2015, None).unwrap()
    }

    #[test]
    fn the_stockpiling_example_is_one_period() {
        let p = run(&[0, 10, 15, 30, 40], &[10.0; 5]);
        assert_eq!(p.len(), 1);
        assert_eq!(p.n_purchases, [5]);
    }

    #[test]
    fn a_long_gap_ends_a_period_and_a_lone_purchase_lasts_its_single_duration() {
        let p = run(&[0, 30, 60, 90, 400], &[30.0; 5]);
        assert_eq!(p.start, [0, 400]);
        // The lone purchase: 30 DDD * 1.2 / 1 DDD a day = 36 days.
        assert!((p.end[1] - 436.0).abs() < 1e-12);
        // A single purchase shorter than the minimum period lasts the minimum.
        let q = run(&[0], &[5.0]);
        assert!((q.end[0] - 15.0).abs() < 1e-12);
    }

    #[test]
    fn refusals() {
        let p = Purchases {
            person_offsets: &[0, 1],
            group: &[Some(0)],
            day: &[0],
            ddd: &[Some(1.0)],
            packages: &[1.0],
            typical_refill_days: &[None],
        };
        let g = GlobalParameters::TANSKANEN_2015;
        assert!(drug_use_periods(&p, &[], &g, None).is_err());
        let bad = GroupParameters {
            lower_ddd_per_day: 2.0,
            upper_ddd_per_day: 1.0,
            ..group()
        };
        assert!(drug_use_periods(&p, &[bad], &g, None).is_err());
        let zero = Purchases {
            ddd: &[Some(0.0)],
            ..p
        };
        assert!(drug_use_periods(&zero, &[group()], &g, None).is_err());
        let stranger = Purchases {
            group: &[Some(3)],
            ..p
        };
        assert!(drug_use_periods(&stranger, &[group()], &g, None).is_err());
    }
}

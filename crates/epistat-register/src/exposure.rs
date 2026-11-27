//! Drug exposure episodes and cumulative dose from dispensings.
//!
//! A dispensing register records that a person collected a quantity of a drug
//! on a date. It records neither consumption nor an end date. Studies of drug
//! use turn dispensings into periods of exposure by assuming how long each
//! dispensing lasts, joining dispensings across a permitted gap and deciding
//! what happens to supply that overlaps the next dispensing \[1, 2\]. This
//! module does that on the assumptions the caller declares. It ships none: a
//! plausible default dose for a drug nobody checked gives periods that look
//! computed. [`Assumption`] has no `Default`, and an empty list of assumptions
//! is an error.
//!
//! # Method
//!
//! Per person and drug group, the dispensings are sorted by day and
//! dispensings on one day are added into one supply. For the supply `i` of
//! quantity `u_i` on day `t_i`, the duration in days is `u_i / d_i`, where the
//! daily dose `d_i` follows the [`DoseRule`]:
//!
//! ```text
//! Default:   d_i = dose_default
//! Observed:  d_i = clamp(u_i / (t_{i+1} - t_i), dose_min, dose_max)   (last: dose_default)
//! Fixed:     duration_i = days * (number of dispensings on day t_i)
//! ```
//!
//! The recurrence keeps the real day `e` on which the supply runs out:
//!
//! ```text
//! if first, or t_i > e + grace:   close the period, start a new one at t_i, s_i = t_i
//! else:                           s_i = t_i + min(max(e - t_i, 0), cap)
//! e = advance(s_i, duration_i, pauses)
//! ```
//!
//! where `cap` is `max_stock_days` under [`Stock::CarryForward`], so that
//! supply carried over from earlier dispensings is kept up to the cap and the
//! rest is lost, and 0 under [`Stock::Discard`]. A dispensing on the day the
//! supply runs out continues the period, and so does one within `grace_days`
//! after it. A grace period does not add days of supply: the supply of a
//! dispensing after a gap starts on the day it is dispensed.
//!
//! The pauses are spans `[start, end)` of a person, typically hospital
//! admissions, in which the supply is not consumed. `advance(s, d, pauses)`
//! walks the person's pauses in start order. For each pause `[a, b)` with
//! `b > position` and `a < position + remaining`, it takes `max(0, a -
//! position)` from `remaining` and sets `position = max(position, b)`, and it
//! returns `position + remaining`. A pause that starts after the supply has
//! run out extends nothing. The start of successive supplies of a person and
//! group never decreases, so a pointer over the pauses only moves forward and
//! the work for a person is `O(dispensings + pauses)`.
//!
//! The end of a period is the real day `e` of its last supply. A period
//! covers a day `t` when `start <= t < e`, so a day on which some supply is
//! left is a day of exposure, and [`Periods::end_day`] is `ceil(e)`, the first
//! whole day with no supply left: a supply that starts on day 0 and runs out
//! at 4.9 covers days 0 to 4 and ends on day 5. [`Periods::last_day`] is the
//! day before, the last day covered. Under this rule [`exposed_on`],
//! [`Periods::end_day`] and [`Periods::span_set`] agree on every day, also
//! when the supply is not a whole number of days.
//!
//! The daily dose of a period is the total quantity over the sum of the
//! durations of its supplies, fixed-order sums in dispensing order. Under
//! [`DoseRule::Default`] it equals `dose_default`, and under
//! [`DoseRule::Observed`] it is the quantity-weighted harmonic mean of the
//! doses. It is [`None`] when the sum of durations is 0.
//!
//! Without cap, pauses and grace, the recurrence has the closed form
//! `e_i = S_i + max_{j<=i}(t_j - S_{j-1})` with `S_i` the cumulative duration,
//! which the tests use as an oracle.
//!
//! # Cumulative dose
//!
//! [`DoseMode::AtDispensing`] is the sum of the quantities dispensed in a
//! window, the usual Danish choice, and needs no assumption.
//! [`DoseMode::ProratedOverSupply`] spreads each supply uniformly over
//! `[s_i, e_i)` and adds the part of it inside the window, `u_i * overlap /
//! (e_i - s_i)` in real days. A lag moves the window back by that many days.
//!
//! # Units
//!
//! Quantity, doses and the cap on stock are numbers in the unit of the
//! caller. The crate does not look at units: a DDD is an amount of substance
//! and a package count times package size is a count of tablets, and a dose in
//! one against a quantity in the other gives durations that are wrong by the
//! strength. Convert before the call.
//!
//! # Scale
//!
//! One sort of each person's dispensings and one pass: `O(n log m)` for `m`
//! dispensings of the largest person. Results do not depend on a thread count:
//! the function is serial and the output is ordered by person, group and
//! start.
//!
//! # Other implementations
//!
//! The tests compare the dose rules, [`exposed_on`] and both cumulative doses
//! with base R and the periods with `AdhereR::compute.treatment.episodes`
//! and with `heaven::medicinMacro` (type `"dynamic"`), and agree exactly where
//! the rules coincide: whole-day durations with AdhereR, and with heaven a
//! fixed daily dose (equal minimum, maximum and default), no permitted gap,
//! with or without a cap on stock. They differ elsewhere:
//!
//! - heaven estimates the daily dose from the purchases around a supply, and
//!   uses its default dose before the second purchase. Its cap on stock is a
//!   quantity, here a number of days. It has no permitted gap.
//! - heaven reads an admission as the span from the admission day to the
//!   discharge day, discharge day excluded, as a pause here. When an admission
//!   starts while a supply is running, heaven's periods end later than here
//!   and can join periods that are separate here, so the comparison uses the
//!   persons whose admissions start with no supply running.
//!
//! # Related methods
//!
//! PRE2DUP \[3\], the person-specific model of drug use periods, is in
//! [`crate::pre2dup`], and the parametric waiting time distribution \[4\],
//! which gives the duration of a prescription, in [`crate::wtd`].
//!
//! # References
//!
//! 1. Pottegard A, Schmidt SAJ, Wallach-Kildemoes H, Sorensen HT, Hallas J,
//!    Schmidt M. Data resource profile: the Danish National Prescription
//!    Registry. Int J Epidemiol. 2017;46(3):798-798f.
//! 2. Gardarsdottir H, Souverein PC, Egberts TCG, Heerdink ER. Construction of
//!    drug treatment episodes from drug-dispensing histories is influenced by
//!    the gap length. J Clin Epidemiol. 2010;63(4):422-7.
//! 3. Tanskanen A, Taipale H, Koponen M, et al. From prescription drug
//!    purchases to drug use periods: a second generation method (PRE2DUP).
//!    BMC Med Inform Decis Mak. 2015;15:21.
//! 4. Hallas J, Gaist D, Bjerrum L. The waiting time distribution as a
//!    graphical approach to epidemiologic measures of drug utilization.
//!    Epidemiology. 1997;8(6):666-70.

use crate::Error;
use crate::error::{csr, length, row_count};
use crate::intervals::SpanSet;
use epistat_core::{InputError, check};

/// How one group of drugs is taken, declared by the caller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Assumption {
    /// The daily dose, in the unit of the quantity, used under
    /// [`DoseRule::Default`] and for the last supply under
    /// [`DoseRule::Observed`]. Positive and finite.
    pub dose_default: f64,
    /// The lowest observed daily dose. Positive and at most `dose_max`.
    pub dose_min: f64,
    /// The highest observed daily dose, finite.
    pub dose_max: f64,
    /// The most carried supply in days, at least 0. `f64::INFINITY` is allowed.
    pub max_stock_days: f64,
    /// The permitted gap in days after the supply runs out, at least 0, finite.
    pub grace_days: f64,
}

/// How the duration of a supply is set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DoseRule {
    /// The quantity over `dose_default`.
    Default,
    /// The quantity over the dose observed from the time to the next
    /// dispensing, clamped to the bounds of the assumption.
    Observed,
    /// Each dispensing lasts a fixed number of days, whatever its quantity.
    Fixed {
        /// The days a dispensing lasts, positive and finite.
        days: f64,
    },
}

/// What becomes of supply that overlaps the next dispensing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stock {
    /// It is carried over, up to `max_stock_days`.
    CarryForward,
    /// It is lost: the new supply starts on the day of the dispensing.
    Discard,
}

/// Dispensings in CSR layout by person.
#[derive(Clone, Copy, Debug)]
pub struct Dispensings<'a> {
    /// `person_offsets[p]..person_offsets[p + 1]` are the rows of person `p`.
    /// The rows of a person need not be sorted.
    pub person_offsets: &'a [usize],
    /// The group of each row, an index into the assumptions, or [`None`] for a
    /// row that matches no group. Such rows are counted in
    /// [`Periods::unassigned`] and used for nothing else.
    pub group: &'a [Option<u32>],
    /// The day of each row.
    pub day: &'a [i64],
    /// The quantity of each row, positive and finite.
    pub quantity: &'a [f64],
}

/// Spans in which supply is not consumed, in CSR layout by person.
#[derive(Clone, Copy, Debug)]
pub struct Pauses<'a> {
    /// `person_offsets[p]..person_offsets[p + 1]` are the pauses of person `p`.
    /// The offsets must give the same number of persons as the dispensings.
    pub person_offsets: &'a [usize],
    /// The first paused day of each pause, sorted within a person.
    pub start: &'a [i64],
    /// The first day after the pause, at least its start.
    pub end: &'a [i64],
}

/// The assumptions, the dose rule, the stock rule and the pauses together.
#[derive(Clone, Copy, Debug)]
pub struct Supply<'a> {
    /// One assumption per group, indexed by [`Dispensings::group`].
    pub assumptions: &'a [Assumption],
    /// How the duration of a supply is set.
    pub rule: DoseRule,
    /// What becomes of overlapping supply.
    pub stock: Stock,
    /// Spans in which the supply is not consumed.
    pub pauses: Option<&'a Pauses<'a>>,
}

/// The first whole day on or after the real day `x`, or [`None`] when `x` is
/// not finite or the day does not fit an `i64`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the value is checked against the i64 range"
)]
pub(crate) fn ceil_day(x: f64) -> Option<i64> {
    let day = x.ceil();
    // 2^63 is exactly representable; every f64 below it converts exactly.
    (day.is_finite() && (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&day))
        .then_some(day as i64)
}

/// The exposure periods, ordered by person, group and start.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Periods {
    /// The person of each period, a position among the persons.
    pub(crate) person: Vec<u32>,
    /// The group of each period.
    pub(crate) group: Vec<u32>,
    /// The day of the first dispensing of the period.
    pub(crate) start: Vec<i64>,
    /// The real day on which the supply of the period runs out.
    pub(crate) end: Vec<f64>,
    /// The number of days with a dispensing in the period, same-day dispensings
    /// counted once.
    pub(crate) n_dispensing_days: Vec<u32>,
    /// The total quantity dispensed in the period.
    pub(crate) quantity: Vec<f64>,
    /// The quantity over the sum of the durations of the supplies, or [`None`]
    /// when the sum is 0.
    pub(crate) daily_dose: Vec<Option<f64>>,
    /// The number of dispensings that belong to no group.
    pub(crate) unassigned: u64,
}

impl Periods {
    /// The person of each period, a position among the persons.
    #[must_use]
    pub fn person(&self) -> &[u32] {
        &self.person
    }

    /// The group of each period.
    #[must_use]
    pub fn group(&self) -> &[u32] {
        &self.group
    }

    /// The day of the first dispensing of the period.
    #[must_use]
    pub fn start(&self) -> &[i64] {
        &self.start
    }

    /// The real day on which the supply of the period runs out.
    #[must_use]
    pub fn end(&self) -> &[f64] {
        &self.end
    }

    /// The number of days with a dispensing in the period, same-day dispensings
    /// counted once.
    #[must_use]
    pub fn n_dispensing_days(&self) -> &[u32] {
        &self.n_dispensing_days
    }

    /// The total quantity dispensed in the period.
    #[must_use]
    pub fn quantity(&self) -> &[f64] {
        &self.quantity
    }

    /// The quantity over the sum of the durations of the supplies, or [`None`]
    /// when the sum is 0.
    #[must_use]
    pub fn daily_dose(&self) -> &[Option<f64>] {
        &self.daily_dose
    }

    /// The number of dispensings that belong to no group.
    #[must_use]
    pub fn unassigned(&self) -> u64 {
        self.unassigned
    }

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

    /// The first whole day of period `i` with no supply left, `ceil(end)`, or
    /// [`None`] when there is no period `i` or its end is not a day an `i64`
    /// holds. The period covers the days `start..end_day`, which are the days
    /// [`exposed_on`] reads as exposed.
    #[must_use]
    pub fn end_day(&self, i: usize) -> Option<i64> {
        ceil_day(*self.end.get(i)?)
    }

    /// The last day of period `i` with some supply left, `end_day - 1`, or
    /// [`None`] as for [`end_day`](Self::end_day).
    #[must_use]
    pub fn last_day(&self, i: usize) -> Option<i64> {
        self.end_day(i)?.checked_sub(1)
    }

    /// The periods of one group as a half-open [`SpanSet`] over `persons`
    /// keys, period `i` as `[start, end_day)`. A person with no period of the
    /// group is an empty key.
    ///
    /// Periods of one person and group never share a day: a period begins
    /// only on a dispensing day later than the end of the one before, which is
    /// at least its `end_day`. Periods of different groups can overlap, so a
    /// set is per group, and [`crate::intervals`] combines them.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when a period names a person at or beyond
    /// `persons`, or `persons` does not fit a `u32`.
    pub fn span_set(&self, group: u32, persons: usize) -> Result<SpanSet, Error> {
        row_count("persons", persons)?;
        let mut start = Vec::new();
        let mut end = Vec::new();
        let mut offsets = vec![0_usize; persons + 1];
        for i in 0..self.len() {
            if self.group[i] != group {
                continue;
            }
            let p = self.person[i] as usize;
            if p >= persons {
                return Err(Error::invalid(format!(
                    "period {i} names person {p} of {persons}"
                )));
            }
            offsets[p + 1] += 1;
            start.push(self.start[i]);
            end.push(ceil_day(self.end[i]).ok_or_else(|| {
                Error::invalid(format!("period {i} has an end that is not a day"))
            })?);
        }
        for p in 0..persons {
            offsets[p + 1] += offsets[p];
        }
        SpanSet::new(start, end, offsets)
    }
}

fn validate_assumptions(a: &[Assumption]) -> Result<(), Error> {
    if a.is_empty() {
        return Err(Error::invalid(
            "no drug assumptions given; none is built in",
        ));
    }
    for (g, x) in a.iter().enumerate() {
        let ok = x.dose_default.is_finite()
            && x.dose_default > 0.0
            && x.dose_min.is_finite()
            && x.dose_min > 0.0
            && x.dose_max.is_finite()
            && x.dose_min <= x.dose_max
            && !x.max_stock_days.is_nan()
            && x.max_stock_days >= 0.0
            && x.grace_days.is_finite()
            && x.grace_days >= 0.0;
        if !ok {
            return Err(Error::invalid(format!(
                "assumption {g} is not valid: doses must be positive and finite with dose_min <= dose_max, max_stock_days at least 0 and grace_days finite and at least 0"
            )));
        }
    }
    Ok(())
}

fn validate(d: &Dispensings<'_>, s: &Supply<'_>) -> Result<usize, Error> {
    validate_assumptions(s.assumptions)?;
    if let DoseRule::Fixed { days } = s.rule
        && !(days.is_finite() && days > 0.0)
    {
        return Err(Error::invalid(
            "a fixed duration must be positive and finite",
        ));
    }
    let o = d.person_offsets;
    let n = d.day.len();
    csr("person_offsets", o, n)?;
    length("group", n, d.group.len())?;
    length("quantity", n, d.quantity.len())?;
    row_count("day", n)?;
    for (row, &q) in d.quantity.iter().enumerate() {
        if !(q.is_finite() && q > 0.0) {
            return Err(Error::invalid(format!(
                "row {row} has the quantity {q}, which is not positive and finite"
            )));
        }
    }
    if let Some(row) = d
        .group
        .iter()
        .position(|g| g.is_some_and(|g| g as usize >= s.assumptions.len()))
    {
        return Err(Error::invalid(format!(
            "row {row} names a group without an assumption"
        )));
    }
    let persons = o.len() - 1;
    if let Some(p) = s.pauses {
        let po = p.person_offsets;
        length("pause offsets", o.len(), po.len())?;
        length("pause end", p.start.len(), p.end.len())?;
        csr("pause offsets", po, p.start.len())?;
        for q in 0..persons {
            let (a, b) = (po[q], po[q + 1]);
            check::sorted_by("pause start", &p.start[a..b], i64::cmp).map_err(
                |error| match error {
                    InputError::Unsorted { what, row } => {
                        InputError::Unsorted { what, row: a + row }
                    }
                    other => other,
                },
            )?;
            if (a..b).any(|i| p.end[i] < p.start[i]) {
                return Err(Error::invalid(format!(
                    "a pause of person {q} ends before it starts"
                )));
            }
        }
    }
    Ok(persons)
}

/// The supply of one dispensing day: its start `s`, its end, the day and
/// quantity, and where the period it belongs to begins.
struct Step {
    person: u32,
    group: u32,
    day: i64,
    quantity: f64,
    duration: f64,
    supply_start: f64,
    supply_end: f64,
    new_period: bool,
}

/// The day on which `days` of consumption have elapsed from `start`, skipping
/// the paused days. `cursor` is the first pause that has not ended before the
/// positions seen so far.
fn advance(start: f64, days: f64, pauses: &[(i64, i64)], cursor: &mut usize) -> f64 {
    let mut remaining = days;
    let mut position = start;
    while *cursor < pauses.len() && (pauses[*cursor].1 as f64) <= position {
        *cursor += 1;
    }
    for &(pause_start, pause_end) in &pauses[*cursor..] {
        if (pause_end as f64) <= position {
            continue;
        }
        if (pause_start as f64) >= position + remaining {
            break;
        }
        remaining -= (pause_start as f64 - position).max(0.0);
        position = position.max(pause_end as f64);
    }
    position + remaining
}

/// Every supply of every person and group, in order of person, group and day.
fn walk(d: &Dispensings<'_>, s: &Supply<'_>, persons: usize) -> (Vec<Step>, u64) {
    let mut steps = Vec::new();
    let mut unassigned = 0_u64;
    let mut order: Vec<(u32, i64, usize)> = Vec::new();
    let mut pauses: Vec<(i64, i64)> = Vec::new();
    for p in 0..persons {
        let (a, b) = (d.person_offsets[p], d.person_offsets[p + 1]);
        order.clear();
        for row in a..b {
            match d.group[row] {
                Some(g) => order.push((g, d.day[row], row)),
                None => unassigned += 1,
            }
        }
        if order.is_empty() {
            continue;
        }
        order.sort_unstable();
        pauses.clear();
        if let Some(pz) = s.pauses {
            for i in pz.person_offsets[p]..pz.person_offsets[p + 1] {
                pauses.push((pz.start[i], pz.end[i]));
            }
        }
        let mut i = 0;
        while i < order.len() {
            let group = order[i].0;
            let mut j = i;
            while j < order.len() && order[j].0 == group {
                j += 1;
            }
            // Same-day rows of this group are one supply.
            let mut days: Vec<(i64, f64, f64)> = Vec::new();
            for &(_, day, row) in &order[i..j] {
                match days.last_mut() {
                    Some(last) if last.0 == day => {
                        last.1 += d.quantity[row];
                        last.2 += 1.0;
                    }
                    _ => days.push((day, d.quantity[row], 1.0)),
                }
            }
            series(
                s,
                group,
                u32::try_from(p).unwrap_or(u32::MAX),
                &days,
                &pauses,
                &mut steps,
            );
            i = j;
        }
    }
    (steps, unassigned)
}

/// The recurrence for one person and group.
fn series(
    s: &Supply<'_>,
    group: u32,
    person: u32,
    days: &[(i64, f64, f64)],
    pauses: &[(i64, i64)],
    steps: &mut Vec<Step>,
) {
    let a = &s.assumptions[group as usize];
    let cap = match s.stock {
        Stock::CarryForward => a.max_stock_days,
        Stock::Discard => 0.0,
    };
    let mut cursor = 0;
    let mut supply_end: Option<f64> = None;
    for (k, &(day, quantity, rows)) in days.iter().enumerate() {
        let t = day as f64;
        let continues = supply_end.is_some_and(|e| t <= e + a.grace_days);
        let supply_start = match supply_end {
            Some(e) if continues => t + (e - t).max(0.0).min(cap),
            _ => t,
        };
        let duration = match s.rule {
            DoseRule::Default => quantity / a.dose_default,
            DoseRule::Observed => {
                let dose = match days.get(k + 1) {
                    Some(&(next, _, _)) => {
                        let gap = (next - day) as f64;
                        (quantity / gap).clamp(a.dose_min, a.dose_max)
                    }
                    None => a.dose_default,
                };
                quantity / dose
            }
            DoseRule::Fixed { days } => days * rows,
        };
        let end = advance(supply_start, duration, pauses, &mut cursor);
        steps.push(Step {
            person,
            group,
            day,
            quantity,
            duration,
            supply_start,
            supply_end: end,
            new_period: !continues,
        });
        supply_end = Some(end);
    }
}

/// The exposure periods of every person and group.
///
/// See the [module](self) for the recurrence. A row with no group is counted in
/// [`Periods::unassigned`].
///
/// # Errors
///
/// [`Error::Input`] when there are no assumptions or one is invalid,
/// the columns differ in length or the offsets are not a CSR layout, a quantity
/// is not positive and finite (with its row), a group has no assumption, a
/// fixed duration is not positive, or the pauses are not sorted or do not cover
/// the same persons.
pub fn exposure_periods(d: &Dispensings<'_>, s: &Supply<'_>) -> Result<Periods, Error> {
    let persons = validate(d, s)?;
    let (steps, unassigned) = walk(d, s, persons);
    let mut out = Periods {
        unassigned,
        ..Periods::default()
    };
    let mut count = 0_u32;
    let mut total = 0.0;
    let mut consumption = 0.0;
    for step in &steps {
        if step.new_period {
            out.person.push(step.person);
            out.group.push(step.group);
            out.start.push(step.day);
            out.end.push(0.0);
            out.n_dispensing_days.push(0);
            out.quantity.push(0.0);
            out.daily_dose.push(None);
            count = 0;
            total = 0.0;
            consumption = 0.0;
        }
        count += 1;
        total += step.quantity;
        consumption += step.duration;
        let last = out.person.len() - 1;
        out.end[last] = step.supply_end;
        out.n_dispensing_days[last] = count;
        out.quantity[last] = total;
        out.daily_dose[last] = (consumption > 0.0).then(|| total / consumption);
    }
    Ok(out)
}

/// What a cumulative dose adds up.
#[derive(Clone, Copy, Debug)]
pub enum DoseMode<'a> {
    /// The quantity dispensed in the window.
    AtDispensing,
    /// Each supply spread uniformly over its days of supply, and the part
    /// inside the window added.
    ProratedOverSupply(&'a Supply<'a>),
}

/// Windows `[start, end)` for which a cumulative dose is wanted.
#[derive(Clone, Copy, Debug)]
pub struct Windows<'a> {
    /// The person of each window, a position among the persons of the
    /// dispensings.
    pub person: &'a [u32],
    /// The group of each window, an index into the assumptions.
    pub group: &'a [u32],
    /// The first day of each window.
    pub start: &'a [i64],
    /// The first day after each window, at least its start.
    pub end: &'a [i64],
}

/// The cumulative dose of each window of a person and group, with the window
/// moved back by `lag_days`.
///
/// Under [`DoseMode::AtDispensing`] the dispensings are the rows with a group
/// and no assumptions are used, so the groups of the windows are matched with
/// the groups of the rows. The result is summed in row order.
///
/// # Errors
///
/// [`Error::Input`] for window columns of different lengths, a window
/// that ends before it starts, a person out of range, a negative lag and, under
/// [`DoseMode::ProratedOverSupply`], for the reasons of [`exposure_periods`].
pub fn cumulative_dose(
    d: &Dispensings<'_>,
    windows: &Windows<'_>,
    lag_days: i64,
    mode: DoseMode<'_>,
) -> Result<Vec<f64>, Error> {
    let m = windows.person.len();
    length("window group", m, windows.group.len())?;
    length("window start", m, windows.start.len())?;
    length("window end", m, windows.end.len())?;
    if lag_days < 0 {
        return Err(Error::invalid("the lag must not be negative"));
    }
    let o = d.person_offsets;
    csr("person_offsets", o, d.day.len())?;
    length("group", d.day.len(), d.group.len())?;
    length("quantity", d.day.len(), d.quantity.len())?;
    let persons = o.len() - 1;
    for w in 0..m {
        if windows.person[w] as usize >= persons {
            return Err(Error::invalid(format!(
                "window {w} names a person out of range"
            )));
        }
        if windows.end[w] < windows.start[w] {
            return Err(Error::invalid(format!("window {w} ends before it starts")));
        }
    }
    let mut out = vec![0.0; m];
    match mode {
        DoseMode::AtDispensing => {
            for (w, slot) in out.iter_mut().enumerate() {
                let p = windows.person[w] as usize;
                let (lo, hi) = (
                    windows.start[w].saturating_sub(lag_days),
                    windows.end[w].saturating_sub(lag_days),
                );
                let mut sum = 0.0;
                for row in o[p]..o[p + 1] {
                    if d.group[row] == Some(windows.group[w]) && d.day[row] >= lo && d.day[row] < hi
                    {
                        sum += d.quantity[row];
                    }
                }
                *slot = sum;
            }
        }
        DoseMode::ProratedOverSupply(supply) => {
            let persons = validate(d, supply)?;
            let (steps, _) = walk(d, supply, persons);
            for (w, slot) in out.iter_mut().enumerate() {
                let (person, group) = (windows.person[w], windows.group[w]);
                let lo = steps.partition_point(|s| (s.person, s.group) < (person, group));
                let hi = steps.partition_point(|s| (s.person, s.group) <= (person, group));
                let (w0, w1) = (
                    windows.start[w].saturating_sub(lag_days) as f64,
                    windows.end[w].saturating_sub(lag_days) as f64,
                );
                let mut sum = 0.0;
                for step in &steps[lo..hi] {
                    let span = step.supply_end - step.supply_start;
                    let overlap = (w1.min(step.supply_end) - w0.max(step.supply_start)).max(0.0);
                    if overlap > 0.0 && span > 0.0 {
                        sum += step.quantity * (overlap / span);
                    }
                }
                *slot = sum;
            }
        }
    }
    Ok(out)
}

/// Days at which exposure status is wanted.
#[derive(Clone, Copy, Debug)]
pub struct PointTable<'a> {
    /// The person of each point.
    pub person: &'a [u32],
    /// The group of each point.
    pub group: &'a [u32],
    /// The day of each point.
    pub day: &'a [i64],
}

/// Whether each point lies in a period of its person and group, with the point
/// moved back by `lag_days`. A period covers a day `t` when `start <= t < end`.
///
/// # Errors
///
/// [`Error::Input`] for columns of different lengths or a negative lag.
pub fn exposed_on(
    periods: &Periods,
    query: &PointTable<'_>,
    lag_days: i64,
) -> Result<Vec<bool>, Error> {
    let m = query.person.len();
    length("point group", m, query.group.len())?;
    length("point day", m, query.day.len())?;
    if lag_days < 0 {
        return Err(Error::invalid("the lag must not be negative"));
    }
    Ok((0..m)
        .map(|i| {
            let key = (query.person[i], query.group[i]);
            let t = query.day[i].saturating_sub(lag_days);
            let upto = partition_by(periods, key, t);
            upto > 0 && {
                let k = upto - 1;
                (periods.person[k], periods.group[k]) == key && (t as f64) < periods.end[k]
            }
        })
        .collect())
}

/// The number of periods that sort before or at `(key, start = t)`.
fn partition_by(periods: &Periods, key: (u32, u32), t: i64) -> usize {
    let (mut lo, mut hi) = (0, periods.len());
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let at = (periods.person[mid], periods.group[mid], periods.start[mid]);
        if at <= (key.0, key.1, t) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(test)]
mod tests;

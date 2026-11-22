//! Event definitions: first event in a window, washout, prevalent status and
//! confirmation by a second record.
//!
//! A register case definition is a rule over the dated events of a person: the
//! day of the first qualifying event after the start of follow-up, whether the
//! person was already a case before it (prevalent, and excluded from an
//! incidence study), how far back the record reaches (the washout or
//! look-back), and whether one event is enough or a second one must confirm
//! it \[1, 2\]. Which rows are events, by code, contact type or diagnosis
//! position, is the caller's: each row carries a mask with bit `k` set when it
//! is a qualifying event of definition `k`, for instance from
//! [`CodeMap::classify`](crate::codes::CodeMap::classify). This module does
//! the time logic, and one pass over a person serves up to 64 definitions.
//!
//! # Windows
//!
//! For a person with entry `e`, exit `x` and history start `h`, and a rule with
//! look-back `L`:
//!
//! ```text
//! window start         w = e + 1 if entry_inclusive else e
//! prior window         [max(s, h), w)        s = e - L (see below)
//! follow-up window     [max(w, h), x]
//! ```
//!
//! An event on the entry day is therefore before entry when `entry_inclusive`
//! is set and in the follow-up window otherwise, never both. The look-back
//! start `s` is by [`Lookback`]:
//!
//! - [`Lookback::None`]: there is no prior window and the status is
//!   [`PriorStatus::NotAssessed`].
//! - [`Lookback::Days`]: `s = e - d`. `heaven` looks back `look.back * 365.25`
//!   days with `data.date >= index - look.back * 365.25`, which for whole days
//!   is `Days(floor(365.25 * look.back))`.
//! - [`Lookback::Years`]: the date `n` years before `e`, by civil years. For
//!   an entry on 29 February in a year `n` years back that is not a leap year
//!   the [`LeapBirthday`] rule says whether it is 28 February or 1 March.
//! - [`Lookback::Unlimited`]: `s = h`, everything the register can have
//!   recorded.
//!
//! An event before `h` lies before the register could have recorded it. It is
//! counted in [`Outcome::n_before_history`] and used for nothing else.
//!
//! # Prior status
//!
//! [`PriorStatus::Prevalent`] when an event lies in the prior window.
//! [`PriorStatus::NoneObserved`] when none does and the window lies inside the
//! recorded history, `s >= h`. [`PriorStatus::HistoryTooShort`] when none does
//! and `s < h`: the person may have had an event before the register began.
//! The caller decides how to treat the third state, as not prevalent, as
//! unknown or as an exclusion. A person with no event and a window that starts
//! before the register is not a person with a window and no event.
//!
//! # Confirmation
//!
//! With `require = 2` the case day is read from the events of the follow-up
//! window sorted by day. Let `i` be the first event for which a later event
//! `j` exists within `confirm_within` days of it, `day[j] - day[i] <=
//! confirm_within` (no bound when it is [`None`]). With [`ConfirmBy::Dates`],
//! `j` must be on a later day than `i`. With [`ConfirmBy::Contacts`], `j` must
//! belong to a different contact, and may be on the same day. The case day is
//! the day of `i` for [`CaseDate::FirstEvent`] and of the earliest such `j` for
//! [`CaseDate::ConfirmingEvent`]. Events outside the follow-up window confirm
//! nothing.
//!
//! # Algorithm
//!
//! Per person the rows are sorted by day. The prior and follow-up windows are
//! two binary searches. The events of a definition in a window are one scan
//! of the window's rows, and the confirmation is one pass over them with a
//! pointer to the next later day, or the next different contact. A person
//! with no row for a definition costs a bit test. The work is `O(rows +
//! persons x definitions + window rows)`, and no result depends on a thread
//! count: the function is serial.
//!
//! # References
//!
//! 1. Schmidt M, Schmidt SAJ, Sandegaard JL, Ehrenstein V, Pedersen L,
//!    Sorensen HT. The Danish National Patient Registry: a review of content,
//!    data quality, and research potential. Clin Epidemiol. 2015;7:449-90.
//! 2. Lund JL, Richardson DB, Sturmer T. The active comparator, new user
//!    study design in pharmacoepidemiology: historical foundations and
//!    contemporary application. Curr Epidemiol Rep. 2015;2(4):221-8.

use crate::person_time::{civil_from_days, days_from_civil};
use crate::{LeapBirthday, RegisterError};

/// How far back from the entry day the prior window reaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lookback {
    /// No prior window.
    None,
    /// A number of days, at least 0.
    Days(i64),
    /// A number of civil years, with the rule for a 29 February entry.
    Years(u32, LeapBirthday),
    /// Back to the start of the recorded history.
    Unlimited,
}

impl Lookback {
    /// The first day of the prior window for an entry day and a history start,
    /// or [`None`] for [`Lookback::None`].
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] for a negative number of days.
    pub fn window_start(
        self,
        entry: i64,
        history_start: i64,
    ) -> Result<Option<i64>, RegisterError> {
        if let Self::Days(days) = self
            && days < 0
        {
            return Err(RegisterError::input(format!(
                "a look-back of {days} days is negative"
            )));
        }
        Ok(self.start(entry, history_start))
    }

    /// [`window_start`](Self::window_start) for a look-back already checked.
    pub(crate) fn start(self, entry: i64, history_start: i64) -> Option<i64> {
        match self {
            Self::None => None,
            Self::Days(days) => Some(entry.saturating_sub(days)),
            Self::Years(years, leap) => Some(years_before(entry, years, leap)),
            Self::Unlimited => Some(history_start),
        }
    }
}

/// The day `years` civil years before `day`.
fn years_before(day: i64, years: u32, leap: LeapBirthday) -> i64 {
    let (y, m, d) = civil_from_days(day);
    let target = y - i64::from(years);
    let common_year = !(target % 4 == 0 && (target % 100 != 0 || target % 400 == 0));
    let (m, d) = if m == 2 && d == 29 && common_year {
        match leap {
            LeapBirthday::March1 => (3, 1),
            LeapBirthday::February28 => (2, 28),
        }
    } else {
        (m, d)
    };
    days_from_civil(target, m, d)
}

/// Which event of a confirmed pair gives the day of the case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaseDate {
    /// The first of the two events.
    FirstEvent,
    /// The confirming event.
    ConfirmingEvent,
}

/// What makes two events a confirmed pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfirmBy {
    /// The events are on different days.
    Dates,
    /// The events belong to different contacts, which needs
    /// [`EventRows::contact`].
    Contacts,
}

/// The rule of one definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rule {
    /// The look-back of the prior window.
    pub lookback: Lookback,
    /// 1 when one event is a case, 2 when a second event must confirm it.
    pub require: u32,
    /// The longest time in days between the two events, or [`None`] for no
    /// limit. Used when `require` is 2.
    pub confirm_within: Option<i64>,
    /// What makes the second event a confirmation. Used when `require` is 2.
    pub confirm_by: ConfirmBy,
    /// Which of the two events dates the case. Used when `require` is 2.
    pub date_of_case: CaseDate,
    /// Whether an event on the entry day is before entry.
    pub entry_inclusive: bool,
}

impl Rule {
    /// A rule where one event is a case and an event on the entry day is in
    /// the follow-up window.
    #[must_use]
    pub const fn single(lookback: Lookback) -> Self {
        Self {
            lookback,
            require: 1,
            confirm_within: None,
            confirm_by: ConfirmBy::Dates,
            date_of_case: CaseDate::FirstEvent,
            entry_inclusive: false,
        }
    }
}

/// The events of all persons, in CSR layout.
#[derive(Clone, Copy, Debug)]
pub struct EventRows<'a> {
    /// `person_offsets[p]..person_offsets[p + 1]` are the rows of person `p`.
    /// Its length is the number of persons plus one.
    pub person_offsets: &'a [usize],
    /// The day of each row. Rows are sorted by day within a person.
    pub day: &'a [i64],
    /// Bit `k` of a row's mask is set when the row is a qualifying event of
    /// definition `k`.
    pub mask: &'a [u64],
    /// The contact of each row, for [`ConfirmBy::Contacts`].
    pub contact: Option<&'a [u32]>,
}

/// The windows of each person, one value per person.
#[derive(Clone, Copy, Debug)]
pub struct PersonWindows<'a> {
    /// The start of follow-up, or the index date.
    pub entry: &'a [i64],
    /// The end of follow-up, at least the entry.
    pub exit: &'a [i64],
    /// The first day the register could have recorded the person.
    pub history_start: &'a [i64],
}

/// What the prior window shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PriorStatus {
    /// The rule has no look-back.
    NotAssessed,
    /// No event in a window that lies inside the recorded history.
    NoneObserved,
    /// An event in the prior window.
    Prevalent,
    /// No event found, but the window starts before the recorded history.
    HistoryTooShort,
}

/// The result for one person and one definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Outcome {
    /// The day of the first event in the follow-up window.
    pub first_in_window: Option<i64>,
    /// The day of the case after the confirmation rule.
    pub case_day: Option<i64>,
    /// What the prior window shows.
    pub prior: PriorStatus,
    /// The number of events in the follow-up window.
    pub n_events_in_window: u32,
    /// The number of events before the start of the recorded history.
    pub n_before_history: u32,
}

fn validate(
    rows: &EventRows<'_>,
    windows: &PersonWindows<'_>,
    rules: &[Rule],
) -> Result<usize, RegisterError> {
    let offsets = rows.person_offsets;
    let persons = offsets.len().saturating_sub(1);
    if offsets.is_empty() || offsets[0] != 0 || offsets[persons] != rows.day.len() {
        return Err(RegisterError::input(
            "person_offsets must start at 0 and end at the number of rows",
        ));
    }
    if rows.mask.len() != rows.day.len() || rows.contact.is_some_and(|c| c.len() != rows.day.len())
    {
        return Err(RegisterError::input(
            "day, mask and contact must have one value per row",
        ));
    }
    for (name, len) in [
        ("entry", windows.entry.len()),
        ("exit", windows.exit.len()),
        ("history_start", windows.history_start.len()),
    ] {
        if len != persons {
            return Err(RegisterError::input(format!(
                "{name} has {len} values for {persons} persons"
            )));
        }
    }
    if rules.is_empty() || rules.len() > 64 {
        return Err(RegisterError::input(format!(
            "between 1 and 64 rules are needed, got {}",
            rules.len()
        )));
    }
    for rule in rules {
        if rule.require != 1 && rule.require != 2 {
            return Err(RegisterError::input(format!(
                "require must be 1 or 2, got {}",
                rule.require
            )));
        }
        if rule.confirm_within.is_some_and(|w| w < 0) {
            return Err(RegisterError::input("confirm_within must not be negative"));
        }
        if rule.require == 2 && rule.confirm_by == ConfirmBy::Contacts && rows.contact.is_none() {
            return Err(RegisterError::input(
                "a rule that confirms by contacts needs the contact column",
            ));
        }
        rule.lookback.window_start(0, 0)?;
    }
    for p in 0..persons {
        let (a, b) = (offsets[p], offsets[p + 1]);
        if a > b {
            return Err(RegisterError::input("person_offsets must not decrease"));
        }
        if rows.day[a..b].windows(2).any(|w| w[0] > w[1]) {
            return Err(RegisterError::input(format!(
                "the rows of person {p} are not sorted by day"
            )));
        }
        if windows.entry[p] > windows.exit[p] {
            return Err(RegisterError::input(format!(
                "person {p} has an entry after its exit"
            )));
        }
    }
    Ok(persons)
}

/// Evaluate every definition for every person.
///
/// `rules[k]` is the rule of the definition with bit `k`. The result is indexed
/// `[definition][person]`.
///
/// # Errors
///
/// [`RegisterError::Input`] when the columns differ in length, the offsets are
/// not a CSR layout, the rows of a person are not sorted by day, a person's
/// entry is after its exit, there are no rules or more than 64, `require` is not
/// 1 or 2, a look-back or `confirm_within` is negative, or a rule that confirms
/// by contacts has no contact column.
pub fn evaluate(
    rows: &EventRows<'_>,
    windows: &PersonWindows<'_>,
    rules: &[Rule],
) -> Result<Vec<Vec<Outcome>>, RegisterError> {
    let persons = validate(rows, windows, rules)?;
    let blank = Outcome {
        first_in_window: None,
        case_day: None,
        prior: PriorStatus::NotAssessed,
        n_events_in_window: 0,
        n_before_history: 0,
    };
    let mut out = vec![vec![blank; persons]; rules.len()];
    let mut events: Vec<usize> = Vec::new();
    let mut next_other: Vec<usize> = Vec::new();
    for (p, span) in rows.person_offsets.windows(2).enumerate() {
        let (a, b) = (span[0], span[1]);
        let days = &rows.day[a..b];
        let masks = &rows.mask[a..b];
        let present = masks.iter().fold(0_u64, |acc, m| acc | m);
        let (entry, exit, history) = (windows.entry[p], windows.exit[p], windows.history_start[p]);
        for (k, rule) in rules.iter().enumerate() {
            let bit = 1_u64 << k;
            let start = rule.lookback.start(entry, history);
            let outcome = &mut out[k][p];
            let first_real = days.partition_point(|&d| d < history);
            let boundary = entry + i64::from(rule.entry_inclusive);
            let prior_lo = start.map(|s| s.max(history));
            let prior_hits = |lo: i64| -> bool {
                let from = days.partition_point(|&d| d < lo);
                let to = days.partition_point(|&d| d < boundary);
                (from..to).any(|i| masks[i] & bit != 0)
            };
            outcome.prior = match start {
                None => PriorStatus::NotAssessed,
                Some(s) => {
                    let found = present & bit != 0 && prior_lo.is_some_and(prior_hits);
                    if found {
                        PriorStatus::Prevalent
                    } else if s < history {
                        PriorStatus::HistoryTooShort
                    } else {
                        PriorStatus::NoneObserved
                    }
                }
            };
            if present & bit == 0 {
                continue;
            }
            outcome.n_before_history =
                u32::try_from((0..first_real).filter(|&i| masks[i] & bit != 0).count())
                    .unwrap_or(u32::MAX);
            let lo = days.partition_point(|&d| d < boundary.max(history));
            let hi = days.partition_point(|&d| d <= exit);
            events.clear();
            events.extend((lo..hi).filter(|&i| masks[i] & bit != 0));
            outcome.n_events_in_window = u32::try_from(events.len()).unwrap_or(u32::MAX);
            let Some(&first) = events.first() else {
                continue;
            };
            outcome.first_in_window = Some(days[first]);
            outcome.case_day = if rule.require == 1 {
                Some(days[first])
            } else {
                confirmed_case(
                    rule,
                    days,
                    rows.contact.map(|c| &c[a..b]),
                    &events,
                    &mut next_other,
                )
            };
        }
    }
    Ok(out)
}

/// The case day of a rule that needs a second event, or [`None`].
fn confirmed_case(
    rule: &Rule,
    days: &[i64],
    contact: Option<&[u32]>,
    events: &[usize],
    next_other: &mut Vec<usize>,
) -> Option<i64> {
    let n = events.len();
    // For each event, the index among `events` of the first later event that
    // can confirm it, or `n`.
    next_other.clear();
    next_other.resize(n, n);
    match (rule.confirm_by, contact) {
        (ConfirmBy::Contacts, Some(contact)) => {
            for i in (0..n.saturating_sub(1)).rev() {
                next_other[i] = if contact[events[i + 1]] == contact[events[i]] {
                    next_other[i + 1]
                } else {
                    i + 1
                };
            }
        }
        _ => {
            for i in (0..n.saturating_sub(1)).rev() {
                next_other[i] = if days[events[i + 1]] == days[events[i]] {
                    next_other[i + 1]
                } else {
                    i + 1
                };
            }
        }
    }
    for i in 0..n {
        let j = next_other[i];
        if j == n {
            continue;
        }
        let gap = days[events[j]] - days[events[i]];
        if rule.confirm_within.is_none_or(|w| gap <= w) {
            return Some(match rule.date_of_case {
                CaseDate::FirstEvent => days[events[i]],
                CaseDate::ConfirmingEvent => days[events[j]],
            });
        }
    }
    None
}

#[cfg(test)]
mod tests;

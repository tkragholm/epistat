//! Sequential trial emulation with matching in each trial.
//!
//! A target trial is emulated at every period at which people are eligible
//! (Hernán and Robins 2016; Danaei et al. 2013). In the trial that starts at
//! period t, everyone eligible at t is assigned the strategy they follow at t,
//! and is followed from t. A person is eligible in many trials, so one person
//! appears in many of them.
//!
//! # The expansion
//!
//! The panel holds one row per person and period, with the eligibility, the
//! treatment and whether the outcome occurred. The expansion follows the
//! documented behaviour of the R package TrialEmulation (version 0.0.5):
//!
//! * rows before the first eligible period of a person and rows after the
//!   person's first outcome are dropped;
//! * a trial starts at every remaining period at which the person is eligible
//!   and the treatment is known, between the first and the last period asked
//!   for;
//! * the assignment of the person in that trial is the treatment at the start;
//! * the person is followed from the start to the last row. The follow-up time
//!   of a row is its period minus the period of the start;
//! * the person deviates from the assignment at the first row after the start
//!   at which the treatment differs from the treatment of the row before, when
//!   both are known. Per-protocol follow-up ends on the row before that, and
//!   the intention-to-treat follow-up does not end there.
//!
//! # Matching
//!
//! In each trial the initiators, who are assigned treatment, are matched to
//! people eligible in the same trial who are not, by risk-set matching on the
//! date of the trial. The units of the trial are the people eligible at its
//! start. An initiator is the case of the set and takes controls from the
//! non-initiators with the options of the matching job: strata, birth-date
//! window, age limit, alive and resident checks, entry. A non-initiator of
//! the trial who initiates in a later trial is a valid control, since the
//! treatment of a later period is unknown at the start of the trial. Matching
//! on time (risk-set matching) follows Li, Propert and Rosenbaum (2001) and
//! Lu (2005).
//!
//! Two options say how people are reused between trials. The default of both
//! follows the sequential trial design, in which each trial is emulated on
//! everyone eligible: a control may serve in several trials, and a person may
//! be a control in a trial before the one in which the person initiates.
//!
//! * [`ControlReuse::OncePerRun`] removes a person who was drawn as a control
//!   from the pool of the later trials, as controls are used once in matching
//!   without replacement. The person stays a possible initiator.
//! * `initiators_as_earlier_controls = false` removes from the pool of a trial
//!   every person who initiates in a later trial. That conditions on the
//!   future and makes the controls of early trials a selected group, so the
//!   option is there to be compared with the default and not as a recommended
//!   design.
//!
//! The output gives, for each trial, the sets, the start of the follow-up of
//! every matched person and the period of the deviation. The analysis censors
//! each person at the deviation. Weights for the censoring are the task of the
//! analysis and are not computed here.

use crate::matching::MatchingRecord;
use crate::types::{MatchOutcome, RoleTransitionRecord};
use chrono::NaiveDate;
use rapidhash::RapidHashMap;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

/// One period of one person.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrialPeriod {
    /// The period number. Strictly increasing within a person.
    pub period: i64,
    /// The first day of the period. Every person has the same date for the
    /// same period, and the dates increase with the period. A trial that starts
    /// at the period has this date as the index date of its sets.
    pub date: NaiveDate,
    /// Whether the person is eligible for a trial that starts at the period.
    pub eligible: bool,
    /// Whether the person is treated in the period, or `None` when it is not
    /// known. A trial does not start at a period without a known treatment.
    pub treatment: Option<bool>,
    /// Whether the outcome occurs in the period. Later periods are dropped.
    pub event: bool,
}

/// A person with the periods in which the person is observed.
#[derive(Debug, Clone)]
pub struct TrialPerson<R> {
    /// The record the matching constraints read: birth date, strata, entry,
    /// death and emigration. It is the same in every trial.
    pub record: R,
    /// The periods, in increasing order.
    pub periods: Vec<TrialPeriod>,
}

/// Whether a person who served as a control may serve again in a later trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlReuse {
    /// A control may serve in any number of trials. This is the sequential
    /// trial design of Danaei et al.
    #[default]
    AcrossTrials,
    /// A person drawn as a control in one trial is not a control in a later
    /// trial. The person can still be an initiator.
    OncePerRun,
}

/// Options of the expansion and of the matching across trials.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrialOptions {
    /// The first period at which a trial may start. No lower limit when
    /// `None`.
    pub first_period: Option<i64>,
    /// The last period at which a trial may start. No upper limit when `None`.
    pub last_period: Option<i64>,
    /// Whether a control may serve in several trials.
    pub control_reuse: ControlReuse,
    /// Whether a person who initiates in a later trial may be a control in an
    /// earlier one. The default is yes.
    pub initiators_as_earlier_controls: bool,
    /// Whether the follow-up of a person ends before the deviation from the
    /// assigned strategy, as in a per-protocol analysis. When false the
    /// follow-up runs to the last period, as in an intention-to-treat
    /// analysis. The period of the deviation is reported either way.
    pub censor_at_deviation: bool,
}

impl Default for TrialOptions {
    fn default() -> Self {
        Self {
            first_period: None,
            last_period: None,
            control_reuse: ControlReuse::AcrossTrials,
            initiators_as_earlier_controls: true,
            censor_at_deviation: true,
        }
    }
}

/// Why the panel could not be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrialInputError {
    /// Two people share an id, so a pair does not say which person it means.
    DuplicateId {
        /// The id that occurs twice.
        id: String,
    },
    /// The periods of a person are not strictly increasing.
    PeriodsNotIncreasing {
        /// The person.
        id: String,
        /// The period that does not follow its predecessor.
        period: i64,
    },
    /// Two people give different dates for one period.
    InconsistentDate {
        /// The period.
        period: i64,
    },
    /// The dates of the periods do not increase with the period number.
    DatesNotIncreasing {
        /// The period whose date is not after the date of the period before.
        period: i64,
    },
}

impl Display for TrialInputError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateId { id } => write!(
                f,
                "the id `{id}` occurs more than once in the panel; give every person their own id"
            ),
            Self::PeriodsNotIncreasing { id, period } => write!(
                f,
                "the periods of `{id}` do not increase: period {period} does not follow the one before it"
            ),
            Self::InconsistentDate { period } => write!(
                f,
                "period {period} has more than one date; every person needs the same date for a period"
            ),
            Self::DatesNotIncreasing { period } => write!(
                f,
                "the date of period {period} is not after the date of the period before it"
            ),
        }
    }
}

impl Error for TrialInputError {}

/// Why trials could not be matched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrialsError<E> {
    /// The panel is not valid.
    Input(TrialInputError),
    /// The matching of one trial failed.
    Matching {
        /// The period at which the trial starts.
        period: i64,
        /// What the matching reported.
        error: E,
    },
}

impl<E: Display> Display for TrialsError<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input(error) => write!(f, "{error}"),
            Self::Matching { period, error } => {
                write!(f, "the trial that starts at period {period}: {error}")
            }
        }
    }
}

impl<E: Error + 'static> Error for TrialsError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Input(error) => Some(error),
            Self::Matching { error, .. } => Some(error),
        }
    }
}

impl<E> From<TrialInputError> for TrialsError<E> {
    fn from(error: TrialInputError) -> Self {
        Self::Input(error)
    }
}

/// One row of the expanded data: a person in a trial at one period of the
/// follow-up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpandedRow {
    /// The period at which the trial starts.
    pub trial_period: i64,
    /// The person.
    pub id: String,
    /// The period of the row minus the period at which the trial starts.
    pub follow_up_time: i64,
    /// The treatment of the person at the start of the trial.
    pub assigned: bool,
    /// The treatment in the period of the row.
    pub treatment: Option<bool>,
    /// Whether the outcome occurs in the row.
    pub outcome: bool,
}

/// A person who was matched in a trial.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrialMember {
    /// The person.
    pub id: String,
    /// Whether the person initiates treatment in the trial.
    pub assigned: bool,
    /// The period at which the follow-up starts, which is the start of the
    /// trial.
    pub follow_up_start_period: i64,
    /// The first day of that period.
    pub follow_up_start: NaiveDate,
    /// The period at which the person first differs from the treatment of the
    /// period before, after the start. `None` when the person never deviates.
    /// The analysis censors the person at this period.
    pub deviation_period: Option<i64>,
    /// The first day of the period of the deviation.
    pub deviation_date: Option<NaiveDate>,
    /// The last period of the follow-up. With censoring at the deviation it is
    /// the period before the deviation.
    pub last_period: i64,
}

/// The sets of one trial.
#[derive(Debug, Clone)]
pub struct TrialResult {
    /// The period at which the trial starts.
    pub period: i64,
    /// The first day of the period, which is the index date of the sets.
    pub index_date: NaiveDate,
    /// How many people were in the pool of the trial.
    pub n_eligible: usize,
    /// How many of them initiate treatment.
    pub n_initiators: usize,
    /// The pairs and diagnostics of the risk-set matching.
    pub outcome: MatchOutcome,
    /// The people with a set, each once, in the order of the pairs.
    pub members: Vec<TrialMember>,
}

/// The trials of a run, in the order of their start.
#[derive(Debug, Clone)]
pub struct TrialOutcome {
    /// The trials that have at least one initiator in the pool.
    pub trials: Vec<TrialResult>,
}

/// A person in a trial, by position in the panel.
#[derive(Debug, Clone, Copy)]
struct Enrolment {
    person: usize,
    /// Index of the start row in the periods of the person.
    start: usize,
    assigned: bool,
    /// Index of the row of the first deviation.
    deviation: Option<usize>,
    /// Index of the last row of the follow-up.
    last: usize,
}

struct Plan {
    /// Trials by start period, each with the date and the enrolments in the
    /// order of the panel.
    trials: BTreeMap<i64, (NaiveDate, Vec<Enrolment>)>,
    /// The last trial in which each person is assigned treatment.
    last_initiation: Vec<Option<i64>>,
}

fn validate<R: MatchingRecord>(panel: &[TrialPerson<R>]) -> Result<(), TrialInputError> {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut dates: BTreeMap<i64, NaiveDate> = BTreeMap::new();
    for person in panel {
        if !seen.insert(person.record.id()) {
            return Err(TrialInputError::DuplicateId {
                id: person.record.id().to_string(),
            });
        }
        for pair in person.periods.windows(2) {
            if pair[1].period <= pair[0].period {
                return Err(TrialInputError::PeriodsNotIncreasing {
                    id: person.record.id().to_string(),
                    period: pair[1].period,
                });
            }
        }
        for row in &person.periods {
            if *dates.entry(row.period).or_insert(row.date) != row.date {
                return Err(TrialInputError::InconsistentDate { period: row.period });
            }
        }
    }
    let mut previous: Option<NaiveDate> = None;
    for (period, date) in &dates {
        if previous.is_some_and(|before| *date <= before) {
            return Err(TrialInputError::DatesNotIncreasing { period: *period });
        }
        previous = Some(*date);
    }
    Ok(())
}

/// Whether the treatment differs from the row before, when both are known.
fn deviates(rows: &[TrialPeriod], at: usize) -> bool {
    at > 0
        && matches!(
            (rows[at - 1].treatment, rows[at].treatment),
            (Some(before), Some(now)) if before != now
        )
}

fn plan<R: MatchingRecord>(panel: &[TrialPerson<R>], options: &TrialOptions) -> Plan {
    let mut trials: BTreeMap<i64, (NaiveDate, Vec<Enrolment>)> = BTreeMap::new();
    let mut last_initiation = vec![None; panel.len()];
    for (person, row) in panel.iter().enumerate() {
        let rows = &row.periods;
        // Rows before the first eligible period and after the first outcome
        // are not used.
        let Some(first) = rows.iter().position(|r| r.eligible) else {
            continue;
        };
        let end = rows[first..]
            .iter()
            .position(|r| r.event)
            .map_or(rows.len(), |at| first + at + 1);
        let used = &rows[first..end];
        // The first deviation after each row, from the back.
        let mut next: Vec<Option<usize>> = vec![None; used.len()];
        for at in (0..used.len().saturating_sub(1)).rev() {
            next[at] = if deviates(used, at + 1) {
                Some(at + 1)
            } else {
                next[at + 1]
            };
        }
        for (at, start) in used.iter().enumerate() {
            let Some(assigned) = start.treatment else {
                continue;
            };
            if !start.eligible
                || options.first_period.is_some_and(|p| start.period < p)
                || options.last_period.is_some_and(|p| start.period > p)
            {
                continue;
            }
            let deviation = next[at].map(|d| first + d);
            let last = match deviation {
                Some(d) if options.censor_at_deviation => d - 1,
                _ => end - 1,
            };
            let entry = trials
                .entry(start.period)
                .or_insert_with(|| (start.date, Vec::new()));
            entry.1.push(Enrolment {
                person,
                start: first + at,
                assigned,
                deviation,
                last,
            });
            if assigned {
                last_initiation[person] = Some(start.period);
            }
        }
    }
    Plan {
        trials,
        last_initiation,
    }
}

/// The expanded data of the panel: one row per person, trial and period of
/// the follow-up, in the order of the panel, then the trial, then the
/// follow-up time.
///
/// # Errors
///
/// A [`TrialInputError`] when two people share an id, or the periods or their
/// dates are not in order.
pub fn expand_trials<R: MatchingRecord>(
    panel: &[TrialPerson<R>],
    options: &TrialOptions,
) -> Result<Vec<ExpandedRow>, TrialInputError> {
    validate(panel)?;
    let plan = plan(panel, options);
    let mut by_person: Vec<Vec<(i64, Enrolment)>> = vec![Vec::new(); panel.len()];
    for (period, (_, enrolments)) in &plan.trials {
        for enrolment in enrolments {
            by_person[enrolment.person].push((*period, *enrolment));
        }
    }
    let mut rows = Vec::new();
    for (person, trials) in by_person.iter().enumerate() {
        let source = &panel[person];
        for (period, enrolment) in trials {
            for row in &source.periods[enrolment.start..=enrolment.last] {
                rows.push(ExpandedRow {
                    trial_period: *period,
                    id: source.record.id().to_string(),
                    follow_up_time: row.period - period,
                    assigned: enrolment.assigned,
                    treatment: row.treatment,
                    outcome: row.event,
                });
            }
        }
    }
    Ok(rows)
}

/// The seed of the trial that starts at `period`, from the seed of the run. A
/// trial has the same seed in every run with the same seed, whatever the other
/// trials are and however many threads run.
#[must_use]
#[expect(
    clippy::cast_sign_loss,
    reason = "the bits of the period are mixed, the sign is not used"
)]
pub fn trial_seed(seed: u64, period: i64) -> u64 {
    // SplitMix64 of the seed combined with the period, so that the seeds of
    // neighbouring periods are not neighbours.
    let mut z = seed.wrapping_add((period as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Match initiators to non-initiators in every trial of the panel.
///
/// `run` matches one trial. It is given the period at which the trial starts
/// and the pool of the trial as a risk-set cohort: the people eligible at the
/// start, with the date of the trial as the transition date of the initiators
/// and none for the rest. It returns the matching of that cohort, for example
/// by running a [`MatchJob::new_transition`](crate::MatchJob::new_transition)
/// with the options of the study and a seed from [`trial_seed`]. The pool
/// holds a reference to the record of each person in the panel, so a record
/// is never copied, however many trials the person is eligible in.
///
/// Trials without an initiator in the pool are left out of the output.
///
/// # Errors
///
/// A [`TrialsError`] when the panel is not valid or `run` fails.
pub fn match_trials<R, E, F>(
    panel: &[TrialPerson<R>],
    options: &TrialOptions,
    mut run: F,
) -> Result<TrialOutcome, TrialsError<E>>
where
    R: MatchingRecord,
    F: FnMut(i64, &[RoleTransitionRecord<&R>]) -> Result<MatchOutcome, E>,
{
    validate(panel)?;
    let plan = plan(panel, options);
    let mut used_as_control = vec![false; panel.len()];
    let mut trials = Vec::new();
    for (period, (date, enrolments)) in &plan.trials {
        let pool: Vec<&Enrolment> = enrolments
            .iter()
            .filter(|e| {
                e.assigned
                    || !((options.control_reuse == ControlReuse::OncePerRun
                        && used_as_control[e.person])
                        || (!options.initiators_as_earlier_controls
                            && plan.last_initiation[e.person].is_some_and(|last| last > *period)))
            })
            .collect();
        let n_initiators = pool.iter().filter(|e| e.assigned).count();
        if n_initiators == 0 {
            continue;
        }
        let cohort: Vec<RoleTransitionRecord<&R>> = pool
            .iter()
            .map(|e| {
                RoleTransitionRecord::from_record(
                    &panel[e.person].record,
                    e.assigned.then_some(*date),
                )
            })
            .collect();
        let outcome = run(*period, &cohort).map_err(|error| TrialsError::Matching {
            period: *period,
            error,
        })?;
        let position: RapidHashMap<&str, usize> = pool
            .iter()
            .enumerate()
            .map(|(at, e)| (panel[e.person].record.id(), at))
            .collect();
        let mut listed = vec![false; pool.len()];
        let mut members = Vec::new();
        let mut add = |id: &str, members: &mut Vec<TrialMember>| {
            let at = position.get(id)?;
            if std::mem::replace(&mut listed[*at], true) {
                return Some(*at);
            }
            let e = pool[*at];
            let rows = &panel[e.person].periods;
            members.push(TrialMember {
                id: id.to_string(),
                assigned: e.assigned,
                follow_up_start_period: *period,
                follow_up_start: *date,
                deviation_period: e.deviation.map(|d| rows[d].period),
                deviation_date: e.deviation.map(|d| rows[d].date),
                last_period: rows[e.last].period,
            });
            Some(*at)
        };
        for pair in &outcome.pairs {
            add(pair.anchor_id(), &mut members);
            if let Some(at) = add(pair.comparator_id(), &mut members) {
                used_as_control[pool[at].person] = true;
            }
        }
        trials.push(TrialResult {
            period: *period,
            index_date: *date,
            n_eligible: pool.len(),
            n_initiators,
            outcome,
            members,
        });
    }
    Ok(TrialOutcome { trials })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BaseRecord;
    use chrono::Days;

    fn day(n: u64) -> NaiveDate {
        NaiveDate::from_ymd_opt(2020, 1, 1)
            .expect("valid")
            .checked_add_days(Days::new(n * 30))
            .expect("in range")
    }

    fn person(id: &str, rows: &[(i64, bool, Option<bool>, bool)]) -> TrialPerson<BaseRecord> {
        TrialPerson {
            record: BaseRecord::new(id, NaiveDate::from_ymd_opt(1980, 1, 1).expect("valid")),
            periods: rows
                .iter()
                .map(|&(period, eligible, treatment, event)| TrialPeriod {
                    period,
                    date: day(u64::try_from(period).expect("non-negative")),
                    eligible,
                    treatment,
                    event,
                })
                .collect(),
        }
    }

    #[test]
    fn expansion_cuts_at_the_first_change_of_treatment() {
        let a = person(
            "a",
            &[
                (0, true, Some(false), false),
                (1, true, Some(false), false),
                (2, false, Some(true), false),
                (3, false, Some(true), true),
            ],
        );
        let per_protocol =
            expand_trials(std::slice::from_ref(&a), &TrialOptions::default()).expect("valid");
        let shape: Vec<(i64, i64)> = per_protocol
            .iter()
            .map(|r| (r.trial_period, r.follow_up_time))
            .collect();
        assert_eq!(shape, vec![(0, 0), (0, 1), (1, 0)]);
        let itt = expand_trials(
            &[a],
            &TrialOptions {
                censor_at_deviation: false,
                ..TrialOptions::default()
            },
        )
        .expect("valid");
        let shape: Vec<(i64, i64, bool)> = itt
            .iter()
            .map(|r| (r.trial_period, r.follow_up_time, r.outcome))
            .collect();
        assert_eq!(
            shape,
            vec![
                (0, 0, false),
                (0, 1, false),
                (0, 2, false),
                (0, 3, true),
                (1, 0, false),
                (1, 1, false),
                (1, 2, true)
            ]
        );
    }

    #[test]
    fn rows_before_eligibility_and_after_the_event_are_dropped() {
        let a = person(
            "a",
            &[
                (0, false, Some(false), false),
                (1, true, Some(false), false),
                (2, true, Some(false), true),
                (3, true, Some(false), false),
            ],
        );
        let rows = expand_trials(&[a], &TrialOptions::default()).expect("valid");
        let shape: Vec<(i64, i64)> = rows
            .iter()
            .map(|r| (r.trial_period, r.follow_up_time))
            .collect();
        assert_eq!(shape, vec![(1, 0), (1, 1), (2, 0)]);
    }

    #[test]
    fn unknown_treatment_starts_no_trial_and_is_not_a_deviation() {
        let a = person(
            "a",
            &[
                (0, true, Some(true), false),
                (1, true, None, false),
                (2, true, Some(false), false),
                (3, true, Some(false), false),
            ],
        );
        let rows = expand_trials(&[a], &TrialOptions::default()).expect("valid");
        let shape: Vec<(i64, i64)> = rows
            .iter()
            .map(|r| (r.trial_period, r.follow_up_time))
            .collect();
        // The change from unknown to false is not compared, and the change
        // from false to false is none.
        assert_eq!(
            shape,
            vec![(0, 0), (0, 1), (0, 2), (0, 3), (2, 0), (2, 1), (3, 0)]
        );
    }

    #[test]
    fn period_limits_restrict_the_trials() {
        let a = person(
            "a",
            &[
                (0, true, Some(false), false),
                (1, true, Some(false), false),
                (2, true, Some(false), false),
            ],
        );
        let rows = expand_trials(
            &[a],
            &TrialOptions {
                first_period: Some(1),
                last_period: Some(1),
                ..TrialOptions::default()
            },
        )
        .expect("valid");
        assert!(rows.iter().all(|r| r.trial_period == 1));
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn invalid_panels_are_errors() {
        let a = person("a", &[(0, true, Some(false), false)]);
        let b = person("a", &[(0, true, Some(false), false)]);
        assert!(matches!(
            expand_trials(&[a.clone(), b], &TrialOptions::default()),
            Err(TrialInputError::DuplicateId { .. })
        ));
        let c = person(
            "c",
            &[(1, true, Some(false), false), (1, true, Some(false), false)],
        );
        assert!(matches!(
            expand_trials(&[c], &TrialOptions::default()),
            Err(TrialInputError::PeriodsNotIncreasing { .. })
        ));
        let mut d = person("d", &[(0, true, Some(false), false)]);
        d.periods[0].date = day(9);
        assert!(matches!(
            expand_trials(&[a, d], &TrialOptions::default()),
            Err(TrialInputError::InconsistentDate { period: 0 })
        ));
    }

    #[test]
    fn trial_seeds_differ_by_period_and_repeat() {
        assert_eq!(trial_seed(5, 3), trial_seed(5, 3));
        assert_ne!(trial_seed(5, 3), trial_seed(5, 4));
        assert_ne!(trial_seed(5, 3), trial_seed(6, 3));
    }
}

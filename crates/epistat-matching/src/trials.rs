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
//! # Dates and strata that differ between people and trials
//!
//! By default every person has the same date for the same period, and the
//! sets of a trial have that date as their index date. With
//! [`PeriodDates::PerPerson`] (through [`match_trials`]) the periods are a
//! time scale of each person's own, such as months since diagnosis, and each
//! person gives the calendar date of each of their periods. The index date of a
//! set is then the date of its initiator's period: the controls are at risk,
//! alive and resident on that date and the age limit is taken at it, as for
//! any risk set, and their own period of the trial says they are eligible.
//!
//! The strata of the record are the same in every trial. [`PeriodStrata`] gives
//! strata of each person and period, such as a flare in the month of the
//! trial, and in a trial a person's value is that of the period at which the
//! trial starts. The matching reads such a key before the record's own, so the
//! job's exact matches name the keys of both alike.
//!
//! With shared dates and no period strata, [`match_trials`] is
//! [`match_trials`].
//!
//! The output gives, for each trial, the sets, the start of the follow-up of
//! every matched person and the period of the deviation. The analysis censors
//! each person at the deviation. Weights for the censoring are the task of the
//! analysis and are not computed here.

use crate::Error;
use crate::matching::job::TransitionMode;
use crate::matching::{CovariateRecord, MatchJob, MatchingRecord, RandomSelection};
use crate::role_transition::IdOrder;
use crate::types::{CovariateRef, CovariateValue};
use crate::types::{MatchOutcome, RoleTransitionRecord};
use chrono::NaiveDate;
use rapidhash::RapidHashMap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

use std::sync::OnceLock;

/// One period of one person.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrialPeriod {
    /// The period number. Strictly increasing within a person.
    pub period: i64,
    /// The first day of the period. For [`match_trials`] every person has the
    /// same date for the same period, the dates increase with the period, and
    /// a trial that starts at the period has this date as the index date of its
    /// sets. The expansion does not read it.
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
    /// The first day of that period, in the person's own dates.
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
    /// The first day of the period, which is the index date of the sets. With
    /// [`PeriodDates::PerPerson`] each set has its initiator's date (the
    /// initiator's [`TrialMember::follow_up_start`]), and this is the earliest
    /// of them.
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

/// Checks the ids and the order of the periods, and the dates as `dates`
/// says: none for the expansion, which reads no date; for
/// [`PeriodDates::Shared`] that the dates of a period agree across persons
/// and increase with the period; for [`PeriodDates::PerPerson`] that each
/// person's dates increase with the period.
fn validate<R: MatchingRecord>(
    panel: &[TrialPerson<R>],
    dates: Option<PeriodDates>,
) -> Result<(), Error> {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut by_period: BTreeMap<i64, NaiveDate> = BTreeMap::new();
    for person in panel {
        if !seen.insert(person.record.id()) {
            return Err(Error::DuplicateId {
                id: person.record.id().to_string(),
            });
        }
        for pair in person.periods.windows(2) {
            if pair[1].period <= pair[0].period {
                return Err(Error::PeriodsNotIncreasing {
                    id: person.record.id().to_string(),
                    period: pair[1].period,
                });
            }
        }
        match dates {
            None => {}
            Some(PeriodDates::Shared) => {
                for row in &person.periods {
                    if *by_period.entry(row.period).or_insert(row.date) != row.date {
                        return Err(Error::InconsistentDate { period: row.period });
                    }
                }
            }
            Some(PeriodDates::PerPerson) => {
                for pair in person.periods.windows(2) {
                    if pair[1].date <= pair[0].date {
                        return Err(Error::DatesNotIncreasing {
                            period: pair[1].period,
                        });
                    }
                }
            }
        }
    }
    let mut previous: Option<NaiveDate> = None;
    for (period, date) in &by_period {
        if previous.is_some_and(|before| *date <= before) {
            return Err(Error::DatesNotIncreasing { period: *period });
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
/// The dates of the periods are not read, so a panel whose periods are times
/// since a person's own origin (months since diagnosis) may give each person
/// the calendar dates of their periods.
///
/// # Errors
///
/// A [`Error`] when two people share an id or the periods of a
/// person are not in order.
pub fn expand_trials<R: MatchingRecord>(
    panel: &[TrialPerson<R>],
    options: &TrialOptions,
) -> Result<Vec<ExpandedRow>, Error> {
    validate(panel, None)?;
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

/// The panel in columns: one row per person and period, in the order of the
/// panel and then of the periods.
///
/// A row of the trials points at these rows ([`TrialColumns::row`] and
/// [`TrialColumns::start_row`]), so covariates kept in columns in the same
/// order are read by the row. The columns of a person-period analysis in
/// `epistat_survival::weighting::trials` (`PersonPeriod`, `Panel`) are these
/// columns with the treatment known.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PanelColumns {
    /// The position of the person in the panel.
    pub person: Vec<u64>,
    /// The period of the row.
    pub period: Vec<i64>,
    /// The treatment in the period, `None` when it is not known.
    pub treatment: Vec<Option<bool>>,
    /// Whether the person is eligible for a trial that starts in the period.
    pub eligible: Vec<bool>,
    /// Whether the outcome occurs in the period.
    pub event: Vec<bool>,
}

impl PanelColumns {
    /// The columns of `panel`, every period of every person.
    #[must_use]
    pub fn of<R>(panel: &[TrialPerson<R>]) -> Self {
        let rows: usize = panel.iter().map(|p| p.periods.len()).sum();
        let mut out = Self {
            person: Vec::with_capacity(rows),
            period: Vec::with_capacity(rows),
            treatment: Vec::with_capacity(rows),
            eligible: Vec::with_capacity(rows),
            event: Vec::with_capacity(rows),
        };
        for (person, p) in (0_u64..).zip(panel) {
            for row in &p.periods {
                out.person.push(person);
                out.period.push(row.period);
                out.treatment.push(row.treatment);
                out.eligible.push(row.eligible);
                out.event.push(row.event);
            }
        }
        out
    }

    /// The number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.person.len()
    }

    /// Whether there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.person.is_empty()
    }

    /// The treatment of every row, `None` when a row's treatment is not known.
    /// The models of the weights need it known.
    #[must_use]
    pub fn known_treatment(&self) -> Option<Vec<bool>> {
        self.treatment.iter().copied().collect()
    }
}

/// The rows of the trials in columns, each with the rows of the panel it is
/// read from: the form that the weights and the outcome model of
/// `epistat_survival::weighting::trials` take.
///
/// The rows are in the order of [`expand_trials`]: by person in the order of
/// the panel, then by trial, then by follow-up time. `person`, `trial_period`
/// and `follow_up_time` are the columns of its `TrialRows`, and the spans of
/// [`Self::spans`] those of its `TrialSpans`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TrialColumns {
    /// The position of the person in the panel.
    pub person: Vec<u64>,
    /// The period at which the trial starts.
    pub trial_period: Vec<i64>,
    /// The period of the row minus the period at which the trial starts.
    pub follow_up_time: Vec<i64>,
    /// The treatment of the person at the start of the trial.
    pub assigned: Vec<bool>,
    /// The treatment in the period of the row.
    pub treatment: Vec<Option<bool>>,
    /// Whether the outcome occurs in the row.
    pub outcome: Vec<bool>,
    /// The row of [`PanelColumns`] this row is read from: the person at the
    /// period of the row, where the time-varying covariates of the row are.
    pub row: Vec<usize>,
    /// The row of [`PanelColumns`] at the start of the trial, where the
    /// covariates at the start (the baseline of the trial) are.
    pub start_row: Vec<usize>,
}

impl TrialColumns {
    /// The number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.person.len()
    }

    /// Whether there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.person.is_empty()
    }

    /// The trials as spans of the rows of [`PanelColumns`]: for each person in
    /// a trial, the row at which the trial starts and the row after its last
    /// row, in the order of the rows. Each span is a run of consecutive panel
    /// rows, since a person is followed in a trial from its start without a
    /// gap.
    #[must_use]
    pub fn spans(&self) -> (Vec<usize>, Vec<usize>) {
        let (mut start, mut end) = (Vec::new(), Vec::new());
        for i in 0..self.len() {
            if i == 0 || self.start_row[i] != self.start_row[i - 1] {
                start.push(self.start_row[i]);
                end.push(self.row[i] + 1);
            } else if let Some(last) = end.last_mut() {
                *last = self.row[i] + 1;
            }
        }
        (start, end)
    }

    fn push(
        &mut self,
        person: u64,
        trial: i64,
        rows: &[TrialPeriod],
        start: usize,
        at: usize,
        e: &Enrolment,
    ) {
        let row = &rows[at];
        self.person.push(person);
        self.trial_period.push(trial);
        self.follow_up_time.push(row.period - trial);
        self.assigned.push(e.assigned);
        self.treatment.push(row.treatment);
        self.outcome.push(row.event);
        self.row.push(start + at);
        self.start_row.push(start + e.start);
    }
}

/// The first row of each person in [`PanelColumns`], and the total.
fn offsets<R>(panel: &[TrialPerson<R>]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(panel.len() + 1);
    let mut at = 0;
    for p in panel {
        offsets.push(at);
        at += p.periods.len();
    }
    offsets.push(at);
    offsets
}

/// The expanded data of [`expand_trials`] in columns, with the rows of
/// [`PanelColumns::of`] that each row of the trials is read from.
///
/// # Errors
///
/// As [`expand_trials`].
pub fn expand_trial_columns<R: MatchingRecord>(
    panel: &[TrialPerson<R>],
    options: &TrialOptions,
) -> Result<TrialColumns, Error> {
    validate(panel, None)?;
    let plan = plan(panel, options);
    let offsets = offsets(panel);
    let mut by_person: Vec<Vec<(i64, Enrolment)>> = vec![Vec::new(); panel.len()];
    for (period, (_, enrolments)) in &plan.trials {
        for enrolment in enrolments {
            by_person[enrolment.person].push((*period, *enrolment));
        }
    }
    let mut out = TrialColumns::default();
    for ((person, trials), number) in by_person.iter().enumerate().zip(0_u64..) {
        let rows = &panel[person].periods;
        for (period, enrolment) in trials {
            for at in enrolment.start..=enrolment.last {
                out.push(number, *period, rows, offsets[person], at, enrolment);
            }
        }
    }
    Ok(out)
}

/// The rows of the people matched in each trial of `matched`, as
/// [`expand_trial_columns`] gives them for everyone eligible: each member of a
/// trial from the start of the trial to the end of the follow-up that
/// `options` sets (to the row before the deviation when
/// [`TrialOptions::censor_at_deviation`], to the first outcome or the last
/// row otherwise), with the position of the trial in `matched.trials`.
///
/// `panel` must be the panel `matched` was matched on. The rows are in the
/// order of the trials and, in each, of the members, so an analysis of the
/// matched trials clusters by [`TrialColumns::person`] as the analysis of all
/// trials does: a person who is a member of several trials has rows in each.
/// `options` may differ from those of the matching in `censor_at_deviation`
/// alone, so that the sets matched once are analysed by intention to treat and
/// per protocol.
///
/// # Errors
///
/// As [`expand_trials`], and [`Error::UnknownMember`] for a member
/// that is not enrolled in its trial under `options`.
pub fn matched_trial_columns<R: MatchingRecord>(
    panel: &[TrialPerson<R>],
    matched: &TrialOutcome,
    options: &TrialOptions,
) -> Result<(TrialColumns, Vec<usize>), Error> {
    validate(panel, None)?;
    let plan = plan(panel, options);
    let offsets = offsets(panel);
    let position: RapidHashMap<&str, usize> = panel
        .iter()
        .enumerate()
        .map(|(at, p)| (p.record.id(), at))
        .collect();
    let mut out = TrialColumns::default();
    let mut trial_of = Vec::new();
    for (k, trial) in matched.trials.iter().enumerate() {
        let enrolments = plan.trials.get(&trial.period).map(|(_, e)| e.as_slice());
        for member in &trial.members {
            let unknown = || Error::UnknownMember {
                id: member.id.clone(),
                period: trial.period,
            };
            let person = *position.get(member.id.as_str()).ok_or_else(unknown)?;
            let enrolment = enrolments
                .and_then(|e| {
                    e.binary_search_by_key(&person, |x| x.person)
                        .ok()
                        .map(|at| e[at])
                })
                .ok_or_else(unknown)?;
            let rows = &panel[person].periods;
            for at in enrolment.start..=enrolment.last {
                out.push(
                    u64::try_from(person).unwrap_or(u64::MAX),
                    trial.period,
                    rows,
                    offsets[person],
                    at,
                    &enrolment,
                );
                trial_of.push(k);
            }
        }
    }
    Ok((out, trial_of))
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
    epistat_rng::mix64(seed.wrapping_add((period as u64).wrapping_mul(epistat_rng::GAMMA)))
}

/// Whether the dates of a period are shared by every person.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PeriodDates {
    /// Every person has the same date for the same period, and the sets of a
    /// trial have that date as their index date.
    #[default]
    Shared,
    /// Each person has dates of their own, increasing with the period, and the
    /// index date of a set is the date of its initiator's period.
    PerPerson,
}

/// Strata of each person and period, read in each trial at the period at
/// which the trial starts.
///
/// The values are in the order of the rows of [`PanelColumns::of`]: person by
/// person in the order of the panel, period by period.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PeriodStrata {
    keys: Vec<String>,
    /// `values[k][row]`, the value of key `k` in panel row `row`.
    values: Vec<Vec<String>>,
}

impl PeriodStrata {
    /// Strata from one column of values per key, each in the order of the rows
    /// of [`PanelColumns::of`]. The lengths are checked against the panel when
    /// it is matched.
    #[must_use]
    pub fn new(keys: Vec<String>, values: Vec<Vec<String>>) -> Self {
        Self { keys, values }
    }

    /// Strata with the value of each key `value(person, period, key)` for the
    /// persons and periods of `panel`.
    #[must_use]
    pub fn of<R>(
        panel: &[TrialPerson<R>],
        keys: &[&str],
        value: impl Fn(&TrialPerson<R>, &TrialPeriod, &str) -> String,
    ) -> Self {
        let values = keys
            .iter()
            .map(|key| {
                panel
                    .iter()
                    .flat_map(|person| person.periods.iter().map(move |row| (person, row)))
                    .map(|(person, row)| value(person, row, key))
                    .collect()
            })
            .collect();
        Self {
            keys: keys.iter().map(|k| (*k).to_string()).collect(),
            values,
        }
    }

    fn check<R>(&self, panel: &[TrialPerson<R>]) -> Result<(), Error> {
        let expected: usize = panel.iter().map(|p| p.periods.len()).sum();
        if self.keys.len() != self.values.len() {
            return Err(Error::StrataRows {
                expected: self.keys.len(),
                found: self.values.len(),
            });
        }
        match self.values.iter().find(|v| v.len() != expected) {
            Some(v) => Err(Error::StrataRows {
                expected,
                found: v.len(),
            }),
            None => Ok(()),
        }
    }

    /// The value of `key` in panel row `row`, `None` for a key these strata do
    /// not have.
    fn get(&self, key: &str, row: usize) -> Option<&str> {
        let k = self.keys.iter().position(|name| name == key)?;
        Some(self.values[k][row].as_str())
    }
}

/// What [`match_trials`] adds to the options: how the dates of the
/// periods are given, and strata that change between periods.
#[derive(Debug, Clone, Copy, Default)]
pub struct TrialMatching<'a> {
    /// Whether the dates of a period are shared by every person.
    pub dates: PeriodDates,
    /// Strata of each person and period, `None` for the record's strata alone.
    pub strata: Option<&'a PeriodStrata>,
}

/// A person of a trial's pool: the person's record, with the strata of the
/// period at which the trial starts read before the record's own.
///
/// [`MatchingRecord::strata`] is the record's map; the period strata are read
/// through [`MatchingRecord::stratum`], which is what the matchers read.
///
/// Under [`PeriodDates::PerPerson`] the initiators of a trial have dates of
/// their own, and an initiator whose date is after another's would be at risk
/// as that one's control by the risk-set rule. The view of an initiator then
/// gives an entry date after every date ([`NaiveDate::MAX`]), so that it is
/// never a control in its own trial, as with shared dates, where every
/// initiator's date is the date of the trial.
#[derive(Debug)]
pub struct TrialView<'p, R> {
    record: &'p R,
    strata: Option<(&'p PeriodStrata, usize)>,
    /// An initiator under per-person dates, which enters no risk set.
    closed: bool,
}

impl<R> Clone for TrialView<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R> Copy for TrialView<'_, R> {}

impl<'p, R> TrialView<'p, R> {
    /// The person's record in the panel.
    #[must_use]
    pub const fn record(&self) -> &'p R {
        self.record
    }
}

impl<R: MatchingRecord> MatchingRecord for TrialView<'_, R> {
    fn id(&self) -> &str {
        self.record.id()
    }
    fn birth_date(&self) -> NaiveDate {
        self.record.birth_date()
    }
    fn strata(&self) -> &HashMap<String, String> {
        self.record.strata()
    }
    fn stratum(&self, key: &str) -> Option<&str> {
        match self.strata.and_then(|(strata, row)| strata.get(key, row)) {
            Some(value) => Some(value),
            None => self.record.stratum(key),
        }
    }
    fn numeric(&self, name: &str) -> Option<f64> {
        self.record.numeric(name)
    }
    fn numeric_table(&self) -> Option<usize> {
        self.record.numeric_table()
    }
    fn numeric_slot(&self, name: &str) -> Option<usize> {
        self.record.numeric_slot(name)
    }
    fn numeric_in_slot(&self, slot: usize) -> Option<f64> {
        self.record.numeric_in_slot(slot)
    }
    fn unique_key(&self) -> Option<&str> {
        self.record.unique_key()
    }
    fn death_date(&self) -> Option<NaiveDate> {
        self.record.death_date()
    }
    fn emigration_date(&self) -> Option<NaiveDate> {
        self.record.emigration_date()
    }
    fn entry_date(&self) -> Option<NaiveDate> {
        if self.closed {
            Some(NaiveDate::MAX)
        } else {
            self.record.entry_date()
        }
    }
}

impl<R: CovariateRecord> CovariateRecord for TrialView<'_, R> {
    delegate_covariate_record!(record);
}

/// A person of a trial's pool as the risk-set matching reads it: the
/// [`TrialView`] with the initiator's date as its transition date.
pub type TrialRecord<'p, R> = RoleTransitionRecord<TrialView<'p, R>>;

/// The pool of one trial, as the matching of the trial is given it.
///
/// The records are the people eligible at the start of the trial, in the
/// order of the panel, with the date of the initiator's period as the
/// transition date of each initiator and none for the rest. Each is a
/// [`TrialView`] of the record of the person in the panel, so a record is never
/// copied, however many trials the person is eligible in.
pub struct TrialPool<'p, R> {
    period: i64,
    records: Vec<TrialRecord<'p, R>>,
    /// The order of the pool's ids, read off the order of the panel's.
    id_order: Option<IdOrder>,
}

impl<'p, R: MatchingRecord> TrialPool<'p, R> {
    /// The period at which the trial starts.
    #[must_use]
    pub const fn period(&self) -> i64 {
        self.period
    }

    /// The people of the pool as a risk-set cohort.
    #[must_use]
    pub fn records(&self) -> &[TrialRecord<'p, R>] {
        &self.records
    }

    /// A risk-set job on the pool: [`MatchJob::new_transition`] on
    /// [`Self::records`] with the order of the pool's ids, which is read off
    /// the order of the panel's ids, so the job does not sort them.
    #[must_use]
    pub fn job(
        &self,
        seed: u64,
    ) -> MatchJob<TransitionMode<'_, TrialRecord<'p, R>>, TrialRecord<'p, R>, RandomSelection> {
        MatchJob::new_transition(&self.records, seed).with_id_order(self.id_order.as_ref())
    }

    /// The order of the pool's ids, when the panel fits one.
    pub(crate) const fn id_order(&self) -> Option<&IdOrder> {
        self.id_order.as_ref()
    }
}

/// Match initiators to non-initiators in every trial of the panel.
///
/// `run` matches one trial. It is given the pool of the trial (see
/// [`TrialPool`]) and returns the matching of its records, for example by
/// running [`TrialPool::job`] with the options of the study and a seed from
/// [`trial_seed`].
///
/// Under [`ControlReuse::AcrossTrials`] the trials do not depend on each
/// other, and they are matched in parallel on the rayon pool; `run` must give
/// the same matching for the same pool whatever else runs, which a seed from
/// [`trial_seed`] does. Under [`ControlReuse::OncePerRun`] a trial depends on
/// the controls drawn before it, and the trials are matched one after the
/// other in order of their start. Either way the output is in order of the
/// start and does not depend on the number of threads.
///
/// Trials without an initiator in the pool are left out of the output.
///
/// `matching` gives how the dates of the periods are given and strata that
/// change between periods (see the module documentation);
/// [`TrialMatching::default`] is shared dates and the record's strata alone.
///
/// # Errors
///
/// A [`Error`] when the panel is not valid or `run` fails. When `run`
/// fails for several trials, the error is that of the earliest of them.
pub fn match_trials<R, F>(
    panel: &[TrialPerson<R>],
    options: &TrialOptions,
    matching: &TrialMatching<'_>,
    run: F,
) -> Result<TrialOutcome, Error>
where
    R: MatchingRecord + Sync,
    F: Fn(&TrialPool<'_, R>) -> Result<MatchOutcome, Error> + Sync,
{
    validate(panel, Some(matching.dates))?;
    if let Some(strata) = matching.strata {
        strata.check(panel)?;
    }
    let plan = plan(panel, options);
    let panel_order = IdOrder::of_ids(panel.len(), |person| panel[person].record.id());
    let trials = match options.control_reuse {
        ControlReuse::AcrossTrials => {
            let starts: Vec<(&i64, &(NaiveDate, Vec<Enrolment>))> = plan.trials.iter().collect();
            let context = TrialContext::new(panel, &plan, options, matching, panel_order.as_ref());
            let results: Vec<Option<Result<TrialResult, Error>>> = starts
                .into_par_iter()
                .map(|(period, (date, enrolments))| {
                    context
                        .run(*period, *date, enrolments, None, &run)
                        .map(|result| result.map(|(trial, _)| trial))
                })
                .collect();
            results
                .into_iter()
                .flatten()
                .collect::<Result<Vec<_>, _>>()?
        }
        ControlReuse::OncePerRun => {
            in_order(panel, &plan, options, matching, panel_order.as_ref(), &run)?
        }
    };
    Ok(TrialOutcome { trials })
}

/// The trials of the plan one after the other, in order of their start. A
/// person drawn as a control leaves the pools of the later trials under
/// [`ControlReuse::OncePerRun`].
fn in_order<R, F>(
    panel: &[TrialPerson<R>],
    plan: &Plan,
    options: &TrialOptions,
    matching: &TrialMatching<'_>,
    panel_order: Option<&IdOrder>,
    run: &F,
) -> Result<Vec<TrialResult>, Error>
where
    R: MatchingRecord,
    F: Fn(&TrialPool<'_, R>) -> Result<MatchOutcome, Error>,
{
    let context = TrialContext::new(panel, plan, options, matching, panel_order);
    let once = options.control_reuse == ControlReuse::OncePerRun;
    let mut used_as_control = vec![false; panel.len()];
    let mut trials = Vec::new();
    for (period, (date, enrolments)) in &plan.trials {
        let Some(result) = context.run(
            *period,
            *date,
            enrolments,
            once.then_some(used_as_control.as_slice()),
            run,
        ) else {
            continue;
        };
        let (trial, controls) = result?;
        for person in controls {
            used_as_control[person] = true;
        }
        trials.push(trial);
    }
    Ok(trials)
}

/// A trial and the panel positions of the people drawn as its controls.
type RunTrial = (TrialResult, Vec<usize>);

/// What every trial of a run reads.
struct TrialContext<'a, R> {
    panel: &'a [TrialPerson<R>],
    plan: &'a Plan,
    options: &'a TrialOptions,
    matching: TrialMatching<'a>,
    /// The first row of each person in [`PanelColumns`], for the strata.
    offsets: Vec<usize>,
    panel_order: Option<&'a IdOrder>,
    /// The panel position of each id, built when a trial first has pairs.
    person_of: OnceLock<RapidHashMap<&'a str, usize>>,
}

impl<'a, R: MatchingRecord> TrialContext<'a, R> {
    fn new(
        panel: &'a [TrialPerson<R>],
        plan: &'a Plan,
        options: &'a TrialOptions,
        matching: &TrialMatching<'a>,
        panel_order: Option<&'a IdOrder>,
    ) -> Self {
        Self {
            panel,
            plan,
            options,
            matching: *matching,
            offsets: offsets(panel),
            panel_order,
            person_of: OnceLock::new(),
        }
    }

    /// The panel position of the person with this id.
    fn person_of(&self, id: &str) -> Option<usize> {
        self.person_of
            .get_or_init(|| {
                self.panel
                    .iter()
                    .enumerate()
                    .map(|(person, row)| (row.record.id(), person))
                    .collect()
            })
            .get(id)
            .copied()
    }

    /// The trial that starts at `period`, and the people drawn as its
    /// controls, or `None` when its pool has no initiator. `used_as_control`
    /// marks the people drawn in earlier trials when they leave the pool.
    fn run<F>(
        &self,
        period: i64,
        date: NaiveDate,
        enrolments: &[Enrolment],
        used_as_control: Option<&[bool]>,
        run: &F,
    ) -> Option<Result<RunTrial, Error>>
    where
        F: Fn(&TrialPool<'_, R>) -> Result<MatchOutcome, Error>,
    {
        let (panel, options) = (self.panel, self.options);
        let pool: Vec<&Enrolment> = enrolments
            .iter()
            .filter(|e| {
                e.assigned
                    || !(used_as_control.is_some_and(|used| used[e.person])
                        || (!options.initiators_as_earlier_controls
                            && self.plan.last_initiation[e.person]
                                .is_some_and(|last| last > period)))
            })
            .collect();
        let n_initiators = pool.iter().filter(|e| e.assigned).count();
        if n_initiators == 0 {
            return None;
        }
        // A panel with an order has at most `u32::MAX` people.
        let members: Vec<u32> = pool
            .iter()
            .filter_map(|e| u32::try_from(e.person).ok())
            .collect();
        let trial_pool = TrialPool {
            period,
            records: pool
                .iter()
                .map(|e| {
                    let view = TrialView {
                        record: &panel[e.person].record,
                        strata: self
                            .matching
                            .strata
                            .map(|strata| (strata, self.offsets[e.person] + e.start)),
                        closed: e.assigned && self.matching.dates == PeriodDates::PerPerson,
                    };
                    // The date of the initiator's own period, which is the
                    // date of the trial when dates are shared.
                    let own = panel[e.person].periods[e.start].date;
                    RoleTransitionRecord::from_record(view, e.assigned.then_some(own))
                })
                .collect(),
            id_order: self.panel_order.map(|order| order.subset(&members)),
        };
        let outcome = match run(&trial_pool) {
            Ok(outcome) => outcome,
            Err(error) => {
                return Some(Err(Error::Trial {
                    period,
                    source: Box::new(error),
                }));
            }
        };
        // The position in the pool of each person of the panel.
        let mut pool_at = vec![usize::MAX; panel.len()];
        for (at, e) in pool.iter().enumerate() {
            pool_at[e.person] = at;
        }
        let mut listed = vec![false; pool.len()];
        let mut members = Vec::new();
        let mut controls = Vec::new();
        let mut add = |id: &str, members: &mut Vec<TrialMember>| {
            let at = pool_at[self.person_of(id)?];
            if at == usize::MAX {
                return None;
            }
            if std::mem::replace(&mut listed[at], true) {
                return Some(at);
            }
            let e = pool[at];
            let rows = &panel[e.person].periods;
            members.push(TrialMember {
                id: id.to_string(),
                assigned: e.assigned,
                follow_up_start_period: period,
                follow_up_start: rows[e.start].date,
                deviation_period: e.deviation.map(|d| rows[d].period),
                deviation_date: e.deviation.map(|d| rows[d].date),
                last_period: rows[e.last].period,
            });
            Some(at)
        };
        for pair in &outcome.pairs {
            add(pair.anchor_id(), &mut members);
            if let Some(at) = add(pair.comparator_id(), &mut members) {
                controls.push(pool[at].person);
            }
        }
        // Under per-person dates the trial has no one date; it reports the
        // earliest of its initiators'.
        let index_date = match self.matching.dates {
            PeriodDates::Shared => date,
            PeriodDates::PerPerson => pool
                .iter()
                .filter(|e| e.assigned)
                .map(|e| panel[e.person].periods[e.start].date)
                .min()
                .unwrap_or(date),
        };
        Some(Ok((
            TrialResult {
                period,
                index_date,
                n_eligible: pool.len(),
                n_initiators,
                outcome,
                members,
            },
            controls,
        )))
    }
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
            Err(Error::DuplicateId { .. })
        ));
        let c = person(
            "c",
            &[(1, true, Some(false), false), (1, true, Some(false), false)],
        );
        assert!(matches!(
            expand_trials(&[c], &TrialOptions::default()),
            Err(Error::PeriodsNotIncreasing { .. })
        ));
        let mut d = person("d", &[(0, true, Some(false), false)]);
        d.periods[0].date = day(9);
        // The expansion reads no date; matching needs one date per period.
        assert!(expand_trials(&[a.clone(), d.clone()], &TrialOptions::default()).is_ok());
        assert!(matches!(
            match_trials(
                &[a, d],
                &TrialOptions::default(),
                &TrialMatching::default(),
                |pool| { pool.job(1).run() }
            ),
            Err(Error::InconsistentDate { period: 0 })
        ));
    }

    #[test]
    fn trial_seeds_differ_by_period_and_repeat() {
        assert_eq!(trial_seed(5, 3), trial_seed(5, 3));
        assert_ne!(trial_seed(5, 3), trial_seed(5, 4));
        assert_ne!(trial_seed(5, 3), trial_seed(6, 3));
    }

    /// A panel of people in strata, with births spread over years, eight
    /// periods, initiation at a person's own period, deaths and outcomes,
    /// and ids out of panel order.
    fn panel(seed: u64, n: usize) -> Vec<TrialPerson<BaseRecord>> {
        let mut state = seed;
        let mut next = |bound: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % bound
        };
        (0..n)
            .map(|i| {
                let birth =
                    NaiveDate::from_ymd_opt(2012, 1, 1).expect("valid") + Days::new(next(1500));
                let mut record = BaseRecord::new(format!("{:x}-{i}", next(1 << 20)), birth);
                record
                    .strata
                    .insert("sex".into(), if next(2) == 0 { "F" } else { "M" }.into());
                if next(10) == 0 {
                    record.death_date = Some(day(next(8)));
                }
                let start = i64::try_from(next(12)).expect("small");
                let event = (next(6) == 0).then(|| i64::try_from(next(8)).expect("small"));
                TrialPerson {
                    record,
                    periods: (0..8)
                        .map(|period| TrialPeriod {
                            period,
                            date: day(u64::try_from(period).expect("small")),
                            eligible: period <= start,
                            treatment: Some(period >= start),
                            event: event == Some(period),
                        })
                        .collect(),
                }
            })
            .collect()
    }

    fn job_on(pool: &TrialPool<'_, BaseRecord>, ordered: bool, ratio: usize) -> MatchOutcome {
        let seed = trial_seed(3, pool.period());
        let job = if ordered {
            pool.job(seed)
        } else {
            MatchJob::new_transition(pool.records(), seed)
        };
        job.with_ratio(crate::types::MatchRatio::new(ratio).expect("ratio"))
            .with_birth_window(crate::types::BirthDateWindowDays::new(60).expect("window"))
            .with_age_limit(crate::types::AgeLimitYears::new(18).expect("limit"))
            .with_exact_matches(["sex"])
            .with_alive_check()
            .run()
            .expect("a valid job")
    }

    /// Trials matched in parallel under the default reuse rule equal the
    /// trials matched one after the other, with or without the panel's id
    /// order, on 1, 2, 4 and 12 threads.
    #[test]
    fn parallel_trials_equal_trials_in_order() {
        for (seed, n, earlier) in [(1, 40, true), (2, 400, true), (3, 3000, false)] {
            let people = panel(seed, n);
            let options = TrialOptions {
                initiators_as_earlier_controls: earlier,
                ..TrialOptions::default()
            };
            let plan = plan(&people, &options);
            let order = IdOrder::of_ids(people.len(), |p| people[p].record.id());
            for ratio in [1, 3] {
                let reference = in_order(
                    &people,
                    &plan,
                    &options,
                    &TrialMatching::default(),
                    None,
                    &|pool| Ok(job_on(pool, false, ratio)),
                )
                .expect("valid");
                let ordered = in_order(
                    &people,
                    &plan,
                    &options,
                    &TrialMatching::default(),
                    order.as_ref(),
                    &|pool| Ok(job_on(pool, true, ratio)),
                )
                .expect("valid");
                let pairs: usize = reference.iter().map(|t| t.outcome.pairs.len()).sum();
                assert!(pairs > n / 20, "seed {seed}: {pairs} pairs");
                let reference = format!("{reference:?}");
                assert_eq!(format!("{ordered:?}"), reference, "seed {seed}");
                let parallel = || {
                    let outcome =
                        match_trials(&people, &options, &TrialMatching::default(), |pool| {
                            Ok(job_on(pool, true, ratio))
                        })
                        .expect("valid");
                    format!("{:?}", outcome.trials)
                };
                assert_eq!(parallel(), reference, "seed {seed}");
                epistat_testkit::assert_same_on_pools(&[1, 2, 4, 12], || {
                    parallel().bytes().map(f64::from).collect()
                });
            }
        }
    }

    /// The rows of the matched members are the rows that the expansion gives
    /// each of them in their trial, by intention to treat and per protocol.
    #[test]
    fn matched_rows_are_the_expanded_rows_of_the_members() {
        let people = panel(7, 600);
        let options = TrialOptions::default();
        let matched = match_trials(&people, &options, &TrialMatching::default(), |pool| {
            Ok(job_on(pool, true, 2))
        })
        .expect("valid");
        let members: usize = matched.trials.iter().map(|t| t.members.len()).sum();
        assert!(members > 100, "{members} members");
        for censor_at_deviation in [true, false] {
            let options = TrialOptions {
                censor_at_deviation,
                ..options
            };
            let all = expand_trial_columns(&people, &options).expect("valid");
            let (rows, trial) = matched_trial_columns(&people, &matched, &options).expect("valid");
            assert_eq!(rows.len(), trial.len());
            let key = |t: &TrialColumns, i: usize| (t.person[i], t.trial_period[i], t.row[i]);
            let expanded: HashSet<(u64, i64, usize)> =
                (0..all.len()).map(|i| key(&all, i)).collect();
            for i in 0..rows.len() {
                assert!(expanded.contains(&key(&rows, i)));
                assert_eq!(rows.trial_period[i], matched.trials[trial[i]].period);
            }
            // Every member has all its rows: the first row is the start of the
            // trial and the count is that of the expansion.
            for (k, t) in matched.trials.iter().enumerate() {
                for m in &t.members {
                    let person = people
                        .iter()
                        .position(|p| p.record.id() == m.id)
                        .and_then(|p| u64::try_from(p).ok())
                        .expect("in the panel");
                    let mine = (0..rows.len())
                        .filter(|&i| trial[i] == k && rows.person[i] == person)
                        .count();
                    let theirs = (0..all.len())
                        .filter(|&i| all.person[i] == person && all.trial_period[i] == t.period)
                        .count();
                    assert_eq!(mine, theirs);
                }
            }
        }
    }

    /// The error of a failing run is that of the earliest failing trial.
    #[test]
    fn the_earliest_failing_trial_is_reported() {
        let people = panel(4, 500);
        let failed = match_trials(
            &people,
            &TrialOptions::default(),
            &TrialMatching::default(),
            |pool| {
                if pool.period() >= 2 {
                    Err(Error::option("test", pool.period().to_string()))
                } else {
                    Ok(job_on(pool, true, 1))
                }
            },
        );
        let Err(Error::Trial { period, source }) = failed else {
            panic!("a trial fails");
        };
        assert_eq!((period, *source), (2, Error::option("test", "2")));
    }
}

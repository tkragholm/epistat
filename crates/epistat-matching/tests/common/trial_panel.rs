//! A trial panel written person by person, as the R references give it, and
//! turned into the columns `TrialPanel` takes.

#![allow(dead_code, reason = "each test file uses part of the helpers")]

use chrono::NaiveDate;
use epistat_core::Panel;
use epistat_matching::{
    MatchingRecord, TrialColumns, TrialOptions, TrialPanel, expand_trial_columns,
};

/// One period of one person.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PanelPeriod {
    pub period: i64,
    pub date: NaiveDate,
    pub eligible: bool,
    pub treatment: Option<bool>,
    pub event: bool,
}

/// A person with the periods in which the person is observed.
#[derive(Debug, Clone)]
pub struct PanelPerson<R> {
    pub record: R,
    pub periods: Vec<PanelPeriod>,
}

/// The panel in columns, one row per person and period in the order of the
/// people and then of the periods.
#[derive(Debug, Clone)]
pub struct PanelRows<R> {
    pub records: Vec<R>,
    pub unit: Vec<u32>,
    /// The unit as the person number the survival weights take.
    pub person: Vec<u64>,
    pub period: Vec<i64>,
    pub date: Vec<NaiveDate>,
    pub eligible: Vec<bool>,
    pub treatment: Vec<Option<bool>>,
    pub event: Vec<bool>,
}

impl<R: MatchingRecord> PanelRows<R> {
    pub fn of(people: &[PanelPerson<R>]) -> Self
    where
        R: Clone,
    {
        let mut rows = Self {
            records: Vec::with_capacity(people.len()),
            unit: Vec::new(),
            person: Vec::new(),
            period: Vec::new(),
            date: Vec::new(),
            eligible: Vec::new(),
            treatment: Vec::new(),
            event: Vec::new(),
        };
        for (unit, person) in (0_u32..).zip(people) {
            rows.records.push(person.record.clone());
            for p in &person.periods {
                rows.unit.push(unit);
                rows.person.push(u64::from(unit));
                rows.period.push(p.period);
                rows.date.push(p.date);
                rows.eligible.push(p.eligible);
                rows.treatment.push(p.treatment);
                rows.event.push(p.event);
            }
        }
        rows
    }

    /// The checked panel, with dates.
    pub fn try_panel(&self) -> Result<TrialPanel<'_, R>, epistat_matching::Error> {
        let key = Panel::new(&self.unit, &self.period)?;
        TrialPanel::builder(
            &self.records,
            key,
            &self.eligible,
            &self.treatment,
            &self.event,
        )
        .dates(&self.date)
        .build()
    }

    pub fn panel(&self) -> TrialPanel<'_, R> {
        self.try_panel().expect("a valid panel")
    }

    /// The number of rows.
    pub fn len(&self) -> usize {
        self.unit.len()
    }

    /// The treatment of every row, `None` when a row's treatment is not known.
    pub fn known_treatment(&self) -> Option<Vec<bool>> {
        self.treatment.iter().copied().collect()
    }
}

/// One row of the expanded trials: a person in a trial at one period of the
/// follow-up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpandedRow {
    pub trial_period: i64,
    pub id: String,
    pub follow_up_time: i64,
    pub assigned: bool,
    pub treatment: Option<bool>,
    pub outcome: bool,
}

/// The expanded trials of `people` as rows.
pub fn expand_trials<R: MatchingRecord + Clone>(
    people: &[PanelPerson<R>],
    options: &TrialOptions,
) -> Result<Vec<ExpandedRow>, epistat_matching::Error> {
    let rows = PanelRows::of(people);
    let columns = expand_trial_columns(&rows.try_panel()?, options)?;
    Ok((0..columns.len())
        .map(|i| ExpandedRow {
            trial_period: columns.trial_period[i],
            id: people[columns.unit[i] as usize].record.id().to_string(),
            follow_up_time: columns.follow_up_time[i],
            assigned: columns.assigned[i],
            treatment: columns.treatment[i],
            outcome: columns.outcome[i],
        })
        .collect())
}

/// The person of every row of the trials as the survival weights number them.
pub fn persons(t: &TrialColumns) -> Vec<u64> {
    t.unit.iter().map(|&u| u64::from(u)).collect()
}

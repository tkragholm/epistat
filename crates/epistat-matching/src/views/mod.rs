//! The input views of the two matching tasks: [`TreatmentData`] for
//! treatment-group matching and [`RiskSetCohort`] for risk-set matching.
//!
//! A view takes the caller's columns ([`Col`]) by name, each once, and checks
//! them when it is built. Ids and categories are any [`Label`]; the ids are
//! coded once with `epistat_core::Codes`, so the rows are ordered by id as a
//! sort of the labels orders them. Dates name their epoch ([`Col::unix_days`],
//! [`Col::sas_days`], [`Col::dates`]). Inside, a view holds the engine's
//! columnar [`Cohort`] and the order of its ids, which the matchers read.
//!
//! What a column is used for, an exact key, a caliper, a covariate, comes
//! from the options of a match and not from the view, so a column used as an
//! exact key and as a balance covariate is entered once.
//! [`TreatmentData::column_kind`] and [`TreatmentData::check_exact_key`], and
//! the same on [`RiskSetCohort`], are the checks of a named column against
//! the data, which refuse a name the data do not have with the names they
//! do have.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the engine side of a view, its cohort and id order, is read by its tests alone \
                  until a matcher runs on views"
    )
)]

mod col;
mod risk_set;
mod treatment;

pub use col::{Col, Label, LabelColumn, UnixDays, UnixDaysNullable};
pub use risk_set::{Residence, RiskSetCohort, RiskSetCohortBuilder, SpellEnd};
pub use treatment::{TreatmentData, TreatmentDataBuilder};

use std::collections::HashSet;
use std::fmt;

use chrono::NaiveDate;
use epistat_core::{Codes, Day};
use serde::{Deserialize, Serialize};

use crate::Error;
use crate::cohort::{Cohort, CohortBuilder};
use crate::matching::MatchingRecord;
use crate::role_transition::IdOrder;
use col::OutOfRange;

/// What a column of a view holds.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColumnKind {
    /// Numbers: a covariate, a caliper field, a score or a weight.
    Numeric,
    /// Labels: a covariate as indicators, an exact or antiexact key.
    Categorical,
    /// Dates, for a constraint within days.
    Date,
}

impl ColumnKind {
    /// The kind as a lower-case word.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Numeric => "numeric",
            Self::Categorical => "categorical",
            Self::Date => "date",
        }
    }
}

impl fmt::Display for ColumnKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How an exact key treats a row without a value.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExactMissing {
    /// A null is not a value: a key with a null is refused before matching.
    #[default]
    Refuse,
    /// A null is a level of its own, so a null matches a null
    /// (`missing_as_level()`).
    OwnLevel,
}

/// A column of a view: its name, its kind and its first null.
#[derive(Clone, Debug)]
struct Column {
    name: String,
    kind: ColumnKind,
    first_null: Option<usize>,
}

/// What both views hold: the ids, the named columns and the engine's cohort.
#[derive(Debug)]
struct Columns {
    named: Vec<Column>,
    cohort: Cohort,
    order: IdOrder,
}

impl Columns {
    fn column(&self, name: &str) -> Result<&Column, Error> {
        self.named
            .iter()
            .find(|column| column.name == name)
            .ok_or_else(|| Error::UnknownColumn {
                name: name.to_string(),
                known: self.named.iter().map(|c| c.name.clone()).collect(),
            })
    }

    fn check_exact_key(&self, name: &str, missing: ExactMissing) -> Result<(), Error> {
        let column = self.column(name)?;
        if column.kind != ColumnKind::Categorical {
            return Err(Error::WrongColumnKind {
                name: name.to_string(),
                expected: ColumnKind::Categorical,
                found: column.kind,
            });
        }
        match (missing, column.first_null) {
            (ExactMissing::Refuse, Some(row)) => Err(Error::NullKey {
                column: name.to_string(),
                row,
                id: self.id(row),
            }),
            _ => Ok(()),
        }
    }

    fn id(&self, row: usize) -> String {
        self.cohort
            .row(row)
            .map_or_else(String::new, |record| record.id().to_string())
    }
}

/// The columns of a view as they are added, read and checked one at a time.
/// The first error is kept and returned by `build`.
struct Pending {
    ids: Vec<String>,
    ranks: Vec<u32>,
    columns: Vec<Column>,
    numeric: Vec<(String, Vec<Option<f64>>)>,
    categorical: Vec<(String, Vec<Option<String>>)>,
    dates: Vec<(String, Vec<Option<Day>>)>,
    error: Option<Error>,
}

impl Pending {
    /// The ids, coded once in the sorted order of the labels.
    fn new<L: Label>(ids: &[L]) -> Self {
        let mut pending = Self {
            ids: Vec::new(),
            ranks: Vec::new(),
            columns: Vec::new(),
            numeric: Vec::new(),
            categorical: Vec::new(),
            dates: Vec::new(),
            error: None,
        };
        match code_ids(ids) {
            Ok((text, ranks)) => {
                pending.ids = text;
                pending.ranks = ranks;
            }
            Err(error) => pending.error = Some(error),
        }
        pending
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    fn fail(&mut self, error: Error) {
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    /// Records a column by name, refusing a name already taken and a column
    /// of another length than the ids.
    fn name(&mut self, name: &str, kind: ColumnKind, len: usize, first_null: Option<usize>) {
        if self.error.is_some() {
            return;
        }
        if self.columns.iter().any(|column| column.name == name) {
            self.fail(Error::DuplicateColumn {
                column: name.to_string(),
            });
        } else if len != self.len() {
            self.fail(length(name, len, self.len()));
        } else {
            self.columns.push(Column {
                name: name.to_string(),
                kind,
                first_null,
            });
        }
    }

    fn numeric(&mut self, name: &str, values: &Col<'_, f64>) {
        if self.error.is_some() {
            return;
        }
        if let Err(error) = check_mask(name, values) {
            return self.fail(error);
        }
        let n = values.len().min(self.len());
        let read: Vec<Option<f64>> = (0..n).map(|row| values.get(row)).collect();
        if let Some(row) = read.iter().position(|v| v.is_some_and(|v| !v.is_finite())) {
            return self.fail(Error::NonFiniteValue {
                column: name.to_string(),
                row,
                id: self.ids[row].clone(),
            });
        }
        let first_null = read.iter().position(Option::is_none);
        self.name(name, ColumnKind::Numeric, values.len(), first_null);
        self.numeric.push((name.to_string(), read));
    }

    fn categorical<L: Label>(&mut self, name: &str, values: &Col<'_, L>) {
        if self.error.is_some() {
            return;
        }
        if let Err(error) = check_mask(name, values) {
            return self.fail(error);
        }
        let n = values.len().min(self.len());
        let read: Vec<Option<L>> = (0..n).map(|row| values.get(row)).collect();
        let present: Vec<L> = read.iter().flatten().cloned().collect();
        if let Err(error) = distinct_text(name, &present) {
            return self.fail(error);
        }
        let first_null = read.iter().position(Option::is_none);
        self.name(name, ColumnKind::Categorical, values.len(), first_null);
        let text = read
            .into_iter()
            .map(|value| value.map(|label| label.to_string()))
            .collect();
        self.categorical.push((name.to_string(), text));
    }

    fn date(&mut self, name: &str, values: &Col<'_, Day>) {
        if self.error.is_some() {
            return;
        }
        match read_days(name, values, self.len()) {
            Ok(days) => {
                let first_null = days.iter().position(Option::is_none);
                self.name(name, ColumnKind::Date, values.len(), first_null);
                self.dates.push((name.to_string(), days));
            }
            Err(error) => self.fail(error),
        }
    }

    /// A date column of the view's own, read but not named: birth, death,
    /// the index date and the like.
    fn own_dates(&mut self, name: &str, values: &Col<'_, Day>) -> Vec<Option<Day>> {
        if self.error.is_some() {
            return Vec::new();
        }
        match read_days(name, values, self.len()) {
            Ok(days) if days.len() == self.len() => days,
            Ok(days) => {
                self.fail(length(name, days.len(), self.len()));
                Vec::new()
            }
            Err(error) => {
                self.fail(error);
                Vec::new()
            }
        }
    }

    /// The columns into the engine's cohort, with the given birth dates.
    fn build(
        self,
        birth: Vec<NaiveDate>,
        finish: impl FnOnce(CohortBuilder) -> CohortBuilder,
    ) -> Result<Columns, Error> {
        if let Some(error) = self.error {
            return Err(error);
        }
        let mut builder = Cohort::builder(&self.ids, birth);
        for (name, values) in &self.numeric {
            builder = builder.numeric_field_and_covariate(name.clone(), values.iter().copied());
        }
        for (name, values) in &self.categorical {
            builder = builder
                .stratum(name.clone(), values.iter().map(Option::as_deref))
                .categorical_covariate(name.clone(), values.iter().map(Option::as_deref));
        }
        for (name, values) in &self.dates {
            #[expect(
                clippy::cast_precision_loss,
                reason = "a Day is within 2^40 days of the epoch, exact in an f64"
            )]
            let days = values
                .iter()
                .map(|day| day.map(|day| day.unix_days() as f64));
            builder = builder.numeric(name.clone(), days);
        }
        let cohort = finish(builder).build()?;
        let order = IdOrder::from_ranks(&self.ranks);
        Ok(Columns {
            named: self.columns,
            cohort,
            order,
        })
    }
}

/// The text of each id and its rank in the sorted order of the labels.
///
/// # Errors
///
/// [`InputError::Empty`] for no ids, [`Error::TooManyUnits`] for more than
/// `u32::MAX`, [`Error::DuplicateIdRows`] for an id on two rows and
/// [`Error::AmbiguousLabel`] for two ids that print alike.
fn code_ids<L: Label>(ids: &[L]) -> Result<(Vec<String>, Vec<u32>), Error> {
    epistat_core::check::non_empty("ids", ids.len())?;
    if u32::try_from(ids.len()).is_err() {
        return Err(Error::TooManyUnits);
    }
    let codes = Codes::from_labels(ids)?;
    if codes.n_codes() < ids.len() {
        let mut first = vec![usize::MAX; codes.n_codes()];
        for (row, &code) in codes.codes().iter().enumerate() {
            let seen = &mut first[code as usize];
            if *seen != usize::MAX {
                return Err(Error::DuplicateIdRows {
                    id: ids[row].to_string(),
                    first: *seen,
                    row,
                });
            }
            *seen = row;
        }
    }
    distinct_text("ids", codes.labels())?;
    let text = ids.iter().map(ToString::to_string).collect();
    Ok((text, codes.into_parts().0))
}

/// Refuses two distinct labels that print alike, which the engine, reading
/// labels as text, would take for one.
fn distinct_text<L: Label>(column: &str, labels: &[L]) -> Result<(), Error> {
    let mut sorted = labels.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut seen = HashSet::with_capacity(sorted.len());
    for label in &sorted {
        let text = label.to_string();
        if !seen.insert(text.clone()) {
            return Err(Error::AmbiguousLabel {
                column: column.to_string(),
                text,
            });
        }
    }
    Ok(())
}

fn length(column: &str, found: usize, expected: usize) -> Error {
    Error::ColumnLength {
        column: column.to_string(),
        found,
        expected,
    }
}

/// Refuses a validity mask of another length than its values.
fn check_mask<T>(name: &str, values: &Col<'_, T>) -> Result<(), Error> {
    match values.mask_mismatch() {
        Some(found) => Err(length(&format!("{name} validity"), found, values.len())),
        None => Ok(()),
    }
}

/// The days of a date column, which must have `n` rows.
fn read_days(name: &str, values: &Col<'_, Day>, n: usize) -> Result<Vec<Option<Day>>, Error> {
    check_mask(name, values)?;
    if values.len() != n {
        return Err(length(name, values.len(), n));
    }
    (0..n)
        .map(|row| {
            values.day(row).map_err(|OutOfRange| Error::DateOutOfRange {
                column: name.to_string(),
                row,
            })
        })
        .collect()
}

/// The chrono date of a day, which the engine holds.
fn naive(column: &str, row: usize, day: Day) -> Result<NaiveDate, Error> {
    i32::try_from(day.unix_days())
        .ok()
        .and_then(|days| days.checked_add(crate::UNIX_EPOCH_DAYS_FROM_CE))
        .and_then(NaiveDate::from_num_days_from_ce_opt)
        .ok_or_else(|| Error::DateOutOfRange {
            column: column.to_string(),
            row,
        })
}

/// The chrono dates of a column of days.
fn naive_column(column: &str, days: &[Option<Day>]) -> Result<Vec<Option<NaiveDate>>, Error> {
    days.iter()
        .enumerate()
        .map(|(row, day)| day.map(|day| naive(column, row, day)).transpose())
        .collect()
}

#[cfg(test)]
mod tests;

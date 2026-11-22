//! A cohort held in columns, whose rows are records; see [`Cohort`].

use std::collections::HashMap;
use std::fmt;
use std::sync::OnceLock;

use chrono::NaiveDate;
use rustc_hash::FxHashMap;

use crate::matching::{CovariateRecord, MatchingRecord};
use crate::types::{CovariateRef, CovariateValue, RoleTransitionRecord};

/// Code of a row without a value in a dictionary column.
const NONE: u32 = u32::MAX;

/// Up to this many names are searched in order; more are looked up by hash.
const LINEAR_NAMES: usize = 8;

/// Why a cohort could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CohortError {
    /// A column has another length than the id column.
    LengthMismatch {
        /// The column.
        column: String,
        /// Its length.
        found: usize,
        /// The number of ids.
        expected: usize,
    },
    /// Two columns of the same kind have the same name: two strata, two
    /// numeric fields or two covariates.
    DuplicateName {
        /// The name.
        column: String,
    },
    /// A column has more distinct values than a dictionary code can number.
    TooManyValues {
        /// The column.
        column: String,
    },
}

impl fmt::Display for CohortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                column,
                found,
                expected,
            } => write!(
                f,
                "column {column} has {found} rows, but there are {expected} ids"
            ),
            Self::DuplicateName { column } => {
                write!(f, "two columns of the same kind are called {column}")
            }
            Self::TooManyValues { column } => write!(
                f,
                "column {column} has more than {} distinct values",
                NONE - 1
            ),
        }
    }
}

impl std::error::Error for CohortError {}

/// Names of the columns of one kind, with the position of each.
#[derive(Debug, Clone, Default)]
struct Names {
    names: Vec<String>,
    by_name: FxHashMap<String, usize>,
}

impl Names {
    fn push(&mut self, name: String) -> Result<(), CohortError> {
        if self.by_name.contains_key(&name) {
            return Err(CohortError::DuplicateName { column: name });
        }
        self.by_name.insert(name.clone(), self.names.len());
        self.names.push(name);
        Ok(())
    }

    fn find(&self, name: &str) -> Option<usize> {
        if self.names.len() <= LINEAR_NAMES {
            self.names.iter().position(|known| known == name)
        } else {
            self.by_name.get(name).copied()
        }
    }
}

/// A text column: the distinct values in order of first appearance and a code
/// per row, [`NONE`] for a row without a value.
#[derive(Debug, Clone, Default)]
struct Dictionary {
    values: Vec<Box<str>>,
    codes: Vec<u32>,
}

impl Dictionary {
    fn from_values<I, S>(name: &str, values: I) -> Result<Self, CohortError>
    where
        I: IntoIterator<Item = Option<S>>,
        S: AsRef<str>,
    {
        let values = values.into_iter();
        let mut codes = Vec::with_capacity(values.size_hint().0);
        let mut seen: FxHashMap<Box<str>, u32> = FxHashMap::default();
        let mut distinct = Vec::new();
        for value in values {
            let code = match value {
                None => NONE,
                Some(value) => {
                    let value = value.as_ref();
                    if let Some(&code) = seen.get(value) {
                        code
                    } else {
                        let code = u32::try_from(distinct.len())
                            .ok()
                            .filter(|&code| code < NONE)
                            .ok_or_else(|| CohortError::TooManyValues {
                                column: name.to_string(),
                            })?;
                        let value: Box<str> = value.into();
                        seen.insert(value.clone(), code);
                        distinct.push(value);
                        code
                    }
                }
            };
            codes.push(code);
        }
        distinct.shrink_to_fit();
        Ok(Self {
            values: distinct,
            codes,
        })
    }

    fn get(&self, row: usize) -> Option<&str> {
        match self.codes[row] {
            NONE => None,
            code => Some(&self.values[code as usize]),
        }
    }

    fn len(&self) -> usize {
        self.codes.len()
    }
}

/// A numeric column: a value per row and, when some row has none, a bit per
/// row that is set where a value is present.
#[derive(Debug, Clone, Default)]
struct Numbers {
    values: Vec<f64>,
    present: Option<Vec<u64>>,
}

impl Numbers {
    fn from_values(values: impl IntoIterator<Item = Option<f64>>) -> Self {
        let values = values.into_iter();
        let mut out = Vec::with_capacity(values.size_hint().0);
        let mut present: Vec<u64> = Vec::new();
        let mut any_missing = false;
        for (row, value) in values.enumerate() {
            if row % 64 == 0 {
                present.push(0);
            }
            if let Some(value) = value {
                out.push(value);
                present[row / 64] |= 1 << (row % 64);
            } else {
                out.push(f64::NAN);
                any_missing = true;
            }
        }
        out.shrink_to_fit();
        Self {
            values: out,
            present: any_missing.then(|| {
                present.shrink_to_fit();
                present
            }),
        }
    }

    fn get(&self, row: usize) -> Option<f64> {
        match &self.present {
            Some(bits) if bits[row / 64] & (1 << (row % 64)) == 0 => None,
            _ => Some(self.values[row]),
        }
    }

    fn len(&self) -> usize {
        self.values.len()
    }
}

/// Where a covariate's values are.
#[derive(Debug, Clone)]
enum Covariate {
    /// A numeric column, shared with a numeric field when both have the name.
    Numeric(usize),
    /// A categorical column.
    Categorical(Dictionary),
}

/// A cohort held in columns, whose rows are records.
///
/// The owned records, [`BaseRecord`](crate::BaseRecord) and
/// [`BalanceRecord`](crate::BalanceRecord), keep their strata, numeric fields
/// and covariates in hash maps keyed by name and their id in a string of its
/// own, which costs about 1.9 KiB per unit with five strata and five
/// covariates. A [`Cohort`] keeps one column per field instead:
///
/// - the ids end to end in one string, with the end of each;
/// - dates as [`NaiveDate`] values, four bytes each;
/// - strata, the unique key and categorical covariates as a dictionary per
///   column, the distinct values once and a four-byte code per row;
/// - numeric fields and numeric covariates as `f64` columns, with a bit per row
///   for a missing value only when the column has one.
///
/// The same five strata and five covariates then take about 90 bytes per unit,
/// and 10^7 units in about 1 GB, with 16 bytes more per row handle.
///
/// A [`CohortRow`] is a reference to the cohort and a row number, 16 bytes,
/// and is `Copy`. It implements [`MatchingRecord`] and [`CovariateRecord`], so
/// a slice of rows goes wherever a slice of owned records goes, and
/// [`Cohort::risk_set_rows`] gives the records of risk-set matching. Every
/// matcher reads a row one value at a time, through
/// [`MatchingRecord::stratum`], [`MatchingRecord::numeric`] and
/// [`CovariateRecord::covariate`], and gives the same result as on owned
/// records with the same values.
///
/// [`MatchingRecord::strata`] and [`CovariateRecord::covariates`] return a
/// whole map. A row has none, so the first call of either builds the maps of
/// every row of the cohort once and keeps them, which costs what owned records
/// cost. No matcher in this crate calls them; [`Cohort::has_built_record_maps`]
/// tells whether something else has.
///
/// # Values
///
/// A row is the record that the same values make through the owned records'
/// constructors and setters, or through
/// [`ArrowRecords::balance_records`](crate::arrow::ArrowRecords::balance_records):
///
/// - a stratum without a value is a stratum the record does not have;
/// - a numeric field without a value is a field the record does not have, and a
///   `NaN` is kept as it is;
/// - a numeric covariate without a value or with a value that is not finite is
///   [`CovariateRef::Missing`], as is a categorical covariate without a value;
///   every row carries every covariate of the cohort.
///
/// ```
/// use chrono::NaiveDate;
/// use epistat_matching::{Cohort, CovariateRecord, CovariateRef, MatchingRecord};
///
/// let day = |d| NaiveDate::from_ymd_opt(2010, 1, d).unwrap();
/// let cohort = Cohort::builder(["a", "b"], vec![day(1), day(2)])
///     .stratum("sex", [Some("F"), None])
///     .numeric_field_and_covariate("x", [Some(1.5), Some(f64::NAN)])
///     .build()
///     .unwrap();
/// let rows: Vec<_> = cohort.rows().collect();
/// assert_eq!(rows[0].stratum("sex"), Some("F"));
/// assert_eq!(rows[1].stratum("sex"), None);
/// assert!(rows[1].numeric("x").unwrap().is_nan());
/// assert_eq!(rows[1].covariate("x"), Some(CovariateRef::Missing));
/// ```
pub struct Cohort {
    len: usize,
    id_text: String,
    id_ends: Vec<usize>,
    birth: Vec<NaiveDate>,
    death: Option<Vec<Option<NaiveDate>>>,
    emigration: Option<Vec<Option<NaiveDate>>>,
    entry: Option<Vec<Option<NaiveDate>>>,
    event: Option<Vec<Option<NaiveDate>>>,
    unique_key: Option<Dictionary>,
    strata_names: Names,
    strata: Vec<Dictionary>,
    numbers: Vec<Numbers>,
    field_names: Names,
    fields: Vec<usize>,
    covariate_names: Names,
    covariates: Vec<Covariate>,
    strata_maps: OnceLock<Vec<HashMap<String, String>>>,
    covariate_maps: OnceLock<Vec<HashMap<String, CovariateValue>>>,
}

impl fmt::Debug for Cohort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cohort")
            .field("rows", &self.len)
            .field("strata", &self.strata_names.names)
            .field("numeric_fields", &self.field_names.names)
            .field("covariates", &self.covariate_names.names)
            .finish_non_exhaustive()
    }
}

impl Cohort {
    /// Start a cohort from the ids and the birth dates, one per row.
    ///
    /// The ids are copied into the cohort, so they can be borrowed from
    /// anywhere.
    pub fn builder<I, S>(ids: I, birth_dates: Vec<NaiveDate>) -> CohortBuilder
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ids = ids.into_iter();
        let mut id_text = String::new();
        let mut id_ends = Vec::with_capacity(ids.size_hint().0);
        for id in ids {
            id_text.push_str(id.as_ref());
            id_ends.push(id_text.len());
        }
        id_text.shrink_to_fit();
        id_ends.shrink_to_fit();
        CohortBuilder {
            cohort: Self {
                len: id_ends.len(),
                id_text,
                id_ends,
                birth: birth_dates,
                death: None,
                emigration: None,
                entry: None,
                event: None,
                unique_key: None,
                strata_names: Names::default(),
                strata: Vec::new(),
                numbers: Vec::new(),
                field_names: Names::default(),
                fields: Vec::new(),
                covariate_names: Names::default(),
                covariates: Vec::new(),
                strata_maps: OnceLock::new(),
                covariate_maps: OnceLock::new(),
            },
            error: None,
        }
    }

    /// The number of rows.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the cohort has no rows.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The row `index`, `None` past the end.
    #[must_use]
    pub const fn row(&self, index: usize) -> Option<CohortRow<'_>> {
        if index < self.len {
            Some(CohortRow {
                cohort: self,
                row: index,
            })
        } else {
            None
        }
    }

    /// Every row, in order.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = CohortRow<'_>> + '_ {
        (0..self.len).map(move |row| CohortRow { cohort: self, row })
    }

    /// The records of risk-set matching: every row, with its event date as
    /// the transition date, `None` for every row when the cohort has no event
    /// date column.
    #[must_use]
    pub fn risk_set_rows(&self) -> Vec<RoleTransitionRecord<CohortRow<'_>>> {
        self.rows()
            .map(|row| {
                let event = self.event.as_ref().and_then(|dates| dates[row.row]);
                RoleTransitionRecord::from_record(row, event)
            })
            .collect()
    }

    /// The names of the strata columns, in the order they were added.
    #[must_use]
    pub fn strata_names(&self) -> &[String] {
        &self.strata_names.names
    }

    /// The names of the numeric fields, in the order they were added.
    #[must_use]
    pub fn numeric_names(&self) -> &[String] {
        &self.field_names.names
    }

    /// The names of the covariates, in the order they were added.
    #[must_use]
    pub fn covariate_names(&self) -> &[String] {
        &self.covariate_names.names
    }

    /// Whether the whole-record maps of [`MatchingRecord::strata`] or
    /// [`CovariateRecord::covariates`] have been built, which happens on the
    /// first call of either on any row and costs about what owned records
    /// cost.
    #[must_use]
    pub fn has_built_record_maps(&self) -> bool {
        self.strata_maps.get().is_some() || self.covariate_maps.get().is_some()
    }

    fn id(&self, row: usize) -> &str {
        let start = if row == 0 { 0 } else { self.id_ends[row - 1] };
        &self.id_text[start..self.id_ends[row]]
    }

    fn stratum(&self, row: usize, key: &str) -> Option<&str> {
        self.strata[self.strata_names.find(key)?].get(row)
    }

    fn numeric(&self, row: usize, name: &str) -> Option<f64> {
        self.numbers[self.fields[self.field_names.find(name)?]].get(row)
    }

    fn covariate(&self, row: usize, name: &str) -> Option<CovariateRef<'_>> {
        Some(self.covariate_in_slot(row, self.covariate_names.find(name)?))
    }

    fn covariate_in_slot(&self, row: usize, slot: usize) -> CovariateRef<'_> {
        match &self.covariates[slot] {
            Covariate::Numeric(column) => match self.numbers[*column].get(row) {
                Some(value) if value.is_finite() => CovariateRef::Numeric(value),
                _ => CovariateRef::Missing,
            },
            Covariate::Categorical(text) => text
                .get(row)
                .map_or(CovariateRef::Missing, CovariateRef::Categorical),
        }
    }

    fn date(column: Option<&Vec<Option<NaiveDate>>>, row: usize) -> Option<NaiveDate> {
        column.and_then(|dates| dates[row])
    }

    fn strata_maps(&self) -> &[HashMap<String, String>] {
        self.strata_maps.get_or_init(|| {
            (0..self.len)
                .map(|row| {
                    self.strata_names
                        .names
                        .iter()
                        .zip(&self.strata)
                        .filter_map(|(name, column)| {
                            column
                                .get(row)
                                .map(|value| (name.clone(), value.to_string()))
                        })
                        .collect()
                })
                .collect()
        })
    }

    fn covariate_maps(&self) -> &[HashMap<String, CovariateValue>] {
        self.covariate_maps.get_or_init(|| {
            (0..self.len)
                .map(|row| {
                    self.covariate_names
                        .names
                        .iter()
                        .filter_map(|name| {
                            self.covariate(row, name)
                                .map(|value| (name.clone(), value.to_value()))
                        })
                        .collect()
                })
                .collect()
        })
    }
}

/// Builds a [`Cohort`] column by column, from [`Cohort::builder`].
///
/// Each column is read and stored when it is added. A column of the wrong
/// length or a repeated name is reported by [`build`](Self::build).
#[must_use]
pub struct CohortBuilder {
    cohort: Cohort,
    error: Option<CohortError>,
}

impl fmt::Debug for CohortBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CohortBuilder")
            .field("cohort", &self.cohort)
            .field("error", &self.error)
            .finish()
    }
}

impl CohortBuilder {
    fn check(&mut self, column: &str, found: usize) {
        if self.error.is_none() && found != self.cohort.len {
            self.error = Some(CohortError::LengthMismatch {
                column: column.to_string(),
                found,
                expected: self.cohort.len,
            });
        }
    }

    fn fail(&mut self, error: Result<(), CohortError>) {
        if let (None, Err(error)) = (&self.error, error) {
            self.error = Some(error);
        }
    }

    /// An exact-match stratum called `name`. A row without a value does not
    /// have the stratum.
    pub fn stratum<I, S>(mut self, name: impl Into<String>, values: I) -> Self
    where
        I: IntoIterator<Item = Option<S>>,
        S: AsRef<str>,
    {
        let name = name.into();
        match Dictionary::from_values(&name, values) {
            Ok(column) => {
                self.check(&name, column.len());
                let pushed = self.cohort.strata_names.push(name);
                self.fail(pushed);
                self.cohort.strata.push(column);
            }
            Err(error) => self.fail(Err(error)),
        }
        self
    }

    /// A numeric field called `name`, for calipers. A row without a value
    /// does not have the field.
    pub fn numeric(
        mut self,
        name: impl Into<String>,
        values: impl IntoIterator<Item = Option<f64>>,
    ) -> Self {
        let name = name.into();
        let column = self.add_numbers(&name, values);
        let pushed = self.cohort.field_names.push(name);
        self.fail(pushed);
        self.cohort.fields.push(column);
        self
    }

    /// A numeric covariate called `name`. A row without a value, or with a
    /// value that is not finite, has the covariate missing.
    pub fn numeric_covariate(
        mut self,
        name: impl Into<String>,
        values: impl IntoIterator<Item = Option<f64>>,
    ) -> Self {
        let name = name.into();
        let column = self.add_numbers(&name, values);
        let pushed = self.cohort.covariate_names.push(name);
        self.fail(pushed);
        self.cohort.covariates.push(Covariate::Numeric(column));
        self
    }

    /// A numeric field and a numeric covariate, both called `name`, over one
    /// column: what [`numeric`](Self::numeric) and
    /// [`numeric_covariate`](Self::numeric_covariate) give with the same
    /// values, stored once.
    pub fn numeric_field_and_covariate(
        mut self,
        name: impl Into<String>,
        values: impl IntoIterator<Item = Option<f64>>,
    ) -> Self {
        let name = name.into();
        let column = self.add_numbers(&name, values);
        let pushed = self.cohort.field_names.push(name.clone());
        self.fail(pushed);
        self.cohort.fields.push(column);
        let pushed = self.cohort.covariate_names.push(name);
        self.fail(pushed);
        self.cohort.covariates.push(Covariate::Numeric(column));
        self
    }

    /// A categorical covariate called `name`. A row without a value has the
    /// covariate missing.
    pub fn categorical_covariate<I, S>(mut self, name: impl Into<String>, values: I) -> Self
    where
        I: IntoIterator<Item = Option<S>>,
        S: AsRef<str>,
    {
        let name = name.into();
        match Dictionary::from_values(&name, values) {
            Ok(column) => {
                self.check(&name, column.len());
                let pushed = self.cohort.covariate_names.push(name);
                self.fail(pushed);
                self.cohort.covariates.push(Covariate::Categorical(column));
            }
            Err(error) => self.fail(Err(error)),
        }
        self
    }

    /// The generic uniqueness key.
    pub fn unique_key<I, S>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = Option<S>>,
        S: AsRef<str>,
    {
        match Dictionary::from_values("unique_key", values) {
            Ok(column) => {
                self.check("unique_key", column.len());
                self.cohort.unique_key = Some(column);
            }
            Err(error) => self.fail(Err(error)),
        }
        self
    }

    /// The death dates.
    pub fn death_date(mut self, dates: Vec<Option<NaiveDate>>) -> Self {
        self.check("death_date", dates.len());
        self.cohort.death = Some(dates);
        self
    }

    /// The emigration dates.
    pub fn emigration_date(mut self, dates: Vec<Option<NaiveDate>>) -> Self {
        self.check("emigration_date", dates.len());
        self.cohort.emigration = Some(dates);
        self
    }

    /// The dates from which a row is in the population at risk.
    pub fn entry_date(mut self, dates: Vec<Option<NaiveDate>>) -> Self {
        self.check("entry_date", dates.len());
        self.cohort.entry = Some(dates);
        self
    }

    /// The event dates of risk-set matching, read by
    /// [`Cohort::risk_set_rows`].
    pub fn event_date(mut self, dates: Vec<Option<NaiveDate>>) -> Self {
        self.check("event_date", dates.len());
        self.cohort.event = Some(dates);
        self
    }

    /// The cohort.
    ///
    /// # Errors
    ///
    /// The first [`CohortError`] met while the columns were added.
    pub fn build(mut self) -> Result<Cohort, CohortError> {
        let birth = self.cohort.birth.len();
        self.check("birth_date", birth);
        match self.error {
            Some(error) => Err(error),
            None => Ok(self.cohort),
        }
    }

    fn add_numbers(&mut self, name: &str, values: impl IntoIterator<Item = Option<f64>>) -> usize {
        let column = Numbers::from_values(values);
        self.check(name, column.len());
        self.cohort.numbers.push(column);
        self.cohort.numbers.len() - 1
    }
}

/// A row of a [`Cohort`]: a reference to the cohort and the row number.
#[derive(Clone, Copy)]
pub struct CohortRow<'a> {
    cohort: &'a Cohort,
    row: usize,
}

impl<'a> CohortRow<'a> {
    /// The row number in the cohort.
    #[must_use]
    pub const fn index(&self) -> usize {
        self.row
    }

    /// The cohort the row belongs to.
    #[must_use]
    pub const fn cohort(&self) -> &'a Cohort {
        self.cohort
    }
}

impl fmt::Debug for CohortRow<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CohortRow")
            .field("row", &self.row)
            .field("id", &self.cohort.id(self.row))
            .finish()
    }
}

impl MatchingRecord for CohortRow<'_> {
    fn id(&self) -> &str {
        self.cohort.id(self.row)
    }

    fn birth_date(&self) -> NaiveDate {
        self.cohort.birth[self.row]
    }

    fn strata(&self) -> &HashMap<String, String> {
        &self.cohort.strata_maps()[self.row]
    }

    fn stratum(&self, key: &str) -> Option<&str> {
        self.cohort.stratum(self.row, key)
    }

    fn numeric(&self, name: &str) -> Option<f64> {
        self.cohort.numeric(self.row, name)
    }

    fn numeric_table(&self) -> Option<usize> {
        Some(std::ptr::from_ref(self.cohort) as usize)
    }

    fn numeric_slot(&self, name: &str) -> Option<usize> {
        self.cohort.field_names.find(name)
    }

    fn numeric_in_slot(&self, slot: usize) -> Option<f64> {
        self.cohort.numbers[self.cohort.fields[slot]].get(self.row)
    }

    fn unique_key(&self) -> Option<&str> {
        self.cohort.unique_key.as_ref()?.get(self.row)
    }

    fn death_date(&self) -> Option<NaiveDate> {
        Cohort::date(self.cohort.death.as_ref(), self.row)
    }

    fn emigration_date(&self) -> Option<NaiveDate> {
        Cohort::date(self.cohort.emigration.as_ref(), self.row)
    }

    fn entry_date(&self) -> Option<NaiveDate> {
        Cohort::date(self.cohort.entry.as_ref(), self.row)
    }
}

impl CovariateRecord for CohortRow<'_> {
    fn covariates(&self) -> &HashMap<String, CovariateValue> {
        &self.cohort.covariate_maps()[self.row]
    }

    fn covariate(&self, name: &str) -> Option<CovariateRef<'_>> {
        self.cohort.covariate(self.row, name)
    }

    fn covariate_names(&self) -> impl Iterator<Item = &str> {
        self.cohort.covariate_names.names.iter().map(String::as_str)
    }

    fn covariate_table(&self) -> Option<usize> {
        Some(std::ptr::from_ref(self.cohort) as usize)
    }

    fn covariate_slot(&self, name: &str) -> Option<usize> {
        self.cohort.covariate_names.find(name)
    }

    fn covariate_in_slot(&self, slot: usize) -> CovariateRef<'_> {
        self.cohort.covariate_in_slot(self.row, slot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::date;

    fn small() -> Cohort {
        Cohort::builder(["a", "bb", ""], vec![date(2010, 1, 1); 3])
            .stratum("s", [Some("x"), None, Some("x")])
            .numeric("f", [Some(1.0), None, Some(f64::NAN)])
            .numeric_covariate("c", [Some(2.0), Some(f64::INFINITY), None])
            .categorical_covariate("k", [None, Some("u"), Some("v")])
            .unique_key([Some("fam"), None, Some("fam")])
            .death_date(vec![None, Some(date(2020, 1, 1)), None])
            .event_date(vec![Some(date(2012, 1, 1)), None, None])
            .build()
            .expect("valid cohort")
    }

    #[test]
    fn a_row_reads_the_values_it_was_given() {
        let cohort = small();
        let rows: Vec<_> = cohort.rows().collect();
        assert_eq!(
            rows.iter().map(MatchingRecord::id).collect::<Vec<_>>(),
            ["a", "bb", ""]
        );
        assert_eq!(rows[0].stratum("s"), Some("x"));
        assert_eq!(rows[1].stratum("s"), None);
        assert_eq!(rows[0].stratum("absent"), None);
        assert_eq!(rows[0].numeric("f"), Some(1.0));
        assert_eq!(rows[1].numeric("f"), None);
        assert!(rows[2].numeric("f").is_some_and(f64::is_nan));
        assert_eq!(rows[0].covariate("c"), Some(CovariateRef::Numeric(2.0)));
        assert_eq!(rows[1].covariate("c"), Some(CovariateRef::Missing));
        assert_eq!(rows[2].covariate("c"), Some(CovariateRef::Missing));
        assert_eq!(rows[0].covariate("k"), Some(CovariateRef::Missing));
        assert_eq!(rows[2].covariate("k"), Some(CovariateRef::Categorical("v")));
        assert_eq!(rows[0].covariate("absent"), None);
        assert_eq!(rows[1].unique_key(), None);
        assert_eq!(rows[2].unique_key(), Some("fam"));
        assert_eq!(rows[1].death_date(), Some(date(2020, 1, 1)));
        assert_eq!(rows[0].entry_date(), None);
        let risk = cohort.risk_set_rows();
        assert_eq!(risk[0].transition_date, Some(date(2012, 1, 1)));
        assert_eq!(risk[1].transition_date, None);
        assert!(!cohort.has_built_record_maps());
    }

    #[test]
    fn the_maps_hold_what_the_columns_hold() {
        let cohort = small();
        let rows: Vec<_> = cohort.rows().collect();
        assert_eq!(rows[1].strata(), &HashMap::new());
        assert_eq!(rows[2].strata()["s"], "x");
        assert!(cohort.has_built_record_maps());
        let names: std::collections::BTreeSet<_> = rows[0].covariate_names().collect();
        for row in &rows {
            let map = row.covariates();
            assert_eq!(
                map.keys()
                    .map(String::as_str)
                    .collect::<std::collections::BTreeSet<_>>(),
                names
            );
            for name in &names {
                assert_eq!(map.get(*name).map(CovariateRef::from), row.covariate(name));
            }
        }
    }

    #[test]
    fn many_names_are_found_by_hash() {
        let names: Vec<String> = (0..20).map(|i| format!("s{i}")).collect();
        let mut builder = Cohort::builder(["a"], vec![date(2010, 1, 1)]);
        for (i, name) in names.iter().enumerate() {
            builder = builder.stratum(name.clone(), [Some(format!("v{i}"))]);
        }
        let cohort = builder.build().expect("valid cohort");
        let row = cohort.row(0).expect("one row");
        for (i, name) in names.iter().enumerate() {
            assert_eq!(row.stratum(name), Some(format!("v{i}").as_str()));
        }
        assert_eq!(row.stratum("s20"), None);
        assert!(cohort.row(1).is_none());
    }

    #[test]
    fn bad_columns_are_errors() {
        let short = Cohort::builder(["a", "b"], vec![date(2010, 1, 1); 2])
            .stratum("s", [Some("x")])
            .build();
        assert_eq!(
            short.err(),
            Some(CohortError::LengthMismatch {
                column: "s".into(),
                found: 1,
                expected: 2
            })
        );
        let births = Cohort::builder(["a", "b"], vec![date(2010, 1, 1)]).build();
        assert!(
            matches!(births, Err(CohortError::LengthMismatch { column, .. }) if column == "birth_date")
        );
        let twice = Cohort::builder(["a"], vec![date(2010, 1, 1)])
            .numeric("x", [Some(1.0)])
            .numeric_field_and_covariate("x", [Some(1.0)])
            .build();
        assert_eq!(
            twice.err(),
            Some(CohortError::DuplicateName { column: "x".into() })
        );
        let kinds = Cohort::builder(["a"], vec![date(2010, 1, 1)])
            .stratum("x", [Some("1")])
            .numeric("x", [Some(1.0)])
            .numeric_covariate("x", [Some(1.0)])
            .build();
        assert!(kinds.is_ok());
    }

    #[test]
    fn a_reference_to_a_row_names_the_same_table() {
        let cohort = small();
        let rows: Vec<_> = cohort.rows().collect();
        let row = &rows[1];
        let by_reference: &&CohortRow<'_> = &row;
        assert!(row.covariate_table().is_some());
        assert_eq!(by_reference.covariate_table(), row.covariate_table());
        let slot = by_reference.covariate_slot("c").expect("covariate c");
        assert_eq!(Some(slot), row.covariate_slot("c"));
        assert_eq!(by_reference.covariate_in_slot(slot), CovariateRef::Missing);
    }
}

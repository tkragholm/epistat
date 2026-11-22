//! The crate's records from Apache Arrow arrays.
//!
//! Enable the cargo feature `arrow`. The module adds `arrow-array` as a
//! dependency and nothing else, at the same version as `epistat-survival`,
//! so both crates take the same Arrow.
//!
//! A cohort of millions lives in columns, in Arrow, Polars or a Parquet
//! reader. [`ArrowRecords`] reads the columns and makes the records the
//! matchers take: [`BaseRecord`] with [`ArrowRecords::base_records`],
//! [`BalanceRecord`] with [`ArrowRecords::balance_records`] and the records of
//! risk-set matching, `RoleTransitionRecord<BaseRecord>`, with
//! [`ArrowRecords::risk_set_records`]. No intermediate row structs or data
//! frames are made, and the records are exactly those of the same values
//! given to [`BaseRecord::new`] and its setters.
//!
//! [`ArrowRecords::cohort`] reads the same columns into a [`Cohort`], which
//! holds them as columns and costs about 120 bytes per row with five strata
//! and five numeric covariates, where the records cost about two kilobytes. Its rows are records too, the same ones
//! [`balance_records`](ArrowRecords::balance_records) makes, and
//! [`Cohort::risk_set_rows`] gives those of
//! [`risk_set_records`](ArrowRecords::risk_set_records).
//!
//! # Columns and types
//!
//! | column | Arrow type |
//! |---|---|
//! | id | `Utf8`, `LargeUtf8`, a dictionary of either, or `Int32`, `Int64`, `UInt32`, `UInt64` |
//! | birth date, death, emigration, entry, event date | `Date32` |
//! | strata, unique key, categorical covariates | `Utf8`, `LargeUtf8` or a dictionary of either |
//! | numeric fields and covariates | `Float64` |
//!
//! An integer id becomes its decimal text. Any other type is an error that
//! names the column, the type found and the types expected. Nothing is cast.
//!
//! # Missing values
//!
//! Each null follows the rule the crate already has for that field.
//!
//! - The id and the birth date are required. A null is an error that names the
//!   column and the row.
//! - A null stratum is a stratum the record does not have, as a record built
//!   without the key. Exact-match constraints on that key refuse the record.
//! - A null numeric field is a field the record does not have, which a
//!   [`Caliper`](crate::constraints::Caliper) on it refuses.
//! - A null unique key, death date, emigration date, entry date or event date
//!   is `None`.
//! - A null or non-finite numeric covariate, and a null categorical covariate,
//!   is [`CovariateValue::Missing`], as [`BalanceRecordBuilder`](crate::BalanceRecordBuilder) makes it.
//!
//! A `NaN` in a numeric field is kept as it is, as it is when given to
//! [`BaseRecord::with_numeric`].
//!
//! # Names in errors
//!
//! The fixed columns are called `id`, `birth_date`, `unique_key`,
//! `death_date`, `emigration_date`, `entry_date` and `event_date` in errors.
//! Strata, numeric fields and covariates are called by the name given with
//! them, which is also the name of the key in the record.
//!
//! # Record batches
//!
//! [`column()`](fn@column) looks a column up in a [`RecordBatch`] by name and lists the
//! columns present when it is missing.
//!
//! ```
//! use std::sync::Arc;
//! use arrow_array::{ArrayRef, Date32Array, RecordBatch, StringArray};
//! use epistat_matching::arrow::{ArrowRecords, column};
//!
//! let batch = RecordBatch::try_from_iter([
//!     ("id", Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef),
//!     ("birth", Arc::new(Date32Array::from(vec![14_000, 14_100])) as ArrayRef),
//!     ("sex", Arc::new(StringArray::from(vec!["1", "2"])) as ArrayRef),
//! ])
//! .unwrap();
//! let records = ArrowRecords::new(column(&batch, "id").unwrap(), column(&batch, "birth").unwrap())
//!     .strata("sex", column(&batch, "sex").unwrap())
//!     .base_records()
//!     .unwrap();
//! assert_eq!(records[1].strata["sex"], "2");
//! ```
//!
//! # Polars and pyarrow
//!
//! No polars dependency is needed. A Polars `Series` exports to Arrow with
//! `to_arrow()`, and a pyarrow table through the Arrow C data interface.
//! Polars stores strings as `Utf8View`, which is not accepted here: cast the
//! column to `large_string` first. A chunked column is several arrays, so
//! rechunk it first.

use std::collections::HashMap;
use std::fmt;

use arrow_array::cast::AsArray;
use arrow_array::{
    Array, Date32Array, Float64Array, Int32Array, Int64Array, LargeStringArray, RecordBatch,
    StringArray, UInt32Array, UInt64Array,
};
use chrono::{Days, NaiveDate};

use crate::cohort::{Cohort, CohortBuilder, CohortError};
use crate::types::{BalanceRecord, BaseRecord, CovariateValue, RoleTransitionRecord};

/// Why records could not be built from Arrow arrays.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArrowInputError {
    /// A column of a record batch is not there.
    MissingColumn {
        /// The name asked for.
        column: String,
        /// The names present.
        present: Vec<String>,
    },
    /// A column has a type that is not accepted for its role.
    WrongType {
        /// The column.
        column: String,
        /// The type found.
        found: String,
        /// The types accepted.
        expected: &'static str,
    },
    /// A column has a null where the record needs a value.
    NullValue {
        /// The column.
        column: String,
        /// The first row with a null, counted from 0.
        row: usize,
    },
    /// A column has another length than the id column.
    LengthMismatch {
        /// The column.
        column: String,
        /// Its length.
        found: usize,
        /// The length of the id column.
        expected: usize,
    },
    /// A date is outside the range a calendar date can hold.
    DateOutOfRange {
        /// The column.
        column: String,
        /// The row, counted from 0.
        row: usize,
    },
    /// The columns do not make a [`Cohort`], as when two strata have one
    /// name.
    Cohort(CohortError),
}

impl fmt::Display for ArrowInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingColumn { column, present } => write!(
                f,
                "column {column} is not in the record batch; it has {}",
                present.join(", ")
            ),
            Self::WrongType {
                column,
                found,
                expected,
            } => write!(
                f,
                "column {column} has type {found}, expected {expected}; cast it before building \
                 the records"
            ),
            Self::NullValue { column, row } => write!(
                f,
                "column {column} has a null at row {row}; every record needs a value there"
            ),
            Self::LengthMismatch {
                column,
                found,
                expected,
            } => write!(
                f,
                "column {column} has {found} rows, but the id column has {expected}"
            ),
            Self::DateOutOfRange { column, row } => {
                write!(f, "column {column} has a date at row {row} out of range")
            }
            Self::Cohort(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ArrowInputError {}

type Result<T> = std::result::Result<T, ArrowInputError>;

/// The column `name` of `batch`.
///
/// # Errors
///
/// [`ArrowInputError::MissingColumn`] naming the column and the columns
/// present, when there is no column of that name.
pub fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a dyn Array> {
    batch
        .column_by_name(name)
        .map(|array| &**array)
        .ok_or_else(|| ArrowInputError::MissingColumn {
            column: name.to_string(),
            present: batch
                .schema_ref()
                .fields()
                .iter()
                .map(|field| field.name().clone())
                .collect(),
        })
}

fn wrong_type(name: &str, array: &dyn Array, expected: &'static str) -> ArrowInputError {
    ArrowInputError::WrongType {
        column: name.to_string(),
        found: array.data_type().to_string(),
        expected,
    }
}

const TEXT: &str = "Utf8, LargeUtf8 or a dictionary of either";
const ID: &str = "Utf8, LargeUtf8, a dictionary of either, Int32, Int64, UInt32 or UInt64";

/// A text column read row by row.
enum Text<'a> {
    Utf8(&'a StringArray),
    Large(&'a LargeStringArray),
    Dictionary {
        array: &'a dyn Array,
        keys: Vec<usize>,
        values: Box<Text<'a>>,
    },
}

impl<'a> Text<'a> {
    fn new(name: &str, array: &'a dyn Array, expected: &'static str) -> Result<Self> {
        if let Some(text) = array.as_string_opt::<i32>() {
            return Ok(Self::Utf8(text));
        }
        if let Some(text) = array.as_string_opt::<i64>() {
            return Ok(Self::Large(text));
        }
        if let Some(dictionary) = array.as_any_dictionary_opt() {
            let values = dictionary.values();
            let inner = if let Some(text) = values.as_string_opt::<i32>() {
                Self::Utf8(text)
            } else if let Some(text) = values.as_string_opt::<i64>() {
                Self::Large(text)
            } else {
                return Err(wrong_type(name, array, expected));
            };
            return Ok(Self::Dictionary {
                array,
                keys: dictionary.normalized_keys(),
                values: Box::new(inner),
            });
        }
        Err(wrong_type(name, array, expected))
    }

    fn get(&self, row: usize) -> Option<&str> {
        match self {
            Self::Utf8(text) => (!text.is_null(row)).then(|| text.value(row)),
            Self::Large(text) => (!text.is_null(row)).then(|| text.value(row)),
            Self::Dictionary {
                array,
                keys,
                values,
            } => {
                if array.is_null(row) {
                    None
                } else {
                    values.get(keys[row])
                }
            }
        }
    }
}

/// An id column read row by row.
enum Ids<'a> {
    Text(Text<'a>),
    I32(&'a Int32Array),
    I64(&'a Int64Array),
    U32(&'a UInt32Array),
    U64(&'a UInt64Array),
}

impl<'a> Ids<'a> {
    fn new(array: &'a dyn Array) -> Result<Self> {
        let any = array.as_any();
        if let Some(ids) = any.downcast_ref::<Int32Array>() {
            return Ok(Self::I32(ids));
        }
        if let Some(ids) = any.downcast_ref::<Int64Array>() {
            return Ok(Self::I64(ids));
        }
        if let Some(ids) = any.downcast_ref::<UInt32Array>() {
            return Ok(Self::U32(ids));
        }
        if let Some(ids) = any.downcast_ref::<UInt64Array>() {
            return Ok(Self::U64(ids));
        }
        Text::new("id", array, ID).map(Self::Text)
    }

    fn get(&self, row: usize) -> Option<String> {
        match self {
            Self::Text(text) => text.get(row).map(str::to_string),
            Self::I32(ids) => (!ids.is_null(row)).then(|| ids.value(row).to_string()),
            Self::I64(ids) => (!ids.is_null(row)).then(|| ids.value(row).to_string()),
            Self::U32(ids) => (!ids.is_null(row)).then(|| ids.value(row).to_string()),
            Self::U64(ids) => (!ids.is_null(row)).then(|| ids.value(row).to_string()),
        }
    }
}

/// A date column, with the name it is reported under.
struct Dates<'a> {
    name: &'static str,
    array: &'a Date32Array,
}

impl<'a> Dates<'a> {
    fn new(name: &'static str, array: &'a dyn Array, len: usize) -> Result<Self> {
        let typed = array
            .as_any()
            .downcast_ref::<Date32Array>()
            .ok_or_else(|| wrong_type(name, array, "Date32"))?;
        check_len(name, array, len)?;
        Ok(Self { name, array: typed })
    }

    fn get(&self, row: usize) -> Result<Option<NaiveDate>> {
        if self.array.is_null(row) {
            return Ok(None);
        }
        let days = self.array.value(row);
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).expect("the epoch is a date");
        let date = if days >= 0 {
            epoch.checked_add_days(Days::new(u64::from(days.unsigned_abs())))
        } else {
            epoch.checked_sub_days(Days::new(u64::from(days.unsigned_abs())))
        };
        date.map(Some)
            .ok_or_else(|| ArrowInputError::DateOutOfRange {
                column: self.name.to_string(),
                row,
            })
    }

    fn required(&self, row: usize) -> Result<NaiveDate> {
        self.get(row)?.ok_or_else(|| ArrowInputError::NullValue {
            column: self.name.to_string(),
            row,
        })
    }
}

fn check_len(name: &str, array: &dyn Array, expected: usize) -> Result<()> {
    if array.len() == expected {
        Ok(())
    } else {
        Err(ArrowInputError::LengthMismatch {
            column: name.to_string(),
            found: array.len(),
            expected,
        })
    }
}

fn floats<'a>(name: &str, array: &'a dyn Array, len: usize) -> Result<&'a Float64Array> {
    let typed = array
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| wrong_type(name, array, "Float64"))?;
    check_len(name, array, len)?;
    Ok(typed)
}

fn texts<'a>(name: &str, array: &'a dyn Array, len: usize) -> Result<Text<'a>> {
    let text = Text::new(name, array, TEXT)?;
    check_len(name, array, len)?;
    Ok(text)
}

/// A covariate column read row by row.
enum Cov<'a> {
    Numeric(&'a Float64Array),
    Categorical(Text<'a>),
}

/// The columns of a cohort, read from Arrow arrays into records.
///
/// Start with [`ArrowRecords::new`] from the id and birth date columns, add
/// the columns the analysis uses, and build the records of the type the
/// matcher takes. The arrays are read when a build method is called, and all
/// must have the length of the id column.
#[derive(Clone)]
#[must_use]
pub struct ArrowRecords<'a> {
    id: &'a dyn Array,
    birth_date: &'a dyn Array,
    strata: Vec<(String, &'a dyn Array)>,
    numerics: Vec<(String, &'a dyn Array)>,
    covariates: Vec<(String, &'a dyn Array, bool)>,
    unique_key: Option<&'a dyn Array>,
    death_date: Option<&'a dyn Array>,
    emigration_date: Option<&'a dyn Array>,
    entry_date: Option<&'a dyn Array>,
    event_date: Option<&'a dyn Array>,
}

impl fmt::Debug for ArrowRecords<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArrowRecords")
            .field("rows", &self.id.len())
            .finish_non_exhaustive()
    }
}

impl<'a> ArrowRecords<'a> {
    /// Start from the id column and the `Date32` birth date column.
    pub fn new(id: &'a dyn Array, birth_date: &'a dyn Array) -> Self {
        Self {
            id,
            birth_date,
            strata: Vec::new(),
            numerics: Vec::new(),
            covariates: Vec::new(),
            unique_key: None,
            death_date: None,
            emigration_date: None,
            entry_date: None,
            event_date: None,
        }
    }

    /// An exact-match stratum, kept under `name`.
    pub fn strata(mut self, name: impl Into<String>, values: &'a dyn Array) -> Self {
        self.strata.push((name.into(), values));
        self
    }

    /// A named numeric field, `Float64`, for calipers.
    pub fn numeric(mut self, name: impl Into<String>, values: &'a dyn Array) -> Self {
        self.numerics.push((name.into(), values));
        self
    }

    /// A numeric covariate for balance records, `Float64`.
    pub fn numeric_covariate(mut self, name: impl Into<String>, values: &'a dyn Array) -> Self {
        self.covariates.push((name.into(), values, true));
        self
    }

    /// A categorical covariate for balance records.
    pub fn categorical_covariate(mut self, name: impl Into<String>, values: &'a dyn Array) -> Self {
        self.covariates.push((name.into(), values, false));
        self
    }

    /// The generic uniqueness key.
    pub fn unique_key(mut self, values: &'a dyn Array) -> Self {
        self.unique_key = Some(values);
        self
    }

    /// The death date, `Date32`.
    pub fn death_date(mut self, dates: &'a dyn Array) -> Self {
        self.death_date = Some(dates);
        self
    }

    /// The emigration date, `Date32`.
    pub fn emigration_date(mut self, dates: &'a dyn Array) -> Self {
        self.emigration_date = Some(dates);
        self
    }

    /// The date from which a record is in the population at risk, `Date32`.
    pub fn entry_date(mut self, dates: &'a dyn Array) -> Self {
        self.entry_date = Some(dates);
        self
    }

    /// The event date of risk-set matching, `Date32`: the date a record
    /// moves from the comparison risk set to the anchor group, which is the
    /// index date of the matching. Read only by
    /// [`risk_set_records`](Self::risk_set_records).
    pub fn event_date(mut self, dates: &'a dyn Array) -> Self {
        self.event_date = Some(dates);
        self
    }

    /// One [`BaseRecord`] per row.
    ///
    /// # Errors
    ///
    /// An [`ArrowInputError`] for a column of the wrong type or length, or a
    /// null in the id or birth date column.
    pub fn base_records(&self) -> Result<Vec<BaseRecord>> {
        self.build(|core, _row| Ok(core))
    }

    /// One [`BalanceRecord`] per row, with the covariates added.
    ///
    /// # Errors
    ///
    /// As [`base_records`](Self::base_records), and for a covariate of the
    /// wrong type or length.
    pub fn balance_records(&self) -> Result<Vec<BalanceRecord>> {
        let len = self.id.len();
        let mut cols = Vec::with_capacity(self.covariates.len());
        for (name, array, numeric) in &self.covariates {
            cols.push((
                name,
                if *numeric {
                    Cov::Numeric(floats(name, *array, len)?)
                } else {
                    Cov::Categorical(texts(name, *array, len)?)
                },
            ));
        }
        self.build(|core, row| {
            let mut covariates = HashMap::with_capacity(cols.len());
            for (name, col) in &cols {
                let value = match col {
                    Cov::Numeric(values) => {
                        if values.is_null(row) || !values.value(row).is_finite() {
                            CovariateValue::Missing
                        } else {
                            CovariateValue::Numeric(values.value(row))
                        }
                    }
                    Cov::Categorical(text) => text.get(row).map_or(CovariateValue::Missing, |v| {
                        CovariateValue::Categorical(v.to_string())
                    }),
                };
                covariates.insert((*name).clone(), value);
            }
            Ok(BalanceRecord { core, covariates })
        })
    }

    /// One `RoleTransitionRecord<BaseRecord>` per row, as risk-set matching
    /// takes. The event date column is optional: without it every
    /// transition date is `None`.
    ///
    /// # Errors
    ///
    /// As [`base_records`](Self::base_records), and for an event date column
    /// of the wrong type or length.
    pub fn risk_set_records(&self) -> Result<Vec<RoleTransitionRecord<BaseRecord>>> {
        let event = self
            .event_date
            .map(|array| Dates::new("event_date", array, self.id.len()))
            .transpose()?;
        self.build(|core, row| {
            let transition = match &event {
                Some(dates) => dates.get(row)?,
                None => None,
            };
            Ok(RoleTransitionRecord::from_record(core, transition))
        })
    }

    /// One [`Cohort`] holding every column, with the event date when it is
    /// given. Its rows are the records of
    /// [`balance_records`](Self::balance_records), and its
    /// [`risk_set_rows`](Cohort::risk_set_rows) those of
    /// [`risk_set_records`](Self::risk_set_records). A numeric field and a
    /// numeric covariate of the same name given the same array share one
    /// column.
    ///
    /// # Errors
    ///
    /// As [`balance_records`](Self::balance_records), and
    /// [`ArrowInputError::Cohort`] when two strata, two numeric fields or two
    /// covariates have one name.
    pub fn cohort(&self) -> Result<Cohort> {
        let len = self.id.len();
        let ids = Ids::new(self.id)?;
        if let Some(row) = (0..len).find(|&row| self.id.is_null(row)) {
            return Err(ArrowInputError::NullValue {
                column: "id".to_string(),
                row,
            });
        }
        let birth = Dates::new("birth_date", self.birth_date, len)?;
        let birth = (0..len)
            .map(|row| birth.required(row))
            .collect::<Result<Vec<_>>>()?;
        let optional_date = |name, array: Option<&'a dyn Array>| -> Result<_> {
            array
                .map(|array| {
                    let dates = Dates::new(name, array, len)?;
                    (0..len)
                        .map(|row| dates.get(row))
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()
        };
        let death = optional_date("death_date", self.death_date)?;
        let emigration = optional_date("emigration_date", self.emigration_date)?;
        let entry = optional_date("entry_date", self.entry_date)?;
        let event = optional_date("event_date", self.event_date)?;

        let mut builder =
            Cohort::builder((0..len).map(|row| ids.get(row).unwrap_or_default()), birth);
        for (name, array) in &self.strata {
            let text = texts(name, *array, len)?;
            builder = builder.stratum(name.clone(), (0..len).map(|row| text.get(row)));
        }
        let shared = |name: &str, array: &dyn Array| {
            self.covariates.iter().any(|(other, values, numeric)| {
                *numeric && other == name && std::ptr::addr_eq(*values, array)
            })
        };
        for (name, array) in &self.numerics {
            let values = floats(name, *array, len)?;
            let column = (0..len).map(|row| (!values.is_null(row)).then(|| values.value(row)));
            builder = if shared(name, *array) {
                builder.numeric_field_and_covariate(name.clone(), column)
            } else {
                builder.numeric(name.clone(), column)
            };
        }
        for (name, array, numeric) in &self.covariates {
            if *numeric {
                let is_field = self
                    .numerics
                    .iter()
                    .any(|(other, values)| other == name && std::ptr::addr_eq(*values, *array));
                if is_field {
                    continue;
                }
                let values = floats(name, *array, len)?;
                builder = builder.numeric_covariate(
                    name.clone(),
                    (0..len).map(|row| (!values.is_null(row)).then(|| values.value(row))),
                );
            } else {
                let text = texts(name, *array, len)?;
                builder =
                    builder.categorical_covariate(name.clone(), (0..len).map(|row| text.get(row)));
            }
        }
        if let Some(array) = self.unique_key {
            let text = texts("unique_key", array, len)?;
            builder = builder.unique_key((0..len).map(|row| text.get(row)));
        }
        for (dates, set) in [
            (death, CohortBuilder::death_date as fn(_, _) -> _),
            (emigration, CohortBuilder::emigration_date),
            (entry, CohortBuilder::entry_date),
            (event, CohortBuilder::event_date),
        ] {
            if let Some(dates) = dates {
                builder = set(builder, dates);
            }
        }
        builder.build().map_err(ArrowInputError::Cohort)
    }

    fn build<T>(&self, finish: impl Fn(BaseRecord, usize) -> Result<T>) -> Result<Vec<T>> {
        let len = self.id.len();
        let ids = Ids::new(self.id)?;
        let birth = Dates::new("birth_date", self.birth_date, len)?;
        let strata = self
            .strata
            .iter()
            .map(|(name, array)| Ok((name, texts(name, *array, len)?)))
            .collect::<Result<Vec<_>>>()?;
        let numerics = self
            .numerics
            .iter()
            .map(|(name, array)| Ok((name, floats(name, *array, len)?)))
            .collect::<Result<Vec<_>>>()?;
        let unique = self
            .unique_key
            .map(|array| texts("unique_key", array, len))
            .transpose()?;
        let optional_date = |name, array: Option<&'a dyn Array>| {
            array.map(|array| Dates::new(name, array, len)).transpose()
        };
        let death = optional_date("death_date", self.death_date)?;
        let emigration = optional_date("emigration_date", self.emigration_date)?;
        let entry = optional_date("entry_date", self.entry_date)?;

        let mut out = Vec::with_capacity(len);
        for row in 0..len {
            let id = ids.get(row).ok_or_else(|| ArrowInputError::NullValue {
                column: "id".to_string(),
                row,
            })?;
            let mut record = BaseRecord::new(id, birth.required(row)?);
            for (name, text) in &strata {
                if let Some(value) = text.get(row) {
                    record.strata.insert((*name).clone(), value.to_string());
                }
            }
            for (name, values) in &numerics {
                if !values.is_null(row) {
                    record.numerics.insert((*name).clone(), values.value(row));
                }
            }
            record.unique_key = unique
                .as_ref()
                .and_then(|text| text.get(row))
                .map(str::to_string);
            for (target, dates) in [
                (&mut record.death_date, &death),
                (&mut record.emigration_date, &emigration),
                (&mut record.entry_date, &entry),
            ] {
                if let Some(dates) = dates {
                    *target = dates.get(row)?;
                }
            }
            out.push(finish(record, row)?);
        }
        Ok(out)
    }
}

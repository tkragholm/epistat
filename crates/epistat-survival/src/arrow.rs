//! [`SurvData`] from Apache Arrow arrays, without copying the columns.
//!
//! Enable the cargo feature `arrow`. The module adds `arrow-array` as a
//! dependency and nothing else.
//!
//! The columns are borrowed from the arrays' values buffers, so an array
//! must outlive the [`SurvData`] built from it, as a slice must. The builders
//! here only look at the arrays and hand the slices to
//! [`SurvData::builder`] and [`SurvData::interval_builder`], so every check of
//! [`crate::surv`] applies unchanged and the estimates are those of the same
//! numbers given as slices.
//!
//! # Columns and types
//!
//! | column | Arrow type |
//! |---|---|
//! | start, stop, weight, offset, lower, upper, time, time2 | `Float64` |
//! | status | `Int32` |
//! | stratum, cluster, id | `UInt32`, or a dictionary with `UInt32` keys |
//! | initial state | `UInt32` |
//!
//! Any other type is an error that names the column, the type found and the
//! type expected. Nothing is cast, since a cast is a copy. The caller casts
//! first, for example with `arrow-cast`, and pays for it knowingly. R and
//! Python layers that hold a column as `Int64` or `Float32` do this at the
//! boundary.
//!
//! # Missing values
//!
//! An array with a null is an error that names the column and the first null
//! row. Nothing is dropped or replaced, as nothing is in [`SurvData`]: the
//! caller filters the rows, or its R or Python layer decides what a missing
//! value means. A `NaN` in a `Float64` array is not a null and is refused by
//! the checks of [`SurvData`], as ever. For a dictionary column a key that
//! points at a null dictionary value counts as a null.
//!
//! # Offsets
//!
//! A sliced array has an offset into its buffer. The slice handed on starts
//! at the offset and has the array's length, so a sliced array is borrowed
//! without a copy and reads the rows it shows. An offset never fits badly,
//! and so is not an error.
//!
//! # Dictionary columns
//!
//! A dictionary column is read through its keys, which are the codes of
//! [`SurvData::builder`]: they need not be dense or ordered. The levels the
//! data reports ([`SurvData::stratum_levels`] and its siblings) are key
//! values, not entries of the dictionary, so a caller who wants the original
//! labels reads the dictionary's values at those keys. Keys of another width
//! are an error, as any other type is: casting them to `UInt32` is a copy the
//! caller makes.
//!
//! # Record batches
//!
//! The builders take arrays, so a [`RecordBatch`] is used by looking its
//! columns up with [`column()`](fn@column), which names the missing column and lists those
//! present. A fixed schema is not imposed because the optional columns differ
//! between analyses.
//!
//! ```
//! use std::sync::Arc;
//! use arrow_array::{ArrayRef, Float64Array, Int32Array, RecordBatch};
//! use epistat_survival::arrow::{column, follow_up};
//!
//! let batch = RecordBatch::try_from_iter([
//!     ("time", Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])) as ArrayRef),
//!     ("event", Arc::new(Int32Array::from(vec![1, 0, 1])) as ArrayRef),
//! ])
//! .unwrap();
//! let data = follow_up(column(&batch, "time").unwrap(), column(&batch, "event").unwrap())
//!     .build()
//!     .unwrap();
//! assert_eq!(data.n_rows(), 3);
//! ```
//!
//! # Polars and pyarrow
//!
//! No polars dependency is needed. A Polars `DataFrame` or a pyarrow table
//! exports its columns through the Arrow C data interface (`Series.to_arrow()`
//! and pyarrow's `_export_to_c`, or the PyCapsule interface), and
//! `arrow_array::ffi::from_ffi` imports the structures as `ArrayData`, whose
//! buffers `Float64Array::from` takes over without a copy. Polars stores a
//! column in chunks, and a chunked column is several arrays: build one
//! [`SurvData`] needs one array per column, so rechunk first
//! (`DataFrame.rechunk()`), which is a copy only when there is more than one
//! chunk. A column with a validity bitmap but no nulls is accepted. Polars
//! stores strings and categoricals as `Utf8View` and `Categorical`, which are
//! not accepted here: encode a stratum as `UInt32` or as a dictionary with
//! `UInt32` keys (Polars' `to_physical` of a categorical is `UInt32`). The
//! import itself is `unsafe` in `arrow-array` and is left to the caller,
//! since this crate forbids unsafe code.

use arrow_array::cast::AsArray;
use arrow_array::types::UInt32Type;
use arrow_array::{Array, Float64Array, Int32Array, RecordBatch, UInt32Array};

use crate::error::{Error, Result};
use crate::surv::{IntervalDataBuilder, SurvData, SurvDataBuilder};

/// The column `name` of `batch`.
///
/// # Errors
///
/// An input error naming the column and the columns present, when there is
/// no column of that name.
pub fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a dyn Array> {
    batch
        .column_by_name(name)
        .map(|array| &**array)
        .ok_or_else(|| {
            let present: Vec<&str> = batch
                .schema_ref()
                .fields()
                .iter()
                .map(|field| field.name().as_str())
                .collect();
            Error::invalid(format!(
                "column {name} is not in the record batch; it has {}",
                present.join(", ")
            ))
        })
}

/// The first row of `array` that is null, as a row number from 0.
fn first_null(array: &dyn Array) -> Option<usize> {
    let nulls = array.logical_nulls()?;
    if nulls.null_count() == 0 {
        return None;
    }
    nulls.iter().position(|valid| !valid)
}

fn check_no_nulls(name: &str, array: &dyn Array) -> Result<()> {
    match first_null(array) {
        Some(row) => Err(Error::invalid(format!(
            "column {name} has a null at row {row}; drop or resolve missing values before \
             building the data"
        ))),
        None => Ok(()),
    }
}

fn wrong_type(name: &str, array: &dyn Array, expected: &str) -> Error {
    Error::invalid(format!(
        "column {name} has type {}, expected {expected}; cast it before building the data",
        array.data_type()
    ))
}

fn floats<'a>(name: &str, array: &'a dyn Array) -> Result<&'a [f64]> {
    let typed: &'a Float64Array = array
        .as_any()
        .downcast_ref()
        .ok_or_else(|| wrong_type(name, array, "Float64"))?;
    check_no_nulls(name, array)?;
    Ok(typed.values())
}

fn ints<'a>(name: &str, array: &'a dyn Array) -> Result<&'a [i32]> {
    let typed: &'a Int32Array = array
        .as_any()
        .downcast_ref()
        .ok_or_else(|| wrong_type(name, array, "Int32"))?;
    check_no_nulls(name, array)?;
    Ok(typed.values())
}

fn codes<'a>(name: &str, array: &'a dyn Array) -> Result<&'a [u32]> {
    let expected = "UInt32 or a dictionary with UInt32 keys";
    if let Some(typed) = array.as_any().downcast_ref::<UInt32Array>() {
        check_no_nulls(name, array)?;
        return Ok(typed.values());
    }
    match array.as_any_dictionary_opt() {
        Some(dictionary) => match array.as_dictionary_opt::<UInt32Type>() {
            Some(typed) => {
                check_no_nulls(name, array)?;
                Ok(typed.keys().values())
            }
            None => Err(Error::invalid(format!(
                "column {name} is a dictionary with {} keys, expected {expected}; cast the keys \
                 before building the data",
                dictionary.keys().data_type()
            ))),
        },
        None => Err(wrong_type(name, array, expected)),
    }
}

/// The columns of a follow-up [`SurvData`], read from Arrow arrays.
///
/// Built by [`follow_up`]. The arrays are read when
/// [`build`](Self::build) is called.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct ArrowSurv<'a> {
    stop: &'a dyn Array,
    status: &'a dyn Array,
    start: Option<&'a dyn Array>,
    n_codes: u32,
    weight: Option<&'a dyn Array>,
    offset: Option<&'a dyn Array>,
    stratum: Option<&'a dyn Array>,
    cluster: Option<&'a dyn Array>,
    id: Option<&'a dyn Array>,
    istate: Option<&'a dyn Array>,
}

/// Start follow-up data from a `Float64` stop column and an `Int32` status
/// column, as [`SurvData::builder`] does from slices.
pub fn follow_up<'a>(stop: &'a dyn Array, status: &'a dyn Array) -> ArrowSurv<'a> {
    ArrowSurv {
        stop,
        status,
        start: None,
        n_codes: 1,
        weight: None,
        offset: None,
        stratum: None,
        cluster: None,
        id: None,
        istate: None,
    }
}

impl<'a> ArrowSurv<'a> {
    /// The start of each risk interval, `Float64`.
    pub fn start(mut self, start: &'a dyn Array) -> Self {
        self.start = Some(start);
        self
    }

    /// The number of event codes in status. The default is 1.
    pub fn n_codes(mut self, n_codes: u32) -> Self {
        self.n_codes = n_codes;
        self
    }

    /// Case weights, `Float64`.
    pub fn weight(mut self, weight: &'a dyn Array) -> Self {
        self.weight = Some(weight);
        self
    }

    /// A term added to the linear predictor of every row, `Float64`.
    pub fn offset(mut self, offset: &'a dyn Array) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Stratum codes, `UInt32` or a dictionary with `UInt32` keys.
    pub fn stratum(mut self, stratum: &'a dyn Array) -> Self {
        self.stratum = Some(stratum);
        self
    }

    /// Cluster codes, `UInt32` or a dictionary with `UInt32` keys.
    pub fn cluster(mut self, cluster: &'a dyn Array) -> Self {
        self.cluster = Some(cluster);
        self
    }

    /// Subject codes, `UInt32` or a dictionary with `UInt32` keys.
    pub fn id(mut self, id: &'a dyn Array) -> Self {
        self.id = Some(id);
        self
    }

    /// The state at the start of each row, `UInt32`.
    pub fn istate(mut self, istate: &'a dyn Array) -> Self {
        self.istate = Some(istate);
        self
    }

    /// Read the arrays, then check and recode them as
    /// [`SurvDataBuilder::build`] does.
    ///
    /// # Errors
    ///
    /// An input error for a column of the wrong type or with a null, in the
    /// order stop, status, start, weight, offset, stratum, cluster, id, istate, and
    /// then for the first rule of [`SurvDataBuilder::build`] that fails.
    pub fn build(self) -> Result<SurvData<'a>> {
        self.builder()?.build()
    }

    /// The slice builder over the arrays' buffers, to add what this one does
    /// not offer before building.
    ///
    /// # Errors
    ///
    /// As for [`build`](Self::build), for the columns read.
    pub fn builder(self) -> Result<SurvDataBuilder<'a>> {
        let mut builder =
            SurvData::builder(floats("stop", self.stop)?, ints("status", self.status)?)
                .n_codes(self.n_codes);
        if let Some(array) = self.start {
            builder = builder.start(floats("start", array)?);
        }
        if let Some(array) = self.weight {
            builder = builder.weight(floats("weight", array)?);
        }
        if let Some(array) = self.offset {
            builder = builder.offset(floats("offset", array)?);
        }
        if let Some(array) = self.stratum {
            builder = builder.stratum(codes("stratum", array)?);
        }
        if let Some(array) = self.cluster {
            builder = builder.cluster(codes("cluster", array)?);
        }
        if let Some(array) = self.id {
            builder = builder.id(codes("id", array)?);
        }
        if let Some(array) = self.istate {
            let typed: &UInt32Array = array
                .as_any()
                .downcast_ref()
                .ok_or_else(|| wrong_type("istate", array, "UInt32"))?;
            check_no_nulls("istate", array)?;
            builder = builder.istate(typed.values());
        }
        Ok(builder)
    }
}

/// How the bounds of an interval-censored outcome are given.
#[derive(Debug, Clone, Copy)]
enum Bounds<'a> {
    Pair(&'a dyn Array, &'a dyn Array),
    Status(&'a dyn Array, &'a dyn Array, &'a dyn Array),
}

/// The columns of an interval-censored [`SurvData`], read from Arrow arrays.
///
/// Built by [`interval`] or [`interval_status`].
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct ArrowInterval<'a> {
    bounds: Bounds<'a>,
    weight: Option<&'a dyn Array>,
    offset: Option<&'a dyn Array>,
    stratum: Option<&'a dyn Array>,
    cluster: Option<&'a dyn Array>,
    id: Option<&'a dyn Array>,
}

/// Start an interval-censored outcome from two `Float64` bounds, as
/// [`SurvData::interval_builder`] does from slices. `-inf` and `+inf` are
/// valid `Float64` values and mark a missing bound, as there.
pub fn interval<'a>(lower: &'a dyn Array, upper: &'a dyn Array) -> ArrowInterval<'a> {
    ArrowInterval {
        bounds: Bounds::Pair(lower, upper),
        weight: None,
        offset: None,
        stratum: None,
        cluster: None,
        id: None,
    }
}

/// Start an interval-censored outcome from `Float64` `time` and `time2` and an
/// `Int32` status, as [`SurvData::interval_status_builder`] does from slices.
/// As there, the bounds are built as owned columns.
pub fn interval_status<'a>(
    time: &'a dyn Array,
    time2: &'a dyn Array,
    status: &'a dyn Array,
) -> ArrowInterval<'a> {
    ArrowInterval {
        bounds: Bounds::Status(time, time2, status),
        weight: None,
        offset: None,
        stratum: None,
        cluster: None,
        id: None,
    }
}

impl<'a> ArrowInterval<'a> {
    /// Case weights, `Float64`.
    pub fn weight(mut self, weight: &'a dyn Array) -> Self {
        self.weight = Some(weight);
        self
    }

    /// A term added to the linear predictor of every row, `Float64`.
    pub fn offset(mut self, offset: &'a dyn Array) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Stratum codes, `UInt32` or a dictionary with `UInt32` keys.
    pub fn stratum(mut self, stratum: &'a dyn Array) -> Self {
        self.stratum = Some(stratum);
        self
    }

    /// Cluster codes, `UInt32` or a dictionary with `UInt32` keys.
    pub fn cluster(mut self, cluster: &'a dyn Array) -> Self {
        self.cluster = Some(cluster);
        self
    }

    /// Subject codes, `UInt32` or a dictionary with `UInt32` keys.
    pub fn id(mut self, id: &'a dyn Array) -> Self {
        self.id = Some(id);
        self
    }

    /// Read the arrays, then check and recode them as
    /// [`IntervalDataBuilder::build`] does.
    ///
    /// # Errors
    ///
    /// An input error for a column of the wrong type or with a null, and
    /// then for the first rule of [`IntervalDataBuilder::build`] that fails.
    pub fn build(self) -> Result<SurvData<'a>> {
        self.builder()?.build()
    }

    /// The slice builder over the arrays' buffers.
    ///
    /// # Errors
    ///
    /// As for [`build`](Self::build), for the columns read.
    pub fn builder(self) -> Result<IntervalDataBuilder<'a>> {
        let mut builder = match self.bounds {
            Bounds::Pair(lower, upper) => {
                SurvData::interval_builder(floats("lower", lower)?, floats("upper", upper)?)
            }
            Bounds::Status(time, time2, status) => SurvData::interval_status_builder(
                floats("time", time)?,
                floats("time2", time2)?,
                ints("status", status)?,
            ),
        };
        if let Some(array) = self.weight {
            builder = builder.weight(floats("weight", array)?);
        }
        if let Some(array) = self.offset {
            builder = builder.offset(floats("offset", array)?);
        }
        if let Some(array) = self.stratum {
            builder = builder.stratum(codes("stratum", array)?);
        }
        if let Some(array) = self.cluster {
            builder = builder.cluster(codes("cluster", array)?);
        }
        if let Some(array) = self.id {
            builder = builder.id(codes("id", array)?);
        }
        Ok(builder)
    }
}

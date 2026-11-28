//! [`TreatmentData`]: one row per unit, a treatment fixed at baseline and
//! named columns, the input of treatment-group matching.

use chrono::NaiveDate;
use epistat_core::Day;

use super::{Col, ColumnKind, Columns, ExactMissing, Label, LabelColumn, Pending, length};
use crate::Error;
use crate::cohort::Cohort;
use crate::role_transition::IdOrder;

/// One row per unit: an id, whether the unit is treated, and named columns,
/// as `matchit()` reads a data frame.
///
/// The ids are any [`Label`] and are coded once, in the sorted order of the
/// labels, so the rows are ordered by id as `str::cmp` orders strings and as
/// numbers are ordered for integers. A column is entered once, by name; what
/// it is used for comes from the options of a match.
///
/// ```
/// use epistat_matching::{ColumnKind, ExactMissing, TreatmentData};
///
/// let ids = ["NSW1", "NSW2", "PSID1", "PSID2"];
/// let treat = [true, true, false, false];
/// let age = [37.0, 22.0, 30.0, 27.0];
/// let race = ["black", "hispan", "white", "white"];
/// let re74 = [Some(0.0), None, Some(20_166.7), Some(25_862.3)];
///
/// let data = TreatmentData::builder(&ids, &treat)
///     .numeric("age", &age[..])
///     .categorical("race", &race[..])
///     .numeric("re74", &re74[..])
///     .build()?;
/// assert_eq!((data.len(), data.n_treated()), (4, 2));
/// assert_eq!(data.column_kind("race")?, ColumnKind::Categorical);
/// data.check_exact_key("race", ExactMissing::Refuse)?;
///
/// // A name the data do not have is refused with the names they have.
/// let error = data.column_kind("sex").unwrap_err();
/// assert_eq!(error.to_string(), "no column `sex`; the data have age, race, re74");
/// # Ok::<(), epistat_matching::Error>(())
/// ```
#[derive(Debug)]
pub struct TreatmentData {
    columns: Columns,
    treated: Vec<bool>,
}

impl TreatmentData {
    /// Start the view from the ids and whether each unit is treated.
    pub fn builder<'a, L: Label>(ids: &'a [L], treated: &'a [bool]) -> TreatmentDataBuilder<'a> {
        TreatmentDataBuilder {
            pending: Pending::new(ids),
            treated,
        }
    }

    /// The number of units.
    #[must_use]
    pub fn len(&self) -> usize {
        self.treated.len()
    }

    /// Whether the view has no units, which `build` refuses.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.treated.is_empty()
    }

    /// The number of treated units.
    #[must_use]
    pub fn n_treated(&self) -> usize {
        self.treated.iter().filter(|treated| **treated).count()
    }

    /// Whether each unit is treated, in row order.
    #[must_use]
    pub fn treated(&self) -> &[bool] {
        &self.treated
    }

    /// The named columns and their kinds, in the order they were added.
    pub fn columns(&self) -> impl Iterator<Item = (&str, ColumnKind)> {
        self.columns
            .named
            .iter()
            .map(|column| (column.name.as_str(), column.kind))
    }

    /// The kind of the column `name`.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownColumn`], naming the columns the view has, when it has
    /// none called `name`.
    pub fn column_kind(&self, name: &str) -> Result<ColumnKind, Error> {
        self.columns.column(name).map(|column| column.kind)
    }

    /// Checks that the column `name` can be an exact key: it is categorical,
    /// and it has no null unless `missing` makes a null a level of its own.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownColumn`] when there is no such column,
    /// [`Error::WrongColumnKind`] when it is not categorical and
    /// [`Error::NullKey`] at its first null under [`ExactMissing::Refuse`].
    pub fn check_exact_key(&self, name: &str, missing: ExactMissing) -> Result<(), Error> {
        self.columns.check_exact_key(name, missing)
    }

    /// The engine's cohort of the units, in row order. Its birth dates are
    /// all 1970-01-01, since the task reads none.
    pub(crate) const fn cohort(&self) -> &Cohort {
        &self.columns.cohort
    }

    /// The order of the units' ids.
    pub(crate) const fn id_order(&self) -> &IdOrder {
        &self.columns.order
    }
}

/// Builds a [`TreatmentData`] column by column, from
/// [`TreatmentData::builder`].
///
/// Each column is read and checked when it is added. The first problem is
/// returned by [`build`](Self::build).
#[must_use]
pub struct TreatmentDataBuilder<'a> {
    pending: Pending,
    treated: &'a [bool],
}

impl std::fmt::Debug for TreatmentDataBuilder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreatmentDataBuilder")
            .field("rows", &self.pending.len())
            .field("error", &self.pending.error)
            .finish_non_exhaustive()
    }
}

impl<'a> TreatmentDataBuilder<'a> {
    /// A numeric column: a covariate, a caliper field, a score or a sampling
    /// weight. A null is a missing value; NaN and infinities are refused.
    pub fn numeric(mut self, name: &str, values: impl Into<Col<'a, f64>>) -> Self {
        self.pending.numeric(name, &values.into());
        self
    }

    /// A categorical column: a covariate, entered as indicators, or an exact
    /// or antiexact key.
    pub fn categorical(mut self, name: &str, values: impl LabelColumn<'a>) -> Self {
        self.pending.categorical(name, &values.into_col());
        self
    }

    /// A date column, for a constraint within days.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "a column is passed as the call builds it, Col::unix_days(..), not by reference"
    )]
    pub fn date(mut self, name: &str, values: Col<'a, Day>) -> Self {
        self.pending.date(name, &values);
        self
    }

    /// The view.
    ///
    /// # Errors
    ///
    /// The first problem met: no ids ([`Error::Input`]), an id on two rows
    /// ([`Error::DuplicateIdRows`]), two labels that print alike
    /// ([`Error::AmbiguousLabel`]), a column of another length than the ids
    /// ([`Error::ColumnLength`]), two columns of one name
    /// ([`Error::DuplicateColumn`]), a number that is not finite
    /// ([`Error::NonFiniteValue`]) or a date out of range
    /// ([`Error::DateOutOfRange`]).
    pub fn build(mut self) -> Result<TreatmentData, Error> {
        let n = self.pending.len();
        if self.pending.error.is_none() && self.treated.len() != n {
            self.pending.fail(length("treated", self.treated.len(), n));
        }
        // NaiveDate's default is 1970-01-01.
        let birth = vec![NaiveDate::default(); n];
        let columns = self.pending.build(birth, |builder| builder)?;
        Ok(TreatmentData {
            columns,
            treated: self.treated.to_vec(),
        })
    }
}

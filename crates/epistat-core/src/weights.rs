//! Case weights, checked once where they enter.
//!
//! A weight column means one of three things, and the kind decides what an
//! estimator does with it beyond the weighted sums:
//!
//! | kind | a weight is | what changes |
//! |---|---|---|
//! | [`Prior`](WeightKind::Prior) | the precision of a row, R's `weights` | the weighted likelihood; the residual degrees of freedom count the rows with a positive weight |
//! | [`Frequency`](WeightKind::Frequency) | the number of times a row occurs, a whole number | the rows count `w` times: the degrees of freedom are `sum(w) - p`, as on the expanded data |
//! | [`Sampling`](WeightKind::Sampling) | the inverse probability that a row was sampled, as in a survey | only the sandwich variance is valid, so model-based standard errors are refused |
//!
//! [`CaseWeights`] holds the column and its kind. Every weight is finite and
//! not negative, and at least one is above zero. A weight of 0 is allowed
//! and means the row leaves the fit, so a bootstrap by resample counts passes
//! its counts as frequency weights in every crate.
//!
//! ```
//! use epistat_core::{CaseWeights, InputError, WeightKind};
//!
//! let w = [1.0, 0.0, 2.5];
//! let weights = CaseWeights::new(&w)?;
//! assert_eq!(weights.kind(), WeightKind::Prior);
//! assert_eq!(weights.n_positive(), 2);
//! assert_eq!(weights.as_slice(), &w);
//!
//! let counts = CaseWeights::frequency(&[2.0, 0.0, 1.0])?;
//! assert_eq!(counts.total(), 3.0);
//! assert!(CaseWeights::frequency(&[0.5]).is_err());
//! assert_eq!(CaseWeights::new(&[0.0, 0.0]), Err(InputError::AllZero { what: "weights" }));
//! assert!(CaseWeights::new(&[1.0, -1.0]).is_err());
//! # Ok::<(), InputError>(())
//! ```

use core::ops::Deref;

use crate::check::{self, InputError};

/// What a case weight means; see the module.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum WeightKind {
    /// The precision of a row, as R's `weights` argument to `glm` and `lm`.
    #[default]
    Prior,
    /// The number of times a row occurs: a whole number, `sum(w)` rows in
    /// all.
    Frequency,
    /// The inverse of a row's probability of being sampled.
    Sampling,
}

/// A column of case weights of one kind: finite, not negative, and not all
/// zero.
///
/// It borrows the caller's column, so building it copies nothing, and
/// derefs to `[f64]`. A view or estimator takes `CaseWeights` in place of
/// `&[f64]`, so the column is checked once, where it enters, and carries its
/// kind with it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CaseWeights<'a> {
    values: &'a [f64],
    kind: WeightKind,
}

impl<'a> CaseWeights<'a> {
    /// Prior weights, R's `weights`.
    ///
    /// # Errors
    ///
    /// At the first weight refused, [`InputError::NotFinite`] for NaN or an
    /// infinity and [`InputError::Domain`] for a negative weight, each with
    /// its row; [`InputError::Empty`] for no rows and
    /// [`InputError::AllZero`] when every weight is zero.
    pub fn new(values: &'a [f64]) -> Result<Self, InputError> {
        Self::of_kind(values, WeightKind::Prior)
    }

    /// Frequency weights, each a whole number of occurrences.
    ///
    /// # Errors
    ///
    /// As [`CaseWeights::new`], and [`InputError::Domain`] at the first
    /// weight that is not a whole number.
    pub fn frequency(values: &'a [f64]) -> Result<Self, InputError> {
        Self::of_kind(values, WeightKind::Frequency)
    }

    /// Sampling weights, the inverse probabilities of selection.
    ///
    /// # Errors
    ///
    /// As [`CaseWeights::new`].
    pub fn sampling(values: &'a [f64]) -> Result<Self, InputError> {
        Self::of_kind(values, WeightKind::Sampling)
    }

    /// Weights of the kind `kind`.
    ///
    /// # Errors
    ///
    /// As the constructor of that kind.
    pub fn of_kind(values: &'a [f64], kind: WeightKind) -> Result<Self, InputError> {
        const WHAT: &str = "weights";
        check::non_empty(WHAT, values.len())?;
        check::weights(WHAT, values)?;
        if kind == WeightKind::Frequency
            && let Some(row) = values.iter().position(|w| w.fract() != 0.0)
        {
            return Err(InputError::Domain {
                what: WHAT,
                value: values[row],
                domain: "the whole numbers 0, 1, 2, ...",
                row: Some(row),
            });
        }
        if values.iter().all(|&w| w == 0.0) {
            return Err(InputError::AllZero { what: WHAT });
        }
        Ok(Self { values, kind })
    }

    /// What the weights mean.
    #[must_use]
    pub const fn kind(&self) -> WeightKind {
        self.kind
    }

    /// The weights, one per row.
    #[must_use]
    pub const fn as_slice(&self) -> &'a [f64] {
        self.values
    }

    /// The number of rows, at least 1.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether there are no rows, which the constructors refuse, so always
    /// false.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The number of rows with a weight above zero, the rows that enter a
    /// fit.
    #[must_use]
    pub fn n_positive(&self) -> usize {
        self.values.iter().filter(|&&w| w > 0.0).count()
    }

    /// The sum of the weights, added in row order: the number of rows the
    /// data stand for under frequency weights.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.values.iter().sum()
    }

    /// Whether there is one weight for each of `rows` rows.
    ///
    /// # Errors
    ///
    /// [`InputError::Length`] otherwise, naming the weights.
    pub const fn check_rows(&self, rows: usize) -> Result<(), InputError> {
        check::same_length("weights", rows, self.values.len())
    }
}

impl Deref for CaseWeights<'_> {
    type Target = [f64];

    fn deref(&self) -> &[f64] {
        self.values
    }
}

impl AsRef<[f64]> for CaseWeights<'_> {
    fn as_ref(&self) -> &[f64] {
        self.values
    }
}

/// The serialised form, `{"values": [...], "kind": "prior"}`. Weights are
/// written, not read back: a column borrowed from the caller cannot be
/// deserialised, so a binding builds `CaseWeights` from its own buffer.
#[cfg(feature = "serde")]
impl serde::Serialize for CaseWeights<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("CaseWeights", 2)?;
        state.serialize_field("values", self.values)?;
        state.serialize_field("kind", &self.kind)?;
        state.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Classify;

    #[test]
    fn valid_weights_keep_the_column_and_its_kind() {
        let w = [0.5, 0.0, 2.0];
        let weights = CaseWeights::sampling(&w).unwrap();
        assert_eq!(weights.kind(), WeightKind::Sampling);
        assert_eq!(weights.len(), 3);
        assert!(!weights.is_empty());
        assert_eq!(weights.n_positive(), 2);
        assert_eq!(weights.total().to_bits(), 2.5_f64.to_bits());
        assert_eq!(weights[2], 2.0);
        assert_eq!(weights.as_slice().as_ptr(), w.as_ptr());
        assert_eq!(weights.check_rows(3), Ok(()));
        assert_eq!(weights.check_rows(4).unwrap_err().code(), "input.length");
        assert_eq!(
            CaseWeights::of_kind(&w, WeightKind::Prior),
            CaseWeights::new(&w)
        );
    }

    #[test]
    fn bad_weights_are_refused_at_their_row() {
        assert_eq!(
            CaseWeights::new(&[1.0, -0.5]),
            Err(InputError::Domain {
                what: "weights",
                value: -0.5,
                domain: "[0, inf)",
                row: Some(1)
            })
        );
        assert_eq!(
            CaseWeights::new(&[1.0, f64::NAN]).unwrap_err().code(),
            "input.not_finite"
        );
        assert_eq!(
            CaseWeights::new(&[]),
            Err(InputError::Empty { what: "weights" })
        );
        assert_eq!(
            CaseWeights::sampling(&[0.0]).unwrap_err().code(),
            "input.all_zero"
        );
        assert_eq!(
            CaseWeights::frequency(&[1.0, 2.5]),
            Err(InputError::Domain {
                what: "weights",
                value: 2.5,
                domain: "the whole numbers 0, 1, 2, ...",
                row: Some(1)
            })
        );
        assert_eq!(
            InputError::AllZero { what: "weights" }.to_string(),
            "weights are zero in every row"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn weights_serialise_with_their_kind() {
        let weights = CaseWeights::frequency(&[1.0, 2.0]).unwrap();
        assert_eq!(
            serde_json::to_string(&weights).unwrap(),
            r#"{"values":[1.0,2.0],"kind":"frequency"}"#
        );
    }
}

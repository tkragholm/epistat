//! A covariance matrix that knows its dimension.
//!
//! [`Covariance`] stores a `dim x dim` matrix row by row, each entry `None`
//! where the method leaves the covariance undefined. Reading it needs no
//! index arithmetic: [`Covariance::get`] takes a row and a column, and
//! [`Covariance::standard_errors`] and [`Covariance::quadratic`] give the
//! standard errors and the variance of a linear combination.
//!
//! # Formulas
//!
//! With `V` the matrix and `w` a vector of `dim` weights,
//!
//! ```text
//! se_i        = sqrt(V_ii)                         for V_ii >= 0
//! quadratic   = sum_i sum_j w_i V_ij w_j
//! ```
//!
//! The quadratic form is accumulated in one sum, `i` outer and `j` inner, in
//! that fixed order, each term formed as `(w_i * V_ij) * w_j`, without fused
//! multiply-add. Terms with `w_i = 0` or `w_j = 0` are skipped, so an
//! undefined entry that a zero weight multiplies does not make the result
//! undefined. The skipped terms are exact zeros, so skipping them changes no
//! value other than the sign of a zero result.
//!
//! The matrix is symmetric by convention. The constructor does not check
//! symmetry, because a covariance formed as `A V A'` by a matrix product can
//! differ from its transpose in the last bits; [`Covariance::is_symmetric`]
//! checks it exactly when a caller needs it.

use crate::check::InputError;

/// A `dim x dim` covariance matrix, row-major, `None` where undefined.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(try_from = "CovarianceParts", into = "CovarianceParts")
)]
pub struct Covariance {
    dim: usize,
    values: Vec<Option<f64>>,
}

/// The serialised form, `{"dim": k, "values": [...]}` with `null` for an
/// undefined entry.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct CovarianceParts {
    dim: usize,
    values: Vec<Option<f64>>,
}

impl TryFrom<CovarianceParts> for Covariance {
    type Error = InputError;

    fn try_from(parts: CovarianceParts) -> Result<Self, InputError> {
        Self::new(parts.dim, parts.values)
    }
}

impl From<Covariance> for CovarianceParts {
    fn from(covariance: Covariance) -> Self {
        Self {
            dim: covariance.dim,
            values: covariance.values,
        }
    }
}

impl Covariance {
    /// The matrix with the row-major `values`, `None` for an undefined entry.
    ///
    /// # Errors
    ///
    /// [`InputError::TooLarge`] when `dim * dim` overflows,
    /// [`InputError::Length`] unless there are `dim * dim` values, and
    /// [`InputError::NotFinite`] for a defined entry that is NaN or infinite,
    /// with its row-major position as the row.
    pub fn new(dim: usize, values: Vec<Option<f64>>) -> Result<Self, InputError> {
        let Some(size) = dim.checked_mul(dim) else {
            return Err(InputError::TooLarge {
                what: "covariance",
                limit: usize::MAX as u64,
                found: u64::MAX,
            });
        };
        crate::check::same_length("covariance", size, values.len())?;
        if let Some(row) = values
            .iter()
            .position(|value| value.is_some_and(|v| !v.is_finite()))
        {
            return Err(InputError::NotFinite {
                what: "covariance",
                row,
            });
        }
        Ok(Self { dim, values })
    }

    /// The matrix with the row-major `values`, every entry defined.
    ///
    /// # Errors
    ///
    /// As [`Covariance::new`].
    pub fn from_values(dim: usize, values: &[f64]) -> Result<Self, InputError> {
        Self::new(dim, values.iter().copied().map(Some).collect())
    }

    /// The number of rows, equal to the number of columns.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Entry `(i, j)`, `None` when it is undefined or outside the matrix.
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> Option<f64> {
        if i < self.dim && j < self.dim {
            self.values[i * self.dim + j]
        } else {
            None
        }
    }

    /// The variance of each coordinate, the diagonal.
    #[must_use]
    pub fn variances(&self) -> Vec<Option<f64>> {
        (0..self.dim).map(|i| self.get(i, i)).collect()
    }

    /// The standard error of each coordinate: the square root of its
    /// variance, `None` where the variance is undefined or negative.
    #[must_use]
    pub fn standard_errors(&self) -> Vec<Option<f64>> {
        (0..self.dim)
            .map(|i| self.get(i, i).filter(|v| *v >= 0.0).map(f64::sqrt))
            .collect()
    }

    /// The quadratic form `w' V w`, the variance of the combination
    /// `sum_i w_i x_i`, in the fixed order of the module documentation.
    ///
    /// `None` when `w` does not have `dim` entries, when a weight is not
    /// finite, or when an entry that a non-zero weight pair multiplies is
    /// undefined.
    #[must_use]
    pub fn quadratic(&self, w: &[f64]) -> Option<f64> {
        if w.len() != self.dim || w.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let mut total = 0.0;
        for (i, &wi) in w.iter().enumerate() {
            if wi == 0.0 {
                continue;
            }
            for (j, &wj) in w.iter().enumerate() {
                if wj == 0.0 {
                    continue;
                }
                total += wi * self.values[i * self.dim + j]? * wj;
            }
        }
        Some(total)
    }

    /// Whether the matrix equals its transpose exactly, undefined entries
    /// matching undefined ones.
    #[must_use]
    pub fn is_symmetric(&self) -> bool {
        (0..self.dim).all(|i| {
            (0..i).all(|j| {
                let (a, b) = (self.values[i * self.dim + j], self.values[j * self.dim + i]);
                a.map(f64::to_bits) == b.map(f64::to_bits)
            })
        })
    }

    /// The row-major entries.
    #[must_use]
    pub fn as_slice(&self) -> &[Option<f64>] {
        &self.values
    }

    /// The row-major entries, consuming the matrix.
    #[must_use]
    pub fn into_values(self) -> Vec<Option<f64>> {
        self.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(value: Option<f64>) -> Option<u64> {
        value.map(f64::to_bits)
    }

    #[test]
    fn the_shape_is_checked() {
        assert!(Covariance::new(2, vec![Some(1.0); 4]).is_ok());
        assert_eq!(
            Covariance::new(2, vec![Some(1.0); 3]),
            Err(InputError::Length {
                what: "covariance",
                expected: 4,
                found: 3
            })
        );
        assert_eq!(
            Covariance::from_values(2, &[1.0, f64::NAN, 0.0, 1.0]),
            Err(InputError::NotFinite {
                what: "covariance",
                row: 1
            })
        );
        assert!(Covariance::new(usize::MAX, Vec::new()).is_err());
        let empty = Covariance::new(0, Vec::new()).unwrap();
        assert_eq!(empty.dim(), 0);
        assert_eq!(bits(empty.quadratic(&[])), Some(0.0_f64.to_bits()));
    }

    #[test]
    fn entries_and_standard_errors_read_without_index_arithmetic() {
        let v = Covariance::new(2, vec![Some(4.0), Some(1.0), Some(1.0), Some(-1.0)]).unwrap();
        assert_eq!(bits(v.get(0, 1)), Some(1.0_f64.to_bits()));
        assert_eq!(v.get(2, 0), None);
        let se = v.standard_errors();
        assert_eq!(bits(se[0]), Some(2.0_f64.to_bits()));
        assert_eq!(se[1], None);
        assert_eq!(v.variances().len(), 2);
        assert!(v.is_symmetric());
    }

    #[test]
    fn the_quadratic_form_is_a_contrast_variance() {
        let v =
            Covariance::from_values(3, &[4.0, 1.0, 0.5, 1.0, 9.0, 0.25, 0.5, 0.25, 1.0]).unwrap();
        // var(x0 - x1) = 4 + 9 - 2.
        assert_eq!(
            bits(v.quadratic(&[1.0, -1.0, 0.0])),
            Some(11.0_f64.to_bits())
        );
        assert_eq!(v.quadratic(&[1.0, 1.0]), None);
        assert_eq!(v.quadratic(&[1.0, f64::NAN, 0.0]), None);
        // The fixed order: i outer, j inner, ((w_i * V_ij) * w_j).
        let w = [0.1, 0.7, -0.3];
        let mut expected = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                expected += w[i] * v.get(i, j).unwrap() * w[j];
            }
        }
        assert_eq!(bits(v.quadratic(&w)), Some(expected.to_bits()));
    }

    #[test]
    fn an_undefined_entry_matters_only_under_non_zero_weights() {
        let v = Covariance::new(2, vec![Some(2.0), None, None, Some(3.0)]).unwrap();
        assert_eq!(bits(v.quadratic(&[1.0, 0.0])), Some(2.0_f64.to_bits()));
        assert_eq!(v.quadratic(&[1.0, 1.0]), None);
        assert!(v.is_symmetric());
        let asymmetric = Covariance::new(2, vec![Some(2.0), Some(1.0), None, Some(3.0)]).unwrap();
        assert!(!asymmetric.is_symmetric());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_writes_null_for_undefined_entries_and_validates() {
        let v = Covariance::new(2, vec![Some(1.0), None, None, Some(2.0)]).unwrap();
        let text = serde_json::to_string(&v).unwrap();
        assert_eq!(text, r#"{"dim":2,"values":[1.0,null,null,2.0]}"#);
        assert_eq!(serde_json::from_str::<Covariance>(&text).unwrap(), v);
        assert!(serde_json::from_str::<Covariance>(r#"{"dim":2,"values":[1.0]}"#).is_err());
    }
}

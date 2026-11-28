//! A rectangular matrix for results, stored row by row.
//!
//! A result that is a table of numbers (a coefficient per outcome and
//! column, a survival probability per time and group, a draw per replicate
//! and estimate) is a [`Matrix`]. It knows its shape, so reading it needs no
//! index arithmetic, and it belongs to this crate, so a result does not
//! depend on the version of a linear algebra library. Inputs stay faer's
//! zero-copy `MatRef`; epistat-linalg converts between the two.
//!
//! # Layout
//!
//! Entries are stored row-major: entry `(i, j)` is at `i * columns + j`, as in
//! [`Covariance`]. A result's rows are its records (a time, a replicate, an
//! outcome), so a row is one contiguous slice and the matrix prints and
//! serialises row by row. A column is read with [`Matrix::column`], which
//! steps through the rows. [`Matrix::from_column_major`] and
//! [`Matrix::to_column_major`] convert from and to the layout of faer, R and
//! Fortran.
//!
//! # Entries
//!
//! An entry is an `f64`, or an `Option<f64>` where a result leaves entries
//! undefined. A defined entry must be finite: undefined numbers are `None`,
//! never NaN, as everywhere in the workspace.
//!
//! ```
//! use epistat_core::Matrix;
//!
//! // Two times by three groups.
//! let surv = Matrix::new(2, 3, vec![0.9, 0.8, 0.95, 0.7, 0.6, 0.9])?;
//! assert_eq!(surv.shape(), (2, 3));
//! assert_eq!(surv.get(1, 2), Some(0.9));
//! assert_eq!(surv.row(0), Some(&[0.9, 0.8, 0.95][..]));
//! assert_eq!(surv.column(1).collect::<Vec<_>>(), [0.8, 0.6]);
//! assert_eq!(surv.to_column_major(), [0.9, 0.7, 0.8, 0.6, 0.95, 0.9]);
//! assert!(Matrix::new(1, 2, vec![1.0, f64::NAN]).is_err());
//! # Ok::<(), epistat_core::InputError>(())
//! ```

use core::fmt;

use crate::check::{self, InputError};
use crate::covariance::Covariance;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for Option<f64> {}
}

/// An entry type of a [`Matrix`]: `f64`, or `Option<f64>` with `None` for an
/// undefined entry. Sealed: no other type is an entry.
pub trait Entry: sealed::Sealed + Copy + PartialEq + fmt::Debug {
    /// Whether the entry may be stored: finite, or undefined.
    fn is_valid(&self) -> bool;

    /// The entry as an optional number.
    fn value(self) -> Option<f64>;
}

impl Entry for f64 {
    fn is_valid(&self) -> bool {
        self.is_finite()
    }

    fn value(self) -> Option<f64> {
        Some(self)
    }
}

impl Entry for Option<f64> {
    fn is_valid(&self) -> bool {
        self.is_none_or(f64::is_finite)
    }

    fn value(self) -> Option<f64> {
        self
    }
}

/// A `rows x columns` matrix, row-major, of finite `f64` or of
/// `Option<f64>`.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<T: Entry = f64> {
    rows: usize,
    columns: usize,
    values: Vec<T>,
}

impl<T: Entry> Matrix<T> {
    /// The matrix with the row-major `values`.
    ///
    /// # Errors
    ///
    /// [`InputError::TooLarge`] when `rows * columns` overflows,
    /// [`InputError::Length`] unless there are `rows * columns` values, and
    /// [`InputError::NotFinite`] for an entry that is NaN or infinite, with
    /// its row and column.
    pub fn new(rows: usize, columns: usize, values: Vec<T>) -> Result<Self, InputError> {
        let size = size(rows, columns)?;
        check::same_length("matrix", size, values.len())?;
        if let Some(at) = values.iter().position(|value| !value.is_valid()) {
            return Err(InputError::NotFinite {
                what: "matrix",
                row: at / columns,
                column: Some(at % columns),
            });
        }
        Ok(Self {
            rows,
            columns,
            values,
        })
    }

    /// The matrix whose columns follow each other in `values`, the layout of
    /// faer, R and Fortran.
    ///
    /// # Errors
    ///
    /// As [`Matrix::new`].
    pub fn from_column_major(
        rows: usize,
        columns: usize,
        values: &[T],
    ) -> Result<Self, InputError> {
        let size = size(rows, columns)?;
        check::same_length("matrix", size, values.len())?;
        Self::from_fn(rows, columns, |i, j| values[j * rows + i])
    }

    /// The matrix from its rows, each a vector of the same length.
    ///
    /// # Errors
    ///
    /// [`InputError::Length`] for a row of another length than the first,
    /// and as [`Matrix::new`].
    pub fn from_rows(rows: &[Vec<T>]) -> Result<Self, InputError> {
        let columns = rows.first().map_or(0, Vec::len);
        let mut values = Vec::with_capacity(rows.len().saturating_mul(columns));
        for row in rows {
            check::same_length("matrix row", columns, row.len())?;
            values.extend_from_slice(row);
        }
        Self::new(rows.len(), columns, values)
    }

    /// The matrix whose entry `(i, j)` is `entry(i, j)`, filled row by row.
    ///
    /// # Errors
    ///
    /// As [`Matrix::new`].
    pub fn from_fn(
        rows: usize,
        columns: usize,
        mut entry: impl FnMut(usize, usize) -> T,
    ) -> Result<Self, InputError> {
        let size = size(rows, columns)?;
        let mut values = Vec::with_capacity(size);
        for i in 0..rows {
            for j in 0..columns {
                values.push(entry(i, j));
            }
        }
        Self::new(rows, columns, values)
    }

    /// The number of rows.
    #[must_use]
    pub const fn n_rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    #[must_use]
    pub const fn n_columns(&self) -> usize {
        self.columns
    }

    /// `(rows, columns)`.
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }

    /// Whether the matrix has no entries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Entry `(i, j)`, `None` outside the matrix.
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> Option<T> {
        (i < self.rows && j < self.columns).then(|| self.values[i * self.columns + j])
    }

    /// Row `i`, `None` outside the matrix.
    #[must_use]
    pub fn row(&self, i: usize) -> Option<&[T]> {
        (i < self.rows).then(|| &self.values[i * self.columns..(i + 1) * self.columns])
    }

    /// The rows in order, each a slice.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = &[T]> + '_ {
        (0..self.rows).map(|i| &self.values[i * self.columns..(i + 1) * self.columns])
    }

    /// The entries of column `j` from the first row to the last; empty for a
    /// column outside the matrix.
    pub fn column(&self, j: usize) -> impl ExactSizeIterator<Item = T> + '_ {
        let rows = if j < self.columns { self.rows } else { 0 };
        (0..rows).map(move |i| self.values[i * self.columns + j])
    }

    /// The columns in order, each an iterator over its rows.
    pub fn columns(&self) -> impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = T> + '_> {
        (0..self.columns).map(move |j| self.column(j))
    }

    /// The transpose, a `columns x rows` matrix.
    #[must_use]
    pub fn transpose(&self) -> Self {
        Self {
            rows: self.columns,
            columns: self.rows,
            values: self.to_column_major(),
        }
    }

    /// The row-major entries.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.values
    }

    /// The row-major entries, consuming the matrix.
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.values
    }

    /// The entries column after column, the layout of faer, R and Fortran.
    #[must_use]
    pub fn to_column_major(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.values.len());
        for j in 0..self.columns {
            out.extend(self.column(j));
        }
        out
    }

    /// The rows as vectors.
    #[must_use]
    pub fn to_rows(&self) -> Vec<Vec<T>> {
        self.rows().map(<[T]>::to_vec).collect()
    }
}

impl Matrix<f64> {
    /// The same matrix with each entry defined, as `Option<f64>`.
    #[must_use]
    pub fn to_optional(&self) -> Matrix<Option<f64>> {
        Matrix {
            rows: self.rows,
            columns: self.columns,
            values: self.values.iter().copied().map(Some).collect(),
        }
    }
}

impl Matrix<Option<f64>> {
    /// The same matrix as `f64`, `None` when an entry is undefined.
    #[must_use]
    pub fn to_defined(&self) -> Option<Matrix<f64>> {
        Some(Matrix {
            rows: self.rows,
            columns: self.columns,
            values: self.values.iter().copied().collect::<Option<Vec<f64>>>()?,
        })
    }
}

fn size(rows: usize, columns: usize) -> Result<usize, InputError> {
    rows.checked_mul(columns).ok_or(InputError::TooLarge {
        what: "matrix",
        limit: u64::try_from(usize::MAX).unwrap_or(u64::MAX),
        found: u64::MAX,
    })
}

impl From<Covariance> for Matrix<Option<f64>> {
    /// The covariance as a square matrix of optional entries.
    fn from(covariance: Covariance) -> Self {
        let dim = covariance.dim();
        Self {
            rows: dim,
            columns: dim,
            values: covariance.into_values(),
        }
    }
}

impl TryFrom<Matrix<Option<f64>>> for Covariance {
    type Error = InputError;

    /// The covariance with the matrix's entries.
    ///
    /// # Errors
    ///
    /// [`InputError::Length`] for a matrix that is not square.
    fn try_from(matrix: Matrix<Option<f64>>) -> Result<Self, InputError> {
        check::same_length("covariance columns", matrix.rows, matrix.columns)?;
        Self::new(matrix.rows, matrix.values)
    }
}

impl TryFrom<Matrix<f64>> for Covariance {
    type Error = InputError;

    /// The covariance with the matrix's entries, all defined.
    ///
    /// # Errors
    ///
    /// [`InputError::Length`] for a matrix that is not square.
    fn try_from(matrix: Matrix<f64>) -> Result<Self, InputError> {
        check::same_length("covariance columns", matrix.rows, matrix.columns)?;
        Self::from_values(matrix.rows, &matrix.values)
    }
}

impl<T: Entry> TryFrom<Vec<Vec<T>>> for Matrix<T> {
    type Error = InputError;

    /// The matrix from its rows; see [`Matrix::from_rows`].
    fn try_from(rows: Vec<Vec<T>>) -> Result<Self, InputError> {
        Self::from_rows(&rows)
    }
}

impl<T: Entry> From<Matrix<T>> for Vec<Vec<T>> {
    fn from(matrix: Matrix<T>) -> Self {
        matrix.to_rows()
    }
}

impl<T: Entry> fmt::Display for Matrix<T> {
    /// The matrix as R prints one, with rows and columns numbered from 1,
    /// four significant digits, or the decimals of a precision such as
    /// `{:.2}`, and `NA` for an undefined entry.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let decimals = f.precision();
        let header: Vec<String> = (1..=self.columns).map(|j| format!("[,{j}]")).collect();
        let header: Vec<&str> = header.iter().map(String::as_str).collect();
        let rows: Vec<(String, Vec<String>)> = self
            .rows()
            .enumerate()
            .map(|(i, row)| {
                let cells = row
                    .iter()
                    .map(|entry| crate::format::optional(entry.value(), 4, decimals))
                    .collect();
                (format!("[{},]", i + 1), cells)
            })
            .collect();
        crate::format::table(f, &header, &rows)
    }
}

#[cfg(feature = "serde")]
mod serde_form {
    use super::{Entry, Matrix};

    /// `{"rows": r, "columns": c, "values": [...]}`, the values row by row,
    /// `null` for an undefined entry.
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Parts<T> {
        rows: usize,
        columns: usize,
        values: Vec<T>,
    }

    #[derive(serde::Serialize)]
    struct PartsRef<'a, T> {
        rows: usize,
        columns: usize,
        values: &'a [T],
    }

    impl<T: Entry + serde::Serialize> serde::Serialize for Matrix<T> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            PartsRef {
                rows: self.rows,
                columns: self.columns,
                values: &self.values,
            }
            .serialize(serializer)
        }
    }

    impl<'de, T: Entry + serde::Deserialize<'de>> serde::Deserialize<'de> for Matrix<T> {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let parts = Parts::<T>::deserialize(deserializer)?;
            Self::new(parts.rows, parts.columns, parts.values).map_err(serde::de::Error::custom)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_matrix_reads_by_row_column_and_entry() {
        let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!((m.n_rows(), m.n_columns()), (2, 3));
        assert_eq!(m.get(0, 2), Some(3.0));
        assert_eq!(m.get(2, 0), None);
        assert_eq!(m.row(1), Some(&[4.0, 5.0, 6.0][..]));
        assert_eq!(m.row(2), None);
        assert_eq!(m.rows().len(), 2);
        assert_eq!(m.column(0).collect::<Vec<_>>(), [1.0, 4.0]);
        assert_eq!(m.column(3).count(), 0);
        let columns: Vec<Vec<f64>> = m.columns().map(Iterator::collect).collect();
        assert_eq!(columns, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
        assert_eq!(m.transpose().row(0), Some(&[1.0, 4.0][..]));
        assert_eq!(m.transpose().transpose(), m);
        assert!(!m.is_empty());
        assert!(Matrix::<f64>::new(0, 3, Vec::new()).unwrap().is_empty());
    }

    #[test]
    fn the_layouts_convert_both_ways() {
        let column_major = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let m = Matrix::from_column_major(2, 3, &column_major).unwrap();
        assert_eq!(m.as_slice(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.to_column_major(), column_major);
        let rows = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        assert_eq!(Matrix::try_from(rows.clone()).unwrap(), m);
        assert_eq!(Vec::<Vec<f64>>::from(m.clone()), rows);
        assert_eq!(m.clone().into_vec().len(), 6);
        assert_eq!(
            Matrix::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f64).unwrap(),
            m
        );
    }

    #[test]
    fn bad_shapes_and_values_are_refused() {
        assert_eq!(
            Matrix::new(2, 2, vec![1.0; 3]),
            Err(InputError::Length {
                what: "matrix",
                expected: 4,
                found: 3
            })
        );
        assert_eq!(
            Matrix::new(2, 2, vec![1.0, 1.0, 1.0, f64::INFINITY]),
            Err(InputError::NotFinite {
                what: "matrix",
                row: 1,
                column: Some(1)
            })
        );
        assert!(Matrix::<f64>::new(usize::MAX, 2, Vec::new()).is_err());
        assert!(Matrix::from_rows(&[vec![1.0], vec![1.0, 2.0]]).is_err());
        assert!(Matrix::new(1, 2, vec![None, Some(f64::NAN)]).is_err());
        assert!(Matrix::from_column_major(2, 2, &[1.0]).is_err());
    }

    #[test]
    fn optional_entries_and_covariances_convert() {
        let m = Matrix::new(2, 2, vec![Some(4.0), None, None, Some(1.0)]).unwrap();
        assert_eq!(m.to_defined(), None);
        let v = Covariance::try_from(m.clone()).unwrap();
        assert_eq!(v.get(0, 0), Some(4.0));
        assert_eq!(Matrix::from(v), m);
        let full = Matrix::new(2, 2, vec![1.0, 0.5, 0.5, 2.0]).unwrap();
        assert_eq!(full.to_optional().to_defined(), Some(full.clone()));
        assert_eq!(Covariance::try_from(full).unwrap().dim(), 2);
        assert!(Covariance::try_from(Matrix::new(1, 2, vec![1.0, 2.0]).unwrap()).is_err());
    }

    #[test]
    fn a_matrix_prints_as_r_prints_one() {
        let m = Matrix::new(2, 2, vec![Some(0.5), None, Some(-1.25), Some(3.0)]).unwrap();
        assert_eq!(
            m.to_string(),
            "        [,1]   [,2]\n[1,]  0.5000     NA\n[2,]  -1.250  3.000"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_writes_the_shape_and_validates() {
        let m = Matrix::new(1, 2, vec![Some(1.0), None]).unwrap();
        let text = serde_json::to_string(&m).unwrap();
        assert_eq!(text, r#"{"rows":1,"columns":2,"values":[1.0,null]}"#);
        assert_eq!(
            serde_json::from_str::<Matrix<Option<f64>>>(&text).unwrap(),
            m
        );
        assert!(
            serde_json::from_str::<Matrix>(r#"{"rows":2,"columns":2,"values":[1.0]}"#).is_err()
        );
    }
}

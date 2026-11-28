//! Conversions between epistat-core's result types and faer's matrices.
//!
//! Results hold [`Matrix`] and [`Covariance`], which belong to epistat-core,
//! so that no result depends on faer's version. Estimators compute with faer;
//! these functions copy between the two at the boundary. Each copy visits the
//! entries once and changes no value.
//!
//! ```
//! use epistat_core::{Covariance, Matrix};
//! use epistat_linalg::{convert, mat};
//!
//! let beta = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
//! let result: Matrix = convert::to_matrix(beta.as_ref())?;
//! assert_eq!(result.row(1), Some(&[3.0, 4.0][..]));
//! assert_eq!(convert::to_mat(&result), beta);
//!
//! let v = convert::covariance_from_mat(mat![[2.0, 0.5], [0.5, 1.0]].as_ref())?;
//! assert_eq!(v.get(0, 1), Some(0.5));
//! assert_eq!(convert::covariance_to_mat(&v), Some(mat![[2.0, 0.5], [0.5, 1.0]]));
//! # Ok::<(), epistat_linalg::Error>(())
//! ```

use epistat_core::{Covariance, Matrix, check};
use faer::{Mat, MatRef};

use crate::Error;

/// The faer matrix `m` as a [`Matrix`].
///
/// # Errors
///
/// [`Error::Input`] with [`InputError::NotFinite`](epistat_core::InputError::NotFinite) for an entry that is NaN
/// or infinite, naming its row and column.
pub fn to_matrix(m: MatRef<'_, f64>) -> Result<Matrix, Error> {
    Ok(Matrix::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)])?)
}

/// The [`Matrix`] `m` as a faer matrix.
#[must_use]
pub fn to_mat(m: &Matrix) -> Mat<f64> {
    Mat::from_fn(m.n_rows(), m.n_columns(), |i, j| {
        m.as_slice()[i * m.n_columns() + j]
    })
}

/// The square faer matrix `m` as a [`Covariance`], every entry defined.
///
/// # Errors
///
/// [`Error::Input`] with [`InputError::Length`](epistat_core::InputError::Length) for a matrix that is not
/// square, and with [`InputError::NotFinite`](epistat_core::InputError::NotFinite) for an entry that is NaN or
/// infinite.
pub fn covariance_from_mat(m: MatRef<'_, f64>) -> Result<Covariance, Error> {
    check::same_length("covariance columns", m.nrows(), m.ncols())?;
    let dim = m.nrows();
    let values: Vec<Option<f64>> = (0..dim * dim)
        .map(|k| Some(m[(k / dim, k % dim)]))
        .collect();
    Ok(Covariance::new(dim, values)?)
}

/// The [`Covariance`] `c` as a faer matrix, `None` when an entry is
/// undefined.
#[must_use]
pub fn covariance_to_mat(c: &Covariance) -> Option<Mat<f64>> {
    let dim = c.dim();
    let values = c.as_slice();
    if values.iter().any(Option::is_none) {
        return None;
    }
    Some(Mat::from_fn(dim, dim, |i, j| {
        values[i * dim + j].unwrap_or_default()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat;
    use epistat_core::InputError;

    #[test]
    fn matrices_round_trip_bit_for_bit() {
        let m = mat![[0.1 + 0.2, -0.0], [1e-300, 7.0]];
        let back = to_mat(&to_matrix(m.as_ref()).unwrap());
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(back[(i, j)].to_bits(), m[(i, j)].to_bits());
            }
        }
        let empty = Mat::<f64>::zeros(0, 3);
        assert_eq!(to_matrix(empty.as_ref()).unwrap().shape(), (0, 3));
    }

    #[test]
    fn non_finite_and_non_square_input_is_refused() {
        let m = mat![[1.0, f64::NAN]];
        assert!(matches!(
            to_matrix(m.as_ref()),
            Err(Error::Input(InputError::NotFinite {
                row: 0,
                column: Some(1),
                ..
            }))
        ));
        assert!(matches!(
            covariance_from_mat(mat![[1.0, 2.0]].as_ref()),
            Err(Error::Input(InputError::Length { .. }))
        ));
        let undefined = Covariance::new(1, vec![None]).unwrap();
        assert_eq!(covariance_to_mat(&undefined), None);
    }
}

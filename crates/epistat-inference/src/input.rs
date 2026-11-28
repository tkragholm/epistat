//! The covariance matrices the tests and bands take.

use epistat_core::{Covariance, InputError, Matrix};
use epistat_linalg::{Mat, MatRef};

use crate::Error;

pub(crate) mod sealed {
    use epistat_linalg::Mat;

    use crate::Error;

    /// The matrix as faer reads it.
    pub trait Sealed {
        /// A copy of the matrix; an undefined entry of a `Covariance` is
        /// [`InputError::NotFinite`](epistat_core::InputError::NotFinite)
        /// with its row and column.
        fn faer(self) -> Result<Mat<f64>, Error>;
    }
}

/// A covariance matrix in one of the forms the workspace hands out: a faer
/// [`MatRef`] or `&Mat`, an [`epistat_core::Covariance`] such as
/// [`Influence::variance`](crate::Influence::variance) returns, or an
/// [`epistat_core::Matrix`].
///
/// [`wald`](crate::wald::wald) and
/// [`sup_t_from_covariance`](crate::simultaneous::sup_t_from_covariance) take
/// any of them, so a covariance from one crate goes into a test from another
/// without a copy by hand. An undefined entry of a `Covariance` is
/// [`InputError::NotFinite`] with its row and column. The trait is sealed.
pub trait CovarianceInput: sealed::Sealed {}

impl<T: sealed::Sealed> CovarianceInput for T {}

impl sealed::Sealed for MatRef<'_, f64> {
    fn faer(self) -> Result<Mat<f64>, Error> {
        Ok(self.to_owned())
    }
}

impl sealed::Sealed for &Mat<f64> {
    fn faer(self) -> Result<Mat<f64>, Error> {
        Ok(self.clone())
    }
}

impl sealed::Sealed for &Covariance {
    fn faer(self) -> Result<Mat<f64>, Error> {
        let dim = self.dim();
        if let Some(at) = self.as_slice().iter().position(Option::is_none) {
            return Err(InputError::NotFinite {
                what: "covariance",
                row: at / dim,
                column: Some(at % dim),
            }
            .into());
        }
        Ok(Mat::from_fn(dim, dim, |i, j| {
            self.get(i, j).unwrap_or_default()
        }))
    }
}

impl sealed::Sealed for &Matrix {
    fn faer(self) -> Result<Mat<f64>, Error> {
        Ok(epistat_linalg::convert::to_mat(self))
    }
}

#[cfg(test)]
mod tests {
    use super::sealed::Sealed;
    use super::*;

    #[test]
    fn every_form_reads_the_same_entries() {
        let m = epistat_linalg::mat![[2.0, 0.5], [0.5, 1.0]];
        let c = Covariance::from_values(2, &[2.0, 0.5, 0.5, 1.0]).unwrap();
        let r = Matrix::from_rows(&[vec![2.0, 0.5], vec![0.5, 1.0]]).unwrap();
        for read in [
            m.as_ref().faer().unwrap(),
            (&m).faer().unwrap(),
            (&c).faer().unwrap(),
            (&r).faer().unwrap(),
        ] {
            assert_eq!(read, m);
        }
    }

    #[test]
    fn an_undefined_entry_is_named() {
        let c = Covariance::new(2, vec![Some(1.0), None, None, Some(1.0)]).unwrap();
        assert!(matches!(
            (&c).faer(),
            Err(Error::Input(InputError::NotFinite {
                row: 0,
                column: Some(1),
                ..
            }))
        ));
    }
}

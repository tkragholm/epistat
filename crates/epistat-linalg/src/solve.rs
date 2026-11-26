//! faer's own factorisations, for callers that solve with one of them
//! directly rather than through the fallback sequence of [`solve`](crate::solve()).
//!
//! Each function makes the one faer call its name says, factorisation and
//! solve, on the crate's one-thread pool, so the result is the one faer gives
//! with one thread and does not depend on the caller's thread count. A caller
//! that made the same call on a pool of one thread gets the same bits; on a
//! larger pool faer could split the work, which these functions do not let it
//! do.
//!
//! - [`lu_partial`]: LU with partial pivoting, `P A = L U`, faer's
//!   `partial_piv_lu`. It factorises every square matrix, so a singular one
//!   shows as a solution that is not finite, which is returned as it is.
//! - [`llt`]: the Cholesky factorisation `A = L L'` of the lower triangle,
//!   faer's `llt`, or `None` when a pivot is not positive.
//! - [`least_squares`]: the least-squares solution of `A x = b` by the QR
//!   factorisation with column pivoting, faer's `col_piv_qr`.
//!
//! These return faer's result without the checks of [`solve`](crate::solve()):
//! a solution that is not finite is not refused. They are for code that
//! applies its own rule to the solution.

use faer::prelude::{Solve, SolveLstsq};
use faer::{Mat, MatRef, Side};

use epistat_core::InputError;

use crate::Error;
use crate::error::check_length;
use crate::factor::on_one_thread;

fn check_square(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Result<(), Error> {
    check_length("columns of a", a.ncols(), a.nrows())?;
    check_length("rows of b", b.nrows(), a.nrows())
}

/// `x` with `A x = b`, by LU with partial pivoting.
///
/// # Errors
///
/// [`Error::Input`] when `a` is not square or `b` does not have one row per
/// row of `a`.
pub fn lu_partial(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Result<Mat<f64>, Error> {
    check_square(a, b)?;
    Ok(on_one_thread(|| a.partial_piv_lu().solve(b)))
}

/// `x` with `A x = b` by the Cholesky factorisation of the lower triangle of
/// `a`, or `None` when the factorisation meets a pivot that is not positive.
///
/// # Errors
///
/// [`Error::Input`] when `a` is not square or `b` does not have one row per
/// row of `a`.
pub fn llt(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Result<Option<Mat<f64>>, Error> {
    check_square(a, b)?;
    Ok(on_one_thread(|| {
        a.llt(Side::Lower).ok().map(|factor| factor.solve(b))
    }))
}

/// The `x` that minimises `|A x - b|` in the Euclidean norm, by the QR
/// factorisation of `a` with column pivoting.
///
/// # Errors
///
/// [`Error::Input`] when `a` has fewer rows than columns or `b` does not have
/// one row per row of `a`.
pub fn least_squares(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Result<Mat<f64>, Error> {
    check_length("rows of b", b.nrows(), a.nrows())?;
    if a.nrows() < a.ncols() {
        return Err(Error::Input(InputError::TooLarge {
            what: "columns of a",
            limit: a.nrows() as u64,
            found: a.ncols() as u64,
        }));
    }
    Ok(on_one_thread(|| a.col_piv_qr().solve_lstsq(b)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat;

    fn bits(x: &Mat<f64>) -> Vec<u64> {
        (0..x.ncols())
            .flat_map(|j| (0..x.nrows()).map(move |i| (i, j)))
            .map(|(i, j)| x[(i, j)].to_bits())
            .collect()
    }

    fn system() -> (Mat<f64>, Mat<f64>) {
        let n = 9;
        let x = Mat::from_fn(30, n, |i, j| ((i * 7 + j * 13) % 17) as f64 / 17.0 - 0.4);
        let a = x.transpose() * &x + Mat::<f64>::identity(n, n);
        let b = Mat::from_fn(n, 2, |i, j| (i + 2 * j) as f64 - 3.0);
        (a, b)
    }

    #[test]
    fn each_wrapper_is_the_faer_call_on_one_thread() {
        let (a, b) = system();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let lu = pool.install(|| a.partial_piv_lu().solve(&b));
        assert_eq!(
            bits(&lu_partial(a.as_ref(), b.as_ref()).unwrap()),
            bits(&lu)
        );
        let chol = pool.install(|| a.llt(Side::Lower).unwrap().solve(&b));
        assert_eq!(
            bits(&llt(a.as_ref(), b.as_ref()).unwrap().unwrap()),
            bits(&chol)
        );
        let tall = Mat::from_fn(12, 3, |i, j| {
            ((i + 1) * (j + 2)) as f64 % 5.0 + 0.1 * i as f64
        });
        let rhs = Mat::from_fn(12, 1, |i, _| i as f64);
        let qr = pool.install(|| tall.col_piv_qr().solve_lstsq(&rhs));
        assert_eq!(
            bits(&least_squares(tall.as_ref(), rhs.as_ref()).unwrap()),
            bits(&qr)
        );
    }

    #[test]
    fn llt_of_an_indefinite_matrix_is_none_and_shapes_are_checked() {
        let a = mat![[1.0, 2.0], [2.0, 1.0]];
        let b = mat![[1.0], [1.0]];
        assert!(matches!(llt(a.as_ref(), b.as_ref()), Ok(None)));
        let short = mat![[1.0]];
        assert!(matches!(
            lu_partial(a.as_ref(), short.as_ref()),
            Err(Error::Input(InputError::Length { .. }))
        ));
        let wide = Mat::<f64>::zeros(2, 3);
        assert!(matches!(
            least_squares(wide.as_ref(), b.as_ref()),
            Err(Error::Input(InputError::TooLarge { .. }))
        ));
        assert!(matches!(
            lu_partial(wide.as_ref(), b.as_ref()),
            Err(Error::Input(InputError::Length { .. }))
        ));
    }
}

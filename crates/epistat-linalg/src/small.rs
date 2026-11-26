//! Small symmetric positive definite systems on row-major slices, in a fixed
//! order of operations with no fused multiply-add, so that the bits of a
//! result are defined by the code below and not by a library's kernels.
//!
//! A matrix of side `k` is a slice of `k * k` values, row `i` at
//! `i * k..(i + 1) * k`. The factor is the lower `L` of `A = L L'` from
//! [`cholesky_lower`](crate::cholesky::cholesky_lower), with zeros above the
//! diagonal; the functions read only its lower triangle.
//!
//! # Algorithm
//!
//! [`forward_solve`] overwrites `b` with `L^-1 b` and [`backward_solve`] with
//! `L'^-1 b`, by substitution:
//!
//! ```text
//! forward:   y_i = (b_i - sum_{j<i} l_ij y_j) / l_ii,     i = 0, ..., k - 1
//! backward:  x_i = (y_i - sum_{j>i} l_ji x_j) / l_ii,     i = k - 1, ..., 0
//! ```
//!
//! each sum taken in increasing `j` as a running subtraction. Then
//!
//! - [`cholesky_solve`] is the two in turn, `A^-1 b`;
//! - [`cholesky_inverse`] solves against each column of the identity and
//!   copies the lower triangle to the upper, so `A^-1` is exactly symmetric;
//! - [`log_det_from_cholesky`] is `ln det A = 2 sum_i ln l_ii`, summed in
//!   order;
//! - [`quadratic_form_inverse`] is `x' A^-1 x = |L^-1 x|^2`, the squares
//!   added in order as the forward substitution produces them.
//!
//! The arithmetic is that of epistat-mixed's helpers and of the pre-trend
//! power code of epistat-did, so either can call these without a change in
//! bits.
//!
//! # Edge cases
//!
//! A slice of the wrong length is an error. The functions do not check the
//! diagonal of the factor: a zero gives infinities or NaN, as the division
//! does, and a factor from `cholesky_lower` with `Pivot::Strict` has a
//! positive diagonal.
//!
//! # References
//!
//! 1. Golub GH, Van Loan CF. Matrix Computations. 4th ed. Baltimore: Johns
//!    Hopkins University Press; 2013. Sections 3.1 and 4.2.

use crate::Error;
use crate::error::check_length;

fn check(l: &[f64], k: usize, b: usize) -> Result<(), Error> {
    check_length("l", l.len(), k.saturating_mul(k))?;
    check_length("b", b, k)
}

/// `L^-1 b` in place, for the row-major lower factor `l` of side `k`.
///
/// # Errors
///
/// [`Error::Input`] when `l` does not have `k * k` entries or `b` has not `k`.
pub fn forward_solve(l: &[f64], k: usize, b: &mut [f64]) -> Result<(), Error> {
    check(l, k, b.len())?;
    forward(l, k, b);
    Ok(())
}

/// `L'^-1 b` in place, for the row-major lower factor `l` of side `k`.
///
/// # Errors
///
/// As [`forward_solve`].
pub fn backward_solve(l: &[f64], k: usize, b: &mut [f64]) -> Result<(), Error> {
    check(l, k, b.len())?;
    backward(l, k, b);
    Ok(())
}

/// `A^-1 b` in place for `A = L L'`.
///
/// # Errors
///
/// As [`forward_solve`].
pub fn cholesky_solve(l: &[f64], k: usize, b: &mut [f64]) -> Result<(), Error> {
    check(l, k, b.len())?;
    forward(l, k, b);
    backward(l, k, b);
    Ok(())
}

/// `A^-1` for `A = L L'`, row-major and exactly symmetric.
///
/// # Errors
///
/// [`Error::Input`] when `l` does not have `k * k` entries.
pub fn cholesky_inverse(l: &[f64], k: usize) -> Result<Vec<f64>, Error> {
    check(l, k, k)?;
    let mut inverse = vec![0.0; k * k];
    let mut column = vec![0.0; k];
    for j in 0..k {
        column.fill(0.0);
        column[j] = 1.0;
        forward(l, k, &mut column);
        backward(l, k, &mut column);
        for i in 0..k {
            inverse[i * k + j] = column[i];
        }
    }
    for i in 0..k {
        for j in 0..i {
            let value = inverse[i * k + j];
            inverse[j * k + i] = value;
        }
    }
    Ok(inverse)
}

/// `ln det A = 2 sum_i ln l_ii` for `A = L L'`.
///
/// # Errors
///
/// [`Error::Input`] when `l` does not have `k * k` entries.
pub fn log_det_from_cholesky(l: &[f64], k: usize) -> Result<f64, Error> {
    check(l, k, k)?;
    let mut sum = 0.0;
    for i in 0..k {
        sum += l[i * k + i].ln();
    }
    Ok(2.0 * sum)
}

/// `x' A^-1 x` for `A = L L'`, as the squared norm of `L^-1 x`.
///
/// # Errors
///
/// [`Error::Input`] when `l` does not have `k * k` entries or `x` has not
/// `k`.
pub fn quadratic_form_inverse(l: &[f64], k: usize, x: &[f64]) -> Result<f64, Error> {
    check(l, k, x.len())?;
    let mut y = vec![0.0; k];
    let mut q = 0.0;
    for i in 0..k {
        let mut s = x[i];
        for j in 0..i {
            s -= l[i * k + j] * y[j];
        }
        y[i] = s / l[i * k + i];
        q += y[i] * y[i];
    }
    Ok(q)
}

fn forward(l: &[f64], k: usize, b: &mut [f64]) {
    for i in 0..k {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * k + j] * b[j];
        }
        b[i] = s / l[i * k + i];
    }
}

fn backward(l: &[f64], k: usize, b: &mut [f64]) {
    for i in (0..k).rev() {
        let mut s = b[i];
        for j in i + 1..k {
            s -= l[j * k + i] * b[j];
        }
        b[i] = s / l[i * k + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cholesky::{Pivot, cholesky_lower};

    const A: [f64; 9] = [4.0, 2.0, 0.4, 2.0, 3.0, 0.5, 0.4, 0.5, 2.0];

    /// epistat-mixed `small.rs`, `forward` and `backward`, as written there.
    fn mixed_solve(r: &[f64], p: usize, b: &mut [f64]) {
        for i in 0..p {
            let mut s = b[i];
            for k in 0..i {
                s -= r[i * p + k] * b[k];
            }
            b[i] = s / r[i * p + i];
        }
        for i in (0..p).rev() {
            let mut s = b[i];
            for k in i + 1..p {
                s -= r[k * p + i] * b[k];
            }
            b[i] = s / r[i * p + i];
        }
    }

    /// epistat-did `pretrends.rs`, `quad_form`, as written there.
    fn did_quad_form(l: &[Vec<f64>], r: &[f64]) -> f64 {
        let n = l.len();
        let mut y = vec![0.0; n];
        let mut q = 0.0;
        for i in 0..n {
            let mut s = r[i];
            for m in 0..i {
                s -= l[i][m] * y[m];
            }
            y[i] = s / l[i][i];
            q += y[i] * y[i];
        }
        q
    }

    #[test]
    fn the_solves_are_the_consumers_solves_to_the_bit() {
        let l = cholesky_lower(&A, 3, Pivot::Strict).unwrap();
        let b = [1.0, -2.0, 0.3];
        let mut ours = b;
        cholesky_solve(&l, 3, &mut ours).unwrap();
        let mut theirs = b;
        mixed_solve(&l, 3, &mut theirs);
        assert_eq!(ours.map(f64::to_bits), theirs.map(f64::to_bits));
        let rows: Vec<Vec<f64>> = l.chunks(3).map(<[f64]>::to_vec).collect();
        assert_eq!(
            quadratic_form_inverse(&l, 3, &b).unwrap().to_bits(),
            did_quad_form(&rows, &b).to_bits()
        );
        let mut split = b;
        forward_solve(&l, 3, &mut split).unwrap();
        backward_solve(&l, 3, &mut split).unwrap();
        assert_eq!(split.map(f64::to_bits), ours.map(f64::to_bits));
    }

    #[test]
    fn inverse_and_determinant() {
        let l = cholesky_lower(&A, 3, Pivot::Strict).unwrap();
        let inverse = cholesky_inverse(&l, 3).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(inverse[i * 3 + j].to_bits(), inverse[j * 3 + i].to_bits());
                let v: f64 = (0..3).map(|m| A[i * 3 + m] * inverse[m * 3 + j]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                // Entries of A and A^-1 are below 5 and the condition number
                // is below 10, so each of the three products and the solve
                // err by a few units of 2^-52 times 10: below 1e-14.
                assert!((v - want).abs() < 1e-14);
            }
        }
        let det = 4.0 * (3.0 * 2.0 - 0.25) - 2.0 * (2.0 * 2.0 - 0.2) + 0.4 * (1.0 - 1.2);
        // Three logarithms of values near 2, each within an ulp, and their sum:
        // a few units of 2^-52 on a value near 3.
        assert!((log_det_from_cholesky(&l, 3).unwrap() - f64::ln(det)).abs() < 1e-14);
    }

    #[test]
    fn lengths_are_checked() {
        let l = cholesky_lower(&A, 3, Pivot::Strict).unwrap();
        assert!(forward_solve(&l, 3, &mut [1.0; 2]).is_err());
        assert!(cholesky_inverse(&l[..8], 3).is_err());
        assert!(quadratic_form_inverse(&l, 2, &[1.0; 2]).is_err());
    }
}

//! Dense helpers: the Cholesky factor of a `p x p` matrix with its solves,
//! and the order of the operations kept fixed, with no fused multiply-add.

use epistat_linalg::cholesky::{Pivot, cholesky_lower};

/// The lower factor `R` of a positive definite row-major `p x p` matrix,
/// `None` when a pivot is not positive.
pub(crate) fn cholesky(a: &[f64], p: usize) -> Option<Vec<f64>> {
    cholesky_lower(a, p, Pivot::Strict)
}

/// `R^{-1} b` for the lower factor `r`, in place.
pub(crate) fn forward(r: &[f64], p: usize, b: &mut [f64]) {
    for i in 0..p {
        let mut s = b[i];
        for k in 0..i {
            s -= r[i * p + k] * b[k];
        }
        b[i] = s / r[i * p + i];
    }
}

/// `R^{-T} b` for the lower factor `r`, in place.
pub(crate) fn backward(r: &[f64], p: usize, b: &mut [f64]) {
    for i in (0..p).rev() {
        let mut s = b[i];
        for k in i + 1..p {
            s -= r[k * p + i] * b[k];
        }
        b[i] = s / r[i * p + i];
    }
}

/// `A^{-1}` for `A = R R'`, row-major and exactly symmetric.
pub(crate) fn inverse(r: &[f64], p: usize) -> Vec<f64> {
    let mut inv = vec![0.0; p * p];
    let mut column = vec![0.0; p];
    for j in 0..p {
        column.fill(0.0);
        column[j] = 1.0;
        forward(r, p, &mut column);
        backward(r, p, &mut column);
        for i in 0..p {
            inv[i * p + j] = column[i];
        }
    }
    for i in 0..p {
        for j in 0..i {
            let v = inv[i * p + j];
            inv[j * p + i] = v;
        }
    }
    inv
}

/// `A^{-1} b` for a symmetric positive definite `A`, `None` when `A` has no
/// Cholesky factor.
pub(crate) fn solve_symmetric(a: &[f64], p: usize, b: &[f64]) -> Option<Vec<f64>> {
    let r = cholesky(a, p)?;
    let mut x = b.to_vec();
    forward(&r, p, &mut x);
    backward(&r, p, &mut x);
    Some(x)
}

/// `ln det A = 2 Σ ln R_ii` for `A = R R'`.
pub(crate) fn log_det(r: &[f64], p: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..p {
        s += r[i * p + i].ln();
    }
    2.0 * s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_solves_invert_the_factor() {
        let a = [4.0, 2.0, 0.4, 2.0, 3.0, 0.5, 0.4, 0.5, 2.0];
        let r = cholesky(&a, 3).unwrap();
        let inv = inverse(&r, 3);
        for i in 0..3 {
            for j in 0..3 {
                let v: f64 = (0..3).map(|k| a[i * 3 + k] * inv[k * 3 + j]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((v - want).abs() < 1e-14);
            }
        }
        let det = 4.0 * (3.0 * 2.0 - 0.25) - 2.0 * (2.0 * 2.0 - 0.2) + 0.4 * (1.0 - 1.2);
        assert!((log_det(&r, 3) - f64::ln(det)).abs() < 1e-14);
        assert!(cholesky(&[1.0, 2.0, 2.0, 1.0], 2).is_none());
    }
}

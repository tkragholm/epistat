//! Gauss-Hermite quadrature for the standard normal distribution.
//!
//! [`gauss_hermite`] returns `k` nodes `z_i` and weights `w_i` with
//!
//! ```text
//! ∫ f(z) φ(z) dz ≈ Σ_i w_i f(z_i),     φ(z) = exp(-z²/2) / sqrt(2π)
//! ```
//!
//! exact for polynomials of degree up to `2k - 1`. These are the nodes and
//! weights of `lme4::GHrule(k)`, the "probabilists'" form, whose nodes are
//! `sqrt(2)` times those of the physicists' rule for `exp(-x²)`.
//!
//! # Algorithm
//!
//! The orthonormal polynomials of `φ` satisfy
//!
//! ```text
//! p_0 = 1,   p_1(z) = z,   sqrt(j + 1) p_{j+1}(z) = z p_j(z) - sqrt(j) p_{j-1}(z)
//! ```
//!
//! so the nodes are the eigenvalues of the symmetric tridiagonal Jacobi matrix
//! with zero diagonal and off-diagonal `sqrt(1), ..., sqrt(k - 1)` (Golub and
//! Welsch \[1\]). Those eigenvalues, from the Jacobi rotations of
//! `epistat_linalg::eigen`, start a Newton iteration on `p_k`, with
//! `p_k'(z) = sqrt(k) p_{k-1}(z)`, which stops when a step no longer shrinks.
//! The weights are the Christoffel numbers `w_i = 1 / Σ_{j<k} p_j(z_i)²`, a sum
//! of positive terms that keeps the relative accuracy of the small weights in
//! the tails, which the squared first components of the eigenvectors do not.
//! The rule is made exactly symmetric: the nodes of the positive half are
//! mirrored, and the middle node of an odd rule is 0.
//!
//! # References
//!
//! 1. Golub GH, Welsch JH. Calculation of Gauss quadrature rules. Math Comput.
//!    1969;23(106):221-30.
//! 2. Liu Q, Pierce DA. A note on Gauss-Hermite quadrature. Biometrika.
//!    1994;81(3):624-9.

use crate::MixedError;

/// The largest number of nodes [`gauss_hermite`] gives.
pub const MAX_NODES: usize = 100;

/// The nodes, increasing, and weights of the `k`-point Gauss-Hermite rule for
/// the standard normal density.
///
/// # Errors
///
/// [`MixedError::Nodes`] when `k` is 0 or above [`MAX_NODES`].
pub fn gauss_hermite(k: usize) -> Result<(Vec<f64>, Vec<f64>), MixedError> {
    if k == 0 || k > MAX_NODES {
        return Err(MixedError::Nodes(k));
    }
    let mut jacobi = vec![0.0_f64; k * k];
    for j in 1..k {
        let b = (j as f64).sqrt();
        jacobi[j * k + j - 1] = b;
        jacobi[(j - 1) * k + j] = b;
    }
    let (mut values, _) = epistat_linalg::eigen::symmetric_eigen_jacobi(&jacobi, k);
    values.sort_by(f64::total_cmp);
    // The positive half, refined; the negative half mirrors it.
    let half = k / 2;
    let mut positive: Vec<f64> = values[k - half..]
        .iter()
        .map(|&z| refine(z.abs(), k))
        .collect();
    positive.sort_by(f64::total_cmp);
    let mut nodes = Vec::with_capacity(k);
    nodes.extend(positive.iter().rev().map(|z| -z));
    if k % 2 == 1 {
        nodes.push(0.0);
    }
    nodes.extend(positive.iter().copied());
    let weights = nodes.iter().map(|&z| christoffel(z, k)).collect();
    Ok((nodes, weights))
}

/// `(p_k(z), p_{k-1}(z))` for the orthonormal polynomials of the normal
/// density.
fn orthonormal(z: f64, k: usize) -> (f64, f64) {
    let (mut previous, mut current) = (0.0_f64, 1.0_f64);
    for j in 0..k {
        let next = (z * current - (j as f64).sqrt() * previous) / ((j + 1) as f64).sqrt();
        previous = current;
        current = next;
    }
    (current, previous)
}

/// Newton's method on `p_k` from `z`, stopping when a step does not shrink.
fn refine(mut z: f64, k: usize) -> f64 {
    let mut last = f64::INFINITY;
    for _ in 0..100 {
        let (pk, pk1) = orthonormal(z, k);
        let derivative = (k as f64).sqrt() * pk1;
        if derivative == 0.0 {
            break;
        }
        let step = pk / derivative;
        if step.abs() >= last || step.is_nan() {
            break;
        }
        z -= step;
        last = step.abs();
        if step == 0.0 {
            break;
        }
    }
    z
}

/// `1 / Σ_{j<k} p_j(z)²`.
fn christoffel(z: f64, k: usize) -> f64 {
    let (mut previous, mut current) = (0.0_f64, 1.0_f64);
    let mut sum = 1.0;
    for j in 0..k - 1 {
        let next = (z * current - (j as f64).sqrt() * previous) / ((j + 1) as f64).sqrt();
        previous = current;
        current = next;
        sum += current * current;
    }
    1.0 / sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rule_integrates_normal_moments_exactly() {
        for k in [1, 2, 3, 5, 10, 25, 60, 100] {
            let (z, w) = gauss_hermite(k).unwrap();
            assert_eq!(z.len(), k);
            // E z^(2m) = (2m - 1)!!, exact up to degree 2k - 1.
            let mut double_factorial = 1.0;
            for m in 0..k.min(12) {
                if m > 0 {
                    double_factorial *= (2 * m - 1) as f64;
                }
                let moment: f64 = z
                    .iter()
                    .zip(&w)
                    .map(|(z, w)| w * z.powi(i32::try_from(2 * m).unwrap()))
                    .sum();
                assert!(
                    (moment - double_factorial).abs() <= 1e-12 * double_factorial,
                    "k {k} moment {m}: {moment}"
                );
            }
            for i in 0..k {
                assert_eq!(z[i], -z[k - 1 - i]);
                assert_eq!(w[i], w[k - 1 - i]);
            }
        }
        assert_eq!(gauss_hermite(1).unwrap(), (vec![0.0], vec![1.0]));
        assert!(gauss_hermite(0).is_err());
        assert!(gauss_hermite(101).is_err());
    }
}

//! Sandwich covariances for M-estimators.
//!
//! An M-estimator solves `sum_i psi(z_i; theta) = 0`. Under regularity
//! conditions its covariance is `A^-1 M A^-T`, the form of Huber \[1\] and
//! White \[2\], where `A` is the derivative of the summed score (the
//! information) and `M` is the variance of the summed score (the meat).
//! The functions here build `M`, form the sandwich and read standard errors
//! from it. The caller supplies `A`, the rows `x_i` and the score residuals.
//!
//! # Method
//!
//! For a score `r_i x_i`, where `r_i` is the residual of row `i` and `x_i` its
//! row of the design matrix, [`meat`] without clusters is the
//! heteroskedasticity-robust form \[2\]:
//!
//! ```text
//! M = sum_i r_i^2 x_i x_i'
//! ```
//!
//! With clusters it sums the scores within each cluster `g` first, as in \[3\]
//! and \[4\]:
//!
//! ```text
//! s_g = sum_{i in g} r_i x_i
//! M   = sum_g s_g s_g'
//! ```
//!
//! No small-sample factor is applied. [`sandwich`] returns
//!
//! ```text
//! V = A^-1 M A^-T
//! ```
//!
//! by two solves. It factorises `A` and solves `A L = M`, then factorises
//! `A'` and solves `A' R = L'`, so that `R'` is `L A^-T`. `A` is never
//! inverted explicitly. The factorisation is tried with the given `attempts`,
//! in order. The two solves round differently, so `R'` is symmetric only up to
//! rounding, and the function returns the midpoint of `R'` and `R`. That matrix
//! is exactly symmetric, its diagonal equals that of `R'` to the bit, and its
//! off-diagonal entries differ from `R'` in the last bits.
//!
//! [`sandwich_with_inverse`] forms the same `V = A^-1 M A^-T` from an inverse
//! the caller already has, by two products, for a symmetric `A`.
//!
//! [`SmallSample`] gives the small-sample factor `c` that scales `V`, with
//! `N` rows, a model of rank `K` and `G` clusters:
//!
//! ```text
//! rule       no clusters      clusters
//! None       1                1
//! Clusters   N / (N - 1)      G / (G - 1)
//! Hc1        N / (N - K)      G / (G - 1) (N - 1) / (N - K)
//! ```
//!
//! `Hc1` is `sandwich::vcovHC(type = "HC1")` without clusters and
//! `sandwich::vcovCL(type = "HC1", cadjust = TRUE)` with them; `Clusters` is
//! `vcovCL(type = "HC0", cadjust = TRUE)`, every row its own cluster when
//! there are none. The factor does not exist, and is `None`, where a
//! denominator is not positive.
//!
//! [`standard_errors`] is `sqrt(max(V_ii, 0))`. A negative diagonal entry,
//! which rounding can produce when `V` is near singular, is read as zero.
//!
//! # Determinism
//!
//! Without clusters the meat is the Gram matrix of the scaled rows from
//! `epistat-linalg`. With clusters the score sums are formed in row order
//! into one row per cluster, in order of first appearance, and the meat is the
//! Gram matrix of those rows.
//!
//! # Edge cases
//!
//! - A `clusters` argument that covers another number of rows is
//!   [`InputError::Length`]. Without clusters [`meat`] does not check the
//!   length of the residuals, since they are a function of the row;
//!   [`meat_from`] checks it.
//! - A singular or non-finite solve is [`Error::Solve`].
//!
//! # References
//!
//! 1. Huber PJ. The behavior of maximum likelihood estimates under
//!    nonstandard conditions. In: Proceedings of the Fifth Berkeley Symposium
//!    on Mathematical Statistics and Probability, vol 1. Berkeley: University
//!    of California Press; 1967. p. 221-33.
//! 2. White H. A heteroskedasticity-consistent covariance matrix estimator and
//!    a direct test for heteroskedasticity. Econometrica. 1980;48(4):817-38.
//! 3. Liang KY, Zeger SL. Longitudinal data analysis using generalized linear
//!    models. Biometrika. 1986;73(1):13-22.
//! 4. Cameron AC, Gelbach JB, Miller DL. Bootstrap-based improvements for
//!    inference with clustered errors. Rev Econ Stat. 2008;90(3):414-27.

use epistat_linalg::{Attempt, Factor, row_scaled_gram};
use epistat_linalg::{Mat, MatRef};

#[cfg(doc)]
use epistat_core::InputError;

use crate::{Clusters, Error, check_length};

/// The meat `M` from per-row score residuals `r_i = residual(i)`.
///
/// Without clusters it is `sum_i r_i^2 x_i x_i'`. With clusters it is
/// `sum_g s_g s_g'` with `s_g = sum_{i in g} r_i x_i`, the score sums formed in
/// row order.
///
/// # Errors
///
/// [`InputError::Length`] when `clusters` covers another number of rows.
pub fn meat(
    x: MatRef<'_, f64>,
    residual: impl Fn(usize) -> f64 + Sync,
    clusters: Option<&Clusters>,
) -> Result<Mat<f64>, Error> {
    let Some(clusters) = clusters else {
        return Ok(row_scaled_gram(x, residual));
    };
    check_length("clusters", x.nrows(), clusters.len())?;
    let p = x.ncols();
    let mut sums = Mat::<f64>::zeros(clusters.count(), p);
    for (row, &cluster) in clusters.of().iter().enumerate() {
        let r = residual(row);
        for j in 0..p {
            sums[(cluster as usize, j)] += r * x[(row, j)];
        }
    }
    Ok(row_scaled_gram(sums.as_ref(), |_| 1.0))
}

/// [`meat`] with the residuals as a column, or every residual 1 for `None`,
/// which is the meat of scores already in the rows of `x`. The bits are those
/// of [`meat`] with `|i| residuals[i]` or `|_| 1.0`.
///
/// # Errors
///
/// [`InputError::Length`] when the residuals or `clusters` cover another
/// number of rows.
pub fn meat_from(
    x: MatRef<'_, f64>,
    residuals: Option<&[f64]>,
    clusters: Option<&Clusters>,
) -> Result<Mat<f64>, Error> {
    match residuals {
        Some(r) => {
            check_length("residuals", x.nrows(), r.len())?;
            meat(x, |i| r[i], clusters)
        }
        None => meat(x, |_| 1.0, clusters),
    }
}

/// The small-sample factor of a sandwich covariance.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SmallSample {
    /// No factor, `HC0`.
    #[default]
    None,
    /// `G / (G - 1)` for `G` clusters.
    Clusters,
    /// `N / (N - K)` without clusters, `G / (G - 1) (N - 1) / (N - K)` with
    /// them.
    Hc1,
}

impl SmallSample {
    /// The factor for `rows` rows, a model of rank `rank` and `clusters`
    /// clusters (`None` without clusters), or `None` where it does not exist:
    /// fewer than two clusters, or no more rows than the rank.
    ///
    /// The arithmetic is `g / (g - 1) * (n - 1) / (n - k)` from left to right
    /// for `Hc1` with clusters, as `sandwich::vcovCL` writes it.
    #[must_use]
    pub fn factor(self, rows: usize, rank: usize, clusters: Option<usize>) -> Option<f64> {
        match (self, clusters) {
            (Self::None, _) => Some(1.0),
            (Self::Clusters, groups) => {
                let g = groups.unwrap_or(rows);
                (g >= 2).then(|| g as f64 / (g - 1) as f64)
            }
            (Self::Hc1, None) => (rows > rank).then(|| rows as f64 / (rows - rank) as f64),
            (Self::Hc1, Some(g)) => (g >= 2 && rows > rank)
                .then(|| g as f64 / (g - 1) as f64 * (rows - 1) as f64 / (rows - rank) as f64),
        }
    }
}

/// `factor * V`, entry by entry.
#[must_use]
pub fn scaled(covariance: MatRef<'_, f64>, factor: f64) -> Mat<f64> {
    Mat::from_fn(covariance.nrows(), covariance.ncols(), |a, b| {
        factor * covariance[(a, b)]
    })
}

/// `A^-1 M A^-T` for a symmetric meat `M`, by two solves with one
/// factorisation of `A` through `attempts`, averaged with its transpose so that
/// the result is exactly symmetric.
///
/// `A` need not be symmetric: the bread of a stacked M-estimator is not. With
/// `L = A^-1 M`, the sandwich is `L A^-T`, whose transpose is `A^-1 L'`, so the
/// second solve is with `A` again.
///
/// # Errors
///
/// [`Error::Solve`] when `A` cannot be factorised or a solve is not finite.
pub fn sandwich(
    information: MatRef<'_, f64>,
    meat: MatRef<'_, f64>,
    attempts: &[Attempt],
) -> Result<Mat<f64>, Error> {
    let factor = Factor::new(information, attempts)?;
    let left = factor.solve(meat)?;
    let transposed = factor.solve(left.transpose())?;
    let n = transposed.nrows();
    Ok(Mat::from_fn(n, n, |i, j| {
        transposed[(j, i)].midpoint(transposed[(i, j)])
    }))
}

/// `V M V'` for a symmetric inverse `V = A^-1` already in hand and a symmetric
/// meat `M`, averaged with its transpose so that the result is exactly
/// symmetric.
///
/// Use it where the inverse is formed anyway, for the model-based covariance.
/// Each entry is a sum over the shared index in ascending order, first `V M`
/// and then `(V M) V`, and entry `(i, j)` of the result is the midpoint of
/// entries `(i, j)` and `(j, i)` of that product. [`sandwich`] solves instead
/// of multiplying and rounds differently.
///
/// # Panics
///
/// If `inverse` or `meat` is not square of the same side.
#[must_use]
pub fn sandwich_with_inverse(inverse: MatRef<'_, f64>, meat: MatRef<'_, f64>) -> Mat<f64> {
    let p = inverse.nrows();
    assert!(
        inverse.ncols() == p && meat.nrows() == p && meat.ncols() == p,
        "the inverse and the meat must be square of the same side"
    );
    let left = Mat::from_fn(p, p, |a, b| {
        (0..p).map(|c| inverse[(a, c)] * meat[(c, b)]).sum::<f64>()
    });
    let full = Mat::from_fn(p, p, |a, b| {
        (0..p).map(|c| left[(a, c)] * inverse[(c, b)]).sum::<f64>()
    });
    Mat::from_fn(p, p, |a, b| full[(a, b)].midpoint(full[(b, a)]))
}

/// The square roots of the diagonal, with negative variances read as zero.
#[must_use]
pub fn standard_errors(covariance: MatRef<'_, f64>) -> Vec<f64> {
    (0..covariance.nrows())
        .map(|i| covariance[(i, i)].max(0.0).sqrt())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_linalg::mat;

    #[test]
    fn the_least_squares_sandwich_is_the_white_covariance() {
        // Regress y on an intercept; the sandwich variance of the mean is
        // sum(r^2) / n^2.
        let x = Mat::from_fn(4, 1, |_, _| 1.0);
        let residuals = [1.0, -1.0, 2.0, -2.0];
        let information = mat![[4.0]];
        let m = meat(x.as_ref(), |i| residuals[i], None).unwrap();
        let covariance = sandwich(information.as_ref(), m.as_ref(), &Attempt::SYMMETRIC).unwrap();
        assert!((covariance[(0, 0)] - 10.0 / 16.0).abs() < 1e-15);
        assert_eq!(
            standard_errors(covariance.as_ref()),
            [covariance[(0, 0)].sqrt()]
        );
    }

    #[test]
    fn clustered_meat_sums_scores_within_clusters() {
        let x = Mat::from_fn(4, 1, |_, _| 1.0);
        let residuals = [1.0, -1.0, 2.0, 1.0];
        let clusters = Clusters::from_labels_first_seen(&["a", "a", "b", "b"]).unwrap();
        let m = meat(x.as_ref(), |i| residuals[i], Some(&clusters)).unwrap();
        assert!((m[(0, 0)] - 9.0).abs() < 1e-15);
    }

    #[test]
    fn the_sandwich_of_a_non_symmetric_bread_is_a_inverse_m_a_inverse_transposed() {
        // A stacked M-estimator has a block triangular bread. The reference is
        // the inverse formed column by column and multiplied out; both are
        // within a few units of kappa(A) eps of the exact value, kappa(A) < 10.
        let a = mat![[2.0, 0.0, 0.0], [0.7, 1.5, 0.0], [0.4, -0.9, 3.0]];
        let x = Mat::from_fn(9, 3, |i, j| ((i * 3 + j) % 5) as f64 - 1.7);
        let m = meat(x.as_ref(), |i| 0.3 * i as f64 - 1.0, None).unwrap();
        let v = sandwich(a.as_ref(), m.as_ref(), &Attempt::SYMMETRIC).unwrap();
        let inverse = Factor::new(a.as_ref(), &Attempt::SYMMETRIC)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3).as_ref())
            .unwrap();
        let reference = &inverse * &m * inverse.transpose();
        let scale = (0..3).map(|i| reference[(i, i)].abs()).fold(0.0, f64::max);
        for i in 0..3 {
            for j in 0..3 {
                let err = (v[(i, j)] - reference[(i, j)]).abs();
                assert!(
                    err <= 64.0 * 10.0 * f64::EPSILON * scale,
                    "({i},{j}): {err:e}"
                );
            }
        }
    }

    #[test]
    fn the_inverse_form_is_symmetric_and_matches_the_solving_form() {
        let a = mat![[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 3.0]];
        let x = Mat::from_fn(9, 3, |i, j| ((i * 3 + j) % 5) as f64 - 1.7);
        let m = meat(x.as_ref(), |i| 0.3 * i as f64 - 1.0, None).unwrap();
        let inverse = Factor::new(a.as_ref(), &Attempt::SYMMETRIC)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3).as_ref())
            .unwrap();
        let v = sandwich_with_inverse(inverse.as_ref(), m.as_ref());
        let solved = sandwich(a.as_ref(), m.as_ref(), &Attempt::SYMMETRIC).unwrap();
        let scale = (0..3).map(|i| v[(i, i)].abs()).fold(0.0, f64::max);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(v[(i, j)].to_bits(), v[(j, i)].to_bits());
                // Both forms are within a few n kappa(A) eps of the exact value.
                let err = (v[(i, j)] - solved[(i, j)]).abs();
                assert!(
                    err <= 64.0 * 10.0 * f64::EPSILON * scale,
                    "({i},{j}): {err:e}"
                );
            }
        }
    }

    #[test]
    fn the_inverse_form_sums_in_ascending_order_of_the_shared_index() {
        // With a diagonal inverse every sum has one nonzero term, so the
        // result is d_i m_ij d_j exactly, then the midpoint with its mirror.
        let d = [0.5, 3.0];
        let inverse = Mat::from_fn(2, 2, |i, j| if i == j { d[i] } else { 0.0 });
        let m = mat![[1.0, 0.25], [0.75, 2.0]];
        let v = sandwich_with_inverse(inverse.as_ref(), m.as_ref());
        let off = (d[0] * 0.25 * d[1]).midpoint(d[1] * 0.75 * d[0]);
        assert_eq!(v[(0, 1)].to_bits(), off.to_bits());
        assert_eq!(v[(1, 0)].to_bits(), off.to_bits());
        assert_eq!(v[(0, 0)].to_bits(), (d[0] * 1.0 * d[0]).to_bits());
    }

    #[test]
    #[should_panic(expected = "square of the same side")]
    fn the_inverse_form_refuses_matrices_of_different_sides() {
        let _ = sandwich_with_inverse(
            Mat::<f64>::identity(2, 2).as_ref(),
            Mat::<f64>::identity(3, 3).as_ref(),
        );
    }

    #[test]
    fn a_symmetric_bread_gives_the_bits_it_gave_before() {
        // The earlier form solved the second time with A transposed, which is
        // the same matrix when A is symmetric, so callers' numbers do not move.
        let a = mat![[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 3.0]];
        let x = Mat::from_fn(7, 3, |i, j| ((i * 3 + j) % 5) as f64 - 1.7);
        let m = meat(x.as_ref(), |i| 0.3 * i as f64 - 1.0, None).unwrap();
        let left = Factor::new(a.as_ref(), &Attempt::SYMMETRIC)
            .unwrap()
            .solve(m.as_ref())
            .unwrap();
        let old = Factor::new(a.transpose(), &Attempt::SYMMETRIC)
            .unwrap()
            .solve(left.transpose())
            .unwrap();
        let old = Mat::from_fn(3, 3, |i, j| old[(j, i)].midpoint(old[(i, j)]));
        let new = sandwich(a.as_ref(), m.as_ref(), &Attempt::SYMMETRIC).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(new[(i, j)].to_bits(), old[(i, j)].to_bits());
            }
        }
    }

    #[test]
    fn meat_from_columns_has_the_bits_of_the_closure_form() {
        let x = Mat::from_fn(7, 3, |i, j| ((i * 3 + j) % 5) as f64 - 1.7);
        let r: Vec<f64> = (0..7).map(|i| 0.3 * f64::from(i) - 1.0).collect();
        let clusters = Clusters::from_codes(&[0, 0, 1, 1, 2, 2, 2]);
        for c in [None, Some(&clusters)] {
            assert_eq!(
                meat_from(x.as_ref(), Some(&r), c).unwrap(),
                meat(x.as_ref(), |i| r[i], c).unwrap()
            );
            assert_eq!(
                meat_from(x.as_ref(), None, c).unwrap(),
                meat(x.as_ref(), |_| 1.0, c).unwrap()
            );
        }
        assert!(meat_from(x.as_ref(), Some(&r[..3]), None).is_err());
    }

    #[test]
    fn small_sample_factors() {
        assert_eq!(SmallSample::None.factor(10, 3, Some(1)), Some(1.0));
        assert_eq!(SmallSample::Clusters.factor(10, 3, Some(5)), Some(1.25));
        assert_eq!(SmallSample::Clusters.factor(5, 3, None), Some(1.25));
        assert_eq!(SmallSample::Clusters.factor(10, 3, Some(1)), None);
        assert_eq!(SmallSample::Hc1.factor(10, 2, None), Some(1.25));
        assert_eq!(SmallSample::Hc1.factor(2, 2, None), None);
        let g = 4.0_f64 / 3.0 * 9.0 / 8.0;
        assert_eq!(
            SmallSample::Hc1.factor(10, 2, Some(4)).map(f64::to_bits),
            Some(g.to_bits())
        );
        let v = mat![[1.0, 2.0], [2.0, 4.0]];
        assert_eq!(scaled(v.as_ref(), 0.5), mat![[0.5, 1.0], [1.0, 2.0]]);
    }

    #[test]
    fn the_sandwich_is_exactly_symmetric() {
        let information = mat![[2.0, 0.3, 0.1], [0.7, 1.5, 0.2], [0.1, 0.9, 3.0]];
        let x = Mat::from_fn(7, 3, |i, j| ((i * 3 + j) % 5) as f64 - 1.7);
        let m = meat(x.as_ref(), |i| 0.3 * i as f64 - 1.0, None).unwrap();
        let v = sandwich(information.as_ref(), m.as_ref(), &Attempt::SYMMETRIC).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(v[(i, j)].to_bits(), v[(j, i)].to_bits());
            }
        }
    }
}

//! Products over the rows of a design matrix, computed in fixed blocks of
//! rows so that the result does not depend on the number of threads.
//!
//! # Algorithm
//!
//! Let `X` have `n` rows and `p` columns. The rows are cut into blocks of
//! [`ROW_BLOCK`] = 4096, the last one shorter. Each product is a sum over
//! rows, so it splits into one partial per block:
//!
//! ```text
//! X' W X = sum over blocks b of  X_b' W_b X_b
//! X' v   = sum over blocks b of  X_b' v_b
//! ```
//!
//! The partials are computed in parallel, one task per block, each with
//! sequential faer kernels. They are then added to a zero matrix in block
//! order on the calling thread. A system of at most one block is computed
//! directly, which is the same sum.
//!
//! # Determinism
//!
//! The rows in a block and the order of the blocks do not depend on the
//! thread count, and nothing is added across blocks except in block order.
//! The floating-point additions are therefore the same for every thread count,
//! and so are [`weighted_gram`], [`row_scaled_gram`], [`xtv`] and
//! [`weighted_xtz`]. [`matvec_into`] needs no addition across blocks, since
//! each output entry comes from one row, and it fills blocks of the output in
//! parallel. Changing [`ROW_BLOCK`] changes the grouping of the additions and
//! so the last bits of the results. faer chooses its SIMD kernels from the CPU
//! at run time, so the bits are not promised to be the same across CPUs.
//!
//! # Symmetry
//!
//! [`row_scaled_gram`] computes
//!
//! ```text
//! G = sum_i s_i^2 x_i x_i'
//! ```
//!
//! where `x_i` is row `i` of `X` as a column and `s_i` is the scale of the
//! row. Each block scales its rows by `s_i` to give `S_b` and adds `S_b' S_b`
//! through a rank-k update of the lower triangle only. After the blocks are
//! added, the lower triangle is copied to the upper. The result is therefore
//! exactly symmetric, entry for entry, as the Cholesky and Bunch-Kaufman
//! factorisations of [`crate::Factor`] assume \[1\].
//!
//! # Weights
//!
//! [`weighted_gram`] uses `s_i = sqrt(w_i)`. [`weighted_xtz`] forms `w_i z_i`
//! and calls [`xtv`]. Both, and [`independent_columns`], refuse a weight that is
//! negative, infinite or NaN with [`DesignError::Weight`], so the Gram matrix
//! and the right-hand side of the normal equations always describe the same
//! weighted problem. A weight of zero is valid.
//!
//! [`independent_columns`]: crate::design::independent_columns
//!
//! # References
//!
//! 1. Golub GH, Van Loan CF. Matrix Computations. 4th ed. Baltimore: Johns
//!    Hopkins University Press; 2013.

use std::ops::Range;

use crate::error::{DesignError, check_length, check_weights};
use faer::linalg::matmul::matmul;
use faer::linalg::matmul::triangular::{self, BlockStructure};
use faer::{Accum, Mat, MatMut, MatRef, Par};
use rayon::prelude::*;

/// The rows per block. Changing it changes the order of additions, and so
/// the last bits of every product below.
pub const ROW_BLOCK: usize = 4096;

/// `sum_i s_i^2 x_i x_i'` over the rows `x_i` of `x` with `s_i = scale(i)`,
/// exactly symmetric.
///
/// Each block forms its rows scaled by `s_i` and adds their cross-product
/// through a rank-k update of the lower triangle, which is then mirrored.
#[must_use]
pub fn row_scaled_gram(x: MatRef<'_, f64>, scale: impl Fn(usize) -> f64 + Sync) -> Mat<f64> {
    let p = x.ncols();
    let mut gram = sum_over_row_blocks(x.nrows(), p, p, |rows, out| {
        let factors: Vec<f64> = rows.clone().map(&scale).collect();
        let mut scaled = Mat::<f64>::zeros(rows.len(), p);
        for j in 0..p {
            for (k, i) in rows.clone().enumerate() {
                scaled[(k, j)] = x[(i, j)] * factors[k];
            }
        }
        triangular::matmul(
            out,
            BlockStructure::TriangularLower,
            Accum::Replace,
            scaled.transpose(),
            BlockStructure::Rectangular,
            scaled.as_ref(),
            BlockStructure::Rectangular,
            1.0,
            Par::Seq,
        );
    });
    for j in 0..p {
        for i in (j + 1)..p {
            gram[(j, i)] = gram[(i, j)];
        }
    }
    gram
}

/// `X' W X` for the diagonal `W` of `weights`.
///
/// # Errors
///
/// [`DesignError::Length`] when `weights` does not have one entry per row of
/// `x`, and [`DesignError::Weight`] for a weight that is negative or not
/// finite.
pub fn weighted_gram(x: MatRef<'_, f64>, weights: &[f64]) -> Result<Mat<f64>, DesignError> {
    check_weights(weights, x.nrows())?;
    Ok(row_scaled_gram(x, |i| weights[i].sqrt()))
}

/// `X' v` as a `p x 1` column.
///
/// # Panics
///
/// If `v` does not have one entry per row of `x`.
#[must_use]
pub fn xtv(x: MatRef<'_, f64>, v: &[f64]) -> Mat<f64> {
    assert_eq!(v.len(), x.nrows(), "one value per row");
    sum_over_row_blocks(x.nrows(), x.ncols(), 1, |rows, out| {
        let len = rows.len();
        let start = rows.start;
        matmul(
            out,
            Accum::Replace,
            x.subrows(start, len).transpose(),
            MatRef::from_column_major_slice(&v[rows], len, 1),
            1.0,
            Par::Seq,
        );
    })
}

/// `X' W z` for the diagonal `W` of `weights`.
///
/// # Errors
///
/// [`DesignError::Length`] when `weights` or `z` does not have one entry per
/// row of `x`, and [`DesignError::Weight`] for a weight that is negative or
/// not finite.
pub fn weighted_xtz(
    x: MatRef<'_, f64>,
    weights: &[f64],
    z: &[f64],
) -> Result<Mat<f64>, DesignError> {
    check_weights(weights, x.nrows())?;
    check_length("z", z.len(), x.nrows())?;
    let weighted: Vec<f64> = weights.iter().zip(z).map(|(w, v)| w * v).collect();
    Ok(xtv(x, &weighted))
}

/// `out <- X beta` for a `p x 1` column `beta`, block by block.
///
/// # Panics
///
/// If `out` does not have one entry per row of `x` or `beta` has the wrong
/// shape.
pub fn matvec_into(out: &mut [f64], x: MatRef<'_, f64>, beta: MatRef<'_, f64>) {
    assert_eq!(out.len(), x.nrows(), "one output per row");
    assert_eq!(
        (beta.nrows(), beta.ncols()),
        (x.ncols(), 1),
        "beta is a p x 1 column"
    );
    out.par_chunks_mut(ROW_BLOCK)
        .enumerate()
        .for_each(|(block, chunk)| {
            let len = chunk.len();
            matmul(
                MatMut::from_column_major_slice_mut(chunk, len, 1),
                Accum::Replace,
                x.subrows(block * ROW_BLOCK, len),
                beta,
                1.0,
                Par::Seq,
            );
        });
}

/// The sum over blocks of `n` rows of the `rows x cols` partials that
/// `block_term` writes, added in block order. Blocks run in parallel.
fn sum_over_row_blocks<F>(n: usize, rows: usize, cols: usize, block_term: F) -> Mat<f64>
where
    F: Fn(Range<usize>, MatMut<'_, f64>) + Sync,
{
    let mut total = Mat::<f64>::zeros(rows, cols);
    if n <= ROW_BLOCK {
        block_term(0..n, total.as_mut());
        return total;
    }
    let partials: Vec<Mat<f64>> = (0..n.div_ceil(ROW_BLOCK))
        .into_par_iter()
        .map(|block| {
            let mut partial = Mat::<f64>::zeros(rows, cols);
            block_term(
                block * ROW_BLOCK..((block + 1) * ROW_BLOCK).min(n),
                partial.as_mut(),
            );
            partial
        })
        .collect();
    for partial in &partials {
        for j in 0..cols {
            for i in 0..rows {
                total[(i, j)] += partial[(i, j)];
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_testkit::{Tolerance, assert_close, assert_same_on_threads};

    fn design(n: usize, p: usize) -> Mat<f64> {
        let mut stream = 0x1234_5678_u64;
        Mat::from_fn(n, p, |_, j| {
            stream = stream
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            if j == 0 {
                1.0
            } else {
                (stream >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            }
        })
    }

    fn flat(m: &Mat<f64>) -> Vec<f64> {
        (0..m.ncols())
            .flat_map(|j| (0..m.nrows()).map(move |i| m[(i, j)]))
            .collect()
    }

    #[test]
    fn products_agree_with_the_definitions() {
        let (n, p) = (9_000, 4);
        let x = design(n, p);
        let w: Vec<f64> = (0..n)
            .map(|i| f64::from(u32::try_from(i % 5).unwrap()) * 0.25)
            .collect();
        let z: Vec<f64> = (0..n)
            .map(|i| f64::from(u32::try_from(i % 7).unwrap()))
            .collect();
        let gram = weighted_gram(x.as_ref(), &w).unwrap();
        let xtz = weighted_xtz(x.as_ref(), &w, &z).unwrap();
        for j in 0..p {
            let direct: f64 = (0..n).map(|i| w[i] * z[i] * x[(i, j)]).sum();
            assert_close(&[xtz[(j, 0)]], &[direct], Tolerance::exact().rel(1e-12));
            for k in 0..p {
                let direct: f64 = (0..n).map(|i| w[i] * x[(i, j)] * x[(i, k)]).sum();
                assert_close(
                    &[gram[(j, k)]],
                    &[direct],
                    Tolerance::exact().rel(1e-12).abs(1e-12),
                );
            }
        }
        let beta = Mat::from_fn(p, 1, |j, _| f64::from(u32::try_from(j).unwrap()) - 1.5);
        let mut eta = vec![0.0; n];
        matvec_into(&mut eta, x.as_ref(), beta.as_ref());
        let direct: Vec<f64> = (0..n)
            .map(|i| (0..p).map(|j| x[(i, j)] * beta[(j, 0)]).sum())
            .collect();
        assert_close(&eta, &direct, Tolerance::exact().abs(1e-14));
    }

    #[test]
    fn products_do_not_depend_on_threads() {
        let (n, p) = (20_000, 6);
        let x = design(n, p);
        let w: Vec<f64> = (0..n)
            .map(|i| 1.0 + f64::from(u32::try_from(i % 3).unwrap()))
            .collect();
        assert_same_on_threads(|| {
            let mut out = flat(&weighted_gram(x.as_ref(), &w).unwrap());
            out.extend(flat(&weighted_xtz(x.as_ref(), &w, &w).unwrap()));
            let mut eta = vec![0.0; n];
            matvec_into(
                &mut eta,
                x.as_ref(),
                Mat::from_fn(p, 1, |_, _| 0.5).as_ref(),
            );
            out.extend(eta);
            out
        });
    }

    #[test]
    fn the_gram_matrix_is_exactly_symmetric() {
        let x = design(5_000, 5);
        let gram = row_scaled_gram(x.as_ref(), |i| {
            1.0 + 1e-3 * f64::from(u32::try_from(i % 11).unwrap())
        });
        for j in 0..5 {
            for k in 0..5 {
                assert_eq!(gram[(j, k)].to_bits(), gram[(k, j)].to_bits());
            }
        }
    }

    #[test]
    fn invalid_weights_are_refused_by_every_product() {
        let x = design(10, 3);
        let z = vec![1.0; 10];
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            let mut w = vec![1.0; 10];
            w[4] = bad;
            let expected = Err(DesignError::Weight { row: 4 });
            assert_eq!(weighted_gram(x.as_ref(), &w).map(|_| ()), expected);
            assert_eq!(weighted_xtz(x.as_ref(), &w, &z).map(|_| ()), expected);
        }
        assert!(matches!(
            weighted_gram(x.as_ref(), &[1.0; 9]),
            Err(DesignError::Length { .. })
        ));
        assert!(matches!(
            weighted_xtz(x.as_ref(), &[1.0; 10], &z[..9]),
            Err(DesignError::Length { .. })
        ));
        assert!(weighted_gram(x.as_ref(), &[0.0; 10]).is_ok());
    }
}

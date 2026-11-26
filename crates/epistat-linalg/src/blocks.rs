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
//! negative, infinite or NaN with [`Error::Input`], so the Gram matrix
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

use crate::error::{Error, check_length, check_weights};
use faer::linalg::matmul::matmul;
use faer::linalg::matmul::triangular::{self, BlockStructure};
use faer::{Accum, Mat, MatMut, MatRef, Par};
use rayon::prelude::*;

/// The rows per block. Changing it changes the order of additions, and so
/// the last bits of every product below.
pub const ROW_BLOCK: usize = 4096;

/// `X'X`, exactly symmetric: [`row_scaled_gram`] with every scale 1, the same
/// blocks and the same bits.
#[must_use]
pub fn gram(x: MatRef<'_, f64>) -> Mat<f64> {
    row_scaled_gram(x, |_| 1.0)
}

/// `A Σ A'` for `a` of `m x p` and a symmetric `sigma` of `p x p`, exactly
/// symmetric: the congruence of a covariance, the covariance of `A θ` when
/// `Σ` is that of `θ`.
///
/// `T = A Σ` is formed entry by entry with the sum over `k` in increasing
/// order, then `(A Σ A')_ij = sum_k T_ik a_jk` for `j <= i` in the same
/// order, and the upper triangle is copied from the lower. No fused
/// multiply-add and no blocking, so the bits do not depend on faer's kernels.
///
/// # Errors
///
/// [`Error::Input`] when `sigma` is not `p x p`.
pub fn congruence(a: MatRef<'_, f64>, sigma: MatRef<'_, f64>) -> Result<Mat<f64>, Error> {
    let (m, p) = (a.nrows(), a.ncols());
    check_length("rows of sigma", sigma.nrows(), p)?;
    check_length("columns of sigma", sigma.ncols(), p)?;
    let mut t = Mat::<f64>::zeros(m, p);
    for i in 0..m {
        for j in 0..p {
            let mut sum = 0.0;
            for k in 0..p {
                sum += a[(i, k)] * sigma[(k, j)];
            }
            t[(i, j)] = sum;
        }
    }
    let mut out = Mat::<f64>::zeros(m, m);
    for i in 0..m {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..p {
                sum += t[(i, k)] * a[(j, k)];
            }
            out[(i, j)] = sum;
            out[(j, i)] = sum;
        }
    }
    Ok(out)
}

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
        scaled_gram_into(x.subrows(rows.start, rows.len()), &factors, out);
    });
    mirror_lower(&mut gram);
    gram
}

/// Writes the lower triangle of `sum_i s_i^2 x_i x_i'` over the rows of `x`
/// into `out`, with `s_i = factors[i]`: the term of one block of
/// [`row_scaled_gram`].
fn scaled_gram_into(x: MatRef<'_, f64>, factors: &[f64], out: MatMut<'_, f64>) {
    let p = x.ncols();
    let mut scaled = Mat::<f64>::zeros(x.nrows(), p);
    for j in 0..p {
        for (k, &f) in factors.iter().enumerate() {
            scaled[(k, j)] = x[(k, j)] * f;
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
}

fn mirror_lower(gram: &mut Mat<f64>) {
    let p = gram.ncols();
    for j in 0..p {
        for i in (j + 1)..p {
            gram[(j, i)] = gram[(i, j)];
        }
    }
}

/// The term of one block of rows of [`row_scaled_gram`]: the lower triangle
/// of `sum_i s_i^2 x_i x_i'` over the rows of `x`, which are the rows of the
/// block, with `s_i = factors[i]`, and zeros above it. Added in block order
/// by [`BlockSum`] and mirrored by [`BlockSum::finish_symmetric`], the terms of
/// the blocks of [`ROW_BLOCK`] rows of a design give [`row_scaled_gram`] of it
/// bit for bit, for rows formed block by block instead of held in full.
///
/// # Panics
///
/// If `factors` does not have one entry per row of `x`.
#[must_use]
pub fn block_scaled_gram(x: MatRef<'_, f64>, factors: &[f64]) -> Mat<f64> {
    assert_eq!(factors.len(), x.nrows(), "one factor per row");
    let mut out = Mat::<f64>::zeros(x.ncols(), x.ncols());
    scaled_gram_into(x, factors, out.as_mut());
    out
}

/// The term of one block of rows of [`xtv`]: `X_b' v_b` for the rows `x` of
/// the block and their values `v`. Added in block order by [`BlockSum`], the
/// terms of the blocks of [`ROW_BLOCK`] rows give [`xtv`] bit for bit.
///
/// # Panics
///
/// If `v` does not have one entry per row of `x`.
#[must_use]
pub fn block_xtv(x: MatRef<'_, f64>, v: &[f64]) -> Mat<f64> {
    assert_eq!(v.len(), x.nrows(), "one value per row");
    let mut out = Mat::<f64>::zeros(x.ncols(), 1);
    xtv_into(x, v, out.as_mut());
    out
}

fn xtv_into(x: MatRef<'_, f64>, v: &[f64], out: MatMut<'_, f64>) {
    let len = x.nrows();
    matmul(
        out,
        Accum::Replace,
        x.transpose(),
        MatRef::from_column_major_slice(v, len, 1),
        1.0,
        Par::Seq,
    );
}

/// The sum of the terms of the blocks of `n` rows, added in block order as
/// [`row_scaled_gram`] and [`xtv`] add them: a single block is the sum, and
/// more are added one after another to a zero matrix.
#[derive(Clone, Debug)]
pub struct BlockSum {
    single: bool,
    total: Mat<f64>,
}

impl BlockSum {
    /// The sum of the `rows x cols` terms of the blocks of `n` rows.
    #[must_use]
    pub fn new(n: usize, rows: usize, cols: usize) -> Self {
        Self {
            single: n <= ROW_BLOCK,
            total: Mat::zeros(rows, cols),
        }
    }

    /// Adds the term of the next block.
    ///
    /// # Panics
    ///
    /// If the term is not of the shape of the sum.
    pub fn add(&mut self, term: &Mat<f64>) {
        assert_eq!(
            (term.nrows(), term.ncols()),
            (self.total.nrows(), self.total.ncols()),
            "a term of the shape of the sum"
        );
        if self.single {
            self.total.clone_from(term);
            return;
        }
        for j in 0..term.ncols() {
            for i in 0..term.nrows() {
                self.total[(i, j)] += term[(i, j)];
            }
        }
    }

    /// The sum.
    #[must_use]
    pub fn finish(self) -> Mat<f64> {
        self.total
    }

    /// The sum with its lower triangle copied to the upper, as
    /// [`row_scaled_gram`] gives it.
    #[must_use]
    pub fn finish_symmetric(self) -> Mat<f64> {
        let mut total = self.total;
        mirror_lower(&mut total);
        total
    }
}

/// `X' W X` for the diagonal `W` of `weights`.
///
/// # Errors
///
/// [`Error::Input`] when `weights` does not have one entry per row of
/// `x`, and [`Error::Input`] for a weight that is negative or not
/// finite.
pub fn weighted_gram(x: MatRef<'_, f64>, weights: &[f64]) -> Result<Mat<f64>, Error> {
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
        xtv_into(x.subrows(rows.start, rows.len()), &v[rows], out);
    })
}

/// `X' W z` for the diagonal `W` of `weights`.
///
/// # Errors
///
/// [`Error::Input`] when `weights` or `z` does not have one entry per
/// row of `x`, and [`Error::Input`] for a weight that is negative or
/// not finite.
pub fn weighted_xtz(x: MatRef<'_, f64>, weights: &[f64], z: &[f64]) -> Result<Mat<f64>, Error> {
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

    /// The products over a design formed block by block, each block copied
    /// into its own matrix and vector, against those over the design held in
    /// full. `offset` places the copy of `v` that many values into its buffer,
    /// to show whether the alignment of the copy matters.
    fn by_blocks(x: MatRef<'_, f64>, w: &[f64], v: &[f64], offset: usize) -> (Mat<f64>, Mat<f64>) {
        let (n, p) = (x.nrows(), x.ncols());
        let mut gram = BlockSum::new(n, p, p);
        let mut cross = BlockSum::new(n, p, 1);
        let mut start = 0;
        while start < n || (n == 0 && start == 0) {
            let len = ROW_BLOCK.min(n - start);
            let block = Mat::from_fn(len, p, |i, j| x[(start + i, j)]);
            let factors: Vec<f64> = w[start..start + len].iter().map(|v| v.sqrt()).collect();
            gram.add(&block_scaled_gram(block.as_ref(), &factors));
            let mut buffer = vec![0.0; offset];
            buffer.extend_from_slice(&v[start..start + len]);
            cross.add(&block_xtv(block.as_ref(), &buffer[offset..]));
            if n == 0 {
                break;
            }
            start += len;
        }
        (gram.finish_symmetric(), cross.finish())
    }

    #[test]
    fn products_formed_by_blocks_have_the_bits_of_the_whole() {
        for n in [
            0,
            1,
            7,
            ROW_BLOCK - 1,
            ROW_BLOCK,
            ROW_BLOCK + 1,
            9_001,
            20_000,
        ] {
            for p in [1, 2, 3, 5, 8] {
                let x = design(n, p);
                let w: Vec<f64> = (0..n)
                    .map(|i| 0.5 + f64::from(u32::try_from(i % 13).unwrap()) * 0.37)
                    .collect();
                let v: Vec<f64> = (0..n)
                    .map(|i| (f64::from(u32::try_from(i % 17).unwrap()) - 8.3) * 1.7)
                    .collect();
                let gram = weighted_gram(x.as_ref(), &w).unwrap();
                let cross = xtv(x.as_ref(), &v);
                for offset in [0, 1, 3] {
                    let (g, c) = by_blocks(x.as_ref(), &w, &v, offset);
                    assert_eq!(
                        epistat_testkit::bits(&flat(&g)),
                        epistat_testkit::bits(&flat(&gram)),
                        "gram n {n} p {p} offset {offset}"
                    );
                    assert_eq!(
                        epistat_testkit::bits(&flat(&c)),
                        epistat_testkit::bits(&flat(&cross)),
                        "xtv n {n} p {p} offset {offset}"
                    );
                }
            }
        }
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
    fn gram_is_the_unit_scaled_gram_and_congruence_is_a_sigma_a_transpose() {
        let x = design(50, 4);
        let ours = gram(x.as_ref());
        let theirs = row_scaled_gram(x.as_ref(), |_| 1.0);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(ours[(i, j)].to_bits(), theirs[(i, j)].to_bits());
            }
        }
        let a = crate::mat![[1.0, 2.0, 0.0], [0.5, -1.0, 3.0]];
        let sigma = crate::mat![[2.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 0.5]];
        let c = congruence(a.as_ref(), sigma.as_ref()).unwrap();
        let direct = &a * &sigma * a.transpose();
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(c[(i, j)].to_bits(), c[(j, i)].to_bits());
                // Sums of three products of values below 4: a few units of
                // 2^-52 times 20.
                assert!((c[(i, j)] - direct[(i, j)]).abs() < 1e-13);
            }
        }
        assert!(congruence(a.as_ref(), a.as_ref()).is_err());
    }

    #[test]
    fn invalid_weights_are_refused_by_every_product() {
        let x = design(10, 3);
        let z = vec![1.0; 10];
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            let mut w = vec![1.0; 10];
            w[4] = bad;
            let expected = Err(crate::weight_error(&w).unwrap());
            assert_eq!(weighted_gram(x.as_ref(), &w).map(|_| ()), expected);
            assert_eq!(weighted_xtz(x.as_ref(), &w, &z).map(|_| ()), expected);
        }
        assert!(matches!(
            weighted_gram(x.as_ref(), &[1.0; 9]),
            Err(crate::Error::Input(epistat_core::InputError::Length { .. }))
        ));
        assert!(matches!(
            weighted_xtz(x.as_ref(), &[1.0; 10], &z[..9]),
            Err(crate::Error::Input(epistat_core::InputError::Length { .. }))
        ));
        assert!(weighted_gram(x.as_ref(), &[0.0; 10]).is_ok());
    }
}

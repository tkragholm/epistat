//! The columns of a design as contiguous slices, and the linear predictor
//! formed from them.

use epistat_linalg::MatRef;

/// Rows per tile of [`linear_predictor`]: the tile of `eta` stays in the
/// first-level cache while every column is added to it.
const TILE: usize = 1024;

/// The columns of `x` as a column-major copy, empty when `x` is already
/// column-major and is read in place by [`column_slices`].
pub(crate) fn contiguous_copy(x: MatRef<'_, f64>) -> Vec<f64> {
    if x.try_as_col_major().is_some() {
        return Vec::new();
    }
    let (n, p) = (x.nrows(), x.ncols());
    let mut copy = Vec::with_capacity(n * p);
    for j in 0..p {
        copy.extend((0..n).map(|i| x[(i, j)]));
    }
    copy
}

/// The columns of `x` as slices, of `x` itself when it is column-major and
/// otherwise of `copy`, which [`contiguous_copy`] made.
pub(crate) fn column_slices<'a>(x: MatRef<'a, f64>, copy: &'a [f64]) -> Vec<&'a [f64]> {
    let (n, p) = (x.nrows(), x.ncols());
    if let Some(major) = x.try_as_col_major() {
        return (0..p).map(|j| major.col(j).as_slice()).collect();
    }
    if n == 0 {
        return vec![&[]; p];
    }
    copy.chunks_exact(n).collect()
}

/// `η_i = Σ_j x_ij β_j + o_i` with each row's terms added in column order from
/// negative zero, so every `η_i` has the bits of a sum taken row by row. The
/// design is read a column at a time, a tile of rows after another.
pub(crate) fn linear_predictor(columns: &[&[f64]], beta: &[f64], offset: &[f64]) -> Vec<f64> {
    let n = offset.len();
    let mut eta = vec![-0.0_f64; n];
    for (t, tile) in eta.chunks_mut(TILE).enumerate() {
        let at = t * TILE;
        for (column, &b) in columns.iter().zip(beta) {
            for (e, v) in tile.iter_mut().zip(&column[at..]) {
                *e += v * b;
            }
        }
        for (e, o) in tile.iter_mut().zip(&offset[at..]) {
            *e += o;
        }
    }
    eta
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_linalg::Mat;

    fn by_rows(x: MatRef<'_, f64>, beta: &[f64], offset: &[f64]) -> Vec<f64> {
        (0..x.nrows())
            .map(|i| (0..x.ncols()).map(|j| x[(i, j)] * beta[j]).sum::<f64>() + offset[i])
            .collect()
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn design(n: usize, p: usize) -> (Mat<f64>, Vec<f64>, Vec<f64>) {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
        };
        let mut x = Mat::from_fn(n, p, |_, _| next() * 1e3);
        for i in (0..n).step_by(7) {
            if p > 0 {
                x[(i, 0)] = -0.0;
            }
        }
        for i in (0..n).step_by(5) {
            if p > 1 {
                x[(i, 1)] = 0.0;
            }
        }
        let mut beta: Vec<f64> = (0..p).map(|_| next()).collect();
        if p > 2 {
            beta[2] = -0.0;
        }
        let offset = (0..n)
            .map(|i| if i % 11 == 0 { -0.0 } else { next() })
            .collect();
        (x, beta, offset)
    }

    #[test]
    fn the_predictor_has_the_bits_of_the_row_by_row_sum() {
        for n in [0, 1, 3, 7, 1023, 1024, 1025, 5000] {
            for p in [0, 1, 2, 3, 6, 13] {
                let (x, beta, offset) = design(n, p);
                let copy = contiguous_copy(x.as_ref());
                let cols = column_slices(x.as_ref(), &copy);
                assert_eq!(
                    bits(&linear_predictor(&cols, &beta, &offset)),
                    bits(&by_rows(x.as_ref(), &beta, &offset)),
                    "n {n}, p {p}"
                );
            }
        }
    }

    #[test]
    fn a_nan_and_an_infinity_pass_through_alike() {
        let (mut x, beta, offset) = design(100, 4);
        x[(10, 1)] = f64::NAN;
        x[(20, 2)] = f64::INFINITY;
        let copy = contiguous_copy(x.as_ref());
        let cols = column_slices(x.as_ref(), &copy);
        assert_eq!(
            bits(&linear_predictor(&cols, &beta, &offset)),
            bits(&by_rows(x.as_ref(), &beta, &offset))
        );
    }

    #[test]
    fn a_transposed_design_is_copied_and_gives_the_same_columns() {
        let (x, beta, offset) = design(300, 5);
        let transposed = Mat::from_fn(5, 300, |j, i| x[(i, j)]);
        let view = transposed.as_ref().transpose();
        let copy = contiguous_copy(view);
        assert_eq!(copy.len(), 5 * 300);
        let cols = column_slices(view, &copy);
        assert_eq!(
            bits(&linear_predictor(&cols, &beta, &offset)),
            bits(&by_rows(x.as_ref(), &beta, &offset))
        );
    }
}

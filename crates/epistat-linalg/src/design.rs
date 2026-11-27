//! Helpers for design matrices: building one from columns, finding the columns that are linearly
//! independent within every group of rows, and selecting columns and rows.
//!
//! # Algorithm
//!
//! [`independent_columns`] grows a Cholesky factorisation \[1\] one column
//! at a time. Let `G_g = X_g' W_g X_g` be the weighted Gram matrix of the rows
//! in group `g`, with the rows added in order. For a candidate column `c`
//! and the kept columns `k(1), ..., k(m)`, each group has a lower triangular
//! factor `L` of the Gram matrix of the kept columns. The projection `q` of
//! the candidate on them solves
//!
//! ```text
//! L q = ( G[c, k(1)], ..., G[c, k(m)] )'
//! ```
//!
//! by forward substitution, and the residual is
//!
//! ```text
//! residual = G[c, c] - q'q
//! ```
//!
//! which is the squared weighted norm of the part of the column that is not
//! in the span of the kept columns. The column is dropped when, in any group,
//!
//! ```text
//! residual <= tolerance * G[c, c].
//! ```
//!
//! Otherwise `sqrt(max(residual, 0))` is appended to `q` as the new row of `L`
//! in every group, and the column is kept. Column 0 is always kept. A
//! `tolerance` of 1e-8 on the Gram matrix drops a column whose relative norm
//! after projection is below 1e-4.
//!
//! # Groups
//!
//! With `groups` set to `None` all rows are one group. Otherwise `groups[i]`
//! numbers the group of row `i` from 0, and a column must be independent in
//! every group to be kept. A column whose diagonal entry `G[c, c]` is not
//! positive in a group is dropped when that group has rows, meaning that its
//! first diagonal entry is positive. In a group without rows the column says
//! nothing and is not tested. Of a dependent set the first column is kept,
//! since columns are tested in order. With one column or no rows every column
//! is kept.
//!
//! # References
//!
//! 1. Golub GH, Van Loan CF. Matrix Computations. 4th ed. Baltimore: Johns
//!    Hopkins University Press; 2013.

use faer::{Mat, MatRef};

use epistat_core::{InputError, check};

use crate::error::{Error, check_length, check_weights};

/// The columns of a design that are linearly independent of the first
/// column and of the columns kept before them, in the weighted Gram matrix
/// `X'WX` of each group of rows.
///
/// Column 0, normally the intercept, is always kept. Each later column is
/// tested in order: it is dropped when its own diagonal entry is not positive
/// in a group that has rows, or when the share of it left after projecting
/// out the kept columns is at most `tolerance` in any group. Of a dependent
/// set, the first column is therefore kept. With `groups` `None` all rows form
/// one group; otherwise `groups[i]` numbers the group of row `i` from 0.
///
/// The test is a Cholesky factorisation grown one column at a time. A
/// `tolerance` of `1e-8` on the Gram matrix drops columns whose relative norm
/// left after projection is below `1e-4`.
///
/// Returns one flag per column, true where the column is kept.
///
/// # Errors
///
/// [`Error::Input`] when `weights` or `groups` does not have one entry per
/// row of `x`, for a weight that is negative or not finite, as in
/// [`weighted_gram`](crate::weighted_gram), and for a tolerance that is
/// negative or not finite.
pub fn independent_columns(
    x: MatRef<'_, f64>,
    weights: &[f64],
    groups: Option<&[usize]>,
    tolerance: f64,
) -> Result<Vec<bool>, Error> {
    let (rows, p) = (x.nrows(), x.ncols());
    check_weights(weights, rows)?;
    if !(tolerance.is_finite() && tolerance >= 0.0) {
        return Err(Error::Input(InputError::Domain {
            what: "tolerance",
            value: tolerance,
            domain: "[0, inf)",
            row: None,
        }));
    }
    if let Some(groups) = groups {
        check_length("groups", groups.len(), rows)?;
    }
    if p <= 1 || rows == 0 {
        return Ok(vec![true; p]);
    }
    let group_of = |row: usize| groups.map_or(0, |g| g[row]);
    let group_count = (0..rows).map(group_of).max().map_or(0, |g| g + 1);

    // One weighted Gram matrix per group, upper triangle, rows added in order.
    let size = p * p;
    let mut grams = vec![0.0; group_count * size];
    let mut nonzero = Vec::with_capacity(p);
    for row in 0..rows {
        let group = group_of(row);
        let gram = &mut grams[group * size..(group + 1) * size];
        let w = weights[row];
        nonzero.clear();
        nonzero.extend((0..p).filter(|&c| x[(row, c)] != 0.0));
        for (a, &i) in nonzero.iter().enumerate() {
            let wi = w * x[(row, i)];
            for &j in &nonzero[a..] {
                gram[i * p + j] += wi * x[(row, j)];
            }
        }
    }
    let entry = |gram: &[f64], i: usize, j: usize| {
        let (a, b) = if i <= j { (i, j) } else { (j, i) };
        gram[a * p + b]
    };

    let mut kept: Vec<usize> = Vec::with_capacity(p);
    let mut factors: Vec<Vec<Vec<f64>>> = vec![Vec::new(); group_count];
    for candidate in 0..p {
        let mut projections = Vec::with_capacity(group_count);
        let mut independent = true;
        for (group, factor) in factors.iter().enumerate() {
            let gram = &grams[group * size..(group + 1) * size];
            let own = entry(gram, candidate, candidate);
            if own <= 0.0 {
                // A group without rows says nothing about the column.
                if gram[0] > 0.0 {
                    independent = false;
                }
                projections.push(Vec::new());
                continue;
            }
            let mut projection = vec![0.0; kept.len()];
            for (r, &k) in kept.iter().enumerate() {
                let mut total = entry(gram, candidate, k);
                for c in 0..r {
                    total -= factor[r][c] * projection[c];
                }
                projection[r] = total / factor[r][r];
            }
            let residual = own - projection.iter().map(|v| v * v).sum::<f64>();
            if candidate > 0 && residual <= tolerance * own {
                independent = false;
            }
            projection.push(residual.max(0.0).sqrt());
            projections.push(projection);
        }
        if independent {
            for (factor, projection) in factors.iter_mut().zip(projections) {
                if !projection.is_empty() {
                    factor.push(projection);
                }
            }
            kept.push(candidate);
        }
    }
    let mut flags = vec![false; p];
    for k in kept {
        flags[k] = true;
    }
    Ok(flags)
}

/// The design matrix with the given columns, in order, after a column of ones
/// when `intercept` is true: column-major, one row per entry of a column.
///
/// This is the way from columns of a table to the `MatRef` the estimators
/// take, without a closure over the row and column index.
///
/// # Errors
///
/// [`Error::Input`] when there are no columns and no intercept, when a column
/// has no rows, when the columns differ in length, or when an entry is not
/// finite.
pub fn from_columns(columns: &[&[f64]], intercept: bool) -> Result<Mat<f64>, Error> {
    let Some(first) = columns.first() else {
        return Err(Error::Input(InputError::Empty { what: "columns" }));
    };
    let n = first.len();
    check::non_empty("columns", n)?;
    for column in columns {
        check::same_length("columns", n, column.len())?;
        check::finite("columns", column)?;
    }
    let offset = usize::from(intercept);
    Ok(Mat::from_fn(n, columns.len() + offset, |i, j| {
        if j < offset {
            1.0
        } else {
            columns[j - offset][i]
        }
    }))
}

/// The columns of `x` whose flag is true, in order.
///
/// # Panics
///
/// If `keep` does not have one flag per column.
#[must_use]
pub fn select_columns(x: MatRef<'_, f64>, keep: &[bool]) -> Mat<f64> {
    assert_eq!(keep.len(), x.ncols(), "one flag per column");
    let columns: Vec<usize> = (0..x.ncols()).filter(|&j| keep[j]).collect();
    Mat::from_fn(x.nrows(), columns.len(), |i, k| x[(i, columns[k])])
}

/// The rows of `x` at `rows`, in that order and with repeats, as a bootstrap
/// resample needs.
#[must_use]
pub fn select_rows(x: MatRef<'_, f64>, rows: &[usize]) -> Mat<f64> {
    Mat::from_fn(rows.len(), x.ncols(), |k, j| x[(rows[k], j)])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn design(rows: &[&[f64]]) -> Mat<f64> {
        Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
    }

    #[test]
    fn full_rank_designs_keep_every_column() {
        let x = design(&[
            &[1.0, 0.0, 2.0],
            &[1.0, 1.0, 3.0],
            &[1.0, 0.0, 5.0],
            &[1.0, 1.0, 1.0],
        ]);
        assert_eq!(
            independent_columns(x.as_ref(), &[1.0; 4], None, 1e-8).unwrap(),
            [true; 3]
        );
    }

    #[test]
    fn dummies_that_sum_to_the_intercept_lose_the_last() {
        let x = design(&[
            &[1.0, 1.0, 0.0],
            &[1.0, 0.0, 1.0],
            &[1.0, 1.0, 0.0],
            &[1.0, 0.0, 1.0],
        ]);
        assert_eq!(
            independent_columns(x.as_ref(), &[1.0; 4], None, 1e-8).unwrap(),
            [true, true, false]
        );
    }

    #[test]
    fn a_column_constant_in_one_group_is_dropped() {
        // Column 1 varies overall but is zero throughout group 1.
        let x = design(&[&[1.0, 1.0], &[1.0, 0.0], &[1.0, 0.0], &[1.0, 0.0]]);
        let pooled = independent_columns(x.as_ref(), &[1.0; 4], None, 1e-8).unwrap();
        let grouped =
            independent_columns(x.as_ref(), &[1.0; 4], Some(&[0, 0, 1, 1]), 1e-8).unwrap();
        assert_eq!(pooled, [true, true]);
        assert_eq!(grouped, [true, false]);
    }

    #[test]
    fn a_design_from_columns_has_the_intercept_first() {
        let age = [30.0, 41.0, 52.0];
        let treated = [0.0, 1.0, 1.0];
        let x = from_columns(&[&age, &treated], true).unwrap();
        assert_eq!((x.nrows(), x.ncols()), (3, 3));
        assert_eq!([x[(1, 0)], x[(1, 1)], x[(1, 2)]], [1.0, 41.0, 1.0]);
        let without = from_columns(&[&age], false).unwrap();
        assert_eq!((without.ncols(), without[(2, 0)]), (1, 52.0));
        assert!(matches!(
            from_columns(&[], true),
            Err(Error::Input(InputError::Empty { .. }))
        ));
        assert!(matches!(
            from_columns(&[&age, &treated[..2]], true),
            Err(Error::Input(InputError::Length { .. }))
        ));
        assert!(matches!(
            from_columns(&[&[1.0, f64::NAN]], false),
            Err(Error::Input(InputError::NotFinite { row: 1, .. }))
        ));
    }

    #[test]
    fn invalid_input_is_an_error() {
        let x = design(&[&[1.0, 1.0], &[1.0, 0.0], &[1.0, 2.0]]);
        let check = |w: &[f64], groups: Option<&[usize]>, tol: f64| {
            independent_columns(x.as_ref(), w, groups, tol).unwrap_err()
        };
        assert_eq!(
            check(&[1.0, -1.0, 1.0], None, 1e-8),
            Error::Input(InputError::Domain {
                what: "weights",
                value: -1.0,
                domain: "[0, inf)",
                row: None
            })
        );
        assert_eq!(
            check(&[1.0, 1.0, f64::NAN], None, 1e-8),
            Error::Input(InputError::NotFinite {
                what: "weights",
                row: 2,
                column: None
            })
        );
        assert!(matches!(
            check(&[1.0; 3], Some(&[0, 1]), 1e-8),
            Error::Input(InputError::Length { .. })
        ));
        assert!(matches!(
            check(&[1.0; 3], None, -1.0),
            Error::Input(InputError::Domain {
                what: "tolerance",
                ..
            })
        ));
    }

    #[test]
    fn selections() {
        let x = Mat::from_fn(3, 3, |i, j| (i * 3 + j) as f64);
        let kept = select_columns(x.as_ref(), &[true, false, true]);
        assert_eq!((kept.ncols(), kept[(2, 1)]), (2, 8.0));
        let rows = select_rows(x.as_ref(), &[2, 2, 0]);
        assert_eq!((rows.nrows(), rows[(1, 0)], rows[(2, 2)]), (3, 6.0, 2.0));
    }
}

//! Counts of an outcome's values and the removal of constant design columns.

use epistat_linalg::{Mat, MatRef};

use crate::utils::usize_to_f64;

/// Counts of an outcome's values by sign and finiteness.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutcomeDiagnostics {
    /// Rows.
    pub n_rows: usize,
    /// Finite values.
    pub n_finite: usize,
    /// NaN or infinite values.
    pub n_non_finite: usize,
    /// Finite negative values.
    pub n_negative: usize,
    /// Zeros.
    pub n_zero: usize,
    /// Positive values.
    pub n_positive: usize,
    /// The share of the finite values that are positive, `None` when no
    /// value is finite.
    pub positive_share: Option<f64>,
    /// The share of the finite values that are zero, `None` when no value is
    /// finite.
    pub zero_share: Option<f64>,
}

/// Counts of the values of `outcome`.
#[must_use]
pub fn outcome_diagnostics(outcome: &[f64]) -> OutcomeDiagnostics {
    let n_rows = outcome.len();
    let mut n_finite = 0usize;
    let mut n_negative = 0usize;
    let mut n_zero = 0usize;
    let mut n_positive = 0usize;

    for &value in outcome {
        if !value.is_finite() {
            continue;
        }
        n_finite += 1;
        if value < 0.0 {
            n_negative += 1;
        } else if value == 0.0 {
            n_zero += 1;
        } else {
            n_positive += 1;
        }
    }

    let n_non_finite = n_rows.saturating_sub(n_finite);
    let share = |count: usize| (n_finite > 0).then(|| usize_to_f64(count) / usize_to_f64(n_finite));
    let positive_share = share(n_positive);
    let zero_share = share(n_zero);

    OutcomeDiagnostics {
        n_rows,
        n_finite,
        n_non_finite,
        n_negative,
        n_zero,
        n_positive,
        positive_share,
        zero_share,
    }
}

/// Whether the range of `column` exceeds `|tolerance|`. False for a column out of range or fewer than two rows.
#[must_use]
pub fn column_has_variation(x: MatRef<'_, f64>, column: usize, tolerance: f64) -> bool {
    if column >= x.ncols() || x.nrows() < 2 {
        return false;
    }
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for row in 0..x.nrows() {
        let value = x[(row, column)];
        min = min.min(value);
        max = max.max(value);
    }
    (max - min).abs() > tolerance.abs()
}

#[must_use]
fn nonconstant_column_indices(
    x: MatRef<'_, f64>,
    tolerance: f64,
    always_keep: &[usize],
) -> Vec<usize> {
    // `col` is always in `0..x.ncols()`, so any out-of-range `always_keep`
    // index can never match — a direct slice lookup needs no `HashSet` or bounds
    // filtering, and `always_keep` is tiny in practice.
    let mut cols = Vec::new();
    for col in 0..x.ncols() {
        if always_keep.contains(&col) || column_has_variation(x, col, tolerance) {
            cols.push(col);
        }
    }
    cols
}

#[must_use]
fn select_columns(x: MatRef<'_, f64>, columns: &[usize]) -> Mat<f64> {
    Mat::from_fn(x.nrows(), columns.len(), |row, col| x[(row, columns[col])])
}

/// The columns of `x` with variation above `tolerance`, and those in `always_keep`, with their indices.
#[must_use]
pub fn drop_constant_columns(
    x: MatRef<'_, f64>,
    tolerance: f64,
    always_keep: &[usize],
) -> (Mat<f64>, Vec<usize>) {
    let kept_columns = nonconstant_column_indices(x, tolerance, always_keep);
    (select_columns(x, &kept_columns), kept_columns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcome_diagnostics_counts_values() {
        let y = [-1.0, 0.0, 2.0, 3.0, f64::NAN];
        let diag = outcome_diagnostics(&y);
        assert_eq!(diag.n_rows, 5);
        assert_eq!(diag.n_finite, 4);
        assert_eq!(diag.n_non_finite, 1);
        assert_eq!(diag.n_negative, 1);
        assert_eq!(diag.n_zero, 1);
        assert_eq!(diag.n_positive, 2);
        assert_eq!(diag.positive_share, Some(0.5));
        assert_eq!(diag.zero_share, Some(0.25));
        assert_eq!(outcome_diagnostics(&[f64::NAN]).positive_share, None);
    }

    #[test]
    fn drop_constant_columns_keeps_forced_and_variable_columns() {
        let x = Mat::from_fn(4, 4, |row, col| match col {
            0 => 1.0,
            1 => {
                if row < 2 {
                    0.0
                } else {
                    1.0
                }
            }
            2 => 3.0,
            _ => usize_to_f64(row),
        });

        let (trimmed, kept) = drop_constant_columns(x.as_ref(), 1e-12, &[0, 1]);
        assert_eq!(kept, vec![0, 1, 3]);
        assert_eq!(trimmed.ncols(), 3);
        assert!((trimmed[(3, 2)] - 3.0).abs() < 1e-12);
    }
}

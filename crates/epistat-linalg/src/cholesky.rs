//! The lower Cholesky factor of a symmetric matrix on a row-major slice.
//!
//! The factor `L` with `A = L L^T` is built row by row. Entry `(i, j)` for
//! `j <= i` is `(a_ij - sum_{k<j} l_ik l_jk) / l_jj`, the sum taken in column
//! order with no fused multiply-add, and the diagonal entry is the square
//! root of the same difference. What counts as a usable pivot is the caller's
//! choice, see [`Pivot`].

/// How a pivot, the difference under the square root of a diagonal entry, is
/// judged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pivot {
    /// A pivot must be finite and positive, otherwise there is no factor. The
    /// matrix must be positive definite.
    Strict,
    /// For a positive semidefinite matrix. A pivot below `-negative_floor`
    /// gives no factor, one in between is taken as zero. An off-diagonal
    /// entry in a column whose diagonal entry is at most `pivot_floor` is
    /// left at zero.
    Semidefinite {
        /// The most negative pivot that is rounding error.
        negative_floor: f64,
        /// The largest diagonal entry that counts as zero.
        pivot_floor: f64,
    },
}

/// The lower Cholesky factor of the row-major `matrix` of side `dimension`,
/// row-major with zeros above the diagonal. `None` when `matrix` does not
/// have `dimension * dimension` entries or a pivot is refused by `pivot`.
#[must_use]
pub fn cholesky_lower(matrix: &[f64], dimension: usize, pivot: Pivot) -> Option<Vec<f64>> {
    if matrix.len() != dimension * dimension {
        return None;
    }
    let n = dimension;
    let mut lower = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i * n + j];
            for k in 0..j {
                sum -= lower[i * n + k] * lower[j * n + k];
            }
            match pivot {
                Pivot::Strict => {
                    if i == j {
                        if !(sum.is_finite() && sum > 0.0) {
                            return None;
                        }
                        lower[i * n + i] = sum.sqrt();
                    } else {
                        lower[i * n + j] = sum / lower[j * n + j];
                    }
                }
                Pivot::Semidefinite {
                    negative_floor,
                    pivot_floor,
                } => {
                    if i == j {
                        if sum < -negative_floor {
                            return None;
                        }
                        lower[i * n + j] = sum.max(0.0).sqrt();
                    } else if lower[j * n + j] > pivot_floor {
                        lower[i * n + j] = sum / lower[j * n + j];
                    }
                }
            }
        }
    }
    Some(lower)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_positive_definite_matrix_factors_in_both_rules() {
        let a = [4.0, 2.0, 2.0, 3.0];
        let strict = cholesky_lower(&a, 2, Pivot::Strict).unwrap();
        let loose = cholesky_lower(
            &a,
            2,
            Pivot::Semidefinite {
                negative_floor: 1e-10,
                pivot_floor: 1e-12,
            },
        )
        .unwrap();
        assert_eq!(strict, loose);
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&strict), bits(&[2.0, 0.0, 1.0, 2.0_f64.sqrt()]));
    }

    #[test]
    fn a_singular_matrix_is_refused_only_by_the_strict_rule() {
        let a = [1.0, 1.0, 1.0, 1.0];
        assert!(cholesky_lower(&a, 2, Pivot::Strict).is_none());
        let lower = cholesky_lower(
            &a,
            2,
            Pivot::Semidefinite {
                negative_floor: 1e-10,
                pivot_floor: 1e-12,
            },
        )
        .unwrap();
        assert_eq!(lower, vec![1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn a_wrong_length_has_no_factor() {
        assert!(cholesky_lower(&[1.0, 2.0, 3.0], 2, Pivot::Strict).is_none());
    }
}

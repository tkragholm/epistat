//! The `L D L'` factorisation of an information matrix as `coxph` and
//! `survreg` take it, shared by [`crate::cox`] and [`crate::survreg`].
//!
//! Matrices are row-major slices of side `p`, factorised in place: the strict
//! lower triangle holds `L` and the diagonal holds `D`. A pivot below the
//! tolerance times the largest diagonal element is set to zero together with
//! the column of `L` under it, and that column is left out of every solve and
//! of the inverse. This is how a linearly dependent column is aliased.
//! Reference: Therneau and Grambsch (2000), *Modeling Survival Data*,
//! Section 3.

/// What the factorisation found.
pub(crate) struct Pivots {
    /// The number of pivots kept.
    pub rank: usize,
    /// False when a pivot is not finite or is clearly negative, that is below
    /// minus 8 times the floor. The matrix is then not positive
    /// semi-definite and the factors are not to be used.
    pub clean: bool,
}

/// The `L D L'` factorisation of a symmetric matrix, in place.
pub(crate) fn factorise(a: &mut [f64], p: usize, tolerance: f64) -> Pivots {
    let largest = (0..p).map(|i| a[i * p + i]).fold(0.0_f64, f64::max);
    let floor = if largest == 0.0 {
        tolerance
    } else {
        largest * tolerance
    };
    let mut rank = 0;
    let mut clean = true;
    for i in 0..p {
        let pivot = a[i * p + i];
        if !pivot.is_finite() || pivot < floor {
            if !pivot.is_finite() || pivot < -8.0 * floor {
                clean = false;
            }
            a[i * p + i] = 0.0;
            for j in i + 1..p {
                a[j * p + i] = 0.0;
            }
        } else {
            rank += 1;
            for j in i + 1..p {
                let t = a[j * p + i] / pivot;
                a[j * p + i] = t;
                a[j * p + j] -= t * t * pivot;
                for k in j + 1..p {
                    a[k * p + j] -= t * a[k * p + i];
                }
            }
        }
    }
    Pivots { rank, clean }
}

/// Solve `L D L' z = y` in place, with zero where `D` is zero.
pub(crate) fn solve(a: &[f64], p: usize, y: &mut [f64]) {
    for i in 0..p {
        let mut t = y[i];
        for j in 0..i {
            t -= y[j] * a[i * p + j];
        }
        y[i] = t;
    }
    for i in (0..p).rev() {
        if a[i * p + i] == 0.0 {
            y[i] = 0.0;
        } else {
            let mut t = y[i] / a[i * p + i];
            for j in i + 1..p {
                t -= y[j] * a[j * p + i];
            }
            y[i] = t;
        }
    }
}

/// The inverse over the kept columns, with zero rows and columns for the
/// others, exactly symmetric.
pub(crate) fn inverse(a: &[f64], p: usize) -> Vec<f64> {
    let mut inverse = vec![0.0_f64; p * p];
    for k in 0..p {
        if a[k * p + k] == 0.0 {
            continue;
        }
        let mut column = vec![0.0_f64; p];
        column[k] = 1.0;
        solve(a, p, &mut column);
        for (i, value) in column.iter().enumerate() {
            inverse[i * p + k] = *value;
        }
    }
    for i in 0..p {
        for j in 0..i {
            inverse[j * p + i] = inverse[i * p + j];
        }
    }
    inverse
}

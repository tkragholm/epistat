//! The `L D L'` factorisation of an information matrix as `coxph` and
//! `survreg` take it, shared by [`crate::cox`] and [`crate::survreg`].
//!
//! Matrices are row-major slices of side `p`, factorised in place: the strict
//! lower triangle holds `L` and the diagonal holds `D`. A pivot below the
//! tolerance times the largest diagonal element is set to zero together with
//! the column of `L` under it, and that column is left out of every solve and
//! of the inverse. This is how a linearly dependent column is aliased.
//! Columns are taken in the order given, so of two dependent columns the
//! later is aliased, as in R's model functions.
//!
//! # References
//!
//! 1. Golub GH, Van Loan CF. Matrix Computations. 4th ed. Baltimore: Johns
//!    Hopkins University Press; 2013. Section 4.1.
//! 2. Higham NJ. Accuracy and Stability of Numerical Algorithms. 2nd ed.
//!    Philadelphia: SIAM; 2002. Chapter 10.
//! 3. Therneau TM, Grambsch PM. Modeling Survival Data. New York: Springer;
//!    2000. Section 3.

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
///
/// Left-looking, column by column in the order given, without pivoting
/// (Golub and Van Loan 2013, Section 4.1). Only the diagonal and the strict
/// lower triangle of `a` are read; the strict upper triangle is left as it
/// is. With `w_k = L[j][k] D[k]`, column `j` is
///
/// ```text
/// d       = a[j][j] - sum_k w_k L[j][k]
/// L[i][j] = (a[i][j] - sum_k L[i][k] w_k) / d          for i > j
/// ```
///
/// with the sums over the kept columns `k < j` in ascending order, each term
/// one product subtracted from the running value, no fused multiply-add.
/// This order decides the last bits of every fit that uses the factors.
///
/// The floor is `tolerance` times the largest diagonal element of the input,
/// or `tolerance` itself when no diagonal element is positive. A pivot `d`
/// that is not above zero and at least the floor, NaN included, is aliased:
/// its `D` and the column of `L` under it are set to zero. The floor is
/// relative to the largest diagonal and not to the column's own, so it is not
/// invariant to rescaling one column; this is the rule of `coxph`'s
/// aliasing. The factors are not clean when a pivot is not finite or is below
/// minus 8 floors: the rounding error in `d` is of the order of
/// `p eps` times the largest diagonal, far below a floor of `eps^(3/4)` unless
/// `p` is in the thousands, so a pivot that negative is evidence that the
/// matrix is not positive semi-definite, not rounding.
pub(crate) fn factorise(a: &mut [f64], p: usize, tolerance: f64) -> Pivots {
    debug_assert_eq!(a.len(), p * p);
    let mut largest = 0.0_f64;
    for j in 0..p {
        let v = a[j * p + j];
        if v > largest {
            largest = v;
        }
    }
    let floor = if largest > 0.0 {
        tolerance * largest
    } else {
        tolerance
    };
    let mut rank = 0;
    let mut clean = true;
    let mut w = vec![0.0_f64; p];
    for j in 0..p {
        let row = j * p;
        let mut d = a[row + j];
        for k in 0..j {
            let dk = a[k * p + k];
            if dk == 0.0 {
                w[k] = 0.0;
                continue;
            }
            let wk = a[row + k] * dk;
            w[k] = wk;
            d -= wk * a[row + k];
        }
        if !d.is_finite() || d < -8.0 * floor {
            clean = false;
        }
        if d > 0.0 && d >= floor {
            a[row + j] = d;
            rank += 1;
            for i in j + 1..p {
                let ri = i * p;
                let mut s = a[ri + j];
                for k in 0..j {
                    if w[k] != 0.0 {
                        s -= a[ri + k] * w[k];
                    }
                }
                a[ri + j] = s / d;
            }
        } else {
            a[row + j] = 0.0;
            for i in j + 1..p {
                a[i * p + j] = 0.0;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `toler.chol` of `coxph.control`, `eps^(3/4)`.
    const TOLER_CHOL: f64 = 1.818_989_403_545_856_5e-12;

    /// An exact fraction, enough for the factors of a small integer matrix.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Q(i128, i128);

    fn gcd(a: i128, b: i128) -> i128 {
        if b == 0 { a.abs() } else { gcd(b, a % b) }
    }

    impl Q {
        fn new(n: i128, d: i128) -> Self {
            let g = gcd(n, d).max(1) * d.signum();
            Self(n / g, d / g)
        }
        fn sub(self, o: Self) -> Self {
            Self::new(self.0 * o.1 - o.0 * self.1, self.1 * o.1)
        }
        fn mul(self, o: Self) -> Self {
            Self::new(self.0 * o.0, self.1 * o.1)
        }
        fn div(self, o: Self) -> Self {
            Self::new(self.0 * o.1, self.1 * o.0)
        }
        #[expect(clippy::cast_precision_loss, reason = "small exact integers")]
        fn value(self) -> f64 {
            self.0 as f64 / self.1 as f64
        }
    }

    /// `X'X` for an integer `X` of six rows whose column 3 is column 0 plus
    /// column 1, so column 3 is exactly dependent on the columns before it.
    const GRAM: [[i128; 5]; 5] = [
        [6, 3, 3, 9, 4],
        [3, 19, -4, 22, -2],
        [3, -4, 7, -1, 3],
        [9, 22, -1, 31, 2],
        [4, -2, 3, 2, 8],
    ];

    fn gram() -> Vec<f64> {
        #[expect(clippy::cast_precision_loss, reason = "small exact integers")]
        GRAM.iter().flatten().map(|&v| v as f64).collect()
    }

    /// The exact `L D L'` with the dependent column's pivot zero and its
    /// column of `L` zero.
    fn exact_factors() -> (Vec<Q>, Vec<Q>) {
        let p = 5;
        let mut l = vec![Q(0, 1); p * p];
        let mut d = vec![Q(0, 1); p];
        for j in 0..p {
            let mut dj = Q(GRAM[j][j], 1);
            for k in 0..j {
                dj = dj.sub(l[j * p + k].mul(l[j * p + k]).mul(d[k]));
            }
            d[j] = dj;
            if dj.0 == 0 {
                continue;
            }
            for i in j + 1..p {
                let mut s = Q(GRAM[i][j], 1);
                for k in 0..j {
                    s = s.sub(l[i * p + k].mul(d[k]).mul(l[j * p + k]));
                }
                l[i * p + j] = s.div(dj);
            }
        }
        (l, d)
    }

    // The factors of the kept columns are within `kappa gamma_{p+1}` of the
    // exact ones, relative to their scale, with `kappa` about 14.3 the
    // condition number of the kept block and `gamma_6 = 6 eps / (1 - 6 eps)`
    // (Higham 2002, Theorem 10.3, carried to the factors to first order):
    // about 86 eps, here rounded up to 128 eps.
    const FACTOR_BOUND: f64 = 128.0 * f64::EPSILON;

    #[test]
    fn an_exactly_dependent_column_is_aliased_and_the_factors_are_exact() {
        let p = 5;
        let mut a = gram();
        let pivots = factorise(&mut a, p, TOLER_CHOL);
        assert_eq!(pivots.rank, 4);
        assert!(pivots.clean);
        let (l, d) = exact_factors();
        assert_eq!(d[3], Q(0, 1));
        assert_eq!(a[3 * p + 3], 0.0);
        for i in 4..p {
            assert_eq!(a[i * p + 3], 0.0);
        }
        for j in 0..p {
            let (got, want) = (a[j * p + j], d[j].value());
            assert!(
                (got - want).abs() <= FACTOR_BOUND * want.abs().max(1.0),
                "D[{j}] {got} {want}"
            );
            for i in j + 1..p {
                let (got, want) = (a[i * p + j], l[i * p + j].value());
                assert!(
                    (got - want).abs() <= FACTOR_BOUND * want.abs().max(1.0),
                    "L[{i}][{j}] {got} {want}"
                );
            }
        }
    }

    #[test]
    fn the_upper_triangle_is_not_read_or_written() {
        let p = 5;
        let mut a = gram();
        let mut b = gram();
        for i in 0..p {
            for j in i + 1..p {
                b[i * p + j] = f64::NAN;
            }
        }
        factorise(&mut a, p, TOLER_CHOL);
        factorise(&mut b, p, TOLER_CHOL);
        for i in 0..p {
            for j in 0..=i {
                assert_eq!(a[i * p + j].to_bits(), b[i * p + j].to_bits());
            }
            for j in i + 1..p {
                assert!(b[i * p + j].is_nan());
            }
        }
    }

    #[test]
    fn solve_and_inverse_equal_the_inverse_of_the_kept_block() {
        let p = 5;
        let kept = [0usize, 1, 2, 4];
        // The exact inverse of the kept block by Gauss-Jordan on fractions.
        let q = kept.len();
        let mut m: Vec<Q> = Vec::new();
        for &r in &kept {
            for &c in &kept {
                m.push(Q(GRAM[r][c], 1));
            }
            for &c in &kept {
                m.push(Q(i128::from(c == r), 1));
            }
        }
        let w = 2 * q;
        for col in 0..q {
            let pivot = (col..q)
                .find(|&r| m[r * w + col].0 != 0)
                .expect("non-singular");
            for c in 0..w {
                m.swap(col * w + c, pivot * w + c);
            }
            let inv = m[col * w + col];
            for c in 0..w {
                m[col * w + c] = m[col * w + c].div(inv);
            }
            for r in 0..q {
                if r != col {
                    let f = m[r * w + col];
                    for c in 0..w {
                        m[r * w + c] = m[r * w + c].sub(f.mul(m[col * w + c]));
                    }
                }
            }
        }
        let mut exact = vec![0.0; p * p];
        for (a, &r) in kept.iter().enumerate() {
            for (b, &c) in kept.iter().enumerate() {
                exact[r * p + c] = m[a * w + q + b].value();
            }
        }
        let mut a = gram();
        factorise(&mut a, p, TOLER_CHOL);
        let inverse = inverse(&a, p);
        // The inverse's entries carry the factors' error times the condition
        // number once more: 14.3 * 128 eps, below 2048 eps, relative to the
        // largest exact entry.
        let scale = exact.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
        let bound = 2048.0 * f64::EPSILON * scale;
        for i in 0..p {
            for j in 0..p {
                assert_eq!(inverse[i * p + j].to_bits(), inverse[j * p + i].to_bits());
                assert!(
                    (inverse[i * p + j] - exact[i * p + j]).abs() <= bound,
                    "({i}, {j})"
                );
            }
        }
        for i in 0..p {
            assert_eq!(inverse[i * p + 3], 0.0);
        }
        let y = [1.0, -2.0, 0.5, 3.0, 4.0];
        let mut z = y;
        solve(&a, p, &mut z);
        for i in 0..p {
            let want: f64 = (0..p).map(|j| exact[i * p + j] * y[j]).sum();
            assert!((z[i] - want).abs() <= bound * 8.0, "z[{i}] {} {want}", z[i]);
        }
        assert_eq!(z[3], 0.0);
    }

    /// `[[1, r], [r, 1]]` with `1 - r^2 = s`: the second pivot is `s` up to a
    /// rounding of about `eps`, far below the floors tried.
    fn pair(s: f64) -> Vec<f64> {
        let r = (1.0 - s).sqrt();
        vec![1.0, r, r, 1.0]
    }

    #[test]
    fn a_nearly_dependent_column_is_kept_above_the_floor_and_aliased_below() {
        let tolerance = 1e-10;
        let mut above = pair(10.0 * tolerance);
        let kept = factorise(&mut above, 2, tolerance);
        assert_eq!((kept.rank, kept.clean), (2, true));
        let mut below = pair(tolerance / 10.0);
        let aliased = factorise(&mut below, 2, tolerance);
        assert_eq!((aliased.rank, aliased.clean), (1, true));
        assert_eq!(below[3], 0.0);
    }

    #[test]
    fn zero_negative_and_nan_matrices() {
        let mut zero = vec![0.0; 9];
        let z = factorise(&mut zero, 3, TOLER_CHOL);
        assert_eq!((z.rank, z.clean), (0, true));
        let mut negative = vec![-1.0, 0.0, 0.0, -2.0];
        let n = factorise(&mut negative, 2, TOLER_CHOL);
        assert_eq!(n.rank, 0);
        assert!(!n.clean);
        let mut nan = vec![2.0, 0.0, f64::NAN, 1.0];
        let m = factorise(&mut nan, 2, TOLER_CHOL);
        assert!(!m.clean);
        assert_eq!(m.rank, 1);
        let mut empty: Vec<f64> = Vec::new();
        let e = factorise(&mut empty, 0, TOLER_CHOL);
        assert_eq!((e.rank, e.clean), (0, true));
    }

    #[test]
    fn a_zero_floor_aliases_an_exactly_zero_pivot() {
        let mut a = vec![1.0, 1.0, 1.0, 1.0];
        let pivots = factorise(&mut a, 2, 0.0);
        assert_eq!((pivots.rank, pivots.clean), (1, true));
        assert_eq!(a[3], 0.0);
    }
}

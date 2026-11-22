//! The natural cubic spline basis of R's `splines::ns`.
//!
//! # Algorithm
//!
//! Let `B(x)` be the `n_b = m + 4` cubic B-splines on the boundary knots
//! `[lower, upper]`, repeated four times each, and `m` interior knots. A natural
//! spline is a cubic spline whose second derivative is zero at both boundary
//! knots, which is two linear constraints on the coefficients of `B`. Let `C` be
//! the `n_b x 2` matrix of `B''(lower)` and `B''(upper)` as columns. With
//! `intercept = false`, `ns` first drops the first B-spline from `B` and from
//! `C`. The Householder QR decomposition `C = Q R` gives an orthogonal `Q`,
//! and the natural basis is the columns of `B Q` after the first two:
//!
//! ```text
//! N(x) = B(x) Q_2,    Q_2 = Q[:, 3..n_b]
//! ```
//!
//! The functions span the constrained space, and are the same numbers as
//! R's when the reflections carry the signs LINPACK's `dqrdc2` gives them:
//! `H = I - u u' / u_1` with `u = x / s`, `u_1 <- u_1 + 1`, where `s` is the
//! norm of the column from its pivot row, with the sign of the pivot entry.
//! The matrix `Q_2` is formed once, so a row costs four multiplications by
//! rows of `Q_2` per function, from the four non-zero B-splines found by
//! the triangular scheme of de Boor \[1\].
//!
//! The second derivatives at the boundary come from the same scheme, extended
//! to derivatives \[1, 2\]. They are exact, not numerical differences.
//!
//! Outside the boundary, `ns` extrapolates linearly from the value and the
//! slope at the boundary knot, `B(k) + (x - k) B'(k)`, before the projection.
//! Here the projection is applied first, which is the same linear map.
//!
//! # Conditioning
//!
//! B-splines have local support and sum to one, so the basis is well
//! conditioned: for 2000 uniform values with three interior knots the design
//! has condition number 8.9, against 543 for the restricted cubic form scaled
//! by `(t_k - t_1)^2` and 1.3e6 unscaled. The QR of the constraint matrix has
//! condition number of order one for knots that are not very unevenly spaced.
//!
//! # Prediction
//!
//! The basis stores its knots, boundary and `Q_2`, and evaluates new values
//! with them. [`NaturalSpline::count_outside`] counts values outside the
//! boundary, where the function is a straight line.
//!
//! # References
//!
//! 1. de Boor C. A Practical Guide to Splines. Revised ed. New York:
//!    Springer; 2001.
//! 2. Piegl L, Tiller W. The NURBS Book. 2nd ed. Berlin: Springer; 1997.
//!    Algorithm A2.3.
//! 3. Dongarra JJ, Bunch JR, Moler CB, Stewart GW. LINPACK Users' Guide.
//!    Philadelphia: SIAM; 1979. The routines `dqrdc` and `dqrsl`.

use super::{BSpline, Basis, KnotError};

/// The natural cubic spline basis of `splines::ns`.
#[derive(Clone, Debug, PartialEq)]
pub struct NaturalSpline {
    bspline: BSpline,
    boundary: (f64, f64),
    intercept: bool,
    /// `Q_2`, row-major, one row per B-spline kept.
    projection: Vec<f64>,
    columns: usize,
    /// The basis and its slope at the lower and the upper boundary.
    lower: (Vec<f64>, Vec<f64>),
    upper: (Vec<f64>, Vec<f64>),
}

fn reflect(u: &[f64], y: &mut [f64]) {
    let t = -u.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f64>() / u[0];
    for (v, a) in y.iter_mut().zip(u) {
        *v += t * a;
    }
}

/// One LINPACK reflection of the leading part of `column`: the vector `u`, and
/// `None` when the column is zero there.
fn reflector(column: &[f64]) -> Option<Vec<f64>> {
    let norm = column.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm == 0.0 {
        return None;
    }
    let norm = norm.copysign(column[0]);
    let mut u: Vec<f64> = column.iter().map(|v| v / norm).collect();
    u[0] += 1.0;
    Some(u)
}

impl NaturalSpline {
    /// The basis with the given interior knots on `[lower, upper]`. With
    /// `intercept = false` the constant function is excluded from the span, as
    /// in `splines::ns`.
    ///
    /// # Errors
    ///
    /// [`KnotError`] as [`BSpline::new`] gives it, for a boundary that is not an
    /// increasing pair of finite numbers and interior knots that are unsorted
    /// or not strictly inside it.
    pub fn new(interior: &[f64], boundary: (f64, f64), intercept: bool) -> Result<Self, KnotError> {
        let bspline = BSpline::new(3, interior, boundary.0, boundary.1)?;
        let full = bspline.len();
        let skip = usize::from(!intercept);
        let kept = full - skip;
        let second = |x: f64| {
            let (first, ders) = bspline
                .nonzero_derivatives(x, 2)
                .ok_or(KnotError::Boundary)?;
            let mut row = vec![0.0; full];
            row[first..first + 4].copy_from_slice(&ders[2]);
            Ok(row.split_off(skip))
        };
        let (mut left, mut right) = (second(boundary.0)?, second(boundary.1)?);
        let h0 = reflector(&left);
        if let Some(u) = &h0 {
            reflect(u, &mut right);
        }
        let h1 = reflector(&right[1..]);
        let columns = kept - 2;
        let mut projection = vec![0.0; kept * columns];
        for j in 0..columns {
            let mut e = vec![0.0; kept];
            e[j + 2] = 1.0;
            if let Some(u) = &h1 {
                reflect(u, &mut e[1..]);
            }
            if let Some(u) = &h0 {
                reflect(u, &mut e);
            }
            for (r, v) in e.iter().enumerate() {
                projection[r * columns + j] = *v;
            }
        }
        left.clear();
        let mut spline = Self {
            bspline,
            boundary,
            intercept,
            projection,
            columns,
            lower: (Vec::new(), Vec::new()),
            upper: (Vec::new(), Vec::new()),
        };
        spline.lower = spline.at_boundary(boundary.0)?;
        spline.upper = spline.at_boundary(boundary.1)?;
        Ok(spline)
    }

    fn at_boundary(&self, x: f64) -> Result<(Vec<f64>, Vec<f64>), KnotError> {
        let mut value = vec![0.0; self.columns];
        let mut slope = vec![0.0; self.columns];
        let (first, ders) = self
            .bspline
            .nonzero_derivatives(x, 1)
            .ok_or(KnotError::Boundary)?;
        self.project(first, &ders[0], &mut value);
        self.project(first, &ders[1], &mut slope);
        Ok((value, slope))
    }

    /// `out = sum_r b[r] Q_2[first + r - skip, :]` over the kept B-splines.
    fn project(&self, first: usize, b: &[f64], out: &mut [f64]) {
        let skip = usize::from(!self.intercept);
        out[..self.columns].fill(0.0);
        for (r, &value) in b.iter().enumerate() {
            let Some(row) = (first + r).checked_sub(skip) else {
                continue;
            };
            let q = &self.projection[row * self.columns..(row + 1) * self.columns];
            for (slot, &w) in out[..self.columns].iter_mut().zip(q) {
                *slot += value * w;
            }
        }
    }

    /// The boundary knots.
    #[must_use]
    pub const fn boundary(&self) -> (f64, f64) {
        self.boundary
    }

    /// The number of values of `x` outside the boundary knots, where the basis
    /// is extrapolated linearly.
    #[must_use]
    pub fn count_outside(&self, x: &[f64]) -> usize {
        x.iter()
            .filter(|&&v| v < self.boundary.0 || v > self.boundary.1)
            .count()
    }
}

impl Basis for NaturalSpline {
    fn len(&self) -> usize {
        self.columns
    }

    fn values_into(&self, x: f64, out: &mut [f64]) {
        let (lower, upper) = self.boundary;
        if x < lower || x > upper {
            let (value, slope) = if x < lower { &self.lower } else { &self.upper };
            let pivot = if x < lower { lower } else { upper };
            for ((slot, &v), &s) in out.iter_mut().zip(value).zip(slope) {
                *slot = v + (x - pivot) * s;
            }
            return;
        }
        let mut b = [0.0; 4];
        match self.bspline.nonzero_values_into(x, &mut b) {
            Some(first) => self.project(first, &b, out),
            None => out[..self.columns].fill(f64::NAN),
        }
    }

    fn derivatives_into(&self, x: f64, out: &mut [f64]) {
        let (lower, upper) = self.boundary;
        if x < lower {
            out[..self.columns].copy_from_slice(&self.lower.1);
        } else if x > upper {
            out[..self.columns].copy_from_slice(&self.upper.1);
        } else if let Some((first, ders)) = self.bspline.nonzero_derivatives(x, 1) {
            self.project(first, &ders[1], out);
        } else {
            out[..self.columns].fill(f64::NAN);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn basis() -> NaturalSpline {
        NaturalSpline::new(&[0.2, 0.5, 0.7], (0.0, 1.0), false).unwrap()
    }

    #[test]
    fn the_dimension_is_the_knots_plus_one_less_the_intercept() {
        assert_eq!(basis().len(), 4);
        assert_eq!(
            NaturalSpline::new(&[0.2, 0.5, 0.7], (0.0, 1.0), true)
                .unwrap()
                .len(),
            5
        );
        assert_eq!(NaturalSpline::new(&[], (0.0, 1.0), false).unwrap().len(), 1);
    }

    #[test]
    fn the_second_derivative_is_zero_at_the_boundary_and_the_ends_are_linear() {
        let b = basis();
        let h = 1e-3;
        for edge in [0.0, 1.0] {
            let (a, m, c) = if edge == 0.0 {
                (b.values(-h), b.values(0.0), b.values(h))
            } else {
                (b.values(1.0 - h), b.values(1.0), b.values(1.0 + h))
            };
            for j in 0..4 {
                // Linear across the boundary: the second difference vanishes
                // up to the third derivative times h.
                assert!((a[j] - 2.0 * m[j] + c[j]).abs() < 1e-5, "{edge} {j}");
            }
        }
        let (u, v, w) = (b.values(2.0), b.values(3.0), b.values(4.0));
        for j in 0..4 {
            assert!((u[j] - 2.0 * v[j] + w[j]).abs() < 1e-12);
        }
    }

    #[test]
    fn derivatives_match_central_differences() {
        let b = basis();
        let h = 1e-6;
        for x in [-0.4, 0.05, 0.3, 0.69, 0.95, 1.6] {
            let (up, down) = (b.values(x + h), b.values(x - h));
            let d = b.derivatives(x);
            for j in 0..4 {
                assert!(
                    ((up[j] - down[j]) / (2.0 * h) - d[j]).abs() < 1e-7,
                    "{x} {j}"
                );
            }
        }
    }

    #[test]
    fn invalid_knots_are_errors() {
        assert_eq!(
            NaturalSpline::new(&[1.5], (0.0, 1.0), false),
            Err(KnotError::OutsideBoundary)
        );
    }
}

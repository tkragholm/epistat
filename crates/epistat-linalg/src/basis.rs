//! Basis expansions of one variable, for series and sieve estimators.

/// A finite set of functions of one variable and their derivatives.
pub trait Basis {
    /// The number of functions.
    fn len(&self) -> usize;

    /// Whether the basis has no functions.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Writes the value of each function at `x` into `out`.
    fn values_into(&self, x: f64, out: &mut [f64]);

    /// Writes the first derivative of each function at `x` into `out`.
    fn derivatives_into(&self, x: f64, out: &mut [f64]);

    fn values(&self, x: f64) -> Vec<f64> {
        let mut out = vec![0.0; self.len()];
        self.values_into(x, &mut out);
        out
    }

    fn derivatives(&self, x: f64) -> Vec<f64> {
        let mut out = vec![0.0; self.len()];
        self.derivatives_into(x, &mut out);
        out
    }
}

/// `1, x, x^2, ..., x^degree`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Polynomial {
    pub degree: usize,
}

impl Basis for Polynomial {
    fn len(&self) -> usize {
        self.degree + 1
    }

    fn values_into(&self, x: f64, out: &mut [f64]) {
        let mut power = 1.0;
        for slot in &mut out[..self.len()] {
            *slot = power;
            power *= x;
        }
    }

    fn derivatives_into(&self, x: f64, out: &mut [f64]) {
        out[0] = 0.0;
        let (mut coefficient, mut power) = (1.0, 1.0);
        for slot in &mut out[1..self.len()] {
            *slot = coefficient * power;
            coefficient += 1.0;
            power *= x;
        }
    }
}

/// B-splines of a given degree on `[lower, upper]` with sorted internal
/// knots, by the Cox-de Boor recursion. The same basis as R's
/// `splines::bs(x, knots, degree, intercept = TRUE, Boundary.knots)`.
///
/// The last span is closed on the right, so the basis sums to one on the
/// whole closed interval.
#[derive(Clone, Debug, PartialEq)]
pub struct BSpline {
    degree: usize,
    knots: Vec<f64>,
}

impl BSpline {
    /// # Panics
    ///
    /// If the internal knots are not sorted or do not lie strictly inside
    /// `(lower, upper)`.
    #[must_use]
    pub fn new(degree: usize, internal_knots: &[f64], lower: f64, upper: f64) -> Self {
        assert!(
            internal_knots.windows(2).all(|pair| pair[0] <= pair[1])
                && internal_knots.iter().all(|&k| lower < k && k < upper),
            "internal knots must be sorted and inside the boundary"
        );
        let mut knots = vec![lower; degree + 1];
        knots.extend_from_slice(internal_knots);
        knots.extend(std::iter::repeat_n(upper, degree + 1));
        Self { degree, knots }
    }

    #[must_use]
    pub const fn degree(&self) -> usize {
        self.degree
    }

    fn spline(&self, i: usize, k: usize, x: f64) -> f64 {
        let t = &self.knots;
        if k == 0 {
            let last = i + 1 == self.len() && (x - t[i + 1]).abs() < 1e-12;
            return if (t[i] <= x && x < t[i + 1]) || last {
                1.0
            } else {
                0.0
            };
        }
        let mut value = 0.0;
        let left = t[i + k] - t[i];
        if left > 0.0 {
            value += (x - t[i]) / left * self.spline(i, k - 1, x);
        }
        let right = t[i + k + 1] - t[i + 1];
        if right > 0.0 {
            value += (t[i + k + 1] - x) / right * self.spline(i + 1, k - 1, x);
        }
        value
    }
}

impl Basis for BSpline {
    fn len(&self) -> usize {
        self.knots.len() - self.degree - 1
    }

    fn values_into(&self, x: f64, out: &mut [f64]) {
        for (i, slot) in out[..self.len()].iter_mut().enumerate() {
            *slot = self.spline(i, self.degree, x);
        }
    }

    fn derivatives_into(&self, x: f64, out: &mut [f64]) {
        let (n, d, t) = (self.len(), self.degree, &self.knots);
        if d == 0 {
            out[..n].fill(0.0);
            return;
        }
        let lower: Vec<f64> = (0..=n).map(|i| self.spline(i, d - 1, x)).collect();
        let p = d as f64;
        for (i, slot) in out[..n].iter_mut().enumerate() {
            let mut value = 0.0;
            let left = t[i + d] - t[i];
            if left > 0.0 {
                value += p / left * lower[i];
            }
            let right = t[i + d + 1] - t[i + 1];
            if right > 0.0 {
                value -= p / right * lower[i + 1];
            }
            *slot = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial_values_and_derivatives() {
        let basis = Polynomial { degree: 3 };
        assert_eq!(basis.values(2.0), [1.0, 2.0, 4.0, 8.0]);
        assert_eq!(basis.derivatives(2.0), [0.0, 1.0, 4.0, 12.0]);
    }

    #[test]
    fn b_splines_sum_to_one_on_the_closed_interval() {
        let basis = BSpline::new(3, &[0.25, 0.5, 0.5, 0.8], -1.0, 2.0);
        assert_eq!(basis.len(), 8);
        for x in [-1.0, -0.3, 0.25, 0.5, 0.79, 1.5, 2.0] {
            let total: f64 = basis.values(x).iter().sum();
            assert!((total - 1.0).abs() < 1e-14, "{x}: {total}");
            assert!(basis.derivatives(x).iter().sum::<f64>().abs() < 1e-12);
        }
    }

    #[test]
    #[should_panic(expected = "inside the boundary")]
    fn knots_outside_the_boundary_are_refused() {
        let _ = BSpline::new(2, &[1.5], 0.0, 1.0);
    }
}

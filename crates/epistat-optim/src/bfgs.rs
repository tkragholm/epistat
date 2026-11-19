//! BFGS with the line search of Moré and Thuente (ACM TOMS 20:286-307,
//! 1994), as in MINPACK's `cvsrch` and `cstep`.

use std::fmt;

/// The Moré-Thuente line search: finds a step that satisfies the strong
/// Wolfe conditions `f(x + a d) <= f(x) + c1 a g'd` and
/// `|g(x + a d)'d| <= c2 |g'd|`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoreThuente {
    /// Sufficient decrease, in `(0, c2)`.
    pub c1: f64,
    /// Curvature, in `(c1, 1)`.
    pub c2: f64,
    pub step_min: f64,
    pub step_max: f64,
    /// Relative width of the interval of uncertainty at which the search
    /// stops.
    pub x_tolerance: f64,
    pub max_evaluations: usize,
}

impl Default for MoreThuente {
    fn default() -> Self {
        Self {
            c1: 1e-4,
            c2: 0.9,
            step_min: f64::EPSILON.sqrt(),
            step_max: f64::INFINITY,
            x_tolerance: 1e-10,
            max_evaluations: 40,
        }
    }
}

/// BFGS on the inverse Hessian, starting from the identity unless told
/// otherwise, with a [`MoreThuente`] search of initial step 1 along each
/// direction.
///
/// Stops when the gradient norm is below `gradient_tolerance`, when the cost
/// changes by less than `cost_tolerance` in an iteration, or after
/// `max_iterations` iterations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bfgs {
    pub max_iterations: usize,
    pub gradient_tolerance: f64,
    pub cost_tolerance: f64,
    pub line_search: MoreThuente,
}

impl Default for Bfgs {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            gradient_tolerance: f64::EPSILON.sqrt(),
            cost_tolerance: f64::EPSILON,
            line_search: MoreThuente::default(),
        }
    }
}

/// The best point found.
#[derive(Clone, Debug, PartialEq)]
pub struct Minimum {
    pub x: Vec<f64>,
    pub value: f64,
    pub gradient: Vec<f64>,
    pub iterations: usize,
    /// Whether a tolerance was met before the iteration limit.
    pub converged: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BfgsError {
    /// The starting point is empty, or the gradient or inverse Hessian has the
    /// wrong length.
    Shape,
    /// The cost or gradient is not finite at the starting point.
    NotFinite,
    /// A search direction is not a descent direction.
    NotDescent,
}

impl fmt::Display for BfgsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Shape => "the starting point, gradient or inverse Hessian has the wrong length",
            Self::NotFinite => "the cost or gradient is not finite at the starting point",
            Self::NotDescent => "the search direction is not a descent direction",
        })
    }
}

impl std::error::Error for BfgsError {}

impl Bfgs {
    /// Minimises `f`, which returns the cost and the gradient at a point,
    /// from `x0`. `inverse_hessian`, row-major, replaces the identity as the
    /// starting inverse Hessian.
    ///
    /// An iteration whose step does not satisfy the curvature condition
    /// `y's > 0` keeps the previous inverse Hessian.
    ///
    /// # Errors
    ///
    /// See [`BfgsError`].
    pub fn minimize(
        &self,
        x0: &[f64],
        mut f: impl FnMut(&[f64]) -> (f64, Vec<f64>),
        inverse_hessian: Option<Vec<f64>>,
    ) -> Result<Minimum, BfgsError> {
        let n = x0.len();
        let mut h = inverse_hessian.unwrap_or_else(|| identity(n));
        if n == 0 || h.len() != n * n {
            return Err(BfgsError::Shape);
        }
        let mut x = x0.to_vec();
        let (mut cost, mut grad) = f(&x);
        if grad.len() != n {
            return Err(BfgsError::Shape);
        }
        if !cost.is_finite() || grad.iter().any(|g| !g.is_finite()) {
            return Err(BfgsError::NotFinite);
        }
        let mut best = Minimum {
            x: x.clone(),
            value: cost,
            gradient: grad.clone(),
            iterations: 0,
            converged: false,
        };
        for iteration in 1..=self.max_iterations {
            if norm(&grad) < self.gradient_tolerance {
                best.converged = true;
                return Ok(best);
            }
            let direction: Vec<f64> = (0..n)
                .map(|i| -dot(&h[i * n..(i + 1) * n], &grad))
                .collect();
            let step = self
                .line_search
                .search(&x, cost, &grad, &direction, &mut f)?;
            let s: Vec<f64> = step.x.iter().zip(&x).map(|(a, b)| a - b).collect();
            let y: Vec<f64> = step
                .gradient
                .iter()
                .zip(&grad)
                .map(|(a, b)| a - b)
                .collect();
            let previous = cost;
            (x, cost, grad) = (step.x, step.value, step.gradient);
            if cost < best.value {
                best = Minimum {
                    x: x.clone(),
                    value: cost,
                    gradient: grad.clone(),
                    iterations: iteration,
                    converged: false,
                };
            }
            best.iterations = iteration;
            if (previous - cost).abs() < self.cost_tolerance {
                best.converged = true;
                return Ok(best);
            }
            let ys = dot(&y, &s);
            if ys > 0.0 {
                h = bfgs_update(&h, &s, &y, 1.0 / ys);
            }
        }
        if norm(&best.gradient) < self.gradient_tolerance {
            best.converged = true;
        }
        Ok(best)
    }
}

/// `(I - r s y') H (I - r y s') + r s s'`.
fn bfgs_update(h: &[f64], s: &[f64], y: &[f64], r: f64) -> Vec<f64> {
    let n = s.len();
    // left[i][j] = delta_ij - r s_i y_j, right = left'.
    let left = |i: usize, j: usize| f64::from(u8::from(i == j)) - r * s[i] * y[j];
    let mut inner = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inner[i * n + j] = (0..n).map(|k| h[i * n + k] * left(j, k)).sum();
        }
    }
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let product: f64 = (0..n).map(|k| left(i, k) * inner[k * n + j]).sum();
            out[i * n + j] = product + r * s[i] * s[j];
        }
    }
    out
}

struct Point {
    x: Vec<f64>,
    value: f64,
    gradient: Vec<f64>,
}

/// One end of an interval in the line search: step, value, derivative.
#[derive(Clone, Copy)]
struct End {
    step: f64,
    value: f64,
    slope: f64,
}

impl MoreThuente {
    /// Searches from `x` along `direction`, returning the last point
    /// evaluated.
    #[expect(
        clippy::float_cmp,
        reason = "the step is clamped, so reaching a bound is an exact equality"
    )]
    fn search(
        &self,
        x: &[f64],
        value: f64,
        gradient: &[f64],
        direction: &[f64],
        f: &mut impl FnMut(&[f64]) -> (f64, Vec<f64>),
    ) -> Result<Point, BfgsError> {
        let slope0 = dot(gradient, direction);
        if slope0 >= 0.0 {
            return Err(BfgsError::NotDescent);
        }
        let test = self.c1 * slope0;
        let mut width = self.step_max - self.step_min;
        let mut width_before = 2.0 * width;
        let mut bracketed = false;
        let mut stage1 = true;
        let mut ok = true;
        let mut lower = End {
            step: 0.0,
            value,
            slope: slope0,
        };
        let mut upper = lower;
        let mut step = 1.0_f64;
        let mut last = None;
        for evaluation in 0..self.max_evaluations {
            let (step_lo, step_hi) = if bracketed {
                (lower.step.min(upper.step), lower.step.max(upper.step))
            } else {
                (lower.step, step + 4.0 * (step - lower.step))
            };
            step = step.clamp(self.step_min, self.step_max);
            if (bracketed
                && (step <= step_lo
                    || step >= step_hi
                    || step_hi - step_lo <= self.x_tolerance * step_hi))
                || !ok
            {
                step = lower.step;
            }
            let trial: Vec<f64> = x
                .iter()
                .zip(direction)
                .map(|(xi, di)| xi + step * di)
                .collect();
            let (trial_value, trial_gradient) = f(&trial);
            let slope = dot(&trial_gradient, direction);
            let sufficient = value + step * test;
            let done = (bracketed && (step <= step_lo || step >= step_hi))
                || !ok
                || (step == self.step_max && trial_value <= sufficient && slope <= test)
                || (step == self.step_min && (trial_value > sufficient || slope >= test))
                || (bracketed && step_hi - step_lo <= self.x_tolerance * step_hi)
                || (trial_value <= sufficient && slope.abs() <= self.c2 * -slope0)
                || evaluation + 1 == self.max_evaluations
                || !trial_value.is_finite();
            last = Some(Point {
                x: trial,
                value: trial_value,
                gradient: trial_gradient,
            });
            if done {
                break;
            }
            if stage1 && trial_value <= sufficient && slope >= self.c1.min(self.c2) * slope0 {
                stage1 = false;
            }
            let trial_end = End {
                step,
                value: trial_value,
                slope,
            };
            if stage1 && trial_value <= lower.value && trial_value > sufficient {
                // The modified function f(a) - test * a, whose minimum the
                // search brackets first.
                let modify = |e: End| End {
                    step: e.step,
                    value: e.value - e.step * test,
                    slope: e.slope - test,
                };
                let (mut l, mut u, mut t) = (modify(lower), modify(upper), modify(trial_end));
                (bracketed, ok) = cstep(&mut l, &mut u, &mut t, bracketed, step_lo, step_hi);
                lower = End {
                    step: l.step,
                    value: l.value + l.step * test,
                    slope: l.slope + test,
                };
                upper = End {
                    step: u.step,
                    value: u.value + u.step * test,
                    slope: u.slope + test,
                };
                step = t.step;
            } else {
                let mut t = trial_end;
                (bracketed, ok) =
                    cstep(&mut lower, &mut upper, &mut t, bracketed, step_lo, step_hi);
                step = t.step;
            }
            if bracketed {
                if (upper.step - lower.step).abs() >= 0.66 * width_before {
                    step = lower.step + 0.5 * (upper.step - lower.step);
                }
                width_before = width;
                width = (upper.step - lower.step).abs();
            }
        }
        last.ok_or(BfgsError::Shape)
    }
}

/// One safeguarded step of Moré and Thuente: updates the interval `[x, y]`
/// with the trial `t` and replaces `t.step` by the next trial step. Returns
/// whether the minimum is bracketed and whether the step was taken (`false`
/// when the inputs were inconsistent and nothing changed).
fn cstep(
    x: &mut End,
    y: &mut End,
    t: &mut End,
    bracketed: bool,
    step_min: f64,
    step_max: f64,
) -> (bool, bool) {
    if (bracketed && (t.step <= x.step.min(y.step) || t.step >= x.step.max(y.step)))
        || x.slope * (t.step - x.step) >= 0.0
        || step_max < step_min
    {
        return (bracketed, false);
    }
    let mut bracketed = bracketed;
    let sign = t.slope * (x.slope / x.slope.abs());
    let cubic_gamma = |theta: f64, a: f64, b: f64, clamp: bool| {
        let s = theta.abs().max(a.abs()).max(b.abs());
        let radicand = (theta / s) * (theta / s) - (a / s) * (b / s);
        s * if clamp { radicand.max(0.0) } else { radicand }.sqrt()
    };
    let bound;
    let mut next;
    if t.value > x.value {
        // A higher value: the minimum is bracketed.
        bound = true;
        let theta = 3.0 * (x.value - t.value) / (t.step - x.step) + x.slope + t.slope;
        let mut gamma = cubic_gamma(theta, x.slope, t.slope, false);
        if t.step < x.step {
            gamma = -gamma;
        }
        let p = (gamma - x.slope) + theta;
        let q = ((gamma - x.slope) + gamma) + t.slope;
        let cubic = x.step + p / q * (t.step - x.step);
        let quadratic = x.step
            + x.slope / ((x.value - t.value) / (t.step - x.step) + x.slope) / 2.0
                * (t.step - x.step);
        next = if (cubic - x.step).abs() < (quadratic - x.step).abs() {
            cubic
        } else {
            cubic + (quadratic - cubic) / 2.0
        };
        bracketed = true;
    } else if sign < 0.0 {
        // A lower value and derivatives of opposite sign: bracketed.
        bound = false;
        let theta = 3.0 * (x.value - t.value) / (t.step - x.step) + x.slope + t.slope;
        let mut gamma = cubic_gamma(theta, x.slope, t.slope, false);
        if t.step > x.step {
            gamma = -gamma;
        }
        let p = (gamma - t.slope) + theta;
        let q = ((gamma - t.slope) + gamma) + x.slope;
        let cubic = t.step + p / q * (x.step - t.step);
        let secant = t.step + t.slope / (t.slope - x.slope) * (x.step - t.step);
        next = if (cubic - t.step).abs() > (secant - t.step).abs() {
            cubic
        } else {
            secant
        };
        bracketed = true;
    } else if t.slope.abs() < x.slope.abs() {
        // A lower value, derivatives of the same sign, and a smaller slope.
        bound = true;
        let theta = 3.0 * (x.value - t.value) / (t.step - x.step) + x.slope + t.slope;
        let mut gamma = cubic_gamma(theta, x.slope, t.slope, true);
        if t.step > x.step {
            gamma = -gamma;
        }
        let p = (gamma - t.slope) + theta;
        let q = (gamma + (x.slope - t.slope)) + gamma;
        let r = p / q;
        let cubic = if r < 0.0 && gamma != 0.0 {
            t.step + r * (x.step - t.step)
        } else if t.step > x.step {
            step_max
        } else {
            step_min
        };
        let secant = t.step + t.slope / (t.slope - x.slope) * (x.step - t.step);
        let (to_cubic, to_secant) = ((t.step - cubic).abs(), (t.step - secant).abs());
        // Bracketed, take the nearer step; otherwise the farther.
        let take_cubic = if bracketed {
            to_cubic < to_secant
        } else {
            to_cubic > to_secant
        };
        next = if take_cubic { cubic } else { secant };
    } else {
        // A lower value, derivatives of the same sign, and no smaller slope.
        bound = false;
        next = if bracketed {
            let theta = 3.0 * (t.value - y.value) / (y.step - t.step) + y.slope + t.slope;
            let mut gamma = cubic_gamma(theta, y.slope, t.slope, false);
            if t.step > y.step {
                gamma = -gamma;
            }
            let p = (gamma - t.slope) + theta;
            let q = ((gamma - t.slope) + gamma) + y.slope;
            t.step + p / q * (y.step - t.step)
        } else if t.step > x.step {
            step_max
        } else {
            step_min
        };
    }
    if t.value > x.value {
        *y = *t;
    } else {
        if sign < 0.0 {
            *y = *x;
        }
        *x = *t;
    }
    next = next.clamp(step_min, step_max);
    if bracketed && bound {
        let limit = x.step + 0.66 * (y.step - x.step);
        next = if y.step > x.step {
            next.min(limit)
        } else {
            next.max(limit)
        };
    }
    t.step = next;
    (bracketed, true)
}

fn identity(n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|k| f64::from(u8::from(k % (n + 1) == 0)))
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rosenbrock(p: &[f64]) -> (f64, Vec<f64>) {
        let (x, y) = (p[0], p[1]);
        let value = (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
        let gradient = vec![
            -2.0 * (1.0 - x) - 400.0 * x * (y - x * x),
            200.0 * (y - x * x),
        ];
        (value, gradient)
    }

    #[test]
    fn rosenbrock_converges() {
        let solver = Bfgs {
            max_iterations: 200,
            ..Bfgs::default()
        };
        let minimum = solver.minimize(&[-1.2, 1.0], rosenbrock, None).unwrap();
        assert!(minimum.converged);
        assert!(
            (minimum.x[0] - 1.0).abs() < 1e-6 && (minimum.x[1] - 1.0).abs() < 1e-6,
            "{:?}",
            minimum.x
        );
    }

    #[test]
    fn a_quadratic_is_solved_in_few_iterations() {
        // f = 0.5 x'Ax - b'x with A = [[3, 1], [1, 2]], b = [1, 1].
        let f = |p: &[f64]| {
            let ax = [3.0 * p[0] + p[1], p[0] + 2.0 * p[1]];
            (
                f64::midpoint(p[0] * ax[0], p[1] * ax[1]) - p[0] - p[1],
                vec![ax[0] - 1.0, ax[1] - 1.0],
            )
        };
        let minimum = Bfgs::default().minimize(&[5.0, -3.0], f, None).unwrap();
        assert!((minimum.x[0] - 0.2).abs() < 1e-8 && (minimum.x[1] - 0.4).abs() < 1e-8);
        assert!(minimum.iterations <= 10);
    }

    #[test]
    fn a_given_inverse_hessian_is_used() {
        let f = |p: &[f64]| (4.0 * p[0] * p[0], vec![8.0 * p[0]]);
        let minimum = Bfgs::default()
            .minimize(&[1.0], f, Some(vec![0.125]))
            .unwrap();
        assert!(minimum.x[0].abs() < 1e-12);
        assert_eq!(minimum.iterations, 1);
    }

    #[test]
    fn errors() {
        let f = |p: &[f64]| (p[0], vec![1.0]);
        assert_eq!(
            Bfgs::default().minimize(&[], f, None).unwrap_err(),
            BfgsError::Shape
        );
        assert_eq!(
            Bfgs::default()
                .minimize(&[1.0], f, Some(vec![1.0, 0.0]))
                .unwrap_err(),
            BfgsError::Shape
        );
        let nan = |_: &[f64]| (f64::NAN, vec![0.0]);
        assert_eq!(
            Bfgs::default().minimize(&[1.0], nan, None).unwrap_err(),
            BfgsError::NotFinite
        );
        // A negative definite starting inverse Hessian points uphill.
        let g = |p: &[f64]| (p[0] * p[0], vec![2.0 * p[0]]);
        assert_eq!(
            Bfgs::default()
                .minimize(&[1.0], g, Some(vec![-1.0]))
                .unwrap_err(),
            BfgsError::NotDescent
        );
    }
}

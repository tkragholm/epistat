//! BFGS minimisation with the line search of Moré and Thuente.
//!
//! [`Bfgs`] minimises a smooth function of `n` variables from its value and
//! gradient, keeping an estimate of the inverse Hessian. [`MoreThuente`] chooses
//! the step along each direction. Both are written for the small problems of
//! statistical estimation, where `n` is a few dozen at most and the inverse
//! Hessian is stored densely.
//!
//! # Algorithm
//!
//! The iterate is `x`, the gradient `g`, and `H` the inverse Hessian estimate. `H`
//! starts as the identity, or as the matrix passed to [`Bfgs::minimize`]. Each
//! iteration takes the direction
//!
//! ```text
//! d = -H g
//! ```
//!
//! and asks the line search for a step `a` along `d` with initial trial `a = 1`.
//! With `s = a d` and `y = g(x + s) - g(x)`, and when `y's > 0`, the update of
//! Nocedal and Wright \[2\] (eq. 6.17) is
//!
//! ```text
//! r  = 1 / y's
//! H+ = (I - r s y') H (I - r y s') + r s s'
//! ```
//!
//! When `y's <= 0` the update is skipped and `H` is kept, so `H` stays positive
//! definite as long as it started so \[2\]. If `d` is not a descent direction,
//! `g'd >= 0`, the search fails with [`BfgsError::NotDescent`]. This can
//! happen when an inverse Hessian passed by the caller is not positive
//! definite.
//!
//! # Stopping rules
//!
//! [`Bfgs`] checks, in this order:
//!
//! 1. At the start of an iteration, `|g| < gradient_tolerance`, the Euclidean
//!    norm. The result is converged. This also holds at iteration 0.
//! 2. After the step, `|f_old - f_new| < cost_tolerance`, an absolute change.
//!    The result is converged.
//!    This holds also for a step whose value is above the current one by less
//!    than `cost_tolerance`: the cost has stopped changing beyond rounding, and
//!    the result is the current point.
//! 3. A step whose value is not finite, whose gradient is not finite, or whose
//!    value is above the current one by `cost_tolerance` or more ends the
//!    search at the current point, which is not converged.
//! 4. After `max_iterations` iterations the loop ends. The result is converged
//!    only if the gradient at the last point has norm below
//!    `gradient_tolerance`.
//!
//! The defaults are 100 iterations, `gradient_tolerance = sqrt(ε)` and
//! `cost_tolerance = ε`, with `ε = f64::EPSILON`. Because rule 3 refuses a step
//! that raises the cost, the costs of the iterates do not increase and the
//! returned [`Minimum`] is the last iterate, with its value and gradient.
//!
//! # Line search
//!
//! [`MoreThuente`] follows the algorithm of Moré and Thuente \[1\] as it is
//! implemented in MINPACK (`cvsrch` and `cstep`). Write `f(a) = f(x + a d)`.
//! It looks for a step that satisfies the strong Wolfe conditions
//!
//! ```text
//! f(a)   <= f(0) + c1 a g'd          sufficient decrease
//! |f'(a)| <= c2 |g'd|                 curvature,   f'(a) = g(x + a d)'d
//! ```
//!
//! with `0 < c1 < c2 < 1`. The defaults are `c1 = 1e-4` and `c2 = 0.9`, the usual
//! choice for quasi-Newton directions \[2\] \[3\]. The step is kept in
//! `[step_min, step_max]`, by default `[sqrt(ε), ∞)`.
//!
//! The search keeps an interval of uncertainty with end points `l` and `u`, each
//! a step with its value and derivative, and a trial step `t`. Until a step has
//! been found whose derivative has the opposite sign to the end point's, the
//! interval is not bracketed and the trial is extrapolated. While it is not
//! bracketed, the trial is limited to
//!
//! ```text
//! [ l, t + 4 (t - l) ]
//! ```
//!
//! so the step grows by a factor of at most 4 beyond the last trial. In the
//! first stage the search works on the modified function
//!
//! ```text
//! ψ(a) = f(a) - c1 a g'd,        ψ'(a) = f'(a) - c1 g'd
//! ```
//!
//! whenever the trial has `f(t) <= f(l)` but fails sufficient decrease. A
//! minimiser of `ψ` that is bracketed this way lies in an interval that contains
//! steps satisfying both conditions \[1\]. The first stage ends when
//! `f(t)` satisfies sufficient decrease and `f'(t) >= min(c1, c2) g'd`. After
//! that the search uses `f` itself.
//!
//! Each trial is evaluated, tested against the termination rules below, and
//! otherwise passed to a safeguarded step `cstep` that updates the interval and
//! proposes the next trial. Write `l` for the end point with the lowest value,
//! `f` for values and `d` for derivatives. The four cases of `cstep` are:
//!
//! 1. `f(t) > f(l)`. The minimum is bracketed. The cubic through `l` and `t`
//!    that matches both values and derivatives gives `a_c`, and the quadratic
//!    through `f(l)`, `f(t)` and `d(l)` gives `a_q`. The next trial is `a_c`
//!    when it is closer to `l` than `a_q`, and the midpoint of `a_c` and `a_q`
//!    otherwise. The interval becomes `[l, t]`.
//! 2. `f(t) <= f(l)` and `d(t)` has the opposite sign to `d(l)`. The minimum is
//!    bracketed. The cubic `a_c` and the secant step `a_s` on the derivatives
//!    are formed, and the one farther from `t` is the next trial. The interval
//!    becomes `[t, l]`.
//! 3. `f(t) <= f(l)`, the derivatives have the same sign, and `|d(t)| < |d(l)|`.
//!    The cubic is computed with its discriminant clamped at 0. It is used when
//!    its extrapolation parameter is negative and the discriminant is not zero,
//!    and otherwise the step limit in the direction of the search, `step_max` or
//!    `step_min`, is used. The secant step `a_s` is also formed. If the minimum is
//!    bracketed the nearer of the two to `t` is taken, and otherwise the farther.
//!    The interval becomes `[t, u]`.
//! 4. `f(t) <= f(l)`, the derivatives have the same sign, and
//!    `|d(t)| >= |d(l)|`. If the minimum is bracketed, the next trial is the
//!    cubic through `t` and `u`. Otherwise it is `step_max` or `step_min`,
//!    whichever lies in the direction of descent. The interval becomes `[t, u]`.
//!
//! The proposal is then clamped to the current interval. In cases 1 and 3, where
//! the step is bounded, and when the minimum is bracketed, it is also limited to
//!
//! ```text
//! l + 0.66 (u - l)
//! ```
//!
//! so the next trial stays well inside the interval. After `cstep`, if the
//! minimum is bracketed and the interval has not shrunk to below 0.66 times the
//! width two trials ago, the next trial is the midpoint of the interval. This
//! bisection guarantees that the width falls geometrically \[1\].
//!
//! `cstep` changes nothing and reports failure when its inputs are inconsistent:
//! the trial lies outside a bracketed interval, the derivative at `l` does not
//! point toward `t`, or the step limits cross. The next trial is then `l`, and
//! the search ends after evaluating it.
//!
//! # Termination of the line search
//!
//! The search ends, returning the last point evaluated, when any of these holds:
//!
//! - the trial satisfies both strong Wolfe conditions;
//! - the interval is bracketed and the trial lies on its boundary, or the interval
//!   is narrower than `x_tolerance` times its upper end, by default `1e-10`;
//! - the trial is at `step_max` and still decreasing enough, or at `step_min` and
//!   still not decreasing enough;
//! - `cstep` failed;
//! - the cost at the trial is not finite;
//! - `max_evaluations` evaluations have been made, by default 40.
//!
//! The point returned need not satisfy the Wolfe conditions. In particular a
//! non-finite point is returned as it is.
//!
//! # References
//!
//! 1. Moré JJ, Thuente DJ. Line search algorithms with guaranteed sufficient
//!    decrease. ACM Trans Math Softw. 1994;20(3):286-307.
//! 2. Nocedal J, Wright SJ. Numerical Optimization. 2nd ed. New York: Springer;
//!    2006. Chapters 3 and 6.
//! 3. Fletcher R. Practical Methods of Optimization. 2nd ed. Chichester: Wiley;
//!    1987. Chapter 3.

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
            // A step that raises the cost by less than the cost tolerance is a
            // change of the cost within its rounding: the search has
            // converged, at the current point, which is the best.
            if step.value.is_finite()
                && step.gradient.iter().all(|g| g.is_finite())
                && (cost - step.value).abs() < self.cost_tolerance
                && step.value > cost
            {
                best.iterations = iteration;
                best.converged = true;
                return Ok(best);
            }
            // A step that leaves the finite region or does not decrease the
            // cost ends the search at the current point, which is the best.
            if step.value.is_nan()
                || step.value > cost
                || step.gradient.iter().any(|g| !g.is_finite())
            {
                best.iterations = iteration;
                return Ok(best);
            }
            let previous = cost;
            (x, cost, grad) = (step.x, step.value, step.gradient);
            best = Minimum {
                x: x.clone(),
                value: cost,
                gradient: grad.clone(),
                iterations: iteration,
                converged: false,
            };
            if (previous - cost).abs() < self.cost_tolerance {
                best.converged = true;
                return Ok(best);
            }
            let ys = dot(&y, &s);
            if ys > 0.0 {
                h = bfgs_update(&h, &s, &y, 1.0 / ys);
            }
        }
        best.converged = norm(&best.gradient) < self.gradient_tolerance;
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
    fn a_rise_of_the_cost_within_its_tolerance_is_convergence() {
        // Every point but the start has a cost 1e-12 above x'x, as if the
        // cost were rounded up. No step lowers the cost, and the rise is below
        // the cost tolerance, so the search has converged at the start.
        let start = [1e-8, 1e-8];
        let rounded_up = |p: &[f64]| {
            let bump = if p == start { 0.0 } else { 1e-12 };
            let value: f64 = p.iter().map(|x| x * x).sum::<f64>() + bump;
            let gradient = p.iter().map(|x| 2.0 * x).collect();
            (value, gradient)
        };
        let solver = Bfgs {
            gradient_tolerance: 1e-30,
            cost_tolerance: 1e-9,
            ..Bfgs::default()
        };
        let minimum = solver.minimize(&start, rounded_up, None).unwrap();
        assert!(minimum.converged);
        assert_eq!(minimum.x, start);
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
    fn a_step_into_nan_stops_at_the_last_finite_point() {
        // Finite only on x > -1; from x = 0 a full Newton-like step overshoots.
        let f = |p: &[f64]| {
            if p[0] <= -1.0 {
                (f64::NAN, vec![f64::NAN])
            } else {
                (
                    -(p[0] + 1.0).ln() + 0.01 * p[0] * p[0],
                    vec![-1.0 / (p[0] + 1.0) + 0.02 * p[0]],
                )
            }
        };
        let minimum = Bfgs::default()
            .minimize(&[5.0], f, Some(vec![1e6]))
            .unwrap();
        assert!(minimum.value.is_finite() && minimum.x[0] > -1.0);
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

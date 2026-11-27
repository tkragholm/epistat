//! Ridge-penalised least squares on the probability simplex by the
//! Frank-Wolfe method with an exact line search, the solver of the unit and
//! time weights of synthetic control and synthetic difference-in-differences.
//!
//! Solves
//!
//! ```text
//! min  f(x) = ( |A x - b|^2 + eta |x|^2 ) / m    subject to    x >= 0,  sum(x) = 1
//! ```
//!
//! with `A` a dense `m × n` matrix stored by column, `b` of length `m` and
//! `eta >= 0`. With `eta > 0` the objective is strongly convex and the
//! minimiser is unique.
//!
//! # Algorithm
//!
//! One step from `x`, as `fw.step` of the R package `synthdid` \[3\]:
//!
//! 1. The half gradient `g = A'(A x - b) + eta x`.
//! 2. The vertex `i` of the simplex with the smallest `g_i`, the first such
//!    index on ties. It minimises the linearisation `g'(e_i - x)` over the
//!    simplex \[1\] \[2\].
//! 3. The direction `d = e_i - x`. When `d` is zero, `x` is the vertex `e_i`
//!    and the step leaves it unchanged.
//! 4. The exact line search along `d`. With `r = A e_i - A x = A d`, the
//!    objective along the segment is a parabola in the step `s` with minimum at
//!
//!    ```text
//!    s* = - g'd / ( |r|^2 + eta |d|^2 ),
//!    ```
//!
//!    which is clamped to `[0, 1]` so that `x + s d` stays on the simplex. A
//!    zero denominator makes the objective linear along `d`, and the step is
//!    then `1` when `g'd < 0` and `0` otherwise.
//! 5. `x <- x + s d`, and `A x <- A x + s r`.
//!
//! Every sum over rows or columns (the dot products of the gradient, the
//! norms, the objective) runs over eight fixed lanes in a fixed order, so
//! that it vectorises and has the same bits on every target and thread count.
//!
//! The product `A x` is carried from step to step by the update in 5 rather
//! than formed again, so that a step reads `A` once, for the gradient. The
//! update is a convex combination of `A x` and a column of `A`, so its
//! rounding error grows at most linearly in the number of steps, by about one
//! unit in the last place of `max |A|` per step.
//!
//! [`SimplexLeastSquares::solve`] repeats the step and records `f` after each
//! one. It stops after step `t >= 2` when `f(t-1) - f(t) <= tolerance`, or
//! after `max_iter` steps, which is the loop of `synthdid`'s `sc.weight.fw`:
//! that package passes `tolerance = min.decrease^2`. A step that does not lower
//! `f` (to rounding) therefore ends the search too.
//!
//! # Convergence
//!
//! With the exact line search the gap `f(x_t) - f*` falls as `O(C / t)`, with
//! `C` the curvature constant, at most `2 (λ_max(A'A) + eta) diam^2 / m` with
//! `diam^2 = 2` on the simplex \[1\] \[2\]. The iterates are sparse: after `t`
//! steps from a vertex at most `t + 1` coordinates are positive.
//!
//! # Cost
//!
//! `O(m n)` per step for the gradient, `O(m + n)` for the rest. Nothing is
//! allocated inside the loop and every sum has a fixed order, so the result
//! does not depend on the number of threads. A step of a problem of `10^6`
//! entries takes tens of microseconds, too little to share between threads;
//! callers run independent problems in parallel instead.
//!
//! # References
//!
//! 1. Frank M, Wolfe P. An algorithm for quadratic programming. Nav Res Logist
//!    Q 1956;3(1-2):95-110. doi:10.1002/nav.3800030109
//! 2. Jaggi M. Revisiting Frank-Wolfe: projection-free sparse convex
//!    optimization. Proc 30th Int Conf Mach Learn, PMLR 2013;28(1):427-35.
//! 3. Arkhangelsky D, Athey S, Hirshberg DA, Imbens GW, Wager S. Synthetic
//!    difference-in-differences. Am Econ Rev 2021;111(12):4088-118.
//!    doi:10.1257/aer.20190159. R package `synthdid` 0.0.9, functions
//!    `fw.step` and `sc.weight.fw`.

use epistat_core::check;

use crate::error::{Error, domain};

/// When [`SimplexLeastSquares::solve`] stops.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrankWolfeOptions {
    /// Stop after step `t >= 2` when the objective fell by no more than this.
    pub tolerance: f64,
    /// The largest number of steps.
    pub max_iter: usize,
}

/// The result of [`SimplexLeastSquares::solve`].
#[derive(Debug, Clone, PartialEq)]
pub struct FrankWolfeSolution {
    /// The last iterate, on the simplex.
    pub x: Vec<f64>,
    /// The objective after each step, `values.len()` steps in all.
    pub values: Vec<f64>,
    /// `false` when the search stopped at `max_iter` with the last decrease
    /// still above the tolerance.
    pub converged: bool,
}

/// An iterate with the product `A x` that the steps update.
#[derive(Debug, Clone, PartialEq)]
pub struct FrankWolfeState {
    /// The iterate.
    pub x: Vec<f64>,
    /// `A x`.
    pub ax: Vec<f64>,
    gradient: Vec<f64>,
    residual: Vec<f64>,
}

/// The program `min (|A x - b|^2 + eta |x|^2) / m` over the simplex. See the
/// module documentation.
#[derive(Debug, Clone, Copy)]
pub struct SimplexLeastSquares<'a> {
    a: &'a [f64],
    b: &'a [f64],
    rows: usize,
    cols: usize,
    eta: f64,
}

impl<'a> SimplexLeastSquares<'a> {
    /// The program with `A` of `rows × cols` stored by column (column `j` is
    /// `a[j * rows..(j + 1) * rows]`), target `b` and penalty `eta`.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when `A` has no rows or columns, `a` or `b` has the
    /// wrong length, an entry is not finite, or `eta` is negative or not
    /// finite.
    pub fn new(
        a: &'a [f64],
        rows: usize,
        cols: usize,
        b: &'a [f64],
        eta: f64,
    ) -> Result<Self, Error> {
        check::non_empty("rows", rows)?;
        check::non_empty("cols", cols)?;
        check::same_length("a", rows.saturating_mul(cols), a.len())?;
        check::same_length("b", rows, b.len())?;
        if !(eta.is_finite() && eta >= 0.0) {
            return Err(domain("eta", eta, "[0, inf)"));
        }
        check::finite("a", a)?;
        check::finite("b", b)?;
        Ok(Self {
            a,
            b,
            rows,
            cols,
            eta,
        })
    }

    /// Number of rows of `A`.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns of `A`, the length of `x`.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    fn column(&self, j: usize) -> &'a [f64] {
        &self.a[j * self.rows..(j + 1) * self.rows]
    }

    /// The state at `x`, with `A x` formed column by column.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when `x` has the wrong length, an entry that is not
    /// finite, or is not on the simplex: a negative entry, or a sum further
    /// than `1e-8` from 1.
    pub fn start(&self, x: Vec<f64>) -> Result<FrankWolfeState, Error> {
        check::same_length("x", self.cols, x.len())?;
        check::finite("x", &x)?;
        if let Some(&negative) = x.iter().find(|&&value| value < 0.0) {
            return Err(domain("x", negative, "[0, inf)"));
        }
        let total: f64 = x.iter().sum();
        if (total - 1.0).abs() > 1e-8 {
            return Err(domain("sum of x", total, "within 1e-8 of 1"));
        }
        let mut ax = vec![0.0; self.rows];
        for (j, &xj) in x.iter().enumerate() {
            if xj != 0.0 {
                for (out, &value) in ax.iter_mut().zip(self.column(j)) {
                    *out += xj * value;
                }
            }
        }
        Ok(FrankWolfeState {
            x,
            ax,
            gradient: vec![0.0; self.cols],
            residual: vec![0.0; self.rows],
        })
    }

    /// The objective `(|A x - b|^2 + eta |x|^2) / m` at the state.
    #[must_use]
    pub fn value(&self, state: &FrankWolfeState) -> f64 {
        let fit = sum_map2(&state.ax, self.b, |ax, b| (ax - b) * (ax - b));
        let penalty = dot(&state.x, &state.x);
        (self.eta * penalty + fit) / self.rows as f64
    }

    /// `A' r + eta x` into `out`.
    fn gradient(&self, x: &[f64], residual: &[f64], out: &mut [f64]) {
        for (j, (g, &xj)) in out.iter_mut().zip(x).enumerate() {
            *g = dot(self.column(j), residual) + self.eta * xj;
        }
    }

    /// One step, described in the module documentation.
    pub fn step(&self, state: &mut FrankWolfeState) {
        let FrankWolfeState {
            x,
            ax,
            gradient,
            residual,
        } = state;
        for ((r, &axv), &b) in residual.iter_mut().zip(ax.iter()).zip(self.b) {
            *r = axv - b;
        }
        self.gradient(x, residual, gradient);
        let mut best = 0;
        for (j, &g) in gradient.iter().enumerate() {
            if g < gradient[best] {
                best = j;
            }
        }
        // d = e_best - x
        let mut slope = 0.0;
        let mut d_norm = 0.0;
        let mut moves = false;
        for (j, (&xj, &gj)) in x.iter().zip(gradient.iter()).enumerate() {
            let dj = if j == best { 1.0 - xj } else { -xj };
            moves |= dj != 0.0;
            slope += gj * dj;
            d_norm += dj * dj;
        }
        if !moves {
            return;
        }
        let column = self.column(best);
        for (residual_out, (&a, &axv)) in residual.iter_mut().zip(column.iter().zip(ax.iter())) {
            *residual_out = a - axv;
        }
        let r_norm = dot(residual, residual);
        let curvature = r_norm + self.eta * d_norm;
        let step = if curvature > 0.0 {
            (-slope / curvature).clamp(0.0, 1.0)
        } else if slope < 0.0 {
            1.0
        } else {
            0.0
        };
        for (j, xj) in x.iter_mut().enumerate() {
            let dj = if j == best { 1.0 - *xj } else { -*xj };
            *xj += step * dj;
        }
        for (axv, &r) in ax.iter_mut().zip(residual.iter()) {
            *axv += step * r;
        }
    }

    /// Steps from `x0` until the stopping rule of the module documentation.
    ///
    /// # Errors
    ///
    /// The errors of [`SimplexLeastSquares::start`].
    pub fn solve(
        &self,
        x0: Vec<f64>,
        options: FrankWolfeOptions,
    ) -> Result<FrankWolfeSolution, Error> {
        let mut state = self.start(x0)?;
        let mut values = Vec::with_capacity(options.max_iter.min(1 << 14));
        let mut converged = false;
        while values.len() < options.max_iter {
            self.step(&mut state);
            values.push(self.value(&state));
            let t = values.len();
            if t >= 2 && values[t - 2] - values[t - 1] <= options.tolerance {
                converged = true;
                break;
            }
        }
        Ok(FrankWolfeSolution {
            x: state.x,
            values,
            converged,
        })
    }
}

const LANES: usize = 8;

/// The sum of `f(a_i, b_i)` over eight lanes: lane `j` takes the terms at
/// positions `j`, `j + 8`, ... in order, the lanes are folded as
/// `((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7))`, and the last
/// `len mod 8` terms are summed from left to right and added last. The width is
/// fixed, so the bits do not depend on the target; the loop vectorises because
/// the eight running sums are independent. The order is that of
/// `epistat_stats::sum::lanes`.
#[inline]
fn sum_map2(a: &[f64], b: &[f64], f: impl Fn(f64, f64) -> f64) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let (ca, ra) = a.as_chunks::<LANES>();
    let (cb, rb) = b.as_chunks::<LANES>();
    let mut lanes = [0.0_f64; LANES];
    for (x, y) in ca.iter().zip(cb) {
        for ((lane, &u), &v) in lanes.iter_mut().zip(x).zip(y) {
            *lane += f(u, v);
        }
    }
    let mut tail = 0.0;
    for (&u, &v) in ra.iter().zip(rb) {
        tail += f(u, v);
    }
    ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6]))
        + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]))
        + tail
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    sum_map2(a, b, |u, v| u * v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_core::InputError;

    /// Two columns, so the minimiser on the segment is a parabola's vertex.
    #[test]
    fn two_columns_reach_the_vertex_of_the_parabola() {
        // With A = I, b = (0.3, 0.7) and eta = 0, b lies on the simplex and is
        // the minimiser.
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [0.3, 0.7];
        let problem = SimplexLeastSquares::new(&a, 2, 2, &b, 0.0).unwrap();
        let solution = problem
            .solve(
                vec![0.5, 0.5],
                FrankWolfeOptions {
                    tolerance: 0.0,
                    max_iter: 50,
                },
            )
            .unwrap();
        // One exact line search along the only direction lands on the minimum.
        assert!((solution.x[0] - 0.3).abs() < 1e-15, "{:?}", solution.x);
        assert!((solution.x[1] - 0.7).abs() < 1e-15, "{:?}", solution.x);
        assert!(solution.converged);
    }

    #[test]
    fn a_vertex_start_on_the_minimiser_does_not_move() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 2.0];
        let problem = SimplexLeastSquares::new(&a, 2, 2, &b, 0.0).unwrap();
        let mut state = problem.start(vec![1.0, 0.0]).unwrap();
        problem.step(&mut state);
        assert_eq!(state.x, vec![1.0, 0.0]);
    }

    #[test]
    fn refuses_bad_input() {
        let a = [1.0, 2.0];
        let b = [1.0];
        assert_eq!(
            SimplexLeastSquares::new(&a, 1, 3, &b, 0.0).unwrap_err(),
            Error::Input(InputError::Length {
                what: "a",
                expected: 3,
                found: 2
            })
        );
        assert_eq!(
            SimplexLeastSquares::new(&a, 1, 2, &b, -1.0).unwrap_err(),
            Error::Input(InputError::Domain {
                what: "eta",
                value: -1.0,
                domain: "[0, inf)",
                row: None
            })
        );
        let problem = SimplexLeastSquares::new(&a, 1, 2, &b, 0.0).unwrap();
        assert_eq!(
            problem.start(vec![0.6, 0.6]).unwrap_err(),
            Error::Input(InputError::Domain {
                what: "sum of x",
                value: 1.2,
                domain: "within 1e-8 of 1",
                row: None
            })
        );
    }
}

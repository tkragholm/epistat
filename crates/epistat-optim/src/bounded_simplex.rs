//! A bounded-variable dual simplex method for linear programs with few rows
//! and very many columns.
//!
//! Solves
//!
//! ```text
//! max  c'x    subject to    r_lo ≤ A x ≤ r_hi,    l ≤ x ≤ u
//! ```
//!
//! with `A` an `m × n` dense matrix, `m` a few dozen at most and `n` up to
//! about a million. Every bound may be infinite, a row may be an equality
//! (`r_lo = r_hi`), and a column may be fixed (`l = u`). This is the shape of the
//! relaxation of cardinality matching: one column in `[0, 1]` per unit and a
//! row per balance constraint.
//!
//! [`DenseSimplex`](crate::simplex::DenseSimplex) is the primal method for
//! equality programs with `x ≥ 0` and a few dozen columns. This module is
//! its counterpart for programs whose columns have upper bounds, whose rows are
//! inequalities, and whose bounds change between solves, which is what branch
//! and bound does ([`crate::branch_and_bound`]).
//!
//! # Algorithm
//!
//! Each row `i` gets a slack `s_i = a_i'x` with the bounds `[r_lo_i, r_hi_i]`, so
//! the program is `A x - s = 0` with bounds on all `n + m` variables. A basis
//! `B` is a set of `m` variables, kept as the explicit inverse `B^{-1}`. A
//! non-basic variable sits at its lower or upper bound (or at 0 when it is
//! free). With the costs written for minimisation, `q = -c`, the duals are
//! `y = B^{-T} q_B`, the reduced cost of a non-basic structural column is
//! `d_j = q_j - a_j'y`, and that of a slack is `d_{n+i} = q_{n+i} + y_i`. A basis
//! is dual feasible when `d_j ≥ 0` for variables at their lower bound and
//! `d_j ≤ 0` at their upper bound, and it is optimal when it is also primal
//! feasible, which means every basic variable is within its bounds.
//!
//! The method is the dual simplex method \[1\] \[2\] with the bound flipping ratio
//! test of Koberstein \[3\]. It starts from the all-slack basis with every
//! structural column at the bound its cost prefers, which is dual feasible by
//! construction, and repairs primal infeasibility one basic variable at a time.
//!
//! 1. Leaving variable. The basic variable with the largest bound violation, as
//!    a multiple of its own tolerance, leaves. Ties go to the lowest variable
//!    index.
//! 2. Pivot row. With `ρ` the row of `B^{-1}` of the leaving position, every
//!    non-basic variable has `α_j = ρ'a_j`. Moving `x_j` away from its bound
//!    changes the leaving variable by `-α_j` per unit. Let `t_j` be `α_j` signed
//!    so that `t_j > 0` means moving `x_j` away from its bound reduces the
//!    violation. Only such columns are candidates. The dual step at which `d_j`
//!    reaches zero is `|d_j| / |t_j|`.
//! 3. Ratio test with bound flips. Candidates are taken in order of increasing
//!    ratio. Passing candidate `j` changes the rate of improvement of the
//!    leaving variable by `|t_j| (u_j - l_j)`. While the violation would still
//!    shrink after passing `j` by more than the leaving variable's tolerance,
//!    `j` is flipped to its other bound instead of
//!    entering, which is what keeps the iteration count near `m` when `n` is a
//!    million. The first candidate that would take the rate below zero enters.
//!    The candidates are found by a selection that splits the list at its
//!    median, which costs `O(n)` instead of `O(n log n)`. Among candidates
//!    whose ratio is within the dual tolerance of the entering one, the entering
//!    column is the one with the largest `|t_j|`, for the sake of the pivot.
//! 4. Update. The flipped columns change the basic values by `-B^{-1} Σ a_j Δ_j`.
//!    The entering column then moves until the leaving variable reaches its
//!    violated bound, the basic values and `B^{-1}` are updated, and the
//!    leaving variable becomes non-basic at that bound.
//!
//! `B^{-1}` is rebuilt from the basis by Gauss-Jordan elimination with partial
//! pivoting after every 40 pivots, together with the basic values. The duals and
//! reduced costs are recomputed from `B^{-1}` at every iteration, not
//! updated, so they do not drift. A pivot below `1e-12` in the elimination is
//! [`BoundedLpError::Singular`]. After a refactorisation, a non-basic column whose
//! reduced cost has the wrong sign beyond the dual tolerance is flipped to its
//! other bound, which restores dual feasibility when both bounds are finite.
//!
//! # Degeneracy
//!
//! A step of length at most the dual tolerance is degenerate. After 25 in a row,
//! the leaving variable becomes the infeasible basic variable of lowest index
//! and the entering column the candidate of lowest index among the minimum
//! ratios, with no flips, for the rest of that solve, which cannot cycle \[4\].
//! A solve is limited to `100 (m + n) + 1000` pivots, after which it fails with
//! [`BoundedLpError::IterationLimit`].
//!
//! # Warm starts
//!
//! The bounds of a column may be changed between solves
//! ([`BoundedSimplex::set_bounds`]) and the next [`BoundedSimplex::solve`] starts
//! from the last basis. A basis that was optimal stays dual feasible when
//! bounds change, so the dual simplex needs a few pivots. [`BoundedSimplex::snapshot`]
//! and [`BoundedSimplex::restore`] save and reload a basis.
//!
//! # Tolerances
//!
//! All are derived from the scale of the data, not tuned.
//!
//! - A basic slack of row `i` is within its bounds when it is within
//!   `1e-9 · max(1, ‖a_i ∘ w‖₂)`, where `w_j` is the width `u_j - l_j` of column `j`
//!   (1 when the width is infinite). The norm is the size of the rounding error
//!   of a sum of such terms, in units of `1e-9`.
//! - A basic structural column `j` is within its bounds when it is within
//!   `1e-9 · (1 + |b|)` of the bound `b`.
//! - A reduced cost has the wrong sign only when it is beyond
//!   `1e-9 · max(1, max|q|)`.
//! - A candidate with `|t_j|` below `1e-9` times the largest `|t|` of the row is
//!   not a pivot.
//!
//! # Determinism
//!
//! With the `rayon` feature the scan over the columns runs on the thread pool in
//! fixed chunks of 4096 columns. Every column is handled on its own and the
//! results are joined in column order, and sums over columns are taken per
//! chunk and then added in chunk order, so the pivots, and the result, are the
//! same on any number of threads.
//!
//! # Example
//!
//! ```
//! use epistat_optim::bounded_simplex::BoundedSimplex;
//!
//! // max x0 + x1   s.t.  2 x0 + x1 <= 3,  0 <= x <= 1
//! let columns = vec![2.0, 1.0]; // m = 1 row, n = 2 columns
//! let mut lp = BoundedSimplex::new(
//!     1,
//!     &columns,
//!     &[1.0, 1.0],
//!     &[0.0, 0.0],
//!     &[1.0, 1.0],
//!     &[f64::NEG_INFINITY],
//!     &[3.0],
//! )
//! .unwrap();
//! lp.solve(None).unwrap();
//! assert!((lp.objective() - 2.0).abs() < 1e-9);
//! ```
//!
//! # References
//!
//! 1. Lemke CE. The dual method of solving the linear programming problem. Nav
//!    Res Logist Q. 1954;1(1):36-47.
//! 2. Chvátal V. Linear Programming. New York: W. H. Freeman; 1983.
//! 3. Koberstein A. The dual simplex method, techniques for a fast and stable
//!    implementation. PhD thesis, Universität Paderborn; 2005.
//! 4. Bland RG. New finite pivoting rules for the simplex method. Math Oper Res.
//!    1977;2(2):103-7.

#![expect(
    clippy::float_cmp,
    reason = "bounds and costs are compared as given: a fixed column has equal \
              bounds, an integer cost equals its rounding, and ties are exact"
)]

use std::fmt;
use std::time::Instant;

const CHUNK: usize = 4096;
const REFACTOR_EVERY: usize = 40;
const BLAND_AFTER_DEGENERATE: usize = 25;
/// Relative tolerance of every comparison, in units of the data scale.
const REL_TOL: f64 = 1e-9;
/// A pivot smaller than this in the elimination makes the basis singular.
const TOL_SINGULAR: f64 = 1e-12;
/// A candidate below this absolute size cannot be a pivot.
const TOL_PIVOT_ABS: f64 = 1e-11;

/// Why a solve did not reach an optimum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoundedLpError {
    /// The input is malformed, with the reason.
    InvalidInput(String),
    /// No point satisfies the rows and bounds.
    Infeasible,
    /// A non-basic column has a reduced cost that its bounds cannot absorb, so
    /// the starting basis is not dual feasible. The program is unbounded or
    /// infeasible. No primal phase is run.
    DualInfeasible,
    /// The pivot budget ran out.
    IterationLimit,
    /// The deadline passed.
    TimeLimit,
    /// The basis could not be factorised.
    Singular,
}

impl fmt::Display for BoundedLpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(why) => write!(f, "InvalidInput: {why}"),
            Self::Infeasible => f.write_str("Infeasible"),
            Self::DualInfeasible => f.write_str("DualInfeasible"),
            Self::IterationLimit => f.write_str("IterationLimit"),
            Self::TimeLimit => f.write_str("TimeLimit"),
            Self::Singular => f.write_str("Singular"),
        }
    }
}

impl std::error::Error for BoundedLpError {}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Status {
    Basic,
    Lower,
    Upper,
    /// Non-basic with both bounds infinite, held at 0.
    Free,
}

/// A saved basis, see [`BoundedSimplex::snapshot`].
#[derive(Clone)]
pub struct Snapshot {
    basis: Vec<usize>,
    /// One bit per variable, set when it is non-basic at its upper bound.
    at_upper: Vec<u64>,
}

#[derive(Clone, Copy)]
struct Candidate {
    j: usize,
    ratio: f64,
    t: f64,
    weight: f64,
    flip: bool,
}

/// The order of the ratio test: ratio, then column index. Total, so the
/// selection is the same whatever the order of the input.
fn candidate_order(a: &Candidate, b: &Candidate) -> std::cmp::Ordering {
    a.ratio.total_cmp(&b.ratio).then(a.j.cmp(&b.j))
}

/// A bounded-variable dual simplex solver. See the [module](self) documentation.
#[derive(Clone)]
pub struct BoundedSimplex {
    m: usize,
    n: usize,
    /// Column-major `m * n`.
    a: Vec<f64>,
    /// Minimisation costs of the `n + m` variables.
    q: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    /// Tolerance on the bounds of each of the `n + m` variables.
    tol: Vec<f64>,
    dual_tol: f64,
    status: Vec<Status>,
    basis: Vec<usize>,
    /// Row-major `m * m`.
    binv: Vec<f64>,
    xb: Vec<f64>,
    since_refactor: usize,
    iterations: u64,
}

fn finite_or(value: f64, otherwise: f64) -> f64 {
    if value.is_finite() { value } else { otherwise }
}

/// Run `f` over the chunks of `0..n` and join the results in chunk order.
fn chunked<T, F>(n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize, usize) -> Vec<T> + Sync + Send,
{
    let chunks = n.div_ceil(CHUNK);
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        let parts: Vec<Vec<T>> = (0..chunks)
            .into_par_iter()
            .map(|c| f(c * CHUNK, ((c + 1) * CHUNK).min(n)))
            .collect();
        parts.into_iter().flatten().collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        let mut out = Vec::new();
        for c in 0..chunks {
            out.extend(f(c * CHUNK, ((c + 1) * CHUNK).min(n)));
        }
        out
    }
}

impl BoundedSimplex {
    /// Builds a solver for `max c'x  s.t.  r_lo ≤ A x ≤ r_hi,  l ≤ x ≤ u`.
    ///
    /// `columns` holds `A` column by column, `n * m` values, so column `j` is
    /// `columns[j * m..(j + 1) * m]`. `n` is `cost.len()`.
    ///
    /// # Errors
    /// [`BoundedLpError::InvalidInput`] for a length that does not match, a NaN
    /// or infinite matrix entry or cost, a NaN bound, or a lower bound above its
    /// upper bound.
    pub fn new(
        m: usize,
        columns: &[f64],
        cost: &[f64],
        col_lo: &[f64],
        col_hi: &[f64],
        row_lo: &[f64],
        row_hi: &[f64],
    ) -> Result<Self, BoundedLpError> {
        let n = cost.len();
        let bad = |why: &str| Err(BoundedLpError::InvalidInput(why.to_string()));
        if columns.len() != n * m {
            return bad("the matrix does not have n * m entries");
        }
        if col_lo.len() != n || col_hi.len() != n {
            return bad("the column bounds do not have one entry per column");
        }
        if row_lo.len() != m || row_hi.len() != m {
            return bad("the row bounds do not have one entry per row");
        }
        if columns.iter().any(|v| !v.is_finite()) {
            return bad("the matrix has a non-finite entry");
        }
        if cost.iter().any(|v| !v.is_finite()) {
            return bad("a cost is not finite");
        }
        let mut lo = Vec::with_capacity(n + m);
        let mut hi = Vec::with_capacity(n + m);
        lo.extend_from_slice(col_lo);
        hi.extend_from_slice(col_hi);
        lo.extend_from_slice(row_lo);
        hi.extend_from_slice(row_hi);
        for j in 0..n + m {
            if lo[j].is_nan() || hi[j].is_nan() {
                return bad("a bound is NaN");
            }
            if lo[j] > hi[j] || lo[j] == f64::INFINITY || hi[j] == f64::NEG_INFINITY {
                return bad("a lower bound exceeds its upper bound");
            }
        }
        let mut q = Vec::with_capacity(n + m);
        q.extend(cost.iter().map(|c| -c));
        q.resize(n + m, 0.0);
        let q_max = q.iter().fold(1.0_f64, |acc, v| acc.max(v.abs()));

        let mut solver = Self {
            m,
            n,
            a: columns.to_vec(),
            q,
            lo,
            hi,
            tol: vec![0.0; n + m],
            dual_tol: REL_TOL * q_max,
            status: vec![Status::Lower; n + m],
            basis: (n..n + m).collect(),
            binv: vec![0.0; m * m],
            xb: vec![0.0; m],
            since_refactor: 0,
            iterations: 0,
        };
        solver.set_tolerances();
        for j in 0..n {
            solver.status[j] = solver.default_status(j);
        }
        for i in 0..m {
            solver.status[n + i] = Status::Basic;
            solver.binv[i * m + i] = -1.0;
        }
        Ok(solver)
    }

    /// The width used in the tolerance of a row: the bound width, or 1 when it
    /// is infinite.
    fn width(&self, j: usize) -> f64 {
        finite_or(self.hi[j] - self.lo[j], 1.0)
    }

    fn set_tolerances(&mut self) {
        let (m, n) = (self.m, self.n);
        let widths: Vec<f64> = (0..n).map(|j| self.width(j)).collect();
        let mut sq = vec![0.0; m];
        for j in 0..n {
            let w2 = widths[j] * widths[j];
            let column = &self.a[j * m..(j + 1) * m];
            for i in 0..m {
                sq[i] += column[i] * column[i] * w2;
            }
        }
        for j in 0..n {
            self.tol[j] = self.bound_tol(j);
        }
        for i in 0..m {
            self.tol[n + i] = REL_TOL * sq[i].sqrt().max(1.0);
        }
    }

    fn bound_tol(&self, j: usize) -> f64 {
        let scale = finite_or(self.lo[j].abs(), 0.0).max(finite_or(self.hi[j].abs(), 0.0));
        REL_TOL * (1.0 + scale)
    }

    /// The non-basic status for a column that has no preference: the lower
    /// bound, or the upper one when the lower is infinite, or free.
    fn default_status(&self, j: usize) -> Status {
        if self.lo[j].is_finite() {
            Status::Lower
        } else if self.hi[j].is_finite() {
            Status::Upper
        } else {
            Status::Free
        }
    }

    /// Number of structural columns.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.n
    }

    /// Number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.m
    }

    /// Total simplex pivots over every solve.
    #[must_use]
    pub const fn iterations(&self) -> u64 {
        self.iterations
    }

    /// The tolerance on the bounds of variable `j` (`j >= n` is the slack of
    /// row `j - n`).
    #[must_use]
    pub fn tolerance(&self, j: usize) -> f64 {
        self.tol[j]
    }

    /// The bounds of structural column `j`.
    #[must_use]
    pub fn bounds(&self, j: usize) -> (f64, f64) {
        (self.lo[j], self.hi[j])
    }

    /// Changes the bounds of structural column `j`. The next solve starts from
    /// the current basis.
    ///
    /// # Errors
    /// [`BoundedLpError::InvalidInput`] for a NaN bound or `lo > hi`.
    pub fn set_bounds(&mut self, j: usize, lo: f64, hi: f64) -> Result<(), BoundedLpError> {
        if j >= self.n {
            return Err(BoundedLpError::InvalidInput(
                "column index out of range".to_string(),
            ));
        }
        if lo.is_nan() || hi.is_nan() || lo > hi || lo == f64::INFINITY || hi == f64::NEG_INFINITY {
            return Err(BoundedLpError::InvalidInput(
                "a lower bound exceeds its upper bound".to_string(),
            ));
        }
        self.lo[j] = lo;
        self.hi[j] = hi;
        self.tol[j] = self.bound_tol(j);
        if self.status[j] != Status::Basic {
            self.status[j] = match self.status[j] {
                Status::Lower if lo.is_finite() => Status::Lower,
                Status::Upper if hi.is_finite() => Status::Upper,
                _ => self.default_status(j),
            };
        }
        Ok(())
    }

    /// Saves the basis and the bound each non-basic variable sits at.
    #[must_use]
    pub fn snapshot(&self) -> Snapshot {
        let total = self.n + self.m;
        let mut at_upper = vec![0_u64; total.div_ceil(64)];
        for (j, status) in self.status.iter().enumerate() {
            if *status == Status::Upper {
                at_upper[j / 64] |= 1 << (j % 64);
            }
        }
        Snapshot {
            basis: self.basis.clone(),
            at_upper,
        }
    }

    /// Reloads a saved basis at the current bounds. The next solve factorises it.
    pub fn restore(&mut self, snapshot: &Snapshot) {
        for j in 0..self.n + self.m {
            self.status[j] = if (snapshot.at_upper[j / 64] >> (j % 64)) & 1 == 1 {
                Status::Upper
            } else {
                Status::Lower
            };
            if self.status[j] == Status::Upper && !self.hi[j].is_finite() {
                self.status[j] = self.default_status(j);
            }
            if self.status[j] == Status::Lower && !self.lo[j].is_finite() {
                self.status[j] = self.default_status(j);
            }
        }
        self.basis.clone_from(&snapshot.basis);
        for &j in &snapshot.basis {
            self.status[j] = Status::Basic;
        }
        self.since_refactor = REFACTOR_EVERY;
    }

    fn nonbasic_value(&self, j: usize) -> f64 {
        match self.status[j] {
            Status::Lower => self.lo[j],
            Status::Upper => self.hi[j],
            Status::Free | Status::Basic => 0.0,
        }
    }

    /// The value of structural column `j`.
    #[must_use]
    pub fn value(&self, j: usize) -> f64 {
        if self.status[j] == Status::Basic {
            for (position, &b) in self.basis.iter().enumerate() {
                if b == j {
                    return self.xb[position];
                }
            }
        }
        self.nonbasic_value(j)
    }

    /// The values of all structural columns.
    #[must_use]
    pub fn values(&self) -> Vec<f64> {
        let mut x: Vec<f64> = (0..self.n).map(|j| self.nonbasic_value(j)).collect();
        for (position, &b) in self.basis.iter().enumerate() {
            if b < self.n {
                x[b] = self.xb[position];
            }
        }
        x
    }

    /// The structural columns that are basic, with their values, in basis
    /// order. At most `m`.
    pub fn basic_columns(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.basis
            .iter()
            .zip(&self.xb)
            .filter(|(b, _)| **b < self.n)
            .map(|(b, v)| (*b, *v))
    }

    /// `c'x` at the current point. Meaningful after a successful solve.
    #[must_use]
    pub fn objective(&self) -> f64 {
        let x = self.values();
        // Sum in column order so the value does not depend on the basis.
        -x.iter().zip(&self.q).map(|(v, q)| v * q).sum::<f64>()
    }

    /// The activity `A x` of each row at the current point.
    #[must_use]
    pub fn row_activity(&self) -> Vec<f64> {
        let x = self.values();
        let mut out = vec![0.0; self.m];
        for j in 0..self.n {
            if x[j] != 0.0 {
                let column = &self.a[j * self.m..(j + 1) * self.m];
                for i in 0..self.m {
                    out[i] += column[i] * x[j];
                }
            }
        }
        out
    }

    /// Column `j` of the matrix, `m` entries.
    #[must_use]
    pub fn column(&self, j: usize) -> &[f64] {
        &self.a[j * self.m..(j + 1) * self.m]
    }

    /// The scale `1e-9`-tolerance of row `i` is built on, see the
    /// [module](self) documentation.
    #[must_use]
    pub fn row_tolerance(&self, i: usize) -> f64 {
        self.tol[self.n + i]
    }

    // ----- linear algebra -----

    /// Column of variable `j` into `out`: a stored column, or `-e_i` for the
    /// slack of row `i`.
    fn load_column(&self, j: usize, out: &mut [f64]) {
        if j < self.n {
            out.copy_from_slice(&self.a[j * self.m..(j + 1) * self.m]);
        } else {
            out.fill(0.0);
            out[j - self.n] = -1.0;
        }
    }

    /// Rebuilds `B^{-1}` from the basis by Gauss-Jordan elimination with
    /// partial pivoting, then the basic values.
    fn refactor(&mut self) -> Result<(), BoundedLpError> {
        let m = self.m;
        if m > 0 {
            let w = 2 * m;
            let mut aug = vec![0.0; m * w];
            let mut column = vec![0.0; m];
            for (k, &j) in self.basis.iter().enumerate() {
                self.load_column(j, &mut column);
                for i in 0..m {
                    aug[i * w + k] = column[i];
                }
            }
            for i in 0..m {
                aug[i * w + m + i] = 1.0;
            }
            for c in 0..m {
                let mut p = c;
                let mut best = aug[c * w + c].abs();
                for i in c + 1..m {
                    let v = aug[i * w + c].abs();
                    if v > best {
                        best = v;
                        p = i;
                    }
                }
                if best < TOL_SINGULAR {
                    return Err(BoundedLpError::Singular);
                }
                if p != c {
                    for k in 0..w {
                        aug.swap(c * w + k, p * w + k);
                    }
                }
                let pivot = aug[c * w + c];
                for k in 0..w {
                    aug[c * w + k] /= pivot;
                }
                for i in 0..m {
                    if i != c {
                        let f = aug[i * w + c];
                        if f != 0.0 {
                            for k in 0..w {
                                aug[i * w + k] -= f * aug[c * w + k];
                            }
                        }
                    }
                }
            }
            for i in 0..m {
                self.binv[i * m..(i + 1) * m].copy_from_slice(&aug[i * w + m..(i + 1) * w]);
            }
        }
        self.since_refactor = 0;
        self.recompute_basic_values();
        Ok(())
    }

    /// `x_B = -B^{-1} (N x_N)`, with `N x_N` summed per chunk and the chunk
    /// sums added in order.
    fn recompute_basic_values(&mut self) {
        let m = self.m;
        let n = self.n;
        let parts: Vec<Vec<f64>> = {
            let this = &*self;
            let chunks = n.div_ceil(CHUNK);
            let one = |c: usize| {
                let mut v = vec![0.0; m];
                for j in c * CHUNK..((c + 1) * CHUNK).min(n) {
                    if this.status[j] == Status::Basic {
                        continue;
                    }
                    let x = this.nonbasic_value(j);
                    if x != 0.0 {
                        let column = &this.a[j * m..(j + 1) * m];
                        for i in 0..m {
                            v[i] += column[i] * x;
                        }
                    }
                }
                v
            };
            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;
                (0..chunks).into_par_iter().map(one).collect()
            }
            #[cfg(not(feature = "rayon"))]
            {
                (0..chunks).map(one).collect()
            }
        };
        let mut v = vec![0.0; m];
        for part in &parts {
            for i in 0..m {
                v[i] += part[i];
            }
        }
        for i in 0..m {
            if self.status[n + i] != Status::Basic {
                v[i] -= self.nonbasic_value(n + i);
            }
        }
        for r in 0..m {
            let row = &self.binv[r * m..(r + 1) * m];
            let mut s = 0.0;
            for i in 0..m {
                s += row[i] * v[i];
            }
            self.xb[r] = -s;
        }
    }

    /// `y = B^{-T} q_B`.
    fn duals(&self) -> Vec<f64> {
        let m = self.m;
        let mut y = vec![0.0; m];
        for (r, &b) in self.basis.iter().enumerate() {
            let c = self.q[b];
            if c != 0.0 {
                let row = &self.binv[r * m..(r + 1) * m];
                for k in 0..m {
                    y[k] += c * row[k];
                }
            }
        }
        y
    }

    /// Flip every non-basic variable whose reduced cost has the wrong sign
    /// beyond the dual tolerance to its other bound.
    fn restore_dual_feasibility(&mut self) -> Result<(), BoundedLpError> {
        let m = self.m;
        let n = self.n;
        let y = self.duals();
        let this = &*self;
        let wrong: Vec<usize> = chunked(n, |start, end| {
            let mut out = Vec::new();
            for j in start..end {
                let st = this.status[j];
                if st == Status::Basic || this.lo[j] == this.hi[j] {
                    continue;
                }
                let column = &this.a[j * m..(j + 1) * m];
                let mut ay = 0.0;
                for i in 0..m {
                    ay += column[i] * y[i];
                }
                if this.wrong_sign(st, this.q[j] - ay) {
                    out.push(j);
                }
            }
            out
        });
        let mut wrong = wrong;
        for (i, &yi) in y.iter().enumerate() {
            let j = n + i;
            let st = self.status[j];
            if st != Status::Basic
                && self.lo[j] != self.hi[j]
                && self.wrong_sign(st, self.q[j] + yi)
            {
                wrong.push(j);
            }
        }
        let mut changed = false;
        for j in wrong {
            let flipped = match self.status[j] {
                Status::Lower if self.hi[j].is_finite() => Status::Upper,
                Status::Upper if self.lo[j].is_finite() => Status::Lower,
                _ => return Err(BoundedLpError::DualInfeasible),
            };
            self.status[j] = flipped;
            changed = true;
        }
        if changed {
            self.recompute_basic_values();
        }
        Ok(())
    }

    fn wrong_sign(&self, status: Status, d: f64) -> bool {
        match status {
            Status::Lower => d < -self.dual_tol,
            Status::Upper => d > self.dual_tol,
            Status::Free => d.abs() > self.dual_tol,
            Status::Basic => false,
        }
    }

    /// The scan of the pivot row: candidates for entering, for a leaving
    /// variable that must move up (`increase`) or down.
    fn candidates(&self, rho: &[f64], y: &[f64], increase: bool) -> Vec<Candidate> {
        let m = self.m;
        let n = self.n;
        let sign = if increase { -1.0 } else { 1.0 };
        let consider = |j: usize, alpha: f64, d: f64| -> Option<Candidate> {
            let st = self.status[j];
            if st == Status::Basic || self.lo[j] == self.hi[j] {
                return None;
            }
            let t = sign * alpha;
            let (eligible, signed_d) = match st {
                Status::Lower => (t > TOL_PIVOT_ABS, d.max(0.0)),
                Status::Upper => (t < -TOL_PIVOT_ABS, (-d).max(0.0)),
                Status::Free => (t.abs() > TOL_PIVOT_ABS, 0.0),
                Status::Basic => (false, 0.0),
            };
            if !eligible {
                return None;
            }
            let width = self.hi[j] - self.lo[j];
            let weight = if width.is_finite() {
                t.abs() * width
            } else {
                f64::INFINITY
            };
            Some(Candidate {
                j,
                ratio: signed_d / t.abs(),
                t,
                weight,
                flip: false,
            })
        };
        let mut out = chunked(n, |start, end| {
            let mut v = Vec::new();
            for j in start..end {
                if self.status[j] == Status::Basic {
                    continue;
                }
                let column = &self.a[j * m..(j + 1) * m];
                let mut alpha = 0.0;
                let mut ay = 0.0;
                for i in 0..m {
                    alpha += rho[i] * column[i];
                    ay += y[i] * column[i];
                }
                if let Some(c) = consider(j, alpha, self.q[j] - ay) {
                    v.push(c);
                }
            }
            v
        });
        for i in 0..m {
            let j = n + i;
            if let Some(c) = consider(j, -rho[i], self.q[j] + y[i]) {
                out.push(c);
            }
        }
        out
    }

    /// The bound flipping ratio test. Marks the candidates to flip and returns
    /// the entering candidate, or `None` when the row cannot be repaired.
    fn ratio_test(cands: &mut [Candidate], mut slope: f64, tol: f64) -> Option<Candidate> {
        let mut start = 0;
        let mut end = cands.len();
        loop {
            let len = end - start;
            if len == 0 {
                return None;
            }
            if len <= 32 {
                let slice = &mut cands[start..end];
                slice.sort_by(candidate_order);
                for c in slice.iter_mut() {
                    if slope - c.weight > tol {
                        slope -= c.weight;
                        c.flip = true;
                    } else {
                        return Some(*c);
                    }
                }
                return None;
            }
            let mid = len / 2;
            cands[start..end].select_nth_unstable_by(mid, candidate_order);
            let lower: f64 = cands[start..start + mid].iter().map(|c| c.weight).sum();
            if slope - lower > tol {
                for c in &mut cands[start..start + mid] {
                    c.flip = true;
                }
                slope -= lower;
                start += mid;
            } else {
                end = start + mid;
            }
        }
    }

    /// Solves from the current basis.
    ///
    /// Starts with a refactorisation, so bound changes made since the last solve
    /// take effect. `deadline`, when given, is checked once per pivot.
    ///
    /// # Errors
    /// See [`BoundedLpError`]. After `Infeasible` the basis is dual feasible
    /// and can be reused after the bounds change back.
    pub fn solve(&mut self, deadline: Option<Instant>) -> Result<(), BoundedLpError> {
        let (m, n) = (self.m, self.n);
        self.refactor()?;
        self.restore_dual_feasibility()?;
        let limit = 100 * (m + n) + 1000;
        let mut pivots = 0_usize;
        let mut degenerate_run = 0_usize;
        let mut bland = false;
        let mut col = vec![0.0; m];
        let mut a_q = vec![0.0; m];
        loop {
            if let Some(d) = deadline
                && Instant::now() >= d
            {
                return Err(BoundedLpError::TimeLimit);
            }
            if self.since_refactor >= REFACTOR_EVERY {
                self.refactor()?;
                self.restore_dual_feasibility()?;
            }
            // Leaving position.
            let mut leave: Option<(usize, f64)> = None;
            for r in 0..m {
                let b = self.basis[r];
                let v = self.xb[r];
                let violation = if v < self.lo[b] - self.tol[b] {
                    self.lo[b] - v
                } else if v > self.hi[b] + self.tol[b] {
                    v - self.hi[b]
                } else {
                    continue;
                };
                let score = if bland {
                    -(b as f64)
                } else {
                    violation / self.tol[b]
                };
                let better = match leave {
                    None => true,
                    Some((best_r, best)) => {
                        score > best || (score == best && b < self.basis[best_r])
                    }
                };
                if better {
                    leave = Some((r, score));
                }
            }
            let Some((r, _)) = leave else {
                return Ok(());
            };
            pivots += 1;
            self.iterations += 1;
            if pivots > limit {
                return Err(BoundedLpError::IterationLimit);
            }
            let p = self.basis[r];
            let increase = self.xb[r] < self.lo[p];
            let target = if increase { self.lo[p] } else { self.hi[p] };
            let slope0 = (target - self.xb[r]).abs();

            let rho: Vec<f64> = self.binv[r * m..(r + 1) * m].to_vec();
            let y = self.duals();
            let mut cands = self.candidates(&rho, &y, increase);
            let t_max = cands.iter().fold(0.0_f64, |acc, c| acc.max(c.t.abs()));
            cands.retain(|c| c.t.abs() >= REL_TOL * t_max);
            if cands.is_empty() {
                if self.since_refactor > 0 {
                    self.refactor()?;
                    continue;
                }
                return Err(BoundedLpError::Infeasible);
            }

            let entering = if bland {
                let min_ratio = cands.iter().fold(f64::INFINITY, |acc, c| acc.min(c.ratio));
                let lowest = cands
                    .iter()
                    .filter(|c| c.ratio <= min_ratio + self.dual_tol * 1e-3)
                    .min_by_key(|c| c.j);
                let Some(&c) = lowest else {
                    return Err(BoundedLpError::Infeasible);
                };
                c
            } else {
                let Some(e) = Self::ratio_test(&mut cands, slope0, self.tol[p]) else {
                    if self.since_refactor > 0 {
                        self.refactor()?;
                        continue;
                    }
                    return Err(BoundedLpError::Infeasible);
                };
                // Among the candidates not flipped whose ratio is within the dual
                // tolerance of the entering one, the largest pivot.
                let window = e.ratio + self.dual_tol / e.t.abs().max(TOL_PIVOT_ABS);
                let mut best = e;
                for c in cands.iter().filter(|c| !c.flip && c.ratio <= window) {
                    if c.t.abs() > best.t.abs() || (c.t.abs() == best.t.abs() && c.j < best.j) {
                        best = *c;
                    }
                }
                best
            };

            // Flips.
            if !bland {
                let mut delta_b = vec![0.0; m];
                let mut any = false;
                for c in cands.iter().filter(|c| c.flip && c.j != entering.j) {
                    let delta = match self.status[c.j] {
                        Status::Lower => {
                            self.status[c.j] = Status::Upper;
                            self.hi[c.j] - self.lo[c.j]
                        }
                        Status::Upper => {
                            self.status[c.j] = Status::Lower;
                            self.lo[c.j] - self.hi[c.j]
                        }
                        _ => continue,
                    };
                    any = true;
                    if c.j < n {
                        let column = &self.a[c.j * m..(c.j + 1) * m];
                        for i in 0..m {
                            delta_b[i] += column[i] * delta;
                        }
                    } else {
                        delta_b[c.j - n] -= delta;
                    }
                }
                if any {
                    for k in 0..m {
                        let row = &self.binv[k * m..(k + 1) * m];
                        let mut s = 0.0;
                        for i in 0..m {
                            s += row[i] * delta_b[i];
                        }
                        self.xb[k] -= s;
                    }
                }
            }

            // Pivot.
            let q = entering.j;
            self.load_column(q, &mut col);
            for k in 0..m {
                let row = &self.binv[k * m..(k + 1) * m];
                let mut s = 0.0;
                for i in 0..m {
                    s += row[i] * col[i];
                }
                a_q[k] = s;
            }
            let alpha_r = a_q[r];
            let from_row = if q < n {
                let column = &self.a[q * m..(q + 1) * m];
                let mut s = 0.0;
                for i in 0..m {
                    s += rho[i] * column[i];
                }
                s
            } else {
                -rho[q - n]
            };
            if alpha_r.abs() < TOL_PIVOT_ABS
                || (alpha_r - from_row).abs() > 1e-7 * (1.0 + alpha_r.abs())
            {
                if self.since_refactor > 0 {
                    self.refactor()?;
                    self.restore_dual_feasibility()?;
                    continue;
                }
                return Err(BoundedLpError::Singular);
            }
            let x_q0 = self.nonbasic_value(q);
            let delta_q = (self.xb[r] - target) / alpha_r;
            for k in 0..m {
                self.xb[k] -= a_q[k] * delta_q;
            }
            self.xb[r] = x_q0 + delta_q;
            // B^{-1} update.
            let inv_pivot = 1.0 / alpha_r;
            for k in 0..m {
                self.binv[r * m + k] *= inv_pivot;
            }
            let pivot_row: Vec<f64> = self.binv[r * m..(r + 1) * m].to_vec();
            for i in 0..m {
                if i != r && a_q[i] != 0.0 {
                    let f = a_q[i];
                    let row = &mut self.binv[i * m..(i + 1) * m];
                    for k in 0..m {
                        row[k] -= f * pivot_row[k];
                    }
                }
            }
            self.basis[r] = q;
            self.status[q] = Status::Basic;
            self.status[p] = if increase || self.lo[p] == self.hi[p] {
                Status::Lower
            } else {
                Status::Upper
            };
            self.since_refactor += 1;

            if entering.ratio <= self.dual_tol {
                degenerate_run += 1;
                if degenerate_run >= BLAND_AFTER_DEGENERATE {
                    bland = true;
                }
            } else {
                degenerate_run = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64, enough for generating test problems.
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
            let unit = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            lo + (hi - lo) * unit
        }
    }

    const INF: f64 = f64::INFINITY;

    #[test]
    fn small_example_with_a_bound_flip() {
        // max x0 + x1 + x2  s.t.  x0 + 2 x1 + 3 x2 <= 4,  x in [0, 1]^3
        let mut lp = BoundedSimplex::new(
            1,
            &[1.0, 2.0, 3.0],
            &[1.0, 1.0, 1.0],
            &[0.0; 3],
            &[1.0; 3],
            &[-INF],
            &[4.0],
        )
        .unwrap();
        lp.solve(None).unwrap();
        // x0 = x1 = 1, x2 = 1/3.
        assert!((lp.objective() - (2.0 + 1.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn infeasible_rows_are_reported() {
        let mut lp = BoundedSimplex::new(
            1,
            &[1.0, 1.0],
            &[1.0, 1.0],
            &[0.0; 2],
            &[1.0; 2],
            &[3.0],
            &[INF],
        )
        .unwrap();
        assert_eq!(lp.solve(None), Err(BoundedLpError::Infeasible));
    }

    #[test]
    fn invalid_input_is_an_error() {
        let bad = BoundedSimplex::new(1, &[1.0], &[1.0], &[1.0], &[0.0], &[0.0], &[1.0]);
        assert!(matches!(bad, Err(BoundedLpError::InvalidInput(_))));
        let bad = BoundedSimplex::new(1, &[f64::NAN], &[1.0], &[0.0], &[1.0], &[0.0], &[1.0]);
        assert!(matches!(bad, Err(BoundedLpError::InvalidInput(_))));
        let bad = BoundedSimplex::new(1, &[1.0, 1.0], &[1.0], &[0.0], &[1.0], &[0.0], &[1.0]);
        assert!(matches!(bad, Err(BoundedLpError::InvalidInput(_))));
    }

    /// Compare with the primal simplex of this crate on random equality
    /// programs: `max c'x  s.t.  A x = g, x >= 0` has the form of this
    /// module's programs with equality rows, lower bound 0 and no upper bound.
    #[test]
    fn agrees_with_the_primal_simplex() {
        use crate::simplex::DenseSimplex;
        let mut rng = Rng(7);
        let mut compared = 0;
        for trial in 0..200 {
            let m = 1 + (trial % 4);
            let n = m + 2 + (trial % 7);
            let columns: Vec<Vec<f64>> = (0..n)
                .map(|_| (0..m).map(|_| rng.uniform(-1.0, 2.0)).collect())
                .collect();
            let x0: Vec<f64> = (0..n).map(|_| rng.uniform(0.0, 1.0)).collect();
            let g: Vec<f64> = (0..m)
                .map(|i| (0..n).map(|j| columns[j][i] * x0[j]).sum())
                .collect();
            // A cost that is bounded: below a_j'y for some y.
            let y: Vec<f64> = (0..m).map(|_| rng.uniform(-1.0, 1.0)).collect();
            let cost: Vec<f64> = columns
                .iter()
                .map(|c| c.iter().zip(&y).map(|(a, y)| a * y).sum::<f64>() - rng.uniform(0.0, 1.0))
                .collect();
            let mut primal = DenseSimplex::new(&columns, &g);
            primal.set_cost(&cost);
            let Ok(()) = primal.solve() else { continue };
            let flat: Vec<f64> = columns.iter().flatten().copied().collect();
            // Large finite upper bounds keep every starting basis dual feasible.
            let mut lp =
                BoundedSimplex::new(m, &flat, &cost, &vec![0.0; n], &vec![1e3; n], &g, &g).unwrap();
            lp.solve(None).unwrap();
            assert!(
                (lp.objective() - primal.objective()).abs() < 1e-7,
                "trial {trial}: {} vs {}",
                lp.objective(),
                primal.objective()
            );
            compared += 1;
        }
        assert!(compared > 100);
    }

    #[test]
    fn warm_start_after_bound_change_matches_a_fresh_solve() {
        let mut rng = Rng(11);
        let (m, n) = (3, 40);
        let columns: Vec<f64> = (0..m * n).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let cost: Vec<f64> = (0..n).map(|_| 1.0).collect();
        let row_hi = vec![0.3; m];
        let row_lo = vec![-0.3; m];
        let build = || {
            BoundedSimplex::new(
                m,
                &columns,
                &cost,
                &vec![0.0; n],
                &vec![1.0; n],
                &row_lo,
                &row_hi,
            )
            .unwrap()
        };
        let mut warm = build();
        warm.solve(None).unwrap();
        for step in 0..15 {
            let j = (step * 7) % n;
            let v = if step % 2 == 0 { 0.0 } else { 1.0 };
            warm.set_bounds(j, v, v).unwrap();
            let mut fresh = build();
            for s in 0..=step {
                let jj = (s * 7) % n;
                let vv = if s % 2 == 0 { 0.0 } else { 1.0 };
                fresh.set_bounds(jj, vv, vv).unwrap();
            }
            let a = warm.solve(None);
            let b = fresh.solve(None);
            assert_eq!(a.is_ok(), b.is_ok(), "step {step}");
            if a.is_ok() {
                assert!((warm.objective() - fresh.objective()).abs() < 1e-8);
            } else {
                break;
            }
        }
    }
}

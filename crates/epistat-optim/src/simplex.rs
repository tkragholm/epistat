//! A dense primal simplex method for small linear programs.
//!
//! Solves
//!
//! ```text
//! max  c'λ    subject to    A λ = g,    λ ≥ 0
//! ```
//!
//! with `A` an `m × n` dense matrix, `m` small (a handful of rows) and `n` up
//! to a few dozen columns. The dual of this program is
//! `min g'y  s.t.  A'y ≥ c`, with `y` free, and the optimal `y` is available
//! from [`DenseSimplex::duals`].
//!
//! The constraint matrix and `g` are fixed when the solver is built and only
//! `c` changes between solves. A basis stays primal feasible when `c` changes,
//! so each solve starts from the basis the previous one ended with (the warm
//! start). The first solve goes through a phase one on artificial columns.
//!
//! # Algorithm
//!
//! The method is the revised primal simplex method of Dantzig \[1\] \[3\]. A basis
//! `B` is a set of `m` columns. It is kept as the explicit inverse `B^{-1}`,
//! with `x_B = B^{-1} g` the basic values and `y = B^{-T} c_B` the duals. The
//! reduced cost of a column `j` that is not basic is
//!
//! ```text
//! d_j = c_j - a_j'y
//! ```
//!
//! and the basis is optimal when no `d_j` exceeds `1e-9` times the largest `|c_j|`
//! (`1e-9` in phase one, whose costs are 0 and -1).
//!
//! Phase one. Row `i` gets an artificial column `±e_i`, with the sign of `g_i`,
//! so the all-artificial basis has `x_B = |g|`. Phase one maximises minus the sum
//! of the artificials, with cost 0 on the original columns and -1 on the
//! artificial ones. If the optimum leaves artificials with a total above `1e-8`
//! times the largest `|g_i|`,
//! the problem is [`LpError::Infeasible`]. Otherwise the basis is feasible for
//! every `c`. [`DenseSimplex::prepare`] runs phase one alone, so a solver can be
//! cloned after it and each clone solve for its own `c`.
//!
//! Each pivot, in either phase, does the following.
//!
//! 1. Pricing. Only original columns may enter. Dantzig's rule \[1\] takes the
//!    column with the largest `d_j` above its tolerance. If none has one, the basis is
//!    optimal and the solve ends.
//! 2. Ratio test. With `α = B^{-1} a_q` for the entering column `q`, the leaving
//!    row is the one with `α_i > 1e-10` that minimises `x_{B,i} / α_i`. Ratios
//!    within a step tolerance of `1e-12` times the largest `|g_i|` over the largest
//!    `|A_ij|` of each other are ties, and the row whose basic column has the
//!    lowest index wins. If no row has `α_i > 1e-10`, the problem is
//!    [`LpError::Unbounded`]. A basic artificial in phase two sits at zero, and it
//!    leaves at step 0 as soon as the entering column has `|α_i| > 1e-10` in its
//!    row, whichever way the pivot would move it, so it cannot become positive.
//! 3. Update. The basic values and `B^{-1}` are updated by the elementary row
//!    operations of the pivot, and `y` by the entering column's reduced cost times
//!    the scaled pivot row, which costs `m` operations instead of the `m²` of
//!    forming `B^{-T} c_B` again. Basic values that round below zero are set
//!    to 0.
//!
//! `B^{-1}` is rebuilt from the basis by Gauss-Jordan elimination with partial
//! pivoting after every 40 pivots, and `x_B` and `y` with it. A pivot below
//! `1e-13` in that elimination is [`LpError::Singular`].
//!
//! # Degeneracy
//!
//! A pivot with step length at most the step tolerance is degenerate. Dantzig's rule can
//! cycle through degenerate bases \[3\]. After 25 degenerate pivots in a row,
//! pricing switches to Bland's rule \[2\] for the rest of that phase: the entering
//! column is the one of lowest index with `d_j` above its tolerance. Together with the
//! tie-break of the ratio test, this cannot cycle. A pivot with a longer step
//! resets the count, but not the switch.
//!
//! Each phase is limited to `100 (m + n) + 1000` pivots, after which the solve
//! fails with [`LpError::IterationLimit`].
//!
//! # Warm starts
//!
//! Feasibility does not depend on `c`, so after the first solve, phase one is not
//! repeated. [`DenseSimplex::solve`] starts phase two from the basis the last
//! solve ended with. When it fails with anything but `Singular`, that basis is
//! still primal feasible. [`DenseSimplex::set_memory`] keeps further optimal bases
//! for starting points. [`DenseSimplex::optimality_interval`] gives the range of a
//! cost parameter over which the current basis stays optimal.
//!
//! # Example
//!
//! ```
//! use epistat_optim::simplex::DenseSimplex;
//!
//! // max 3 l0 + 2 l1  s.t.  l0 + l1 + l2 = 4,  l0 + 3 l1 + l3 = 6
//! let columns = vec![
//!     vec![1.0, 1.0],
//!     vec![1.0, 3.0],
//!     vec![1.0, 0.0],
//!     vec![0.0, 1.0],
//! ];
//! let mut lp = DenseSimplex::new(&columns, &[4.0, 6.0]);
//! lp.set_cost(&[3.0, 2.0, 0.0, 0.0]);
//! lp.solve().unwrap();
//! assert!((lp.objective() - 12.0).abs() < 1e-9);
//! ```
//!
//! # References
//!
//! 1. Dantzig GB. Linear Programming and Extensions. Princeton: Princeton
//!    University Press; 1963.
//! 2. Bland RG. New finite pivoting rules for the simplex method. Math Oper Res.
//!    1977;2(2):103-7.
//! 3. Chvátal V. Linear Programming. New York: W. H. Freeman; 1983.

use std::fmt;

/// Relative tolerances. Each is multiplied by the size of the quantity it
/// judges: the largest cost for a reduced cost, the largest right-hand side
/// for the artificials of phase one, and the largest right-hand side over the
/// largest matrix entry for a step length. The pivot tolerance judges a ratio
/// of matrix entries and has no unit.
const TOL_REDUCED_COST: f64 = 1e-9;
const TOL_PIVOT: f64 = 1e-10;
const TOL_DEGENERATE: f64 = 1e-12;
const TOL_PHASE_ONE: f64 = 1e-8;
const REFACTOR_EVERY: usize = 40;
/// Consecutive degenerate pivots before pricing switches to Bland's rule,
/// which cannot cycle.
const BLAND_AFTER_DEGENERATE: usize = 25;

/// Why a solve did not reach an optimum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LpError {
    /// No `λ ≥ 0` satisfies `A λ = g`.
    Infeasible,
    /// The objective increases without bound.
    Unbounded,
    /// The pivot budget ran out.
    IterationLimit,
    /// The basis could not be factorised.
    Singular,
}

impl fmt::Display for LpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Infeasible => "Infeasible",
            Self::Unbounded => "Unbounded",
            Self::IterationLimit => "IterationLimit",
            Self::Singular => "Singular",
        })
    }
}

impl std::error::Error for LpError {}

#[derive(Clone)]
struct RememberedBasis {
    /// The basis as a sorted set, for the membership test.
    key: Vec<usize>,
    basis: Vec<usize>,
    binv: Vec<f64>,
    xb: Vec<f64>,
    /// How many pivots the stored inverse is past its last factorisation, so
    /// a reload keeps counting from there rather than from zero.
    pivots_since_refactor: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    One,
    Two,
}

/// A dense primal simplex solver for `max c'λ  s.t.  A λ = g, λ ≥ 0`.
///
/// Build it once from the columns of `A` and `g`, set a cost vector with
/// [`set_cost`](Self::set_cost) and call [`solve`](Self::solve). Further calls
/// with other cost vectors start from the previous basis.
#[derive(Clone)]
pub struct DenseSimplex {
    m: usize,
    n: usize,
    /// Column-major, `m * n`: column `j` occupies `a[j * m..(j + 1) * m]`.
    a: Vec<f64>,
    cost: Vec<f64>,
    g: Vec<f64>,
    /// Artificial column `i` is `art_sign[i] * e_i`, so the all-artificial
    /// basis carries `|g|`.
    art_sign: Vec<f64>,
    /// Column index per basis position: `< n` is an original column, `n + i`
    /// the artificial for row `i`.
    basis: Vec<usize>,
    is_basic: Vec<bool>,
    /// Row-major `m * m`.
    binv: Vec<f64>,
    xb: Vec<f64>,
    y: Vec<f64>,
    col: Vec<f64>,
    pivot_row: Vec<f64>,
    lambda: Vec<f64>,
    objective: f64,
    has_feasible_basis: bool,
    pivots_since_refactor: usize,
    /// Optimal bases seen by earlier solves, each with its inverse and basic
    /// values, so a new cost vector can start from whichever of them it likes
    /// best rather than from wherever the last solve stopped. See
    /// [`Self::set_memory`].
    memory: Vec<RememberedBasis>,
    memory_cap: usize,
    memory_next: usize,
    key_scratch: Vec<usize>,
    duals_current: bool,
    /// The largest `|g|`, and the largest `|A|`, each 1 when all are zero.
    g_scale: f64,
    a_scale: f64,
    /// The largest `|c|` of the cost vector set last, 1 when all are zero.
    cost_scale: f64,
}

/// The largest absolute value, or 1 when every value is zero.
fn scale_of(values: &[f64]) -> f64 {
    let largest = values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    if largest > 0.0 { largest } else { 1.0 }
}

impl DenseSimplex {
    /// Builds a solver for the columns of `A` and the right-hand side `g`.
    /// `columns[j]` is column `j` of `A`, each of length `g.len()`. The cost
    /// vector starts at zero.
    #[must_use]
    pub fn new(columns: &[Vec<f64>], g: &[f64]) -> Self {
        let m = g.len();
        let n = columns.len();
        let mut a = Vec::with_capacity(m * n);
        for column in columns {
            debug_assert_eq!(column.len(), m);
            a.extend_from_slice(column);
        }
        Self {
            m,
            n,
            a,
            cost: vec![0.0; n],
            g: g.to_vec(),
            art_sign: vec![1.0; m],
            basis: vec![0; m],
            is_basic: vec![false; n + m],
            binv: vec![0.0; m * m],
            xb: vec![0.0; m],
            y: vec![0.0; m],
            col: vec![0.0; m],
            pivot_row: vec![0.0; m],
            lambda: vec![0.0; n],
            objective: 0.0,
            has_feasible_basis: false,
            pivots_since_refactor: 0,
            memory: Vec::new(),
            memory_cap: 0,
            memory_next: 0,
            key_scratch: vec![0; m],
            duals_current: false,
            g_scale: 1.0,
            a_scale: 1.0,
            cost_scale: 1.0,
        }
        .scaled()
    }

    fn scaled(mut self) -> Self {
        self.g_scale = scale_of(&self.g);
        self.a_scale = scale_of(&self.a);
        self
    }

    /// The tolerance on a reduced cost in `phase`. Phase one has costs of 0 and
    /// -1.
    fn tol_reduced_cost(&self, phase: Phase) -> f64 {
        match phase {
            Phase::One => TOL_REDUCED_COST,
            Phase::Two => TOL_REDUCED_COST * self.cost_scale,
        }
    }

    /// The tolerance on a step length, in the units of `λ`.
    fn tol_degenerate(&self) -> f64 {
        TOL_DEGENERATE * self.g_scale / self.a_scale
    }

    /// Remember up to `cap` optimal bases across solves, and start each solve
    /// from the remembered basis with the best objective under the new costs
    /// when that beats the current one.
    ///
    /// For a sequence of unrelated cost vectors over one polytope, the optimum
    /// often lands on a small set of vertices. Starting from the best known
    /// vertex then turns most solves into a single pricing pass. Zero disables
    /// it, and calling this clears anything remembered.
    pub fn set_memory(&mut self, cap: usize) {
        self.memory_cap = cap;
        self.memory.clear();
        self.memory_next = 0;
    }

    /// Sets the cost vector `c`, of length `n`, used by the next solve.
    ///
    /// # Panics
    /// In debug builds, when `cost.len()` is not the number of columns.
    pub fn set_cost(&mut self, cost: &[f64]) {
        debug_assert_eq!(cost.len(), self.n);
        self.cost.copy_from_slice(cost);
        self.cost_scale = scale_of(cost);
    }

    /// `c'λ` at the last optimum.
    #[must_use]
    pub const fn objective(&self) -> f64 {
        self.objective
    }

    /// The optimal `λ`, length `n`.
    #[must_use]
    pub fn lambda(&self) -> &[f64] {
        &self.lambda
    }

    /// The optimal duals `y`, length `m`: the solution of the program this is
    /// the dual of, `min g'y  s.t.  A'y ≥ c`. Computed from the final basis on
    /// demand. Only meaningful after a successful [`solve`](Self::solve).
    pub fn duals(&mut self) -> &[f64] {
        if !self.duals_current {
            self.compute_duals(Phase::Two);
            self.duals_current = true;
        }
        &self.y
    }

    /// The first dual alone, `Σ_r c_Br (B^{-1})_{r0}`, in the same order of
    /// summation as [`Self::duals`] so the two agree to the bit. It is the
    /// optimal value of the dual program when `g = e_0`.
    #[must_use]
    pub fn dual_0(&self) -> f64 {
        if self.duals_current {
            return self.y[0];
        }
        let m = self.m;
        let mut y0 = 0.0;
        for r in 0..m {
            let c_b = self.cost_of(self.basis[r], Phase::Two);
            if c_b == 0.0 {
                continue;
            }
            y0 += c_b * self.binv[r * m];
        }
        y0
    }

    /// Solves for the current cost vector, from the last basis when there is
    /// one.
    ///
    /// # Errors
    /// See [`LpError`]. After an error the basis is still primal feasible
    /// (except after `Singular`), so a later call can start from it.
    pub fn solve(&mut self) -> Result<(), LpError> {
        self.prepare()?;
        if !self.memory.is_empty() {
            self.start_from_best_remembered();
        }
        self.iterate(Phase::Two)?;
        if self.memory_cap > 0 {
            self.remember_current()?;
        }
        self.duals_current = false;
        self.lambda.fill(0.0);
        let mut objective = 0.0;
        for (position, &j) in self.basis.iter().enumerate() {
            if j < self.n {
                let value = self.xb[position].max(0.0);
                self.lambda[j] = value;
                objective += self.cost[j] * value;
            }
        }
        self.objective = objective;
        Ok(())
    }

    /// The current basis, one column index per position; `>= n` is an
    /// artificial column.
    #[must_use]
    pub fn basis(&self) -> &[usize] {
        &self.basis
    }

    /// The basic values, by basis position.
    #[must_use]
    pub fn basic_values(&self) -> &[f64] {
        &self.xb
    }

    /// The number of columns `n` of `A`.
    #[must_use]
    pub const fn num_columns(&self) -> usize {
        self.n
    }

    /// The interval of `t` on which the current basis stays optimal for the
    /// costs `c0 + t c1`.
    ///
    /// A basis is optimal when every nonbasic reduced cost is at most the
    /// pricing tolerance, and for affine costs each reduced cost is affine in
    /// `t`, so the interval is read off two reduced-cost vectors. Meant to be
    /// called right after a solve at some `t` in the interval; `y0` and `y1`
    /// are scratch vectors that are resized to `m`. Returns the lower and upper
    /// end, which are infinite when no column bounds that side.
    #[must_use]
    pub fn optimality_interval(
        &self,
        c0: &[f64],
        c1: &[f64],
        y0: &mut Vec<f64>,
        y1: &mut Vec<f64>,
    ) -> (f64, f64) {
        let m = self.m;
        y0.clear();
        y0.resize(m, 0.0);
        y1.clear();
        y1.resize(m, 0.0);
        for (r, &j) in self.basis.iter().enumerate() {
            if j >= self.n {
                continue;
            }
            let row = &self.binv[r * m..(r + 1) * m];
            let (cb0, cb1) = (c0[j], c1[j]);
            if cb0 != 0.0 {
                for (y, b) in y0.iter_mut().zip(row) {
                    *y += cb0 * b;
                }
            }
            if cb1 != 0.0 {
                for (y, b) in y1.iter_mut().zip(row) {
                    *y += cb1 * b;
                }
            }
        }
        let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
        let tol = TOL_REDUCED_COST * scale_of(c0);
        for j in 0..self.n {
            if self.is_basic[j] {
                continue;
            }
            let d0 = c0[j] - dot(self.column(j), y0);
            let d1 = c1[j] - dot(self.column(j), y1);
            // d0 + t d1 <= TOL
            if d1 > 0.0 {
                hi = hi.min((tol - d0) / d1);
            } else if d1 < 0.0 {
                lo = lo.max((tol - d0) / d1);
            }
        }
        (lo, hi)
    }

    /// The objective of a basis under the current costs: `Σ c_j x_Bj` over its
    /// original columns.
    fn basis_objective(&self, basis: &[usize], xb: &[f64]) -> f64 {
        basis
            .iter()
            .zip(xb)
            .filter(|(j, _)| **j < self.n)
            .map(|(j, x)| self.cost[*j] * x)
            .sum()
    }

    fn start_from_best_remembered(&mut self) {
        let current = self.basis_objective(&self.basis, &self.xb);
        let mut best = None;
        let mut best_value = current;
        for (index, entry) in self.memory.iter().enumerate() {
            let value = self.basis_objective(&entry.basis, &entry.xb);
            if value > best_value {
                best_value = value;
                best = Some(index);
            }
        }
        let Some(index) = best else {
            return;
        };
        let entry = &self.memory[index];
        self.basis.copy_from_slice(&entry.basis);
        self.binv.copy_from_slice(&entry.binv);
        self.xb.copy_from_slice(&entry.xb);
        self.pivots_since_refactor = entry.pivots_since_refactor;
        self.is_basic.fill(false);
        for &j in &self.basis {
            self.is_basic[j] = true;
        }
    }

    fn remember_current(&mut self) -> Result<(), LpError> {
        self.key_scratch.copy_from_slice(&self.basis);
        self.key_scratch.sort_unstable();
        if self
            .memory
            .iter()
            .any(|entry| entry.key == self.key_scratch)
        {
            return Ok(());
        }
        // A stored inverse is reloaded and pivoted on again and again, so it
        // goes in fresh when it is more than half way to its next
        // factorisation; otherwise drift could accumulate across cycles.
        if self.pivots_since_refactor > REFACTOR_EVERY / 2 {
            self.refactor()?;
        }
        let entry = RememberedBasis {
            key: self.key_scratch.clone(),
            basis: self.basis.clone(),
            binv: self.binv.clone(),
            xb: self.xb.clone(),
            pivots_since_refactor: self.pivots_since_refactor,
        };
        if self.memory.len() < self.memory_cap {
            self.memory.push(entry);
        } else {
            self.memory[self.memory_next] = entry;
            self.memory_next = (self.memory_next + 1) % self.memory_cap;
        }
        Ok(())
    }

    /// Phase one on its own: find a feasible basis without a cost vector, so
    /// that a solver can be cloned into many with the work done once.
    /// Feasibility does not depend on the costs, so this is never wasted.
    ///
    /// # Errors
    /// `Infeasible` when no `λ ≥ 0` satisfies `A λ = g`.
    pub fn prepare(&mut self) -> Result<(), LpError> {
        if self.has_feasible_basis {
            return Ok(());
        }
        self.install_artificial_basis();
        self.iterate(Phase::One)?;
        let infeasibility: f64 = self
            .basis
            .iter()
            .zip(&self.xb)
            .filter(|(j, _)| **j >= self.n)
            .map(|(_, x)| *x)
            .sum();
        if infeasibility > TOL_PHASE_ONE * self.g_scale {
            return Err(LpError::Infeasible);
        }
        self.has_feasible_basis = true;
        Ok(())
    }

    fn install_artificial_basis(&mut self) {
        let m = self.m;
        self.is_basic.fill(false);
        self.binv.fill(0.0);
        for i in 0..m {
            self.art_sign[i] = if self.g[i] < 0.0 { -1.0 } else { 1.0 };
            self.basis[i] = self.n + i;
            self.is_basic[self.n + i] = true;
            self.binv[i * m + i] = self.art_sign[i];
            self.xb[i] = self.g[i].abs();
        }
        self.pivots_since_refactor = 0;
    }

    fn cost_of(&self, j: usize, phase: Phase) -> f64 {
        match phase {
            Phase::One => {
                if j < self.n {
                    0.0
                } else {
                    -1.0
                }
            }
            Phase::Two => {
                if j < self.n {
                    self.cost[j]
                } else {
                    0.0
                }
            }
        }
    }

    /// `y = B^{-T} c_B`.
    fn compute_duals(&mut self, phase: Phase) {
        let m = self.m;
        self.y.fill(0.0);
        for r in 0..m {
            let c_b = self.cost_of(self.basis[r], phase);
            if c_b == 0.0 {
                continue;
            }
            let row = &self.binv[r * m..(r + 1) * m];
            for (y, b) in self.y.iter_mut().zip(row) {
                *y += c_b * b;
            }
        }
    }

    fn column(&self, j: usize) -> &[f64] {
        &self.a[j * self.m..(j + 1) * self.m]
    }

    fn iterate(&mut self, phase: Phase) -> Result<(), LpError> {
        let m = self.m;
        let n = self.n;
        let limit = 100 * (m + n) + 1_000;
        let mut degenerate_streak = 0usize;
        let mut bland = false;
        // `y` is rebuilt from the basis here and after every refactorisation,
        // and moved along with each pivot in between: the entering column's
        // reduced cost times the scaled pivot row is exactly the change in
        // `B^{-T} c_B`, and it costs `m` operations instead of `m²`.
        let mut duals_stale = true;
        for _ in 0..limit {
            if duals_stale {
                self.compute_duals(phase);
                duals_stale = false;
            }
            // Pricing. Only original columns may enter: artificials start
            // basic and, once out, have no business coming back.
            let mut entering = None;
            let tol_cost = self.tol_reduced_cost(phase);
            let tol_step = self.tol_degenerate();
            let mut best = tol_cost;
            for j in 0..n {
                if self.is_basic[j] {
                    continue;
                }
                let d = self.cost_of(j, phase) - dot(self.column(j), &self.y);
                if bland {
                    if d > tol_cost {
                        entering = Some(j);
                        best = d;
                        break;
                    }
                } else if d > best {
                    best = d;
                    entering = Some(j);
                }
            }
            let Some(q) = entering else {
                return Ok(());
            };
            let d_q = best;
            // col = B^{-1} a_q
            for i in 0..m {
                self.col[i] = dot(&self.binv[i * m..(i + 1) * m], &self.a[q * m..(q + 1) * m]);
            }
            // Ratio test. In phase two a basic artificial sits at zero and must
            // not be allowed to grow, so it blocks at zero whichever way the
            // pivot would move it.
            let mut leaving: Option<usize> = None;
            let mut theta = f64::INFINITY;
            for i in 0..m {
                let alpha = self.col[i];
                if phase == Phase::Two && self.basis[i] >= n && alpha.abs() > TOL_PIVOT {
                    leaving = Some(i);
                    theta = 0.0;
                    break;
                }
                if alpha > TOL_PIVOT {
                    let ratio = self.xb[i] / alpha;
                    let better = match leaving {
                        None => true,
                        Some(r) => {
                            ratio < theta - tol_step
                                || ((ratio - theta).abs() <= tol_step
                                    && self.basis[i] < self.basis[r])
                        }
                    };
                    if better {
                        theta = ratio;
                        leaving = Some(i);
                    }
                }
            }
            let Some(r) = leaving else {
                return Err(LpError::Unbounded);
            };
            if theta <= tol_step {
                degenerate_streak += 1;
                if degenerate_streak >= BLAND_AFTER_DEGENERATE {
                    bland = true;
                }
            } else {
                degenerate_streak = 0;
            }
            self.pivot(r, q, theta);
            for (y, pivot_row) in self.y.iter_mut().zip(&self.pivot_row) {
                *y += d_q * pivot_row;
            }
            if self.pivots_since_refactor >= REFACTOR_EVERY {
                self.refactor()?;
                duals_stale = true;
            }
        }
        Err(LpError::IterationLimit)
    }

    fn pivot(&mut self, r: usize, q: usize, theta: f64) {
        let m = self.m;
        let p = self.col[r];
        for i in 0..m {
            if i != r {
                self.xb[i] -= theta * self.col[i];
                if self.xb[i] < 0.0 {
                    self.xb[i] = 0.0;
                }
            }
        }
        self.xb[r] = theta;
        let inv_p = 1.0 / p;
        for (scratch, value) in self
            .pivot_row
            .iter_mut()
            .zip(&self.binv[r * m..(r + 1) * m])
        {
            *scratch = value * inv_p;
        }
        for i in 0..m {
            let factor = self.col[i];
            if i == r || factor == 0.0 {
                continue;
            }
            for (target, pivot_row) in self.binv[i * m..(i + 1) * m]
                .iter_mut()
                .zip(&self.pivot_row)
            {
                *target -= factor * pivot_row;
            }
        }
        self.binv[r * m..(r + 1) * m].copy_from_slice(&self.pivot_row);
        let old = self.basis[r];
        self.is_basic[old] = false;
        self.is_basic[q] = true;
        self.basis[r] = q;
        self.pivots_since_refactor += 1;
    }

    /// Rebuild `B^{-1}` from the basis by Gauss-Jordan elimination with partial
    /// pivoting, and `x_B = B^{-1} g` with it.
    fn refactor(&mut self) -> Result<(), LpError> {
        let m = self.m;
        let mut b = vec![0.0; m * m];
        for (position, &j) in self.basis.iter().enumerate() {
            if j < self.n {
                for (i, &value) in self.column(j).iter().enumerate() {
                    b[i * m + position] = value;
                }
            } else {
                let row = j - self.n;
                b[row * m + position] = self.art_sign[row];
            }
        }
        let mut inv = vec![0.0; m * m];
        for i in 0..m {
            inv[i * m + i] = 1.0;
        }
        for k in 0..m {
            let mut pivot_row = k;
            let mut pivot_abs = b[k * m + k].abs();
            for i in k + 1..m {
                let candidate = b[i * m + k].abs();
                if candidate > pivot_abs {
                    pivot_abs = candidate;
                    pivot_row = i;
                }
            }
            if pivot_abs < 1e-13 {
                return Err(LpError::Singular);
            }
            if pivot_row != k {
                for c in 0..m {
                    b.swap(k * m + c, pivot_row * m + c);
                    inv.swap(k * m + c, pivot_row * m + c);
                }
            }
            let inv_p = 1.0 / b[k * m + k];
            for c in 0..m {
                b[k * m + c] *= inv_p;
                inv[k * m + c] *= inv_p;
            }
            for i in 0..m {
                if i == k {
                    continue;
                }
                let factor = b[i * m + k];
                if factor == 0.0 {
                    continue;
                }
                for c in 0..m {
                    b[i * m + c] -= factor * b[k * m + c];
                    inv[i * m + c] -= factor * inv[k * m + c];
                }
            }
        }
        self.binv = inv;
        for i in 0..m {
            let row = &self.binv[i * m..(i + 1) * m];
            let value: f64 = row.iter().zip(&self.g).map(|(b, g)| b * g).sum();
            self.xb[i] = value.max(0.0);
        }
        self.pivots_since_refactor = 0;
        Ok(())
    }
}

/// Four independent accumulators, so the multiply-adds pipeline instead of
/// serialising on one sum. The summation order differs from a plain fold, which
/// moves a reduced cost by a rounding unit and nothing a pivot choice can see
/// except on an exact tie.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0_f64; 4];
    let (chunks_a, rest_a) = a.as_chunks::<4>();
    let (chunks_b, rest_b) = b.as_chunks::<4>();
    for (ca, cb) in chunks_a.iter().zip(chunks_b) {
        acc[0] += ca[0] * cb[0];
        acc[1] += ca[1] * cb[1];
        acc[2] += ca[2] * cb[2];
        acc[3] += ca[3] * cb[3];
    }
    let mut tail = 0.0;
    for (x, y) in rest_a.iter().zip(rest_b) {
        tail += x * y;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

#[cfg(test)]
mod tests {
    use super::{DenseSimplex, LpError};

    /// `SplitMix64`, enough for generating test problems.
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Uniform in `[lo, hi)`.
        fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
            let unit = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            lo + (hi - lo) * unit
        }

        /// Uniform integer in `[lo, hi)`.
        #[expect(clippy::cast_possible_truncation, reason = "the range is small")]
        fn range(&mut self, lo: usize, hi: usize) -> usize {
            lo + (self.next_u64() % (hi - lo) as u64) as usize
        }

        fn chance(&mut self, p: f64) -> bool {
            self.uniform(0.0, 1.0) < p
        }
    }

    /// Costs that keep `max c'λ` bounded on these columns: each `c_j` sits
    /// below `a_j'y` for one `y`, so `y` is dual feasible.
    fn bounded_cost(rng: &mut Rng, columns: &[Vec<f64>], m: usize) -> Vec<f64> {
        let y: Vec<f64> = (0..m).map(|_| rng.uniform(-1.0, 1.0)).collect();
        columns
            .iter()
            .map(|column| {
                let ay: f64 = column.iter().zip(&y).map(|(a, y)| a * y).sum();
                ay - rng.uniform(0.0, 1.0)
            })
            .collect()
    }

    type Problem = (Vec<Vec<f64>>, Vec<f64>, Vec<f64>);

    fn random_feasible_problem(rng: &mut Rng, m: usize, n: usize) -> Problem {
        let columns: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..m).map(|_| rng.uniform(-1.0, 1.0)).collect())
            .collect();
        // g = A λ0 for a sparse nonnegative λ0, so the problem is feasible.
        let mut g = vec![0.0; m];
        for column in &columns {
            if rng.chance(0.4) {
                let weight = rng.uniform(0.0, 2.0);
                for (g, a) in g.iter_mut().zip(column) {
                    *g += weight * a;
                }
            }
        }
        let c = bounded_cost(rng, &columns, m);
        (columns, g, c)
    }

    /// Checks primal feasibility, dual feasibility and strong duality of the
    /// solver's answer.
    fn assert_optimal(lp: &mut DenseSimplex, columns: &[Vec<f64>], g: &[f64], c: &[f64]) {
        for (i, target) in g.iter().enumerate() {
            let activity: f64 = columns
                .iter()
                .zip(lp.lambda())
                .map(|(column, l)| column[i] * l)
                .sum();
            assert!((activity - target).abs() < 1e-9, "row {i}");
        }
        assert!(lp.lambda().iter().all(|l| *l >= 0.0));
        let objective = lp.objective();
        let duals = lp.duals().to_vec();
        for (j, column) in columns.iter().enumerate() {
            let ay: f64 = column.iter().zip(&duals).map(|(a, y)| a * y).sum();
            assert!(ay >= c[j] - 1e-8, "reduced cost of column {j}");
        }
        let dual_objective: f64 = g.iter().zip(&duals).map(|(g, y)| g * y).sum();
        assert!(
            (dual_objective - objective).abs() < 1e-7 * (1.0 + objective.abs()),
            "primal {objective} vs dual {dual_objective}"
        );
    }

    /// Solves the `m x m` system `B x = g` by Gaussian elimination, or `None`
    /// when `B` is singular.
    fn solve_square(b: &[Vec<f64>], g: &[f64]) -> Option<Vec<f64>> {
        let m = g.len();
        let mut aug: Vec<Vec<f64>> = (0..m)
            .map(|i| {
                let mut row: Vec<f64> = b.iter().map(|column| column[i]).collect();
                row.push(g[i]);
                row
            })
            .collect();
        for k in 0..m {
            let p = (k..m).max_by(|&x, &y| aug[x][k].abs().total_cmp(&aug[y][k].abs()))?;
            if aug[p][k].abs() < 1e-12 {
                return None;
            }
            aug.swap(k, p);
            for i in 0..m {
                if i != k {
                    let f = aug[i][k] / aug[k][k];
                    let pivot_row = aug[k].clone();
                    for (target, v) in aug[i].iter_mut().zip(&pivot_row).skip(k) {
                        *target -= f * v;
                    }
                }
            }
        }
        Some((0..m).map(|i| aug[i][m] / aug[i][i]).collect())
    }

    /// The best objective over all basic feasible solutions, or `None` when
    /// there is none.
    fn brute_force(columns: &[Vec<f64>], g: &[f64], c: &[f64]) -> Option<f64> {
        let (m, n) = (g.len(), columns.len());
        let mut best: Option<f64> = None;
        for mask in 0u32..(1 << n) {
            if mask.count_ones() as usize != m {
                continue;
            }
            let chosen: Vec<usize> = (0..n).filter(|j| mask >> j & 1 == 1).collect();
            let basis: Vec<Vec<f64>> = chosen.iter().map(|&j| columns[j].clone()).collect();
            let Some(x) = solve_square(&basis, g) else {
                continue;
            };
            if x.iter().any(|v| *v < -1e-10) {
                continue;
            }
            let value: f64 = chosen.iter().zip(&x).map(|(&j, v)| c[j] * v).sum();
            best = Some(best.map_or(value, |b| b.max(value)));
        }
        best
    }

    #[test]
    fn random_feasible_bounded_programs_are_optimal() {
        let mut rng = Rng(20_260_904);
        for trial in 0..300 {
            let m = rng.range(2, 9);
            let n = rng.range(m, m + 24);
            let (columns, g, c) = random_feasible_problem(&mut rng, m, n);
            let mut lp = DenseSimplex::new(&columns, &g);
            lp.set_cost(&c);
            lp.solve().unwrap_or_else(|e| panic!("trial {trial}: {e}"));
            assert_optimal(&mut lp, &columns, &g, &c);
        }
    }

    #[test]
    fn small_programs_match_enumeration_of_basic_solutions() {
        let mut rng = Rng(31);
        for trial in 0..300 {
            let m = rng.range(1, 4);
            let n = rng.range(m, 8);
            let (columns, g, c) = random_feasible_problem(&mut rng, m, n);
            let reference = brute_force(&columns, &g, &c);
            let mut lp = DenseSimplex::new(&columns, &g);
            lp.set_cost(&c);
            match (lp.solve(), reference) {
                (Ok(()), Some(reference)) => assert!(
                    (lp.objective() - reference).abs() < 1e-8,
                    "trial {trial}: simplex {} vs enumeration {reference}",
                    lp.objective()
                ),
                (result, reference) => {
                    panic!("trial {trial}: simplex {result:?} vs enumeration {reference:?}")
                }
            }
        }
    }

    #[test]
    fn warm_start_after_a_cost_change_matches_a_cold_solve() {
        let mut rng = Rng(7);
        for _ in 0..100 {
            let m = rng.range(2, 8);
            let n = rng.range(m + 2, m + 20);
            let (columns, g, c1) = random_feasible_problem(&mut rng, m, n);
            let c2 = bounded_cost(&mut rng, &columns, m);
            let mut warm = DenseSimplex::new(&columns, &g);
            warm.set_cost(&c1);
            warm.solve().unwrap();
            warm.set_cost(&c2);
            warm.solve().unwrap();
            assert_optimal(&mut warm, &columns, &g, &c2);
            let mut cold = DenseSimplex::new(&columns, &g);
            cold.set_cost(&c2);
            cold.solve().unwrap();
            assert!((warm.objective() - cold.objective()).abs() < 1e-9);
        }
    }

    #[test]
    fn remembered_bases_give_the_same_optimum() {
        let mut rng = Rng(11);
        for _ in 0..50 {
            let m = rng.range(2, 8);
            let n = rng.range(m + 2, m + 20);
            let (columns, g, _) = random_feasible_problem(&mut rng, m, n);
            let mut with_memory = DenseSimplex::new(&columns, &g);
            with_memory.set_memory(4);
            let mut without = DenseSimplex::new(&columns, &g);
            for _ in 0..20 {
                let c = bounded_cost(&mut rng, &columns, m);
                with_memory.set_cost(&c);
                with_memory.solve().unwrap();
                without.set_cost(&c);
                without.solve().unwrap();
                assert!((with_memory.objective() - without.objective()).abs() < 1e-9);
                assert_optimal(&mut with_memory, &columns, &g, &c);
            }
        }
    }

    #[test]
    fn reports_infeasibility() {
        // Columns all in the positive orthant cannot reach a negative g.
        let columns = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let mut lp = DenseSimplex::new(&columns, &[-1.0, 1.0]);
        lp.set_cost(&[1.0, 1.0, 1.0]);
        assert_eq!(lp.solve(), Err(LpError::Infeasible));
    }

    #[test]
    fn reports_unboundedness() {
        // λ_1 - λ_2 = 1 with both free to grow, and a cost that rewards it.
        let columns = vec![vec![1.0], vec![-1.0]];
        let mut lp = DenseSimplex::new(&columns, &[1.0]);
        lp.set_cost(&[1.0, 1.0]);
        assert_eq!(lp.solve(), Err(LpError::Unbounded));
    }

    #[test]
    fn error_displays_and_is_an_error() {
        let error: Box<dyn std::error::Error> = Box::new(LpError::Infeasible);
        assert_eq!(error.to_string(), "Infeasible");
    }

    #[test]
    fn solves_a_known_program() {
        let columns = vec![
            vec![1.0, 1.0],
            vec![1.0, 3.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let mut lp = DenseSimplex::new(&columns, &[4.0, 6.0]);
        lp.set_cost(&[3.0, 2.0, 0.0, 0.0]);
        lp.solve().unwrap();
        assert!((lp.objective() - 12.0).abs() < 1e-9);
        assert!((lp.lambda()[0] - 4.0).abs() < 1e-9);
        assert_eq!(lp.num_columns(), 4);
    }

    #[test]
    fn survives_a_degenerate_start_and_many_pivots() {
        // g = e_0 makes every artificial but one start at zero, so phase one is
        // degenerate from the first pivot.
        let mut rng = Rng(99);
        let mut feasible_programs = 0;
        for _ in 0..50 {
            let m = 20;
            let n = 40;
            // The first column is e_0, which keeps the program feasible.
            let mut columns: Vec<Vec<f64>> = (0..n)
                .map(|_| {
                    let mut column: Vec<f64> = (0..m).map(|_| rng.uniform(-1.0, 1.0)).collect();
                    column[0] = rng.uniform(0.1, 1.0);
                    column
                })
                .collect();
            columns[0] = (0..m).map(|i| f64::from(u8::from(i == 0))).collect();
            let mut g = vec![0.0; m];
            g[0] = 1.0;
            // Random costs can be unbounded, which is accepted here.
            let mut lp = DenseSimplex::new(&columns, &g);
            let mut solved = false;
            for _ in 0..100 {
                let c: Vec<f64> = (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect();
                lp.set_cost(&c);
                match lp.solve() {
                    Ok(()) => {
                        solved = true;
                        assert_optimal(&mut lp, &columns, &g, &c);
                    }
                    Err(LpError::Unbounded) => {}
                    // Feasibility does not depend on the costs.
                    Err(LpError::Infeasible) => break,
                    Err(error) => panic!("unexpected {error}"),
                }
            }
            feasible_programs += usize::from(solved);
        }
        assert!(feasible_programs > 0);
    }

    #[test]
    fn handles_a_degenerate_vertex() {
        // Two rows of g are zero and the columns are repeated, so many bases
        // describe the same vertex.
        let columns = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 1.0, 0.0],
            vec![1.0, -1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 0.0, -1.0],
        ];
        let mut lp = DenseSimplex::new(&columns, &[1.0, 0.0, 0.0]);
        let c = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0];
        lp.set_cost(&c);
        lp.solve().unwrap();
        assert!((lp.objective() - 0.5).abs() < 1e-9);
        assert_optimal(&mut lp, &columns, &[1.0, 0.0, 0.0], &c);
    }
}

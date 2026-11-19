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
//! The basis inverse is held explicitly and updated by the pivot's elementary
//! row operations, with a fresh Gauss-Jordan factorisation every 40 pivots.
//! Pricing is Dantzig's rule, with a switch to Bland's rule after a run of
//! degenerate pivots, which cannot cycle.
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

use std::fmt;

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
        }
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
        for j in 0..self.n {
            if self.is_basic[j] {
                continue;
            }
            let d0 = c0[j] - dot(self.column(j), y0);
            let d1 = c1[j] - dot(self.column(j), y1);
            // d0 + t d1 <= TOL
            if d1 > 0.0 {
                hi = hi.min((TOL_REDUCED_COST - d0) / d1);
            } else if d1 < 0.0 {
                lo = lo.max((TOL_REDUCED_COST - d0) / d1);
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
        if infeasibility > TOL_PHASE_ONE {
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
            let mut best = TOL_REDUCED_COST;
            for j in 0..n {
                if self.is_basic[j] {
                    continue;
                }
                let d = self.cost_of(j, phase) - dot(self.column(j), &self.y);
                if bland {
                    if d > TOL_REDUCED_COST {
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
                            ratio < theta - TOL_DEGENERATE
                                || ((ratio - theta).abs() <= TOL_DEGENERATE
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
            if theta <= TOL_DEGENERATE {
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
    let chunks_a = a.chunks_exact(4);
    let chunks_b = b.chunks_exact(4);
    let (rest_a, rest_b) = (chunks_a.remainder(), chunks_b.remainder());
    for (ca, cb) in chunks_a.zip(chunks_b) {
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

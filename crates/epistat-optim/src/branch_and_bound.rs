//! Branch and bound for linear programs with binary columns.
//!
//! Solves
//!
//! ```text
//! max  c'x    subject to    r_lo ≤ A x ≤ r_hi,    l ≤ x ≤ u,
//!                           x_j ∈ {0, 1} for j in a given set
//! ```
//!
//! where the other columns are continuous. Every column must have finite bounds
//! when the starting basis is built, see [`crate::bounded_simplex`]. This is the
//! shape of cardinality and profile matching: one binary column per unit, and a
//! few continuous columns that carry the size of the matched sample.
//!
//! # Algorithm
//!
//! Branch and bound of Land and Doig \[1\] on the bounded dual simplex method of
//! [`crate::bounded_simplex`].
//!
//! 1. The relaxation, with the binary columns in `[0, 1]`, is solved. Its value
//!    is an upper bound on the value of every integer solution in the node.
//! 2. If no binary column is fractional, the rounded point is an integer
//!    solution. It is checked against the rows, and it replaces the incumbent
//!    when it is better.
//! 3. Otherwise a fractional column is chosen: a general integer column before
//!    a binary one, then the one furthest from a whole value, the lowest index
//!    among ties. Two children bound it above by `floor(v)` and below by
//!    `ceil(v)`. The child on the side the value is nearer to (up on a tie) is
//!    solved at once and the other is kept.
//! 4. After a node is closed, by bound, by infeasibility or as an integer
//!    solution, the search takes the most recently kept node, for 50 times in a
//!    row, and then the kept node with the best bound, the latest of equals.
//!    The first finds integer solutions by diving, and the second keeps the
//!    search from staying in a region that has none. A node whose bound cannot
//!    beat the incumbent is dropped without branching, and a kept node is
//!    dropped, unsolved, when its parent's bound cannot. Each solve starts from
//!    the basis of the node it branches from: the child solved at once
//!    continues from the parent's final basis, and a kept node reloads the basis
//!    saved when it was put away, so a node costs a few dual pivots.
//!
//! When every binary column has an integer cost and every continuous column
//! has cost zero, the value of any integer solution is an integer, so a node is
//! dropped unless `floor(bound) > incumbent`. This is detected from the data.
//!
//! # Rounding dive and neighbourhood search
//!
//! Before the search starts, the root relaxation is rounded by diving: all its
//! fractional columns are fixed to their nearest whole values at once, the
//! relaxation is solved again, and this repeats until it is whole. When fixing
//! all of them is infeasible, the most fractional one is fixed alone, and the
//! dive stops when that fails too. A whole point is an integer solution.
//!
//! In a matching problem many units are interchangeable, and the search
//! alone can stay a few units below the optimum for a long time. After the root
//! relaxation, the binary columns are ordered by the size of their reduced cost,
//! the basic ones, which include every fractional one, first. The first 64
//! stay free and the others are fixed at the root values rounded, and the
//! smaller program is solved by this search within 2000 relaxations. Then 256,
//! 1024, 4096 and 16384 free columns are tried. Each solution is a solution
//! of the whole program, so it can only raise the incumbent, and the proof of
//! optimality, by bound, is unchanged. [`Options::neighbourhood_search`] turns it
//! off.
//!
//! # Limits and the gap
//!
//! The search ends when no node is left, or when the node limit or the time
//! limit is reached. Then [`Solution::status`] says which, and
//! [`Solution::bound`] is the best upper bound proved: the largest bound among
//! the nodes not yet closed, and never below the incumbent's value. The gap
//! `(bound - objective) / max(1, |objective|)` is [`Solution::gap`]. It is zero
//! for a proven optimum, and a solution that stops at a limit says so and is
//! not presented as optimal. The time limit makes the result depend on the
//! machine and the load, and the node limit does not.
//!
//! # Determinism
//!
//! Branching and node order depend only on the data, and the simplex is
//! independent of the number of threads, so the solution is the same on any
//! number of threads, unless the time limit ends the search.
//!
//! # Tolerances
//!
//! A basic binary column is called integral when it is within `1e-8` of 0 or 1,
//! ten times the tolerance the simplex uses for the bounds of a column. The
//! rounded point is checked against every row with the row's own tolerance plus
//! the movement the rounding caused, `Σ_j |a_ij| |x_j - round(x_j)|`. An
//! objective bound is compared with the incumbent to `1e-9 max(1, |z|)`.
//!
//! # Example
//!
//! ```
//! use epistat_optim::branch_and_bound::{MixedBinaryProgram, Options, Status, solve};
//!
//! // max 5 x0 + 4 x1 + 3 x2  s.t.  2 x0 + 3 x1 + x2 <= 5
//! let program = MixedBinaryProgram {
//!     rows: 1,
//!     columns: &[2.0, 3.0, 1.0],
//!     cost: &[5.0, 4.0, 3.0],
//!     col_lo: &[0.0; 3],
//!     col_hi: &[1.0; 3],
//!     row_lo: &[f64::NEG_INFINITY],
//!     row_hi: &[5.0],
//!     binary: &[true; 3],
//!     integer: &[],
//! };
//! let solution = solve(&program, &Options::default()).unwrap();
//! assert_eq!(solution.status, Status::Optimal);
//! assert!((solution.objective - 9.0).abs() < 1e-9);
//! ```
//!
//! # References
//!
//! 1. Land AH, Doig AG. An automatic method of solving discrete programming
//!    problems. Econometrica. 1960;28(3):497-520.

#![expect(
    clippy::float_cmp,
    reason = "bounds and costs are compared as given: a fixed column has equal \
              bounds, an integer cost equals its rounding, and ties are exact"
)]

use std::fmt;
use std::time::{Duration, Instant};

use crate::bounded_simplex::{BoundedLpError, BoundedSimplex, Snapshot};

/// Backtracks to the latest open node before the one with the best bound is taken.
const PLUNGE: usize = 50;

/// A basic binary column within this of 0 or 1 is integral.
const INT_TOL: f64 = 1e-8;
const OBJECTIVE_TOL: f64 = 1e-9;

/// The program, see the [module](self) documentation.
#[derive(Debug, Clone, Copy)]
pub struct MixedBinaryProgram<'a> {
    /// Number of rows `m`.
    pub rows: usize,
    /// `A`, column by column: `n * m` values.
    pub columns: &'a [f64],
    /// `c`, one per column.
    pub cost: &'a [f64],
    /// `l`.
    pub col_lo: &'a [f64],
    /// `u`.
    pub col_hi: &'a [f64],
    /// `r_lo`, one per row.
    pub row_lo: &'a [f64],
    /// `r_hi`, one per row.
    pub row_hi: &'a [f64],
    /// Which columns must be 0 or 1. Their bounds must be within `[0, 1]`.
    pub binary: &'a [bool],
    /// Further columns that must take whole values, within their bounds, which
    /// are then finite. Empty for none, otherwise one entry per column. Branching
    /// goes to these before the binary columns.
    pub integer: &'a [bool],
}

/// Limits on the search, and whether to run the root heuristic. `None` is no
/// limit.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// The most relaxations to solve, the root included.
    pub node_limit: Option<u64>,
    /// The most time to spend. The result then depends on the machine.
    pub time_limit: Option<Duration>,
    /// Run the neighbourhood search at the root, see the [module](self)
    /// documentation. It only ever improves the incumbent, so it changes how
    /// fast the search ends and not the optimum it proves.
    pub neighbourhood_search: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            node_limit: None,
            time_limit: None,
            neighbourhood_search: true,
        }
    }
}

/// How the search ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// The incumbent is optimal: no open node could beat it.
    Optimal,
    /// The node limit ended the search. The solution is the best found, not
    /// proved optimal.
    NodeLimit,
    /// The time limit ended the search. The solution is the best found, not
    /// proved optimal.
    TimeLimit,
}

/// The result of [`solve`].
#[derive(Debug, Clone)]
pub struct Solution {
    /// The values of the columns, the binary ones exactly 0 or 1.
    pub x: Vec<f64>,
    /// `c'x`.
    pub objective: f64,
    /// An upper bound on the optimum, proved by the search.
    pub bound: f64,
    /// How the search ended.
    pub status: Status,
    /// Relaxations solved.
    pub nodes: u64,
    /// Simplex pivots over all of them.
    pub pivots: u64,
    /// The largest amount by which a row's activity is outside its bounds at
    /// `x`, zero when every row holds.
    pub max_row_violation: f64,
}

impl Solution {
    /// `(bound - objective) / max(1, |objective|)`, zero for a proved optimum.
    #[must_use]
    pub fn gap(&self) -> f64 {
        ((self.bound - self.objective) / self.objective.abs().max(1.0)).max(0.0)
    }
}

/// Why no solution is returned.
#[derive(Debug, Clone, PartialEq)]
pub enum MipError {
    /// The input is malformed, with the reason.
    InvalidInput(String),
    /// The relaxation has no feasible point, so the program has none.
    Infeasible,
    /// A limit was reached before any integer solution was found. `bound` is
    /// the best upper bound proved.
    NoSolution {
        /// How the search ended.
        status: Status,
        /// The best upper bound proved.
        bound: f64,
    },
    /// The relaxation failed for a reason other than infeasibility.
    Lp(BoundedLpError),
    /// An integer solution broke a row by more than rounding explains.
    Numerical,
}

impl fmt::Display for MipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(why) => write!(f, "InvalidInput: {why}"),
            Self::Infeasible => f.write_str("Infeasible"),
            Self::NoSolution { status, bound } => {
                write!(f, "NoSolution after {status:?}, bound {bound}")
            }
            Self::Lp(e) => write!(f, "the relaxation failed: {e}"),
            Self::Numerical => f.write_str("Numerical"),
        }
    }
}

impl std::error::Error for MipError {}

struct Node {
    var: usize,
    /// The bounds of the sibling to solve.
    lo: f64,
    hi: f64,
    parent_bound: f64,
    /// The bounds fixed by the branches above it, in order.
    path: Vec<(usize, f64, f64)>,
    snapshot: Snapshot,
}

/// Solves the program by branch and bound.
///
/// # Errors
/// See [`MipError`].
pub fn solve(program: &MixedBinaryProgram<'_>, options: &Options) -> Result<Solution, MipError> {
    solve_with_deadline(
        program,
        options,
        options.time_limit.map(|d| Instant::now() + d),
    )
}

fn solve_with_deadline(
    program: &MixedBinaryProgram<'_>,
    options: &Options,
    deadline: Option<Instant>,
) -> Result<Solution, MipError> {
    let n = program.cost.len();
    if program.binary.len() != n {
        return Err(MipError::InvalidInput(
            "binary needs one entry per column".to_string(),
        ));
    }
    if !(program.integer.is_empty() || program.integer.len() == n) {
        return Err(MipError::InvalidInput(
            "integer needs no entries or one per column".to_string(),
        ));
    }
    let general = |j: usize| program.integer.get(j).copied().unwrap_or(false) && !program.binary[j];
    for j in 0..n {
        if general(j) && !(program.col_lo[j].is_finite() && program.col_hi[j].is_finite()) {
            return Err(MipError::InvalidInput(
                "an integer column needs finite bounds".to_string(),
            ));
        }
        if program.binary[j]
            && !((program.col_lo[j] == 0.0 || program.col_lo[j] == 1.0)
                && (program.col_hi[j] == 0.0 || program.col_hi[j] == 1.0)
                && program.col_lo[j] <= program.col_hi[j])
        {
            return Err(MipError::InvalidInput(
                "a binary column needs bounds in {0, 1}".to_string(),
            ));
        }
    }
    let mut lp = BoundedSimplex::new(
        program.rows,
        program.columns,
        program.cost,
        program.col_lo,
        program.col_hi,
        program.row_lo,
        program.row_hi,
    )
    .map_err(|e| match e {
        BoundedLpError::InvalidInput(why) => MipError::InvalidInput(why),
        other => MipError::Lp(other),
    })?;

    lp.perturb_costs();
    let integral_objective = (0..n).all(|j| {
        if program.binary[j] || general(j) {
            program.cost[j] == program.cost[j].round()
        } else {
            program.cost[j] == 0.0
        }
    });

    let mut nodes: u64 = 0;
    let mut incumbent: Option<(Vec<f64>, f64)> = None;
    let mut path: Vec<(usize, f64, f64)> = Vec::new();
    let mut local_backtracks = 0_usize;
    let mut stack: Vec<Node> = Vec::new();
    let mut status = Status::Optimal;
    let root_bound;
    // The bound of the node being processed when a limit stops the search.
    let mut open_bound = f64::NEG_INFINITY;

    let tighten = |z: f64| {
        if integral_objective {
            (z + OBJECTIVE_TOL * z.abs().max(1.0)).floor()
        } else {
            z
        }
    };
    let beats = |bound: f64, incumbent: &Option<(Vec<f64>, f64)>| match incumbent {
        None => true,
        Some((_, value)) => {
            if integral_objective {
                tighten(bound) > *value + 0.5
            } else {
                bound > *value + OBJECTIVE_TOL * value.abs().max(1.0)
            }
        }
    };

    // The solved node's value, or `None` when it is infeasible.
    let mut solved: Option<f64>;
    match lp.solve(deadline) {
        Ok(()) => {
            nodes += 1;
            solved = Some(lp.relaxation_bound());
            root_bound = lp.relaxation_bound();
        }
        Err(BoundedLpError::Infeasible) => return Err(MipError::Infeasible),
        Err(BoundedLpError::TimeLimit) => {
            return Err(MipError::NoSolution {
                status: Status::TimeLimit,
                bound: f64::INFINITY,
            });
        }
        Err(e) => return Err(MipError::Lp(e)),
    }

    if options.neighbourhood_search && solved.is_some() {
        rounding_dive(program, &mut lp, deadline, &mut incumbent, &mut nodes)?;
    }
    if let (true, Some(z)) = (options.neighbourhood_search, solved) {
        let target = tighten(z);
        neighbourhood_search(
            program,
            &lp,
            deadline,
            integral_objective.then_some(target),
            &mut incumbent,
            &mut nodes,
        );
        if integral_objective && incumbent.as_ref().is_some_and(|(_, v)| *v >= target - 0.5) {
            // Nothing better than the incumbent can exist.
            let (x, objective) = incumbent.expect("checked above");
            let max_row_violation = row_violation(program, &x);
            return Ok(Solution {
                x,
                objective,
                bound: objective,
                status: Status::Optimal,
                nodes,
                pivots: lp.iterations(),
                max_row_violation,
            });
        }
    }

    'search: loop {
        if let Some(z) = solved {
            let limit_hit = options
                .node_limit
                .is_some_and(|l| nodes >= l)
                .then_some(Status::NodeLimit)
                .or_else(|| {
                    deadline
                        .is_some_and(|d| Instant::now() >= d)
                        .then_some(Status::TimeLimit)
                });
            if beats(z, &incumbent) {
                // Fractional whole columns among the basic ones: the general
                // integer columns first, then the furthest from a whole value,
                // then the lowest index.
                let mut branch: Option<(usize, f64, (bool, f64))> = None;
                for (j, v) in lp.basic_columns() {
                    let is_general = general(j);
                    if !(program.binary[j] || is_general) {
                        continue;
                    }
                    let (lo, hi) = lp.bounds(j);
                    if lo == hi {
                        continue;
                    }
                    let dist = (v - v.round()).abs();
                    if dist > INT_TOL * v.abs().max(1.0) {
                        let key = (is_general, dist);
                        let better = match branch {
                            None => true,
                            Some((bj, _, bk)) => {
                                (key.0 && !bk.0)
                                    || (key.0 == bk.0
                                        && (key.1 > bk.1 || (key.1 == bk.1 && j < bj)))
                            }
                        };
                        if better {
                            branch = Some((j, v, key));
                        }
                    }
                }
                match branch {
                    None => {
                        let (x, objective) = rounded_point(&lp, program)?;
                        if incumbent.as_ref().is_none_or(|(_, v)| objective > *v) {
                            incumbent = Some((x, objective));
                        }
                    }
                    Some((j, v, _)) => {
                        if let Some(hit) = limit_hit {
                            status = hit;
                            open_bound = open_bound.max(z);
                            break 'search;
                        }
                        let (lo, hi) = lp.bounds(j);
                        // Down: `x <= floor(v)`. Up: `x >= ceil(v)`. The side
                        // the value is nearer to, up on a tie, is solved first.
                        let down = (lo, v.floor());
                        let up = (v.ceil(), hi);
                        let (near, far) = if v - v.floor() >= 0.5 {
                            (up, down)
                        } else {
                            (down, up)
                        };
                        stack.push(Node {
                            var: j,
                            lo: far.0,
                            hi: far.1,
                            parent_bound: z,
                            path: path.clone(),
                            snapshot: lp.snapshot(),
                        });
                        path.push((j, near.0, near.1));
                        lp.set_bounds(j, near.0, near.1).map_err(MipError::Lp)?;
                        match lp.solve(deadline) {
                            Ok(()) => {
                                nodes += 1;
                                solved = Some(lp.relaxation_bound());
                                continue 'search;
                            }
                            Err(BoundedLpError::Infeasible) => {}
                            Err(BoundedLpError::TimeLimit) => {
                                status = Status::TimeLimit;
                                open_bound = open_bound.max(z);
                                // The node just branched on is not closed.
                                stack.pop();
                                break 'search;
                            }
                            Err(e) => return Err(MipError::Lp(e)),
                        }
                    }
                }
            }
        }
        // Backtrack to the open node with the best bound, the latest of equals.
        loop {
            stack.retain(|node| beats(node.parent_bound, &incumbent));
            // The latest node, for `PLUNGE` backtracks in a row, then the best.
            let mut pick: Option<usize> = None;
            if local_backtracks < PLUNGE {
                local_backtracks += 1;
                pick = stack.len().checked_sub(1);
            } else {
                local_backtracks = 0;
                for (i, node) in stack.iter().enumerate() {
                    if pick.is_none_or(|p| node.parent_bound >= stack[p].parent_bound) {
                        pick = Some(i);
                    }
                }
            }
            let Some(pick) = pick else {
                break 'search;
            };
            let node = stack.remove(pick);
            if options.node_limit.is_some_and(|l| nodes >= l) {
                status = Status::NodeLimit;
                open_bound = open_bound.max(node.parent_bound);
                stack.push(node);
                break 'search;
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                status = Status::TimeLimit;
                open_bound = open_bound.max(node.parent_bound);
                stack.push(node);
                break 'search;
            }
            for &(j, _, _) in &path {
                lp.set_bounds(j, program.col_lo[j], program.col_hi[j])
                    .map_err(MipError::Lp)?;
            }
            for &(j, lo, hi) in &node.path {
                lp.set_bounds(j, lo, hi).map_err(MipError::Lp)?;
            }
            lp.restore(&node.snapshot);
            lp.set_bounds(node.var, node.lo, node.hi)
                .map_err(MipError::Lp)?;
            path.clone_from(&node.path);
            path.push((node.var, node.lo, node.hi));
            match lp.solve(deadline) {
                Ok(()) => {
                    nodes += 1;
                    solved = Some(lp.relaxation_bound());
                    continue 'search;
                }
                Err(BoundedLpError::Infeasible) => {}
                Err(BoundedLpError::TimeLimit) => {
                    status = Status::TimeLimit;
                    open_bound = open_bound.max(node.parent_bound);
                    break 'search;
                }
                Err(e) => return Err(MipError::Lp(e)),
            }
        }
    }

    // The best bound proved: the open nodes' bounds, never below the incumbent.
    let mut bound = open_bound;
    for node in &stack {
        bound = bound.max(node.parent_bound);
    }
    let Some((x, objective)) = incumbent else {
        if status == Status::Optimal {
            return Err(MipError::Infeasible);
        }
        return Err(MipError::NoSolution {
            status,
            bound: tighten(bound.min(root_bound)),
        });
    };
    let bound = if status == Status::Optimal {
        objective
    } else {
        tighten(bound.min(root_bound)).max(objective)
    };
    let max_row_violation = row_violation(program, &x);
    Ok(Solution {
        x,
        objective,
        bound,
        status,
        nodes,
        pivots: lp.iterations(),
        max_row_violation,
    })
}

/// The most relaxations a rounding dive may solve.
const DIVE_LIMIT: u64 = 3000;

/// Fractional whole columns among the basic ones, general integer columns
/// first and then by distance from a whole value, furthest first, ties to the
/// lowest index.
fn fractional_columns(lp: &BoundedSimplex, program: &MixedBinaryProgram<'_>) -> Vec<(usize, f64)> {
    let mut out: Vec<(usize, f64, bool, f64)> = Vec::new();
    for (j, v) in lp.basic_columns() {
        let general = program.integer.get(j).copied().unwrap_or(false) && !program.binary[j];
        if !(program.binary[j] || general) {
            continue;
        }
        let (lo, hi) = lp.bounds(j);
        let dist = (v - v.round()).abs();
        if lo != hi && dist > INT_TOL * v.abs().max(1.0) {
            out.push((j, v, general, dist));
        }
    }
    out.sort_by(|a, b| b.2.cmp(&a.2).then(b.3.total_cmp(&a.3)).then(a.0.cmp(&b.0)));
    out.into_iter().map(|(j, v, _, _)| (j, v)).collect()
}

/// Rounds the root relaxation to an integer solution by diving.
///
/// All the fractional columns of the relaxation are fixed to their nearest whole
/// value at once and the relaxation is solved again, and this repeats until it is
/// whole. When fixing them all makes the relaxation infeasible, only the most
/// fractional is fixed, to its nearest value and else to the other. The dive
/// gives up when that fails too or after [`DIVE_LIMIT`] relaxations. The bounds
/// and the basis are put back afterwards. A whole point is a solution of the
/// program and can only raise the incumbent.
fn rounding_dive(
    program: &MixedBinaryProgram<'_>,
    lp: &mut BoundedSimplex,
    deadline: Option<Instant>,
    incumbent: &mut Option<(Vec<f64>, f64)>,
    nodes: &mut u64,
) -> Result<(), MipError> {
    let snapshot = lp.snapshot();
    let mut changed: Vec<usize> = Vec::new();
    let mut solves = 0_u64;
    'dive: while solves < DIVE_LIMIT {
        let fractional = fractional_columns(lp, program);
        if fractional.is_empty() {
            let (x, objective) = rounded_point(lp, program)?;
            if incumbent.as_ref().is_none_or(|(_, v)| objective > *v) {
                *incumbent = Some((x, objective));
            }
            break;
        }
        let try_fix = |lp: &mut BoundedSimplex, fixes: &[(usize, f64)]| -> Result<bool, MipError> {
            for &(j, value) in fixes {
                lp.set_bounds(j, value, value).map_err(MipError::Lp)?;
            }
            match lp.solve(deadline) {
                Ok(()) => Ok(true),
                Err(BoundedLpError::Infeasible) => Ok(false),
                Err(BoundedLpError::TimeLimit) => Err(MipError::NoSolution {
                    status: Status::TimeLimit,
                    bound: f64::INFINITY,
                }),
                Err(e) => Err(MipError::Lp(e)),
            }
        };
        let restore = |lp: &mut BoundedSimplex, fixes: &[(usize, f64)]| -> Result<(), MipError> {
            for &(j, _) in fixes {
                lp.set_bounds(j, program.col_lo[j], program.col_hi[j])
                    .map_err(MipError::Lp)?;
            }
            Ok(())
        };
        let all: Vec<(usize, f64)> = fractional.iter().map(|&(j, v)| (j, v.round())).collect();
        solves += 1;
        let before = lp.snapshot();
        let ok = match try_fix(lp, &all) {
            Err(MipError::NoSolution { .. }) => break 'dive,
            other => other?,
        };
        if ok {
            changed.extend(all.iter().map(|&(j, _)| j));
            continue;
        }
        restore(lp, &all)?;
        lp.restore(&before);
        let (j, v) = fractional[0];
        for value in [v.round(), if v.round() > v { v.floor() } else { v.ceil() }] {
            solves += 1;
            let single = [(j, value)];
            let r = try_fix(lp, &single);
            if matches!(r, Ok(true)) {
                changed.push(j);
                continue 'dive;
            }
            restore(lp, &single)?;
            lp.restore(&before);
        }
        break;
    }
    *nodes += solves;
    for j in changed {
        lp.set_bounds(j, program.col_lo[j], program.col_hi[j])
            .map_err(MipError::Lp)?;
    }
    lp.restore(&snapshot);
    match lp.solve(deadline) {
        Ok(()) | Err(BoundedLpError::TimeLimit) => Ok(()),
        Err(e) => Err(MipError::Lp(e)),
    }
}

/// Sizes of the free sets of the neighbourhood search, in binary columns.
const NEIGHBOURHOODS: [usize; 5] = [64, 256, 1024, 4096, 16384];

/// The relaxations a neighbourhood's own search may solve.
const NEIGHBOURHOOD_NODES: u64 = 2000;

/// Looks for integer solutions near the root relaxation.
///
/// The binary columns are ordered by the size of their reduced cost at the
/// root optimum, which is zero for the basic ones, and the `f` first are kept
/// free while the rest are fixed to their rounded relaxation values. The
/// program on the free columns, with the fixed ones moved to the row bounds, is
/// solved by the same search without this step, within a node limit and a
/// quarter of the time left. The sizes `f` grow through
/// [`NEIGHBOURHOODS`]. The search ends when a solution reaches `target`.
fn neighbourhood_search(
    program: &MixedBinaryProgram<'_>,
    lp: &BoundedSimplex,
    deadline: Option<Instant>,
    target: Option<f64>,
    incumbent: &mut Option<(Vec<f64>, f64)>,
    nodes: &mut u64,
) {
    let n = program.cost.len();
    let m = program.rows;
    let x = lp.values();
    let d = lp.reduced_costs();
    let mut order: Vec<usize> = (0..n)
        .filter(|&j| program.binary[j] && program.col_lo[j] != program.col_hi[j])
        .collect();
    order.sort_by(|&a, &b| d[a].abs().total_cmp(&d[b].abs()).then(a.cmp(&b)));
    for &size in &NEIGHBOURHOODS {
        if size >= order.len() {
            break;
        }
        if deadline.is_some_and(|dl| Instant::now() >= dl) {
            return;
        }
        let mut free = vec![false; n];
        for &j in &order[..size] {
            free[j] = true;
        }
        for (f, &b) in free.iter_mut().zip(program.binary) {
            if !b {
                *f = true;
            }
        }
        let kept: Vec<usize> = (0..n).filter(|&j| free[j]).collect();
        let mut fixed_activity = vec![0.0; m];
        let mut fixed_value = vec![0.0; n];
        for j in (0..n).filter(|&j| !free[j]) {
            let v = if program.col_lo[j] == program.col_hi[j] {
                program.col_lo[j]
            } else {
                x[j].round()
            };
            fixed_value[j] = v;
            if v != 0.0 {
                let column = &program.columns[j * m..(j + 1) * m];
                for i in 0..m {
                    fixed_activity[i] += column[i] * v;
                }
            }
        }
        let mut columns = Vec::with_capacity(kept.len() * m);
        let mut integer = Vec::with_capacity(kept.len());
        let (mut cost, mut col_lo, mut col_hi, mut binary) = (
            Vec::with_capacity(kept.len()),
            Vec::with_capacity(kept.len()),
            Vec::with_capacity(kept.len()),
            Vec::with_capacity(kept.len()),
        );
        for &j in &kept {
            columns.extend_from_slice(&program.columns[j * m..(j + 1) * m]);
            cost.push(program.cost[j]);
            col_lo.push(program.col_lo[j]);
            col_hi.push(program.col_hi[j]);
            binary.push(program.binary[j]);
            integer.push(program.integer.get(j).copied().unwrap_or(false));
        }
        let row_lo: Vec<f64> = (0..m)
            .map(|i| program.row_lo[i] - fixed_activity[i])
            .collect();
        let row_hi: Vec<f64> = (0..m)
            .map(|i| program.row_hi[i] - fixed_activity[i])
            .collect();
        let sub = MixedBinaryProgram {
            rows: m,
            columns: &columns,
            cost: &cost,
            col_lo: &col_lo,
            col_hi: &col_hi,
            row_lo: &row_lo,
            row_hi: &row_hi,
            binary: &binary,
            integer: &integer,
        };
        let sub_deadline = deadline.map(|dl| {
            let now = Instant::now();
            now + dl.saturating_duration_since(now) / 4
        });
        let sub_options = Options {
            node_limit: Some(NEIGHBOURHOOD_NODES),
            time_limit: None,
            neighbourhood_search: false,
        };
        let Ok(solution) = solve_with_deadline(&sub, &sub_options, sub_deadline) else {
            continue;
        };
        *nodes += solution.nodes;
        let mut full = fixed_value;
        for (k, &j) in kept.iter().enumerate() {
            full[j] = solution.x[k];
        }
        let objective: f64 = full.iter().zip(program.cost).map(|(v, c)| v * c).sum();
        // The sub-problem's rows are the program's rows, so the point is
        // feasible for it to the sub-problem's tolerance, checked again here.
        let act = activity(program, &full);
        let ok = (0..m).all(|i| {
            let tol = lp.row_tolerance(i) * 10.0;
            act[i] >= program.row_lo[i] - tol && act[i] <= program.row_hi[i] + tol
        });
        if ok && incumbent.as_ref().is_none_or(|(_, v)| objective > *v) {
            *incumbent = Some((full, objective));
        }
        if let (Some(t), Some((_, v))) = (target, incumbent.as_ref()) {
            if *v >= t - 0.5 {
                return;
            }
        }
    }
}

/// The LP point with its binary columns rounded, checked against the rows.
fn rounded_point(
    lp: &BoundedSimplex,
    program: &MixedBinaryProgram<'_>,
) -> Result<(Vec<f64>, f64), MipError> {
    let mut x = lp.values();
    let mut moved = vec![0.0; program.rows];
    let m = program.rows;
    for (j, xj) in x.iter_mut().enumerate() {
        if program.binary[j] || program.integer.get(j).copied().unwrap_or(false) {
            let r = xj.round();
            let d = (*xj - r).abs();
            if d > 0.0 {
                let column = &program.columns[j * m..(j + 1) * m];
                for (mv, c) in moved.iter_mut().zip(column) {
                    *mv += c.abs() * d;
                }
            }
            *xj = r;
        }
    }
    let activity = activity(program, &x);
    for i in 0..m {
        let allowed = lp.row_tolerance(i) + moved[i];
        if activity[i] < program.row_lo[i] - allowed || activity[i] > program.row_hi[i] + allowed {
            return Err(MipError::Numerical);
        }
    }
    let objective: f64 = x.iter().zip(program.cost).map(|(v, c)| v * c).sum();
    Ok((x, objective))
}

fn activity(program: &MixedBinaryProgram<'_>, x: &[f64]) -> Vec<f64> {
    let m = program.rows;
    let mut out = vec![0.0; m];
    for (j, &v) in x.iter().enumerate() {
        if v != 0.0 {
            let column = &program.columns[j * m..(j + 1) * m];
            for i in 0..m {
                out[i] += column[i] * v;
            }
        }
    }
    out
}

fn row_violation(program: &MixedBinaryProgram<'_>, x: &[f64]) -> f64 {
    let act = activity(program, x);
    let mut worst = 0.0_f64;
    for (i, a) in act.iter().enumerate() {
        worst = worst.max(program.row_lo[i] - a).max(a - program.row_hi[i]);
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The best value over every 0/1 point, by enumeration, of
    /// `max c'x  s.t.  lo <= A x <= hi` with one continuous column `t` in
    /// `[0, t_hi]` of cost 0 as the last column. For a fixed 0/1 part the rows
    /// are an interval for `t`.
    fn brute_force(
        m: usize,
        columns: &[f64],
        cost: &[f64],
        row_lo: &[f64],
        row_hi: &[f64],
        t_hi: f64,
        nbin: usize,
    ) -> Option<f64> {
        let mut best: Option<f64> = None;
        for mask in 0..(1_u32 << nbin) {
            let mut lo_t = 0.0_f64;
            let mut hi_t = t_hi;
            let mut ok = true;
            for i in 0..m {
                let mut s = 0.0;
                for j in 0..nbin {
                    if mask >> j & 1 == 1 {
                        s += columns[j * m + i];
                    }
                }
                let b = columns[nbin * m + i];
                // row_lo <= s + b t <= row_hi
                if b == 0.0 {
                    if s < row_lo[i] - 1e-12 || s > row_hi[i] + 1e-12 {
                        ok = false;
                    }
                } else {
                    let (u, v) = ((row_lo[i] - s) / b, (row_hi[i] - s) / b);
                    let (u, v) = if b > 0.0 { (u, v) } else { (v, u) };
                    lo_t = lo_t.max(u);
                    hi_t = hi_t.min(v);
                }
            }
            if ok && lo_t <= hi_t + 1e-12 {
                let value: f64 = (0..nbin)
                    .filter(|j| mask >> j & 1 == 1)
                    .map(|j| cost[j])
                    .sum();
                if best.is_none_or(|b| value > b) {
                    best = Some(value);
                }
            }
        }
        best
    }

    #[test]
    fn small_knapsack() {
        let program = MixedBinaryProgram {
            rows: 1,
            columns: &[2.0, 3.0, 1.0],
            cost: &[5.0, 4.0, 3.0],
            col_lo: &[0.0; 3],
            col_hi: &[1.0; 3],
            row_lo: &[-INF],
            row_hi: &[5.0],
            binary: &[true; 3],
            integer: &[],
        };
        let s = solve(&program, &Options::default()).unwrap();
        assert_eq!(s.status, Status::Optimal);
        assert_eq!(s.objective, 9.0);
        assert_eq!(s.gap(), 0.0);
    }

    #[test]
    fn infeasible_program() {
        let program = MixedBinaryProgram {
            rows: 1,
            columns: &[1.0, 1.0],
            cost: &[1.0, 1.0],
            col_lo: &[0.0; 2],
            col_hi: &[1.0; 2],
            row_lo: &[1.5],
            row_hi: &[1.5],
            binary: &[true; 2],
            integer: &[],
        };
        assert!(matches!(
            solve(&program, &Options::default()),
            Err(MipError::Infeasible)
        ));
    }

    /// Random programs of the matching shape with one continuous column: equal
    /// counts of two groups tied to `t`, and balance rows with `t` in them.
    #[test]
    fn agrees_with_exhaustive_search() {
        let mut rng = Rng(2024);
        let mut solved = 0;
        for trial in 0..600 {
            let nbin = 6 + trial % 9;
            let p = 1 + trial % 3;
            // Rows: group-1 count - t = 0, group-2 count - t = 0, then for each
            // covariate two rows: sum(sign * x * v) -/+ tol t.
            let m = 2 + 2 * p;
            let n = nbin + 1;
            let group: Vec<bool> = (0..nbin).map(|j| j % 2 == 0).collect();
            let mut columns = vec![0.0; n * m];
            for j in 0..nbin {
                columns[j * m + usize::from(!group[j])] = 1.0;
            }
            columns[nbin * m] = -1.0;
            columns[nbin * m + 1] = -1.0;
            let tol = rng.uniform(0.2, 0.8);
            for c in 0..p {
                let vals: Vec<f64> = (0..nbin).map(|_| rng.uniform(-1.0, 1.0)).collect();
                for j in 0..nbin {
                    let v = if group[j] { vals[j] } else { -vals[j] };
                    columns[j * m + 2 + 2 * c] = v;
                    columns[j * m + 3 + 2 * c] = v;
                }
                // sum - tol t <= 0 and sum + tol t >= 0
                columns[nbin * m + 2 + 2 * c] = -tol;
                columns[nbin * m + 3 + 2 * c] = tol;
            }
            let mut row_lo = vec![0.0; m];
            let mut row_hi = vec![0.0; m];
            for c in 0..p {
                row_lo[2 + 2 * c] = -INF;
                row_hi[3 + 2 * c] = INF;
            }
            let mut cost = vec![1.0; n];
            cost[nbin] = 0.0;
            let col_lo = vec![0.0; n];
            let mut col_hi = vec![1.0; n];
            col_hi[nbin] = nbin as f64;
            let mut binary = vec![true; n];
            binary[nbin] = false;
            let program = MixedBinaryProgram {
                rows: m,
                columns: &columns,
                cost: &cost,
                col_lo: &col_lo,
                col_hi: &col_hi,
                row_lo: &row_lo,
                row_hi: &row_hi,
                binary: &binary,
                integer: &[],
            };
            let reference = brute_force(m, &columns, &cost, &row_lo, &row_hi, col_hi[nbin], nbin);
            let got = solve(&program, &Options::default());
            match (reference, got) {
                (Some(r), Ok(s)) => {
                    assert_eq!(s.status, Status::Optimal);
                    assert!((s.objective - r).abs() < 1e-9, "trial {trial}");
                    assert!(s.max_row_violation < 1e-7, "trial {trial}");
                    solved += 1;
                }
                (None, Err(MipError::Infeasible)) => {}
                (r, g) => panic!("trial {trial}: reference {r:?}, solver {g:?}"),
            }
        }
        assert!(solved > 200);
    }

    #[test]
    fn node_limit_reports_a_gap_and_is_not_optimal() {
        // A knapsack that needs branching.
        let n = 40;
        let mut rng = Rng(5);
        let w: Vec<f64> = (0..n).map(|_| rng.uniform(1.0, 10.0)).collect();
        let c: Vec<f64> = w.iter().map(|v| (v * 3.0).round()).collect();
        let program = MixedBinaryProgram {
            rows: 1,
            columns: &w,
            cost: &c,
            col_lo: &vec![0.0; n],
            col_hi: &vec![1.0; n],
            row_lo: &[-INF],
            row_hi: &[50.0],
            binary: &vec![true; n],
            integer: &[],
        };
        let full = solve(&program, &Options::default()).unwrap();
        assert_eq!(full.status, Status::Optimal);
        let cut = solve(
            &program,
            &Options {
                node_limit: Some(3),
                time_limit: None,
                ..Options::default()
            },
        );
        match cut {
            Ok(s) => {
                assert_eq!(s.status, Status::NodeLimit);
                assert!(s.bound >= full.objective - 1e-9);
                assert!(s.objective <= full.objective + 1e-9);
            }
            Err(MipError::NoSolution { status, bound }) => {
                assert_eq!(status, Status::NodeLimit);
                assert!(bound >= full.objective - 1e-9);
            }
            Err(e) => panic!("{e}"),
        }
    }
}

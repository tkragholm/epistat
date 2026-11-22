//! Branch and bound for linear programs with binary columns.
//!
//! Solves
//!
//! ```text
//! max  c'x    subject to    r_lo ≤ A x ≤ r_hi,    l ≤ x ≤ u,
//!                           x_j ∈ {0, 1} for j in a given set
//! ```
//!
//! where further columns may have to take whole values and the others are
//! continuous. Every column must have finite bounds when the starting basis is
//! built, see [`crate::bounded_simplex`]. This is the shape of cardinality and
//! profile matching: one binary column per unit, a dozen rows, and a few
//! columns that carry the size of the matched sample.
//!
//! # Algorithm
//!
//! Branch and bound of Land and Doig \[1\] on the bounded dual simplex method of
//! [`crate::bounded_simplex`], with the standard aids of an integer programming
//! solver as Wolsey \[2\] and Achterberg \[3\] describe them, fitted to programs
//! with few rows and many columns.
//!
//! 1. **Root.** The relaxation, the binary columns in `[0, 1]`, is solved with
//!    perturbed costs ([`BoundedSimplex::perturb_costs`]), then again from that
//!    basis with a hundredth of the perturbation, which takes few pivots.
//! 2. **Bounds.** Every bound is the weak-duality bound
//!    ([`BoundedSimplex::dual_bound`]) `L(y) = Σ_j max(r_j l_j, r_j u_j) +
//!    Σ_i max(-y_i r_lo_i, -y_i r_hi_i)` with `r_j = c_j + a_j'y`, from the duals
//!    `y` of the basis. It holds for any `y`, so it does not depend on the
//!    perturbation or on the tolerances of the simplex method, and it includes
//!    a bound on its own rounding error.
//! 3. **Step of the objective.** When every whole column has an integer cost
//!    and every other column cost zero, the value of every integer solution is
//!    a multiple of the greatest common divisor `g` of the costs, so no
//!    solution is above `U = g ⌊L/g⌋`.
//! 4. **Cuts.** At the root of a program with at most 20000 whole columns,
//!    rounds of Gomory mixed-integer cuts \[4\] from the rows of the optimal
//!    tableau are added as rows, see `cut_rounds`, and the relaxation is solved
//!    again. Cuts the relaxation no longer holds tight are dropped in the next
//!    round.
//! 5. **Primal heuristics.** A repair search (see below) from the root point
//!    for a solution of value `U`, then `U - g`, for up to 16 values, followed
//!    at each value by the local search (see below) in a program with at most
//!    20000 free whole columns, and a neighbourhood search (see below). A
//!    solution of value `U` ends the search with a proof.
//! 6. **Levels.** With a step, the search asks, from `z = U` down: is there a
//!    solution of value at least `z`? Every level above has been shown empty, so
//!    the first solution found is optimal, and the bound at any time is the
//!    level being searched. A level ends when a solution reaches it or when
//!    every node is closed, and then the next level down is searched, until the
//!    level is no better than the best solution of the heuristics. Without a
//!    step, one search improves on each solution it finds.
//!
//! The search of a level with threshold `z`:
//!
//! - **Reduced-cost fixing** \[3, section 7.7\]. Since
//!   `c'x ≤ L - Σ_j |r_j| |x_j - b_j|` within the bounds, with `b_j` the bound
//!   `r_j` points to, a whole column with `|r_j| > L - z` takes the value `b_j`
//!   in every solution of value at least `z`. This is applied with the root's
//!   duals at the level, and with each node's own at the node, for its subtree.
//! - **Reduction.** When the fixing leaves at most half of the open whole
//!   columns free, or, in a program that is not itself already a reduction,
//!   at most 20000 of them, the level is solved as a
//!   program of its own on the free columns, the fixed ones moved into the row
//!   bounds, with the row `z ≤ c'x ≤ z` (or `z ≤ c'x` when improving), and with
//!   its own root, cuts and fixing.
//! - **Repair search** from the relaxation's point, and the local search after
//!   it in a program with at most 20000 free whole columns, see below.
//! - **Branch and bound.** A fractional whole column is chosen among the basic
//!   ones: a general integer column before a binary one, then the one furthest
//!   from a whole value, the lowest index among ties. The child on the side the
//!   value is nearer to (up on a tie) is solved at once and the other is kept.
//!   After a node is closed, by bound, by infeasibility or as an integer
//!   solution, the search takes the most recently kept node, for 50 times in a
//!   row, and then the kept node with the best bound, the latest of equals.
//!   Each solve starts from the basis of the node it branches from, so a node
//!   costs a few dual pivots. A node whose bound is below the threshold is
//!   closed. Every 100 nodes the repair search, and in a small program the
//!   local search, runs from the node's point.
//!
//! # Repair search
//!
//! A basic solution has at most as many fractional columns as the program has
//! rows, so the relaxation's point is a few roundings away from an integer
//! point. Every combination of the roundings of up to 14 fractional columns is
//! tried, and the one that breaks the rows least is kept. In a matching problem
//! that point is a few units out of balance. The binary columns with the
//! smallest reduced costs, 2048 and then 8192 of them, are the units nearest the
//! margin of the relaxation, and an exchange of two of them that have the same
//! coefficients on the equality rows keeps the count rows and changes the
//! objective's bound by little. The search makes the flip or exchange within
//! that pool that lowers the violation of the rows most, until every row holds
//! or no move helps. It costs one pass over the matrix and `O(p²)` per move for
//! a pool of `p`, so it runs on programs of any size, where the local search
//! costs a pass over every column per move. See `repair_search`.
//!
//! # Local search
//!
//! The binary columns of the relaxation's point are rounded, and the search
//! moves to lower the weighted violation of the rows and of
//! `z ≤ c'x ≤ ceiling`, each in units of its largest coefficient. A move flips
//! one binary column, or exchanges one at 0 for one at 1 among the 32 best flips
//! each way, whichever lowers the weighted violation most, ties to the lowest
//! indices. After each move every other column moves to the value in its bounds
//! that minimises the weighted violation of its rows, a whole value for a whole
//! column. When no move helps, the weight of every broken row grows by one, as
//! in feasibility jump \[5\]. The search stops when every row holds, or after
//! 400 moves. Exchanges keep the count rows of matching problems, which is what
//! a dive on single columns cannot.
//!
//! # Neighbourhood search
//!
//! In a matching problem many units are interchangeable, and a search can stay a
//! few units below the optimum for a long time. The general integer columns of
//! the root relaxation are first fixed in turn to their nearest whole value and
//! the relaxation solved again. The binary columns are then ordered by the size
//! of their reduced cost there, the basic ones, which include every fractional
//! one, first. The first 64 stay free and the others are fixed at their values
//! rounded, and the smaller program is searched within 250 relaxations. Then
//! 256, 1024, 4096 and 16384 free columns are tried \[6\]. Each solution is a
//! solution of the whole program, so it can only raise the incumbent, and the
//! proof of optimality, by bound, is unchanged. [`Options::neighbourhood_search`]
//! turns this and the root repair and local searches off.
//!
//! # Limits and the gap
//!
//! The search ends when no node is left, or when the node limit or the time
//! limit is reached. Every relaxation solved counts as a node, those of the
//! cuts and of the heuristics included. Then [`Solution::status`] says which
//! limit, and [`Solution::bound`] is the best upper bound proved, never below
//! the incumbent's value. The gap `(bound - objective) / max(1, |objective|)`
//! is [`Solution::gap`]. It is zero for a proven optimum, and a solution that
//! stops at a limit says so and is not presented as optimal. The time limit
//! makes the result depend on the machine and the load, and the node limit
//! does not.
//!
//! # Determinism
//!
//! Branching, node order, cuts and the heuristics depend only on the data, and
//! the simplex is independent of the number of threads, so the solution is the
//! same on any number of threads, unless the time limit ends the search.
//!
//! # Tolerances
//!
//! A basic whole column is called integral when it is within `1e-8` of a whole
//! value, ten times the tolerance the simplex uses for the bounds of a column.
//! The rounded point is checked against every row with the row's own tolerance
//! plus the movement the rounding caused, `Σ_j |a_ij| |x_j - round(x_j)|`.
//! Objective values are compared to `1e-9 max(1, |z|)`. A cut is lowered by
//! `1e-9` times the sum of the magnitudes of its terms at the column bounds. The
//! cut is a combination of rows that the simplex holds to a relative `1e-9` of
//! such a sum, so the combination holds to the same relative amount. Its
//! coefficients below `1e-7` times the largest are dropped with the bound moved
//! by their largest effect, and it is kept only when its coefficients span at most `10^6` and the relaxation's point is
//! at least `10^-4` times their norm from it.
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
//! 2. Wolsey LA. Integer Programming. New York: Wiley; 1998.
//! 3. Achterberg T. Constraint integer programming. PhD thesis, Technische
//!    Universität Berlin; 2007.
//! 4. Gomory RE. An algorithm for the mixed integer problem. Research
//!    memorandum RM-2597. Santa Monica: RAND Corporation; 1960.
//! 5. Luteberget B, Sartor G. Feasibility jump: an LP-free Lagrangian MIP
//!    heuristic. Math Program Comput. 2023;15:365-88.
//! 6. Danna E, Rothberg E, Le Pape C. Exploring relaxation induced
//!    neighborhoods to improve MIP solutions. Math Program. 2005;102(1):71-90.

#![expect(
    clippy::float_cmp,
    reason = "bounds and costs are compared as given: a fixed column has equal \
              bounds, an integer cost equals its rounding, and ties are exact"
)]

use std::fmt;
use std::rc::Rc;
use std::time::{Duration, Instant};

use crate::bounded_simplex::{BoundedLpError, BoundedSimplex, DualBound, Snapshot};

/// Backtracks to the latest open node before the one with the best bound is taken.
const PLUNGE: usize = 50;

/// A basic whole column within this of a whole value is integral.
const INT_TOL: f64 = 1e-8;
/// The relative tolerance of a comparison of objective values.
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

/// Limits on the search, and whether to run the root heuristics. `None` is no
/// limit.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// The most relaxations to solve, the root included.
    pub node_limit: Option<u64>,
    /// The most time to spend. The result then depends on the machine.
    pub time_limit: Option<Duration>,
    /// Run the repair, local and neighbourhood searches at the root, see the
    /// [module](self) documentation. They only ever improve the incumbent, so
    /// they change how fast the search ends and not the optimum it proves.
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
            Self::Numerical => {
                f.write_str("an integer solution broke a row by more than rounding explains")
            }
        }
    }
}

impl std::error::Error for MipError {}

/// Solves the program by branch and bound.
///
/// # Errors
/// See [`MipError`].
pub fn solve(program: &MixedBinaryProgram<'_>, options: &Options) -> Result<Solution, MipError> {
    let deadline = options.time_limit.map(|d| Instant::now() + d);
    validate(program)?;
    let mut budget = Budget {
        node_limit: options.node_limit,
        deadline,
        nodes: 0,
        pivots: 0,
    };
    let goal = Goal {
        cutoff: f64::NEG_INFINITY,
        ceiling: f64::INFINITY,
        heuristics: options.neighbourhood_search,
        cuts: true,
        bounded: false,
        levels: true,
    };
    let outcome = search_program(program, &goal, &mut budget)?;
    match outcome.best {
        None => {
            if outcome.status == Status::Optimal {
                Err(MipError::Infeasible)
            } else {
                Err(MipError::NoSolution {
                    status: outcome.status,
                    bound: outcome.bound,
                })
            }
        }
        Some((x, objective)) => {
            let bound = if outcome.status == Status::Optimal {
                objective
            } else {
                outcome.bound.max(objective)
            };
            let max_row_violation = row_violation(program, &x);
            Ok(Solution {
                x,
                objective,
                bound,
                status: outcome.status,
                nodes: budget.nodes,
                pivots: budget.pivots,
                max_row_violation,
            })
        }
    }
}

fn validate(program: &MixedBinaryProgram<'_>) -> Result<(), MipError> {
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
    for j in 0..n {
        let general = program.integer.get(j).copied().unwrap_or(false) && !program.binary[j];
        if general && !(program.col_lo[j].is_finite() && program.col_hi[j].is_finite()) {
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
    Ok(())
}

/// The limits, shared by a search and the searches it starts, and what they
/// have used.
struct Budget {
    node_limit: Option<u64>,
    deadline: Option<Instant>,
    nodes: u64,
    pivots: u64,
}

impl Budget {
    fn hit(&self) -> Option<Status> {
        if self.node_limit.is_some_and(|l| self.nodes >= l) {
            Some(Status::NodeLimit)
        } else if self.deadline.is_some_and(|d| Instant::now() >= d) {
            Some(Status::TimeLimit)
        } else {
            None
        }
    }
}

/// What a search looks for.
#[derive(Clone, Copy)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent switches of one search, set together by its caller"
)]
struct Goal {
    /// Only solutions of at least this value are sought.
    cutoff: f64,
    /// No solution has a larger value, as the caller has proved.
    ceiling: f64,
    /// Whether to run the neighbourhood search at the root.
    heuristics: bool,
    /// Whether to add cutting planes at the root.
    cuts: bool,
    /// Whether the program holds the row `cutoff ≤ c'x ≤ ceiling` already.
    bounded: bool,
    /// Whether to search by levels, see the [module](self) documentation, when
    /// the objective has a step. Without, the search improves on each solution
    /// it finds.
    levels: bool,
}

/// How a search ended.
struct Outcome {
    /// The best solution found of value at least the cutoff.
    best: Option<(Vec<f64>, f64)>,
    /// No solution of value at least the cutoff has a larger value. The value
    /// of `best` when the search closed.
    bound: f64,
    status: Status,
}

/// Which columns take whole values, and the step of the objective.
struct Shape {
    whole: Vec<bool>,
    general: Vec<bool>,
    /// When every whole column has an integer cost and every other column cost
    /// zero, the value of every integer solution is a multiple of the greatest
    /// common divisor of the costs, and this is it.
    step: Option<f64>,
}

fn shape(program: &MixedBinaryProgram<'_>) -> Shape {
    let n = program.cost.len();
    let general: Vec<bool> = (0..n)
        .map(|j| program.integer.get(j).copied().unwrap_or(false) && !program.binary[j])
        .collect();
    let whole: Vec<bool> = (0..n).map(|j| program.binary[j] || general[j]).collect();
    // Integers up to 2^53 are exact in a double, and so is their divisor.
    let exact = 2.0_f64.powi(53);
    let mut divisor: u64 = 0;
    let mut lattice = true;
    for j in 0..n {
        let c = program.cost[j];
        if whole[j] {
            if c != c.round() || c.abs() >= exact {
                lattice = false;
                break;
            }
            if c != 0.0 {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "a whole, nonnegative value below 2^53"
                )]
                let whole = c.abs() as u64;
                divisor = gcd(divisor, whole);
            }
        } else if c != 0.0 {
            lattice = false;
            break;
        }
    }
    let step = lattice.then_some(if divisor == 0 { 1.0 } else { divisor as f64 });
    Shape {
        whole,
        general,
        step,
    }
}

const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// The largest multiple of `step` that is at most `z`, `z` itself without a
/// step. A bound within the objective tolerance below a multiple is taken as
/// that multiple.
fn floor_to(z: f64, step: Option<f64>) -> f64 {
    match step {
        Some(g) if z.is_finite() => ((z + OBJECTIVE_TOL * z.abs().max(1.0)) / g).floor() * g,
        _ => z,
    }
}

/// The smallest multiple of `step` that is at least `z`, `z` itself without a
/// step. A cutoff within the objective tolerance above a multiple is taken as
/// that multiple.
fn ceil_to(z: f64, step: Option<f64>) -> f64 {
    match step {
        Some(g) if z.is_finite() => ((z - OBJECTIVE_TOL * z.abs().max(1.0)) / g).ceil() * g,
        _ => z,
    }
}

/// `v ≥ t` up to the objective tolerance.
fn reaches(v: f64, t: f64) -> bool {
    v >= t - OBJECTIVE_TOL * t.abs().max(1.0)
}

/// Builds the relaxation, solves it, and searches.
fn search_program(
    program: &MixedBinaryProgram<'_>,
    goal: &Goal,
    budget: &mut Budget,
) -> Result<Outcome, MipError> {
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
    let result = match lp.solve(budget.deadline) {
        Ok(()) => {
            budget.nodes += 1;
            let whole = (0..program.cost.len())
                .filter(|&j| program.binary[j] || program.integer.get(j).copied().unwrap_or(false))
                .count();
            if goal.cuts && whole <= CUT_COLUMNS {
                let pinned = goal.bounded && goal.cutoff == goal.ceiling;
                match cut_rounds(program, &mut lp, pinned, budget)? {
                    Cuts::None => search_lp(&mut lp, program, goal, budget),
                    Cuts::Infeasible => Ok(Outcome {
                        best: None,
                        bound: f64::NEG_INFINITY,
                        status: Status::Optimal,
                    }),
                    Cuts::Added(added) => {
                        let (rows, mut cut_lp) = *added;
                        let augmented = rows.program(program);
                        let result = search_lp(&mut cut_lp, &augmented, goal, budget);
                        budget.pivots += cut_lp.iterations();
                        result
                    }
                }
            } else {
                search_lp(&mut lp, program, goal, budget)
            }
        }
        Err(BoundedLpError::Infeasible) => Ok(Outcome {
            best: None,
            bound: f64::NEG_INFINITY,
            status: Status::Optimal,
        }),
        Err(BoundedLpError::TimeLimit) => Ok(Outcome {
            best: None,
            bound: goal.ceiling,
            status: Status::TimeLimit,
        }),
        Err(e) => Err(MipError::Lp(e)),
    };
    budget.pivots += lp.iterations();
    result
}

/// The search from a solved root relaxation. See the [module](self)
/// documentation.
fn search_lp(
    lp: &mut BoundedSimplex,
    program: &MixedBinaryProgram<'_>,
    goal: &Goal,
    budget: &mut Budget,
) -> Result<Outcome, MipError> {
    let shape = shape(program);
    let root = tight_bound(lp, budget)?;
    let step = shape.step;
    let upper = floor_to(root.value.min(goal.ceiling), step);
    let cutoff = ceil_to(goal.cutoff, step);
    if upper < cutoff {
        return Ok(Outcome {
            best: None,
            bound: f64::NEG_INFINITY,
            status: Status::Optimal,
        });
    }
    let mut best: Option<(Vec<f64>, f64)> = None;
    if goal.heuristics {
        ladder(program, &shape, lp, &root, upper, cutoff, &mut best);
        if best.as_ref().is_none_or(|(_, v)| !reaches(*v, upper)) {
            neighbourhood_search(program, &shape, lp, &root, upper, cutoff, budget, &mut best)?;
        }
        if best.as_ref().is_some_and(|(_, v)| reaches(*v, upper)) {
            let bound = best.as_ref().map_or(upper, |(_, v)| *v);
            return Ok(Outcome {
                best,
                bound,
                status: Status::Optimal,
            });
        }
    }
    let mut context = Context {
        bounded: goal.bounded,
        lp,
        program,
        shape: &shape,
        root: &root,
        budget,
    };
    if let (Some(g), true) = (step, goal.levels) {
        // The levels, from the top: is there a solution of value at least `level`?
        // No point within the column bounds has a smaller value than `least`.
        let least = ceil_to(
            (0..program.cost.len())
                .map(|j| {
                    let c = program.cost[j];
                    if c > 0.0 {
                        c * program.col_lo[j]
                    } else if c < 0.0 {
                        c * program.col_hi[j]
                    } else {
                        0.0
                    }
                })
                .sum(),
            step,
        );
        let lowest = best
            .as_ref()
            .map_or(cutoff, |(_, v)| v + g)
            .max(cutoff)
            .max(least);
        let mut level = upper;
        while reaches(level, lowest) {
            if let Some(status) = context.budget.hit() {
                return Ok(stopped(best, level, status));
            }
            match context.search(level, false, &mut best)? {
                End::Done => {
                    if let Some((_, v)) = best.as_ref()
                        && reaches(*v, level)
                    {
                        let bound = *v;
                        return Ok(Outcome {
                            best,
                            bound,
                            status: Status::Optimal,
                        });
                    }
                    level -= g;
                }
                End::Stopped { status, .. } => return Ok(stopped(best, level, status)),
            }
        }
        let bound = best.as_ref().map_or(f64::NEG_INFINITY, |(_, v)| *v);
        Ok(Outcome {
            best,
            bound,
            status: Status::Optimal,
        })
    } else {
        let threshold = best
            .as_ref()
            .map_or(cutoff, |(_, v)| improvement(*v).max(cutoff));
        match context.search(threshold, true, &mut best)? {
            End::Done => {
                let bound = best.as_ref().map_or(f64::NEG_INFINITY, |(_, v)| *v);
                Ok(Outcome {
                    best,
                    bound,
                    status: Status::Optimal,
                })
            }
            End::Stopped { status, bound } => {
                let bound = bound
                    .min(upper)
                    .max(best.as_ref().map_or(f64::NEG_INFINITY, |(_, v)| *v));
                Ok(Outcome {
                    best,
                    bound,
                    status,
                })
            }
        }
    }
}

fn stopped(best: Option<(Vec<f64>, f64)>, level: f64, status: Status) -> Outcome {
    let bound = best.as_ref().map_or(level, |(_, v)| level.max(*v));
    Outcome {
        best,
        bound,
        status,
    }
}

/// The least value that improves on `v` when the objective has no step.
fn improvement(v: f64) -> f64 {
    v + OBJECTIVE_TOL * v.abs().max(1.0)
}

/// Cutting planes are added to programs with at most this many whole columns.
const CUT_COLUMNS: usize = 20_000;

/// The most rounds of cutting planes at the root.
const CUT_ROUNDS: usize = 8;

/// A basic whole variable is cut on when its fractional part is at least this
/// far from 0 and 1; nearer, the coefficients of the cut grow as its inverse.
const CUT_FRACTION: f64 = 0.01;

/// A cut coefficient below this times the largest is dropped, with the row bound
/// moved so that the cut stays valid.
const CUT_DROP: f64 = 1e-7;

/// The largest ratio of the largest to the smallest coefficient of a cut.
const CUT_RANGE: f64 = 1e6;

/// The least distance of the relaxation's point from a cut, in units of the
/// norm of its coefficients.
const CUT_EFFICACY: f64 = 1e-4;

/// The rows added by the cutting planes, on top of a program's.
struct CutRows {
    rows: usize,
    columns: Vec<f64>,
    row_lo: Vec<f64>,
    row_hi: Vec<f64>,
}

impl CutRows {
    fn program<'s>(&'s self, base: &MixedBinaryProgram<'s>) -> MixedBinaryProgram<'s> {
        MixedBinaryProgram {
            rows: self.rows,
            columns: &self.columns,
            cost: base.cost,
            col_lo: base.col_lo,
            col_hi: base.col_hi,
            row_lo: &self.row_lo,
            row_hi: &self.row_hi,
            binary: base.binary,
            integer: base.integer,
        }
    }
}

/// What the cut rounds did.
enum Cuts {
    /// No cut was added.
    None,
    /// The cuts leave no integer point.
    Infeasible,
    /// The program with the cuts, and its solved relaxation.
    Added(Box<(CutRows, BoundedSimplex)>),
}

/// Rounds of Gomory mixed-integer cuts at the root.
///
/// Each round takes every row of the optimal tableau whose basic variable must
/// be whole and has a fractional part `f0` in `[CUT_FRACTION, 1 -
/// CUT_FRACTION]`, and adds the Gomory mixed-integer cut of that row (Gomory
/// 1960; Wolsey, Integer Programming, section 8.7). With each non-basic
/// variable written as its distance `y_j ≥ 0` from the bound it sits at, the
/// row reads `x_b + Σ a_j y_j = x̄_b`, and the cut is
///
/// ```text
/// Σ_{y_j whole, f_j ≤ f0} f_j/f0 y_j + Σ_{y_j whole, f_j > f0} (1-f_j)/(1-f0) y_j
///   + Σ_{y_j other, a_j > 0} a_j/f0 y_j + Σ_{y_j other, a_j < 0} -a_j/(1-f0) y_j ≥ 1,
/// ```
///
/// with `f_j` the fractional part of `a_j`. A slack is whole when its row has
/// whole coefficients on whole columns only and whole bounds. The cut is then
/// written in the columns, coefficients below [`CUT_DROP`] times the largest are
/// dropped with the bound moved by their largest effect, and the bound is
/// lowered by `10^-9 (1 + |bound| + Σ |coefficient| · |column bound|)` for the
/// rounding of the tableau, so that no integer point the exact cut keeps is cut
/// off. The relaxation with the cuts is solved again, and the rounds stop when
/// a round adds no cut, after [`CUT_ROUNDS`], or when the bound falls by less
/// than a hundredth of the objective step, or of `10^-6 max(1, |bound|)`.
fn cut_rounds(
    program: &MixedBinaryProgram<'_>,
    lp: &mut BoundedSimplex,
    pinned: bool,
    budget: &mut Budget,
) -> Result<Cuts, MipError> {
    let shape = shape(program);
    let n = program.cost.len();
    let mut rows = CutRows {
        rows: program.rows,
        columns: program.columns.to_vec(),
        row_lo: program.row_lo.to_vec(),
        row_hi: program.row_hi.to_vec(),
    };
    let mut current: Option<BoundedSimplex> = None;
    let mut bound = tight_bound(lp, budget)?.value;
    let progress = shape.step.map_or(1e-6 * bound.abs().max(1.0), |g| 0.01 * g);
    for _ in 0..CUT_ROUNDS {
        if budget.hit().is_some() {
            break;
        }
        let solved = current.as_ref().unwrap_or(lp);
        let cuts = gomory_cuts(solved, &rows.program(program), &shape);
        if cuts.is_empty() {
            break;
        }
        let m = rows.rows;
        let k = cuts.len();
        // The program's rows, the cuts of earlier rounds that the relaxation
        // holds tight, and the new cuts.
        let kept: Vec<usize> = (0..m)
            .filter(|&i| i < program.rows || solved.slack(i).2.is_some())
            .collect();
        let mut columns = Vec::with_capacity(n * (kept.len() + k));
        for j in 0..n {
            let column = &rows.columns[j * m..(j + 1) * m];
            columns.extend(kept.iter().map(|&i| column[i]));
            columns.extend(cuts.iter().map(|(c, _)| c[j]));
        }
        let mut next = CutRows {
            rows: kept.len() + k,
            columns,
            row_lo: kept.iter().map(|&i| rows.row_lo[i]).collect(),
            row_hi: kept.iter().map(|&i| rows.row_hi[i]).collect(),
        };
        for (_, lo) in &cuts {
            next.row_lo.push(*lo);
            next.row_hi.push(f64::INFINITY);
        }
        let mut next_lp = BoundedSimplex::new(
            next.rows,
            &next.columns,
            program.cost,
            program.col_lo,
            program.col_hi,
            &next.row_lo,
            &next.row_hi,
        )
        .map_err(MipError::Lp)?;
        next_lp.perturb_costs();
        match next_lp.solve(budget.deadline) {
            Ok(()) => budget.nodes += 1,
            Err(BoundedLpError::Infeasible) => {
                budget.pivots += next_lp.iterations();
                return Ok(Cuts::Infeasible);
            }
            Err(BoundedLpError::TimeLimit) => {
                budget.pivots += next_lp.iterations();
                break;
            }
            Err(e) => return Err(MipError::Lp(e)),
        }
        let next_bound = tight_bound(&mut next_lp, budget)?.value;
        if let Some(old) = current.replace(next_lp) {
            budget.pivots += old.iterations();
        }
        rows = next;
        let gained = bound - next_bound;
        bound = next_bound.min(bound);
        if gained < progress && !pinned {
            break;
        }
    }
    Ok(match current {
        Some(cut_lp) => Cuts::Added(Box::new((rows, cut_lp))),
        None => Cuts::None,
    })
}

/// The Gomory mixed-integer cuts of the optimal tableau, each as coefficients
/// on the columns and a lower bound. See [`cut_rounds`].
fn gomory_cuts(
    lp: &BoundedSimplex,
    program: &MixedBinaryProgram<'_>,
    shape: &Shape,
) -> Vec<(Vec<f64>, f64)> {
    let n = program.cost.len();
    let m = program.rows;
    let whole_value = |v: f64| v.is_finite() && v == v.round();
    // A slack is whole when its row has whole bounds, or none, and whole
    // coefficients on whole columns only.
    let whole_bound = |v: f64| !v.is_finite() || v == v.round();
    let whole_row: Vec<bool> = (0..m)
        .map(|i| {
            whole_bound(program.row_lo[i])
                && whole_bound(program.row_hi[i])
                && (0..n).all(|j| {
                    let a = program.columns[j * m + i];
                    a == 0.0 || (shape.whole[j] && a == a.round())
                })
        })
        .collect();
    let gmi = |a: f64, f0: f64, whole: bool| -> f64 {
        if whole {
            let f = a - a.floor();
            if f <= f0 {
                f / f0
            } else {
                (1.0 - f) / (1.0 - f0)
            }
        } else if a >= 0.0 {
            a / f0
        } else {
            -a / (1.0 - f0)
        }
    };
    let mut cuts = Vec::new();
    for position in 0..m {
        let row = lp.tableau_row(position);
        let b = row.variable;
        let whole_basic = if b < n {
            shape.whole[b]
        } else {
            whole_row[b - n]
        };
        if !whole_basic {
            continue;
        }
        let f0 = row.value - row.value.floor();
        if !(CUT_FRACTION..=1.0 - CUT_FRACTION).contains(&f0) {
            continue;
        }
        let mut coef = vec![0.0; n];
        let mut lower = 1.0;
        for (j, (c, &alpha)) in coef.iter_mut().zip(&row.columns).enumerate() {
            let (lo, hi) = lp.bounds(j);
            if alpha == 0.0 || lo == hi {
                continue;
            }
            let Some(at) = lp.at_bound(j) else {
                continue;
            };
            let at_lower = at == lo;
            let a = if at_lower { alpha } else { -alpha };
            let pi = gmi(a, f0, shape.whole[j] && whole_value(at));
            if at_lower {
                *c += pi;
                lower += pi * lo;
            } else {
                *c -= pi;
                lower -= pi * hi;
            }
        }
        for (i, (&alpha, &whole_slack)) in row.slacks.iter().zip(&whole_row).enumerate() {
            let (lo, hi, at) = lp.slack(i);
            if alpha == 0.0 || lo == hi {
                continue;
            }
            let Some(at) = at else {
                continue;
            };
            let at_lower = at == lo;
            let a = if at_lower { alpha } else { -alpha };
            let pi = gmi(a, f0, whole_slack);
            let sign = if at_lower { 1.0 } else { -1.0 };
            for (j, c) in coef.iter_mut().enumerate() {
                *c += sign * pi * program.columns[j * m + i];
            }
            lower += sign * pi * at;
        }
        let largest = coef.iter().fold(0.0_f64, |acc, c| acc.max(c.abs()));
        if largest == 0.0 || !largest.is_finite() || !lower.is_finite() {
            continue;
        }
        let mut size = 1.0 + lower.abs();
        for (c, (&lo, &hi)) in coef
            .iter_mut()
            .zip(program.col_lo.iter().zip(program.col_hi))
        {
            if c.abs() < CUT_DROP * largest {
                // Dropping `c x_j` from `Σ c x ≥ lower` is valid with
                // `lower - max(c x_j)`.
                lower -= (*c * lo).max(*c * hi);
                *c = 0.0;
            }
            size += c.abs() * lo.abs().max(hi.abs());
        }
        // The rows the cut is made from hold to a relative 1e-9 of the sum of the
        // magnitudes of their terms, so the cut holds to that much of `size`.
        lower -= 1e-9 * size;
        // A cut whose coefficients span more than CUT_RANGE makes the basis
        // ill-conditioned, and one that the relaxation's point barely breaks
        // moves nothing.
        let largest = coef.iter().fold(0.0_f64, |acc, c| acc.max(c.abs()));
        let smallest = coef
            .iter()
            .filter(|c| **c != 0.0)
            .fold(f64::INFINITY, |acc, c| acc.min(c.abs()));
        let x = lp.values();
        let at_point: f64 = coef.iter().zip(&x).map(|(c, v)| c * v).sum();
        let norm = coef.iter().map(|c| c * c).sum::<f64>().sqrt();
        if lower.is_finite()
            && largest > 0.0
            && largest <= CUT_RANGE * smallest
            && (lower - at_point) > CUT_EFFICACY * norm
        {
            for c in &mut coef {
                *c /= largest;
            }
            cuts.push((coef, lower / largest));
        }
    }
    cuts
}

/// The least value of a solution better than one of value `v`.
fn improved(v: f64, step: Option<f64>) -> f64 {
    match step {
        Some(g) => v + g,
        None => improvement(v),
    }
}

/// The most values the ladder of searches tries.
const LADDER: usize = 16;

/// Searches from the root relaxation for a solution of value in `[z, upper]`,
/// with `z = upper`, then one step of the objective less, and so on, for
/// [`LADDER`] values or down to `cutoff`, until one succeeds: the repair search,
/// and after it the local search when at most [`SEARCH_COLUMNS`] whole columns
/// are free. A solution better than `best` replaces it. Without a step of the
/// objective, one value, `upper`.
fn ladder(
    program: &MixedBinaryProgram<'_>,
    shape: &Shape,
    lp: &BoundedSimplex,
    root: &DualBound,
    upper: f64,
    cutoff: f64,
    best: &mut Option<(Vec<f64>, f64)>,
) {
    let n = program.cost.len();
    let start = lp.values();
    let frozen = vec![false; n];
    let tolerance: Vec<f64> = (0..program.rows).map(|i| lp.row_tolerance(i)).collect();
    let small = free_whole(program, shape, &frozen) <= SEARCH_COLUMNS;
    let mut target = upper;
    for _ in 0..LADDER {
        if !reaches(target, cutoff) || best.as_ref().is_some_and(|(_, v)| reaches(*v, target)) {
            return;
        }
        let found = repair_search(
            program,
            shape,
            &start,
            &root.reduced_costs,
            &frozen,
            target,
            upper,
            &tolerance,
        )
        .or_else(|| {
            small
                .then(|| {
                    local_search(
                        program,
                        shape,
                        &start,
                        &frozen,
                        target,
                        f64::INFINITY,
                        &tolerance,
                    )
                })
                .flatten()
        });
        if let Some(found) = found {
            if best.as_ref().is_none_or(|(_, v)| found.1 > *v) {
                *best = Some(found);
            }
            return;
        }
        match shape.step {
            Some(g) => target -= g,
            None => return,
        }
    }
}

/// The whole columns that are neither fixed nor `frozen`.
fn free_whole(program: &MixedBinaryProgram<'_>, shape: &Shape, frozen: &[bool]) -> usize {
    (0..program.cost.len())
        .filter(|&j| shape.whole[j] && !frozen[j] && program.col_lo[j] != program.col_hi[j])
        .count()
}

/// The local search runs, when the repair search finds nothing, on programs
/// with at most this many whole columns free.
const SEARCH_COLUMNS: usize = 20_000;

/// Branch and bound runs the local search from every this many nodes.
const SEARCH_EVERY: usize = 100;

/// The most moves of the local search.
const SEARCH_STEPS: usize = 400;

/// Columns of each kind of move, adding and dropping, among which the local
/// search tries every exchange.
const SEARCH_PAIRS: usize = 32;

/// Looks for a solution of value in `[cutoff, ceiling]` near `start` by a
/// local search that minimises the weighted violation of the rows.
///
/// The rows are the program's and `cutoff ≤ c'x ≤ ceiling`. The binary columns
/// are rounded, and every move then flips one binary column or exchanges a
/// binary column at 0 for one at 1, the move that lowers the weighted
/// violation most, among every flip and the exchanges between the
/// [`SEARCH_PAIRS`] best flips each way, ties to the lowest indices. After each
/// move every other column that is not `frozen` moves to the value in its
/// bounds that minimises the weighted violation of its rows, a whole value for
/// a general integer column. When no move lowers it, the weight of every
/// broken row grows by one, as in feasibility jump (Luteberget and Sartor).
/// A row is broken when its activity is outside its bounds by more than its
/// `tolerance`, the cut by more than the objective tolerance, and the violation
/// of a row is in units of its largest coefficient.
#[expect(
    clippy::too_many_lines,
    reason = "one loop of moves over shared row state"
)]
fn local_search(
    program: &MixedBinaryProgram<'_>,
    shape: &Shape,
    start: &[f64],
    frozen: &[bool],
    cutoff: f64,
    ceiling: f64,
    tolerance: &[f64],
) -> Option<(Vec<f64>, f64)> {
    let n = program.cost.len();
    let m = program.rows;
    let rows = m + 1;
    let coef = |j: usize, i: usize| {
        if i < m {
            program.columns[j * m + i]
        } else {
            program.cost[j]
        }
    };
    let lo: Vec<f64> = (0..rows)
        .map(|i| if i < m { program.row_lo[i] } else { cutoff })
        .collect();
    let hi: Vec<f64> = (0..rows)
        .map(|i| if i < m { program.row_hi[i] } else { ceiling })
        .collect();
    let tol: Vec<f64> = (0..rows)
        .map(|i| {
            if i < m {
                tolerance[i]
            } else {
                OBJECTIVE_TOL * cutoff.abs().max(1.0)
            }
        })
        .collect();
    let mut scale = vec![0.0_f64; rows];
    for j in 0..n {
        for (i, s) in scale.iter_mut().enumerate() {
            *s = s.max(coef(j, i).abs());
        }
    }
    for s in &mut scale {
        *s = if *s > 0.0 { 1.0 / *s } else { 1.0 };
    }
    let violation = |i: usize, act: f64| -> f64 {
        ((lo[i] - tol[i] - act).max(act - hi[i] - tol[i])).max(0.0) * scale[i]
    };
    let mut x: Vec<f64> = (0..n)
        .map(|j| {
            if program.binary[j] && !frozen[j] {
                start[j].round()
            } else {
                start[j]
            }
        })
        .collect();
    let mut act = vec![0.0; rows];
    for (j, &xj) in x.iter().enumerate() {
        if xj != 0.0 {
            for (i, a) in act.iter_mut().enumerate() {
                *a += coef(j, i) * xj;
            }
        }
    }
    let mut weight = vec![1.0; rows];
    let movable_other: Vec<usize> = (0..n)
        .filter(|&j| !program.binary[j] && !frozen[j] && program.col_lo[j] != program.col_hi[j])
        .collect();
    let binaries: Vec<usize> = (0..n)
        .filter(|&j| program.binary[j] && !frozen[j] && program.col_lo[j] != program.col_hi[j])
        .collect();
    // Moves every other column to its best value, one after the other.
    let settle = |x: &mut [f64], act: &mut [f64], weight: &[f64]| {
        for &j in &movable_other {
            let touched: Vec<usize> = (0..rows).filter(|&i| coef(j, i) != 0.0).collect();
            if touched.is_empty() {
                continue;
            }
            let (l, u) = (program.col_lo[j], program.col_hi[j]);
            let cost_at = |v: f64| -> f64 {
                touched
                    .iter()
                    .map(|&i| weight[i] * violation(i, act[i] + coef(j, i) * (v - x[j])))
                    .sum()
            };
            // The weighted violation is convex and piecewise linear in the
            // column, with its corners where a row reaches a bound.
            let mut candidates = vec![x[j]];
            for &i in &touched {
                let a = coef(j, i);
                for b in [lo[i], hi[i]] {
                    if b.is_finite() {
                        candidates.push(x[j] + (b - act[i]) / a);
                    }
                }
            }
            if l.is_finite() {
                candidates.push(l);
            }
            if u.is_finite() {
                candidates.push(u);
            }
            let mut best_v = x[j];
            let mut best_c = cost_at(x[j]);
            for c in candidates {
                let options: Vec<f64> = if shape.general[j] {
                    vec![c.floor(), c.ceil()]
                } else {
                    vec![c]
                };
                for v in options {
                    let v = v.clamp(l, u);
                    if shape.general[j] && v != v.round() {
                        continue;
                    }
                    let cv = cost_at(v);
                    if cv < best_c - 1e-15 {
                        best_c = cv;
                        best_v = v;
                    }
                }
            }
            if best_v != x[j] {
                let d = best_v - x[j];
                for &i in &touched {
                    act[i] += coef(j, i) * d;
                }
                x[j] = best_v;
            }
        }
    };
    settle(&mut x, &mut act, &weight);
    for _ in 0..SEARCH_STEPS {
        let broken: Vec<usize> = (0..rows).filter(|&i| violation(i, act[i]) > 0.0).collect();
        if broken.is_empty() {
            break;
        }
        // The change of the weighted violation for each flip.
        let delta = |j: usize, sign: f64, act: &[f64], weight: &[f64]| -> f64 {
            (0..rows)
                .map(|i| {
                    let a = coef(j, i);
                    if a == 0.0 {
                        0.0
                    } else {
                        weight[i] * (violation(i, act[i] + sign * a) - violation(i, act[i]))
                    }
                })
                .sum()
        };
        let mut adds: Vec<(f64, usize)> = Vec::new();
        let mut drops: Vec<(f64, usize)> = Vec::new();
        for &j in &binaries {
            if x[j] == 0.0 {
                adds.push((delta(j, 1.0, &act, &weight), j));
            } else {
                drops.push((delta(j, -1.0, &act, &weight), j));
            }
        }
        let order = |a: &(f64, usize), b: &(f64, usize)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
        // Only the first SEARCH_PAIRS of each list, in order, are used.
        for list in [&mut adds, &mut drops] {
            if list.len() > SEARCH_PAIRS {
                list.select_nth_unstable_by(SEARCH_PAIRS, order);
                list.truncate(SEARCH_PAIRS);
            }
            list.sort_by(order);
        }
        let mut best_move: Option<(f64, Option<usize>, Option<usize>)> = None;
        let mut consider = |d: f64, add: Option<usize>, drop: Option<usize>| {
            if best_move.is_none_or(|(bd, _, _)| d < bd) {
                best_move = Some((d, add, drop));
            }
        };
        if let Some(&(d, j)) = adds.first() {
            consider(d, Some(j), None);
        }
        if let Some(&(d, j)) = drops.first() {
            consider(d, None, Some(j));
        }
        for &(_, u) in adds.iter().take(SEARCH_PAIRS) {
            for &(_, v) in drops.iter().take(SEARCH_PAIRS) {
                let d: f64 = (0..rows)
                    .map(|i| {
                        let change = coef(u, i) - coef(v, i);
                        if change == 0.0 {
                            0.0
                        } else {
                            weight[i] * (violation(i, act[i] + change) - violation(i, act[i]))
                        }
                    })
                    .sum();
                consider(d, Some(u), Some(v));
            }
        }
        match best_move {
            Some((d, add, drop)) if d < -1e-12 => {
                for (j, value) in [(add, 1.0), (drop, 0.0)] {
                    if let Some(j) = j {
                        let change = value - x[j];
                        for (i, a) in act.iter_mut().enumerate() {
                            *a += coef(j, i) * change;
                        }
                        x[j] = value;
                    }
                }
                settle(&mut x, &mut act, &weight);
            }
            _ => {
                for &i in &broken {
                    weight[i] += 1.0;
                }
            }
        }
    }
    // A point is a solution when every row holds to its tolerance.
    let act = activity(program, &x);
    let holds = (0..m).all(|i| {
        act[i] >= program.row_lo[i] - tolerance[i] && act[i] <= program.row_hi[i] + tolerance[i]
    });
    let value = objective(program, &x);
    let whole = (0..n).all(|j| !shape.whole[j] || x[j] == x[j].round());
    (holds && whole && reaches(value, cutoff)).then_some((x, value))
}

/// The pools of the repair search, in binary columns, see [`repair_search`].
const REPAIR_POOLS: [usize; 2] = [2048, 8192];

/// The most moves of the repair search with one pool.
const REPAIR_STEPS: usize = 48;

/// The most fractional columns whose roundings the repair search tries in
/// every combination.
const ROUND_EXHAUSTIVE: usize = 14;

/// The rows of a repair search: the program's and `cutoff ≤ c'x ≤ ceiling`
/// last, each with its bounds, its tolerance, and the inverse of its largest
/// coefficient, which puts a violation in units of one column's effect.
struct RepairRows {
    lo: Vec<f64>,
    hi: Vec<f64>,
    tol: Vec<f64>,
    scale: Vec<f64>,
}

impl RepairRows {
    /// The violation of row `i` at activity `a`, in units of its largest
    /// coefficient.
    fn violation(&self, i: usize, a: f64) -> f64 {
        ((self.lo[i] - self.tol[i] - a).max(a - self.hi[i] - self.tol[i])).max(0.0) * self.scale[i]
    }

    /// How far activity `a` is from breaking row `i`, in the row's units.
    fn room(&self, i: usize, a: f64) -> f64 {
        (a - (self.lo[i] - self.tol[i])).min(self.hi[i] + self.tol[i] - a)
    }
}

/// Moves every continuous column that is not `frozen` to the value in its
/// bounds that minimises the violation of its rows, one after the other.
fn settle_continuous(
    program: &MixedBinaryProgram<'_>,
    shape: &Shape,
    frozen: &[bool],
    rows: &RepairRows,
    x: &mut [f64],
    act: &mut [f64],
) {
    let m = program.rows;
    let coef = |j: usize, i: usize| {
        if i < m {
            program.columns[j * m + i]
        } else {
            program.cost[j]
        }
    };
    for j in 0..x.len() {
        if shape.whole[j] || frozen[j] || program.col_lo[j] == program.col_hi[j] {
            continue;
        }
        let touched: Vec<usize> = (0..=m).filter(|&i| coef(j, i) != 0.0).collect();
        if touched.is_empty() {
            continue;
        }
        let (l, u) = (program.col_lo[j], program.col_hi[j]);
        let cost_at = |v: f64, x: &[f64], act: &[f64]| -> f64 {
            touched
                .iter()
                .map(|&i| rows.violation(i, act[i] + coef(j, i) * (v - x[j])))
                .sum()
        };
        // The violation is convex and piecewise linear in the column, with its
        // corners where a row reaches a bound.
        let mut candidates = vec![l, u];
        for &i in &touched {
            for b in [rows.lo[i], rows.hi[i]] {
                if b.is_finite() {
                    candidates.push(x[j] + (b - act[i]) / coef(j, i));
                }
            }
        }
        let mut best_v = x[j];
        let mut best_c = cost_at(x[j], x, act);
        for v in candidates {
            if !v.is_finite() {
                continue;
            }
            let v = v.clamp(l, u);
            let c = cost_at(v, x, act);
            if c < best_c {
                best_c = c;
                best_v = v;
            }
        }
        if best_v != x[j] {
            let d = best_v - x[j];
            for &i in &touched {
                act[i] += coef(j, i) * d;
            }
            x[j] = best_v;
        }
    }
}

/// Looks for a solution of value in `[cutoff, ceiling]` near the relaxation's
/// point `start` by rounding it and repairing the rounding by exchanges.
///
/// A basic solution of a program with `m` rows has at most `m` fractional
/// columns, so the rounding is small:
///
/// 1. **Rounding.** Every whole column within `1e-8` of a whole value takes it.
///    For the fractional ones, up to [`ROUND_EXHAUSTIVE`] of the furthest from
///    a whole value, every combination of their two nearest whole values is
///    tried, in Gray-code order, and the one with the least violation of the
///    rows, the first of equals, is kept; the others are rounded to the
///    nearest. The continuous columns then move to the values that minimise the
///    violation of their rows.
/// 2. **Exchanges.** The binary columns that are not `frozen` are ordered by
///    `priority`, the size of their reduced cost in the relaxation, smallest
///    first, and the first 2048 form a pool. Two columns of the pool are
///    interchangeable when they have the same coefficients on every equality
///    row, the row of the objective included when `cutoff = ceiling`, so that
///    setting one to 1 and the other to 0 keeps those rows. Each move is the
///    flip of one pool column or such an exchange, the one that lowers the sum
///    of the rows' violations, each in units of the row's largest coefficient,
///    the most, ties to the lowest indices; the general integer and continuous
///    columns stay. Only rows that one move can break are evaluated: those
///    broken and those within twice the pool's largest coefficient of a bound.
///    The search stops when every row holds, or when no move helps, and then
///    goes on with a pool of 8192, for at most [`REPAIR_STEPS`] moves with each.
///
/// With small reduced costs an exchange costs little of the gap between the
/// relaxation's bound and `cutoff`, and in a matching problem the pool holds
/// many units near the margin of the relaxation, so the exchanges can move the
/// balance rows in small steps. A move costs `O(p² r)` for a pool of `p` and
/// `r` rows evaluated, and the search `O(n m)` besides, for the activity.
#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the program, the point and the search's limits; one pass of rounding and moves"
)]
fn repair_search(
    program: &MixedBinaryProgram<'_>,
    shape: &Shape,
    start: &[f64],
    priority: &[f64],
    frozen: &[bool],
    cutoff: f64,
    ceiling: f64,
    tolerance: &[f64],
) -> Option<(Vec<f64>, f64)> {
    let n = program.cost.len();
    let m = program.rows;
    let total_rows = m + 1;
    let coef = |j: usize, i: usize| {
        if i < m {
            program.columns[j * m + i]
        } else {
            program.cost[j]
        }
    };
    let mut rows = RepairRows {
        lo: (0..total_rows)
            .map(|i| if i < m { program.row_lo[i] } else { cutoff })
            .collect(),
        hi: (0..total_rows)
            .map(|i| if i < m { program.row_hi[i] } else { ceiling })
            .collect(),
        tol: (0..total_rows)
            .map(|i| {
                if i < m {
                    tolerance[i]
                } else {
                    OBJECTIVE_TOL * cutoff.abs().max(1.0)
                }
            })
            .collect(),
        scale: vec![0.0; total_rows],
    };
    // Rounding: the whole columns near a whole value take it.
    let mut x = start.to_vec();
    let mut fractional: Vec<(f64, usize)> = Vec::new();
    for j in 0..n {
        if !shape.whole[j] {
            continue;
        }
        let v = start[j];
        let d = (v - v.round()).abs();
        if frozen[j] || d <= INT_TOL * v.abs().max(1.0) {
            x[j] = v.round();
        } else {
            fractional.push((d, j));
        }
    }
    fractional.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    // The columns whose roundings are enumerated start at their floor.
    let mut enumerated: Vec<usize> = Vec::new();
    for (k, &(_, j)) in fractional.iter().enumerate() {
        let v = start[j];
        let (fl, ce) = (v.floor(), v.ceil());
        if k < ROUND_EXHAUSTIVE && fl >= program.col_lo[j] && ce <= program.col_hi[j] {
            x[j] = fl;
            enumerated.push(j);
        } else {
            x[j] = v.round().clamp(program.col_lo[j], program.col_hi[j]);
        }
    }
    let mut act = vec![0.0; total_rows];
    for (j, &xj) in x.iter().enumerate() {
        for (i, s) in rows.scale.iter_mut().enumerate() {
            let a = coef(j, i);
            *s = s.max(a.abs());
            if xj != 0.0 {
                act[i] += a * xj;
            }
        }
    }
    for s in &mut rows.scale {
        *s = if *s > 0.0 { 1.0 / *s } else { 1.0 };
    }
    let total = |act: &[f64]| -> f64 { (0..total_rows).map(|i| rows.violation(i, act[i])).sum() };
    if !enumerated.is_empty() {
        let mut current = act.clone();
        let mut best = (total(&current), 0_u64);
        let mut up = vec![false; enumerated.len()];
        for step in 1..(1_u64 << enumerated.len()) {
            let bit = step.trailing_zeros() as usize;
            let j = enumerated[bit];
            let sign = if up[bit] { -1.0 } else { 1.0 };
            up[bit] = !up[bit];
            for (i, a) in current.iter_mut().enumerate() {
                *a += sign * coef(j, i);
            }
            let v = total(&current);
            if v < best.0 {
                best = (v, step ^ (step >> 1));
            }
        }
        for (bit, &j) in enumerated.iter().enumerate() {
            if best.1 >> bit & 1 == 1 {
                x[j] += 1.0;
                for (i, a) in act.iter_mut().enumerate() {
                    *a += coef(j, i);
                }
            }
        }
    }
    settle_continuous(program, shape, frozen, &rows, &mut x, &mut act);

    // Exchanges within growing pools of the binary columns nearest the margin.
    let equality: Vec<usize> = (0..total_rows)
        .filter(|&i| rows.lo[i] == rows.hi[i] && rows.lo[i].is_finite())
        .collect();
    let mut movable: Vec<usize> = (0..n)
        .filter(|&j| program.binary[j] && !frozen[j] && program.col_lo[j] != program.col_hi[j])
        .collect();
    let order = |&a: &usize, &b: &usize| {
        priority[a]
            .abs()
            .total_cmp(&priority[b].abs())
            .then(a.cmp(&b))
    };
    let mut steps_left;
    let mut covered = 0;
    for &size in &REPAIR_POOLS {
        if total(&act) == 0.0 {
            break;
        }
        if size == 0 || covered >= movable.len() {
            break;
        }
        let size = size.min(movable.len());
        covered = size;
        if size < movable.len() {
            movable.select_nth_unstable_by(size, order);
        }
        let mut pool: Vec<usize> = movable[..size].to_vec();
        pool.sort_unstable();
        // The largest effect of one pool column on each row.
        let mut reach = vec![0.0_f64; total_rows];
        for &j in &pool {
            for (i, r) in reach.iter_mut().enumerate() {
                *r = r.max(coef(j, i).abs());
            }
        }
        // Classes of interchangeable columns, by their equality coefficients.
        let mut classes: std::collections::BTreeMap<Vec<u64>, Vec<usize>> =
            std::collections::BTreeMap::new();
        for &j in &pool {
            let key: Vec<u64> = equality.iter().map(|&i| coef(j, i).to_bits()).collect();
            classes.entry(key).or_default().push(j);
        }
        steps_left = REPAIR_STEPS;
        while steps_left > 0 {
            steps_left -= 1;
            let now = total(&act);
            if now == 0.0 {
                break;
            }
            let active: Vec<usize> = (0..total_rows)
                .filter(|&i| {
                    rows.violation(i, act[i]) > 0.0 || rows.room(i, act[i]) <= 2.0 * reach[i]
                })
                .collect();
            let base: f64 = active.iter().map(|&i| rows.violation(i, act[i])).sum();
            // (new total over the active rows, column set to 1, column set to 0).
            let mut best: Option<(f64, usize, usize)> = None;
            let mut consider = |v: f64, add: usize, drop: usize| {
                let better =
                    best.is_none_or(|(bv, ba, bd)| v < bv || (v == bv && (add, drop) < (ba, bd)));
                if better {
                    best = Some((v, add, drop));
                }
            };
            for &j in &pool {
                let sign = if x[j] == 0.0 { 1.0 } else { -1.0 };
                let v: f64 = active
                    .iter()
                    .map(|&i| rows.violation(i, act[i] + sign * coef(j, i)))
                    .sum();
                if x[j] == 0.0 {
                    consider(v, j, usize::MAX);
                } else {
                    consider(v, usize::MAX, j);
                }
            }
            // Exchanges change no equality row, so only the others are evaluated.
            let free_rows: Vec<usize> = active
                .iter()
                .copied()
                .filter(|i| !equality.contains(i))
                .collect();
            let fixed_part: f64 = active
                .iter()
                .filter(|i| equality.contains(i))
                .map(|&i| rows.violation(i, act[i]))
                .sum();
            let r = free_rows.len();
            for members in classes.values() {
                let ins: Vec<usize> = members.iter().copied().filter(|&j| x[j] == 0.0).collect();
                let outs: Vec<usize> = members.iter().copied().filter(|&j| x[j] == 1.0).collect();
                if ins.is_empty() || outs.is_empty() {
                    continue;
                }
                let block = |list: &[usize]| -> Vec<f64> {
                    let mut out = Vec::with_capacity(list.len() * r);
                    for &j in list {
                        out.extend(free_rows.iter().map(|&i| coef(j, i)));
                    }
                    out
                };
                let (a_in, a_out) = (block(&ins), block(&outs));
                let here: Vec<f64> = free_rows.iter().map(|&i| act[i]).collect();
                for (q, &u) in ins.iter().enumerate() {
                    let au = &a_in[q * r..(q + 1) * r];
                    for (w, &v) in outs.iter().enumerate() {
                        let av = &a_out[w * r..(w + 1) * r];
                        let mut s = fixed_part;
                        for k in 0..r {
                            s += rows.violation(free_rows[k], here[k] + au[k] - av[k]);
                        }
                        consider(s, u, v);
                    }
                }
            }
            match best {
                Some((v, add, drop)) if v < base * (1.0 - 1e-9) => {
                    for (j, value) in [(add, 1.0), (drop, 0.0)] {
                        if j != usize::MAX {
                            for (i, a) in act.iter_mut().enumerate() {
                                *a += coef(j, i) * (value - x[j]);
                            }
                            x[j] = value;
                        }
                    }
                }
                _ => break,
            }
        }
    }
    // A point is a solution when every row holds to its tolerance.
    let act = activity(program, &x);
    let holds = (0..m).all(|i| {
        act[i] >= program.row_lo[i] - tolerance[i] && act[i] <= program.row_hi[i] + tolerance[i]
    });
    let value = objective(program, &x);
    let within = value <= ceiling + OBJECTIVE_TOL * ceiling.abs().max(1.0);
    let whole = (0..n).all(|j| !shape.whole[j] || x[j] == x[j].round());
    (holds && whole && within && reaches(value, cutoff)).then_some((x, value))
}

/// The fraction of the perturbation kept for the clean-up solve, see
/// [`tight_bound`].
const CLEANUP: f64 = 1e-2;

/// The dual bound of the solved relaxation, after a clean-up solve with a
/// hundredth of the cost perturbation, see
/// [`BoundedSimplex::scale_perturbation`]. The clean-up is limited to `10 m +
/// 100` pivots, and whatever basis it ends with gives a valid bound. The basis
/// is then optimal for the clean-up costs, or, when the clean-up did not
/// finish, solved again with the full perturbation.
fn tight_bound(lp: &mut BoundedSimplex, budget: &Budget) -> Result<DualBound, MipError> {
    let first = lp.dual_bound();
    if lp.perturbation_total() == 0.0 {
        return Ok(first);
    }
    lp.scale_perturbation(CLEANUP);
    let finished = lp.solve_within(budget.deadline, 10 * lp.rows() + 100);
    let second = lp.dual_bound();
    lp.scale_perturbation(1.0 / CLEANUP);
    if finished.is_err() {
        match lp.solve(budget.deadline) {
            Ok(()) | Err(BoundedLpError::TimeLimit) => {}
            Err(e) => return Err(MipError::Lp(e)),
        }
    }
    Ok(if second.value <= first.value {
        second
    } else {
        first
    })
}

/// How the search of one level ended.
enum End {
    /// Every node is closed, or, when not improving, a solution reached the
    /// level.
    Done,
    /// A limit ended it. No solution of value at least the threshold has a
    /// larger value than `bound`.
    Stopped { status: Status, bound: f64 },
}

/// Bound changes from the root of a level, shared by the nodes below.
struct Changes {
    list: Vec<(usize, f64, f64)>,
    parent: Option<Rc<Changes>>,
}

impl Changes {
    /// Every change from the root, oldest first.
    fn flatten(self: &Rc<Self>) -> Vec<(usize, f64, f64)> {
        let mut chain: Vec<&Rc<Self>> = Vec::new();
        let mut at = Some(self);
        while let Some(c) = at {
            chain.push(c);
            at = c.parent.as_ref();
        }
        chain
            .iter()
            .rev()
            .flat_map(|c| c.list.iter().copied())
            .collect()
    }
}

/// An open node: the bounds that define it, its parent's bound, and the basis
/// to start from.
struct Node {
    changes: Rc<Changes>,
    bound: f64,
    snapshot: Snapshot,
}

/// The value a whole column must take in every solution of value at least
/// `bound.value - gap`: the bound its reduced cost points to, when moving it
/// from there by 1 would take the dual bound down by more than `gap`. `None`
/// when it is fixed already or free to move.
fn forced_value(bound: &DualBound, lo: f64, hi: f64, j: usize, gap: f64) -> Option<f64> {
    if lo == hi {
        return None;
    }
    let r = bound.reduced_costs[j];
    (r.abs() - bound.reduced_cost_error > gap).then_some(if r > 0.0 { hi } else { lo })
}

struct Context<'s, 'p> {
    /// Whether the program holds the row of the objective's range.
    bounded: bool,
    lp: &'s mut BoundedSimplex,
    program: &'s MixedBinaryProgram<'p>,
    shape: &'s Shape,
    root: &'s DualBound,
    budget: &'s mut Budget,
}

impl Context<'_, '_> {
    /// Looks for solutions of value at least `threshold`. Without `improve`,
    /// the first one found ends the search. With it, each one found raises the
    /// threshold above its value.
    fn search(
        &mut self,
        threshold: f64,
        improve: bool,
        best: &mut Option<(Vec<f64>, f64)>,
    ) -> Result<End, MipError> {
        let n = self.program.cost.len();
        let gap = self.root.value - threshold;
        let fixed: Vec<(usize, f64)> = (0..n)
            .filter(|&j| self.shape.whole[j])
            .filter_map(|j| {
                let (lo, hi) = self.lp.bounds(j);
                forced_value(self.root, lo, hi, j, gap).map(|b| (j, b))
            })
            .collect();
        let open = (0..n)
            .filter(|&j| self.shape.whole[j] && self.lp.bounds(j).0 != self.lp.bounds(j).1)
            .count();
        let shrinks =
            !fixed.is_empty() && ((open - fixed.len()) as f64) <= COMPRESS_SHARE * open as f64;
        if shrinks || (!self.bounded && open - fixed.len() <= CUT_COLUMNS) {
            return self.search_compressed(&fixed, threshold, improve, best);
        }
        if !improve {
            let mut start = self.lp.values();
            let mut frozen = vec![false; n];
            for &(j, b) in &fixed {
                start[j] = b;
                frozen[j] = true;
            }
            let tolerance: Vec<f64> = (0..self.program.rows)
                .map(|i| self.lp.row_tolerance(i))
                .collect();
            let found = repair_search(
                self.program,
                self.shape,
                &start,
                &self.root.reduced_costs,
                &frozen,
                threshold,
                threshold,
                &tolerance,
            )
            .or_else(|| {
                (free_whole(self.program, self.shape, &frozen) <= SEARCH_COLUMNS)
                    .then(|| {
                        local_search(
                            self.program,
                            self.shape,
                            &start,
                            &frozen,
                            threshold,
                            threshold,
                            &tolerance,
                        )
                    })
                    .flatten()
            });
            if let Some(found) = found {
                *best = Some(found);
                return Ok(End::Done);
            }
        }
        self.search_in_place(&fixed, threshold, improve, best)
    }

    /// The search on the columns that are not fixed, as a program of its own.
    fn search_compressed(
        &mut self,
        fixed: &[(usize, f64)],
        threshold: f64,
        improve: bool,
        best: &mut Option<(Vec<f64>, f64)>,
    ) -> Result<End, MipError> {
        let n = self.program.cost.len();
        let mut keep = vec![true; n];
        let mut value: Vec<f64> = vec![0.0; n];
        for j in 0..n {
            let (lo, hi) = self.lp.bounds(j);
            if lo == hi {
                keep[j] = false;
                value[j] = lo;
            }
        }
        for &(j, b) in fixed {
            keep[j] = false;
            value[j] = b;
        }
        let mut sub = Reduced::new(self.program, &keep, &value);
        let goal = Goal {
            cutoff: threshold - sub.fixed_objective,
            ceiling: if improve {
                f64::INFINITY
            } else {
                threshold - sub.fixed_objective
            },
            heuristics: false,
            cuts: true,
            bounded: true,
            levels: !improve,
        };
        let mut rows = self.program.rows;
        if !self.bounded {
            sub.add_objective_row(rows, goal.cutoff, goal.ceiling);
            rows += 1;
        }
        let outcome = search_program(&sub.program(rows), &goal, self.budget)?;
        if let Some((x, _)) = outcome.best {
            let full = sub.expand(&x);
            let objective = objective(self.program, &full);
            if best.as_ref().is_none_or(|(_, v)| objective > *v) {
                *best = Some((full, objective));
            }
        }
        Ok(match outcome.status {
            Status::Optimal => End::Done,
            status => End::Stopped {
                status,
                bound: outcome.bound + sub.fixed_objective,
            },
        })
    }

    /// Branch and bound on the relaxation itself, with `fixed` as the bounds of
    /// the level. The relaxation is put back to the root afterwards.
    #[expect(
        clippy::too_many_lines,
        reason = "one loop over the nodes, whose steps share their state"
    )]
    fn search_in_place(
        &mut self,
        fixed: &[(usize, f64)],
        mut threshold: f64,
        improve: bool,
        best: &mut Option<(Vec<f64>, f64)>,
    ) -> Result<End, MipError> {
        let n = self.program.cost.len();
        let root_snapshot = self.lp.snapshot();
        let mut base_lo = Vec::with_capacity(n);
        let mut base_hi = Vec::with_capacity(n);
        for j in 0..n {
            let (lo, hi) = self.lp.bounds(j);
            base_lo.push(lo);
            base_hi.push(hi);
        }
        for &(j, b) in fixed {
            self.lp.set_bounds(j, b, b).map_err(MipError::Lp)?;
            base_lo[j] = b;
            base_hi[j] = b;
        }
        let step = self.shape.step;
        // The fixed columns may sit at their other bound in the root basis.
        match self.lp.solve(self.budget.deadline) {
            Ok(()) => {}
            Err(BoundedLpError::Infeasible) => {
                self.restore_root(&[], fixed, &root_snapshot)?;
                return Ok(End::Done);
            }
            Err(BoundedLpError::TimeLimit) => {
                self.restore_root(&[], fixed, &root_snapshot)?;
                return Ok(End::Stopped {
                    status: Status::TimeLimit,
                    bound: self.root.value,
                });
            }
            Err(e) => return Err(MipError::Lp(e)),
        }
        let mut bound = tight_bound(self.lp, self.budget)?;
        let mut current = Rc::new(Changes {
            list: Vec::new(),
            parent: None,
        });
        let mut applied: Vec<usize> = Vec::new();
        let mut stack: Vec<Node> = Vec::new();
        let mut backtracks = 0_usize;
        let mut processed = 0_usize;
        let end = 'search: loop {
            'node: {
                if bound.value < threshold {
                    break 'node;
                }
                // Columns the bound forces, where the relaxation has them
                // already, so that it stays solved.
                let mut fixings: Vec<(usize, f64, f64)> = Vec::new();
                let gap = bound.value - threshold;
                for j in (0..n).filter(|&j| self.shape.whole[j]) {
                    let (lo, hi) = self.lp.bounds(j);
                    let Some(b) = forced_value(&bound, lo, hi, j, gap) else {
                        continue;
                    };
                    if self.lp.at_bound(j) == Some(b) {
                        self.lp.set_bounds(j, b, b).map_err(MipError::Lp)?;
                        applied.push(j);
                        fixings.push((j, b, b));
                    }
                }
                // Now and then, a local search from the node's point.
                processed += 1;
                if processed.is_multiple_of(SEARCH_EVERY) {
                    let start = self.lp.values();
                    let frozen: Vec<bool> = (0..n)
                        .map(|j| {
                            let (lo, hi) = self.lp.bounds(j);
                            lo == hi
                        })
                        .collect();
                    let tolerance: Vec<f64> = (0..self.program.rows)
                        .map(|i| self.lp.row_tolerance(i))
                        .collect();
                    let ceiling = if improve {
                        floor_to(bound.value, step)
                    } else {
                        threshold
                    };
                    let found = repair_search(
                        self.program,
                        self.shape,
                        &start,
                        &bound.reduced_costs,
                        &frozen,
                        threshold,
                        ceiling,
                        &tolerance,
                    )
                    .or_else(|| {
                        (free_whole(self.program, self.shape, &frozen) <= SEARCH_COLUMNS)
                            .then(|| {
                                local_search(
                                    self.program,
                                    self.shape,
                                    &start,
                                    &frozen,
                                    threshold,
                                    ceiling,
                                    &tolerance,
                                )
                            })
                            .flatten()
                    });
                    if let Some((x, value)) = found {
                        if best.as_ref().is_none_or(|(_, b)| value > *b) {
                            *best = Some((x, value));
                        }
                        if !improve {
                            break 'search End::Done;
                        }
                        threshold = improved(value, step);
                        if bound.value < threshold {
                            break 'node;
                        }
                    }
                }
                let Some((j, v)) = branching_column(self.lp, self.shape) else {
                    let (x, value) = rounded_point(self.lp, self.program)?;
                    if reaches(value, threshold) {
                        if best.as_ref().is_none_or(|(_, b)| value > *b) {
                            *best = Some((x, value));
                        }
                        if !improve {
                            break 'search End::Done;
                        }
                        threshold = improved(value, step);
                    }
                    break 'node;
                };
                if let Some(status) = self.budget.hit() {
                    break 'search End::Stopped {
                        status,
                        bound: open_bound(&stack, bound.value),
                    };
                }
                let (lo, hi) = self.lp.bounds(j);
                // Down: `x <= floor(v)`. Up: `x >= ceil(v)`. The side the value
                // is nearer to, up on a tie, is solved first.
                let down = (lo, v.floor());
                let up = (v.ceil(), hi);
                let (near, far) = if v - v.floor() >= 0.5 {
                    (up, down)
                } else {
                    (down, up)
                };
                let parent = if fixings.is_empty() {
                    current.clone()
                } else {
                    Rc::new(Changes {
                        list: fixings,
                        parent: Some(current.clone()),
                    })
                };
                stack.push(Node {
                    changes: Rc::new(Changes {
                        list: vec![(j, far.0, far.1)],
                        parent: Some(parent.clone()),
                    }),
                    bound: bound.value,
                    snapshot: self.lp.snapshot(),
                });
                current = Rc::new(Changes {
                    list: vec![(j, near.0, near.1)],
                    parent: Some(parent),
                });
                self.lp
                    .set_bounds(j, near.0, near.1)
                    .map_err(MipError::Lp)?;
                applied.push(j);
                match self.lp.solve(self.budget.deadline) {
                    Ok(()) => {
                        self.budget.nodes += 1;
                        bound = tight_bound(self.lp, self.budget)?;
                        continue 'search;
                    }
                    Err(BoundedLpError::Infeasible) => {}
                    Err(BoundedLpError::TimeLimit) => {
                        break 'search End::Stopped {
                            status: Status::TimeLimit,
                            bound: open_bound(&stack, bound.value),
                        };
                    }
                    Err(e) => return Err(MipError::Lp(e)),
                }
            }
            // Backtrack: the latest open node, for `PLUNGE` times in a row, then
            // the one with the best bound, the latest of equals.
            loop {
                stack.retain(|node| node.bound >= threshold);
                let pick = if backtracks < PLUNGE {
                    backtracks += 1;
                    stack.len().checked_sub(1)
                } else {
                    backtracks = 0;
                    let mut pick: Option<usize> = None;
                    for (i, node) in stack.iter().enumerate() {
                        if pick.is_none_or(|p| node.bound >= stack[p].bound) {
                            pick = Some(i);
                        }
                    }
                    pick
                };
                let Some(pick) = pick else {
                    break 'search End::Done;
                };
                if let Some(status) = self.budget.hit() {
                    break 'search End::Stopped {
                        status,
                        bound: open_bound(&stack, f64::NEG_INFINITY),
                    };
                }
                let node = stack.remove(pick);
                for &j in &applied {
                    self.lp
                        .set_bounds(j, base_lo[j], base_hi[j])
                        .map_err(MipError::Lp)?;
                }
                applied.clear();
                for (j, lo, hi) in node.changes.flatten() {
                    self.lp.set_bounds(j, lo, hi).map_err(MipError::Lp)?;
                    applied.push(j);
                }
                current = node.changes;
                self.lp.restore(&node.snapshot);
                match self.lp.solve(self.budget.deadline) {
                    Ok(()) => {
                        self.budget.nodes += 1;
                        bound = tight_bound(self.lp, self.budget)?;
                        continue 'search;
                    }
                    Err(BoundedLpError::Infeasible) => {}
                    Err(BoundedLpError::TimeLimit) => {
                        break 'search End::Stopped {
                            status: Status::TimeLimit,
                            bound: open_bound(&stack, node.bound),
                        };
                    }
                    Err(e) => return Err(MipError::Lp(e)),
                }
            }
        };
        self.restore_root(&applied, fixed, &root_snapshot)?;
        Ok(end)
    }

    /// Puts the bounds of the changed columns back and solves from the root
    /// basis again.
    fn restore_root(
        &mut self,
        applied: &[usize],
        fixed: &[(usize, f64)],
        snapshot: &Snapshot,
    ) -> Result<(), MipError> {
        for j in applied.iter().copied().chain(fixed.iter().map(|f| f.0)) {
            self.lp
                .set_bounds(j, self.program.col_lo[j], self.program.col_hi[j])
                .map_err(MipError::Lp)?;
        }
        self.lp.restore(snapshot);
        match self.lp.solve(self.budget.deadline) {
            Ok(()) | Err(BoundedLpError::TimeLimit) => Ok(()),
            Err(e) => Err(MipError::Lp(e)),
        }
    }
}

/// The largest bound among the open nodes and `current`.
fn open_bound(stack: &[Node], current: f64) -> f64 {
    stack.iter().fold(current, |acc, node| acc.max(node.bound))
}

/// The fractional whole column to branch on among the basic ones: a general
/// integer column before a binary one, then the furthest from a whole value,
/// then the lowest index.
fn branching_column(lp: &BoundedSimplex, shape: &Shape) -> Option<(usize, f64)> {
    let mut branch: Option<(usize, f64, bool, f64)> = None;
    for (j, v) in lp.basic_columns() {
        if !shape.whole[j] {
            continue;
        }
        let (lo, hi) = lp.bounds(j);
        if lo == hi {
            continue;
        }
        let dist = (v - v.round()).abs();
        if dist <= INT_TOL * v.abs().max(1.0) {
            continue;
        }
        let general = shape.general[j];
        let better = match branch {
            None => true,
            Some((bj, _, bg, bd)) => {
                (general && !bg) || (general == bg && (dist > bd || (dist == bd && j < bj)))
            }
        };
        if better {
            branch = Some((j, v, general, dist));
        }
    }
    branch.map(|(j, v, _, _)| (j, v))
}

/// A program on some of the columns of another, the others fixed and moved
/// into the row bounds.
struct Reduced {
    kept: Vec<usize>,
    fixed_value: Vec<f64>,
    fixed_objective: f64,
    columns: Vec<f64>,
    cost: Vec<f64>,
    col_lo: Vec<f64>,
    col_hi: Vec<f64>,
    row_lo: Vec<f64>,
    row_hi: Vec<f64>,
    binary: Vec<bool>,
    integer: Vec<bool>,
}

impl Reduced {
    /// Keeps the columns where `keep` is true and fixes the others at `value`.
    fn new(program: &MixedBinaryProgram<'_>, keep: &[bool], value: &[f64]) -> Self {
        let n = program.cost.len();
        let m = program.rows;
        let kept: Vec<usize> = (0..n).filter(|&j| keep[j]).collect();
        let mut fixed_value = vec![0.0; n];
        let mut activity = vec![0.0; m];
        let mut fixed_objective = 0.0;
        for j in (0..n).filter(|&j| !keep[j]) {
            let v = value[j];
            fixed_value[j] = v;
            if v != 0.0 {
                fixed_objective += program.cost[j] * v;
                let column = &program.columns[j * m..(j + 1) * m];
                for i in 0..m {
                    activity[i] += column[i] * v;
                }
            }
        }
        let mut columns = Vec::with_capacity(kept.len() * m);
        for &j in &kept {
            columns.extend_from_slice(&program.columns[j * m..(j + 1) * m]);
        }
        Self {
            fixed_value,
            fixed_objective,
            columns,
            cost: kept.iter().map(|&j| program.cost[j]).collect(),
            col_lo: kept.iter().map(|&j| program.col_lo[j]).collect(),
            col_hi: kept.iter().map(|&j| program.col_hi[j]).collect(),
            row_lo: (0..m).map(|i| program.row_lo[i] - activity[i]).collect(),
            row_hi: (0..m).map(|i| program.row_hi[i] - activity[i]).collect(),
            binary: kept.iter().map(|&j| program.binary[j]).collect(),
            integer: kept
                .iter()
                .map(|&j| program.integer.get(j).copied().unwrap_or(false))
                .collect(),
            kept,
        }
    }

    /// Adds the row `lo ≤ c'x ≤ hi` on the kept columns, after the `rows` rows.
    fn add_objective_row(&mut self, rows: usize, lo: f64, hi: f64) {
        let k = self.kept.len();
        let mut columns = Vec::with_capacity(k * (rows + 1));
        for c in 0..k {
            columns.extend_from_slice(&self.columns[c * rows..(c + 1) * rows]);
            columns.push(self.cost[c]);
        }
        self.columns = columns;
        self.row_lo.push(lo);
        self.row_hi
            .push(if hi.is_finite() { hi } else { f64::INFINITY });
    }

    fn program(&self, rows: usize) -> MixedBinaryProgram<'_> {
        MixedBinaryProgram {
            rows,
            columns: &self.columns,
            cost: &self.cost,
            col_lo: &self.col_lo,
            col_hi: &self.col_hi,
            row_lo: &self.row_lo,
            row_hi: &self.row_hi,
            binary: &self.binary,
            integer: &self.integer,
        }
    }

    /// A point of the whole program from one of this.
    fn expand(&self, x: &[f64]) -> Vec<f64> {
        let mut full = self.fixed_value.clone();
        for (k, &j) in self.kept.iter().enumerate() {
            full[j] = x[k];
        }
        full
    }
}

/// A share of the whole columns. When the fixing of a level leaves at most this
/// share free, the level is searched on the free columns alone.
const COMPRESS_SHARE: f64 = 0.5;

/// Sizes of the free sets of the neighbourhood search, in binary columns.
const NEIGHBOURHOODS: [usize; 5] = [64, 256, 1024, 4096, 16384];

/// The relaxations a neighbourhood's own search may solve.
const NEIGHBOURHOOD_NODES: u64 = 250;

/// Looks for integer solutions near the root relaxation.
///
/// The binary columns are ordered by the size of their reduced cost at the
/// root, which is zero for the basic ones, so every fractional one comes first,
/// and the `f` first are kept free while the rest are fixed to their rounded
/// relaxation values. The program on the free columns, with the fixed ones moved
/// to the row bounds, is searched for a solution better than the best so far,
/// within [`NEIGHBOURHOOD_NODES`] relaxations and a quarter of the time left.
/// The sizes `f` grow through [`NEIGHBOURHOODS`]. The search ends when a
/// solution reaches `upper`.
#[expect(
    clippy::too_many_arguments,
    reason = "the program, its root and the search state, each used once"
)]
fn neighbourhood_search(
    program: &MixedBinaryProgram<'_>,
    shape: &Shape,
    lp: &mut BoundedSimplex,
    root: &DualBound,
    upper: f64,
    cutoff: f64,
    budget: &mut Budget,
    best: &mut Option<(Vec<f64>, f64)>,
) -> Result<(), MipError> {
    let n = program.cost.len();
    let m = program.rows;
    let Some((x, centre_bound, basic)) = centre(program, shape, lp, root, budget)? else {
        return Ok(());
    };
    let r = &centre_bound.reduced_costs;
    let mut order: Vec<usize> = (0..n)
        .filter(|&j| program.binary[j] && program.col_lo[j] != program.col_hi[j])
        .collect();
    order.sort_by(|&a, &b| {
        basic[b]
            .cmp(&basic[a])
            .then(r[a].abs().total_cmp(&r[b].abs()))
            .then(a.cmp(&b))
    });
    for &size in &NEIGHBOURHOODS {
        if size >= order.len() || budget.hit().is_some() {
            return Ok(());
        }
        let mut keep: Vec<bool> = program.binary.iter().map(|b| !b).collect();
        for &j in &order[..size] {
            keep[j] = true;
        }
        let value: Vec<f64> = (0..n)
            .map(|j| {
                if program.col_lo[j] == program.col_hi[j] {
                    program.col_lo[j]
                } else {
                    x[j].round()
                }
            })
            .collect();
        let sub = Reduced::new(program, &keep, &value);
        let wanted = best.as_ref().map_or(cutoff, |(_, v)| match shape.step {
            Some(g) => (v + g).max(cutoff),
            None => improvement(*v).max(cutoff),
        });
        let goal = Goal {
            cutoff: wanted - sub.fixed_objective,
            ceiling: upper - sub.fixed_objective,
            heuristics: false,
            cuts: false,
            bounded: false,
            levels: false,
        };
        let remaining = budget
            .node_limit
            .map_or(NEIGHBOURHOOD_NODES, |l| l.saturating_sub(budget.nodes))
            .min(NEIGHBOURHOOD_NODES);
        let mut own = Budget {
            node_limit: Some(remaining),
            deadline: budget.deadline.map(|dl| {
                let now = Instant::now();
                now + dl.saturating_duration_since(now) / 4
            }),
            nodes: 0,
            pivots: 0,
        };
        let outcome = search_program(&sub.program(m), &goal, &mut own);
        budget.nodes += own.nodes;
        budget.pivots += own.pivots;
        let Ok(outcome) = outcome else {
            continue;
        };
        if let Some((xs, _)) = outcome.best {
            let full = sub.expand(&xs);
            // The sub-problem's rows are the program's rows moved by the fixed
            // columns, so the point is feasible to the sub-problem's tolerance;
            // it is checked again here with ten times the row tolerance.
            let act = activity(program, &full);
            let ok = (0..m).all(|i| {
                let tol = lp.row_tolerance(i) * 10.0;
                act[i] >= program.row_lo[i] - tol && act[i] <= program.row_hi[i] + tol
            });
            let value = objective(program, &full);
            if ok && best.as_ref().is_none_or(|(_, v)| value > *v) {
                *best = Some((full, value));
            }
        }
        if best.as_ref().is_some_and(|(_, v)| reaches(*v, upper)) {
            return Ok(());
        }
    }
    Ok(())
}

/// The centre of the neighbourhoods: the root relaxation with its fractional
/// general integer columns fixed in turn to their nearest whole value, or the
/// other one when that is infeasible, and solved again. Returns its point, its
/// dual bound and which columns are basic, with the relaxation put back to the
/// root afterwards, or `None` when no whole values are feasible.
#[expect(
    clippy::type_complexity,
    reason = "a point, its bound and its basic columns"
)]
fn centre(
    program: &MixedBinaryProgram<'_>,
    shape: &Shape,
    lp: &mut BoundedSimplex,
    root: &DualBound,
    budget: &mut Budget,
) -> Result<Option<(Vec<f64>, DualBound, Vec<bool>)>, MipError> {
    let n = program.cost.len();
    let snapshot = lp.snapshot();
    let mut changed: Vec<usize> = Vec::new();
    let mut bound = root.clone();
    let mut feasible = true;
    while let Some((j, v)) = lp
        .basic_columns()
        .filter(|&(j, v)| shape.general[j] && (v - v.round()).abs() > INT_TOL * v.abs().max(1.0))
        .min_by_key(|&(j, _)| j)
    {
        let near = v.round();
        let other = if near > v { v.floor() } else { v.ceil() };
        let mut placed = false;
        for value in [near, other] {
            lp.set_bounds(j, value, value).map_err(MipError::Lp)?;
            changed.push(j);
            match lp.solve(budget.deadline) {
                Ok(()) => {
                    budget.nodes += 1;
                    placed = true;
                    break;
                }
                Err(BoundedLpError::Infeasible | BoundedLpError::TimeLimit) => {}
                Err(e) => return Err(MipError::Lp(e)),
            }
        }
        if !placed {
            feasible = false;
            break;
        }
        bound = tight_bound(lp, budget)?;
    }
    let result = feasible.then(|| {
        let mut basic = vec![false; n];
        for (j, _) in lp.basic_columns() {
            basic[j] = true;
        }
        (lp.values(), bound, basic)
    });
    if !changed.is_empty() {
        for &j in &changed {
            lp.set_bounds(j, program.col_lo[j], program.col_hi[j])
                .map_err(MipError::Lp)?;
        }
        lp.restore(&snapshot);
        match lp.solve(budget.deadline) {
            Ok(()) | Err(BoundedLpError::TimeLimit) => {}
            Err(e) => return Err(MipError::Lp(e)),
        }
    }
    Ok(result)
}

fn objective(program: &MixedBinaryProgram<'_>, x: &[f64]) -> f64 {
    x.iter().zip(program.cost).map(|(v, c)| v * c).sum()
}

/// The LP point with its whole columns rounded, checked against the rows.
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
    let objective = objective(program, &x);
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
        // A knapsack with two rows and a fractional root relaxation.
        let n = 120;
        let mut rng = Rng(5);
        let mut w = Vec::with_capacity(2 * n);
        for _ in 0..n {
            w.push(rng.uniform(1.0, 10.0));
            w.push(rng.uniform(1.0, 10.0));
        }
        let c: Vec<f64> = (0..n)
            .map(|j| (w[2 * j] * 3.0 + w[2 * j + 1]).round())
            .collect();
        let program = MixedBinaryProgram {
            rows: 2,
            columns: &w,
            cost: &c,
            col_lo: &vec![0.0; n],
            col_hi: &vec![1.0; n],
            row_lo: &[-INF, -INF],
            row_hi: &[150.0, 140.0],
            binary: &vec![true; n],
            integer: &[],
        };
        let limited = Options {
            node_limit: Some(20_000),
            ..Options::default()
        };
        let full = solve(&program, &limited).unwrap();
        for (heuristics, nodes) in [(false, 1), (true, 2), (true, 3)] {
            let cut = solve(
                &program,
                &Options {
                    node_limit: Some(nodes),
                    time_limit: None,
                    neighbourhood_search: heuristics,
                },
            );
            match cut {
                Ok(s) if s.status == Status::Optimal => {
                    assert!(heuristics, "the root relaxation is fractional");
                    assert_eq!(s.objective, full.objective);
                    assert_eq!(s.gap(), 0.0);
                }
                Ok(s) => {
                    assert_eq!(s.status, Status::NodeLimit);
                    assert!(s.gap() > 0.0);
                    assert!(s.bound >= full.objective - 1e-9);
                    assert!(s.objective <= full.bound + 1e-9);
                }
                Err(MipError::NoSolution { status, bound }) => {
                    assert_eq!(status, Status::NodeLimit);
                    assert!(bound >= full.objective - 1e-9);
                }
                Err(e) => panic!("{e}"),
            }
        }
    }

    /// A random program of the shape of profile matching: `x` binary, `t` a
    /// whole column with `Σ x - t = 0` and, per covariate,
    /// `Σ x v - (μ + tol) t ≤ 0` and `Σ x v - (μ - tol) t ≥ 0`. Or of the shape
    /// of 1:1 cardinality matching, see [`Random::cardinality`].
    struct Random {
        m: usize,
        nbin: usize,
        /// The group of each binary column in a cardinality program, empty in
        /// a profile program.
        group: Vec<bool>,
        columns: Vec<f64>,
        cost: Vec<f64>,
        col_lo: Vec<f64>,
        col_hi: Vec<f64>,
        row_lo: Vec<f64>,
        row_hi: Vec<f64>,
        binary: Vec<bool>,
        integer: Vec<bool>,
    }

    impl Random {
        fn profile(rng: &mut Rng, nbin: usize, p: usize) -> Self {
            let m = 1 + 2 * p;
            let n = nbin + 1;
            let mut columns = vec![0.0; n * m];
            for j in 0..nbin {
                columns[j * m] = 1.0;
            }
            columns[nbin * m] = -1.0;
            let mut row_lo = vec![0.0; m];
            let mut row_hi = vec![0.0; m];
            for c in 0..p {
                let v: Vec<f64> = (0..nbin).map(|_| rng.uniform(-1.0, 1.0)).collect();
                let mu = rng.uniform(-0.3, 0.3);
                let tol = rng.uniform(0.05, 0.3);
                for j in 0..nbin {
                    columns[j * m + 1 + c] = v[j];
                    columns[j * m + 1 + p + c] = v[j];
                }
                columns[nbin * m + 1 + c] = -(mu + tol);
                columns[nbin * m + 1 + p + c] = -(mu - tol);
                row_lo[1 + c] = -INF;
                row_hi[1 + p + c] = INF;
            }
            let mut cost = vec![1.0; n];
            cost[nbin] = 0.0;
            let mut col_hi = vec![1.0; n];
            col_hi[nbin] = nbin as f64;
            let mut binary = vec![true; n];
            binary[nbin] = false;
            let mut integer = vec![false; n];
            integer[nbin] = true;
            Self {
                m,
                nbin,
                group: Vec::new(),
                columns,
                cost,
                col_lo: vec![0.0; n],
                col_hi,
                row_lo,
                row_hi,
                binary,
                integer,
            }
        }

        /// A program of the shape of 1:1 cardinality matching: `x` binary in
        /// two groups, `k` a whole column with `Σ_g x - k = 0` for each group,
        /// and per covariate `|Σ_1 x v - Σ_0 x v| ≤ tol k` as two rows. The
        /// objective is `2 k`, the number of columns selected.
        fn cardinality(rng: &mut Rng, nbin: usize, p: usize) -> Self {
            let m = 2 + 2 * p;
            let n = nbin + 1;
            let group: Vec<bool> = (0..nbin).map(|j| j % 2 == 1).collect();
            let mut columns = vec![0.0; n * m];
            for (j, &g) in group.iter().enumerate() {
                columns[j * m + usize::from(g)] = 1.0;
            }
            columns[nbin * m] = -1.0;
            columns[nbin * m + 1] = -1.0;
            let mut row_lo = vec![0.0; m];
            let mut row_hi = vec![0.0; m];
            for c in 0..p {
                let shift = rng.uniform(0.0, 0.8);
                let tol = rng.uniform(0.02, 0.2);
                for (j, &g) in group.iter().enumerate() {
                    let v = rng.uniform(-1.0, 1.0) + if g { shift } else { 0.0 };
                    let coef = if g { v } else { -v };
                    columns[j * m + 2 + c] = coef;
                    columns[j * m + 2 + p + c] = coef;
                }
                columns[nbin * m + 2 + c] = -tol;
                columns[nbin * m + 2 + p + c] = tol;
                row_lo[2 + c] = -INF;
                row_hi[2 + p + c] = INF;
            }
            let mut cost = vec![0.0; n];
            cost[nbin] = 2.0;
            let mut col_hi = vec![1.0; n];
            col_hi[nbin] = (nbin / 2) as f64;
            let mut binary = vec![true; n];
            binary[nbin] = false;
            let mut integer = vec![false; n];
            integer[nbin] = true;
            Self {
                m,
                nbin,
                group,
                columns,
                cost,
                col_lo: vec![0.0; n],
                col_hi,
                row_lo,
                row_hi,
                binary,
                integer,
            }
        }

        fn program(&self) -> MixedBinaryProgram<'_> {
            MixedBinaryProgram {
                rows: self.m,
                columns: &self.columns,
                cost: &self.cost,
                col_lo: &self.col_lo,
                col_hi: &self.col_hi,
                row_lo: &self.row_lo,
                row_hi: &self.row_hi,
                binary: &self.binary,
                integer: &self.integer,
            }
        }

        /// Every integer point, with its value.
        fn points(&self) -> Vec<(Vec<f64>, f64)> {
            let (m, nbin) = (self.m, self.nbin);
            let mut out = Vec::new();
            for mask in 0..(1_u32 << nbin) {
                let mut x: Vec<f64> = (0..nbin).map(|j| f64::from(mask >> j & 1)).collect();
                let t = if self.group.is_empty() {
                    f64::from(mask.count_ones())
                } else {
                    let ones = |g: bool| {
                        (0..nbin)
                            .filter(|&j| self.group[j] == g && x[j] == 1.0)
                            .count()
                    };
                    if ones(true) != ones(false) {
                        continue;
                    }
                    ones(true) as f64
                };
                x.push(t);
                let ok = (0..m).all(|i| {
                    let a: f64 = (0..=nbin).map(|j| self.columns[j * m + i] * x[j]).sum();
                    a >= self.row_lo[i] - 1e-12 && a <= self.row_hi[i] + 1e-12
                });
                if ok {
                    let value = if self.group.is_empty() { t } else { 2.0 * t };
                    out.push((x, value));
                }
            }
            out
        }
    }

    #[test]
    fn profile_programs_agree_with_exhaustive_search() {
        let mut rng = Rng(77);
        let mut solved = 0;
        for trial in 0..300 {
            let random = Random::profile(&mut rng, 6 + trial % 9, 1 + trial % 4);
            let best = random
                .points()
                .iter()
                .map(|p| p.1)
                .fold(f64::NEG_INFINITY, f64::max);
            for heuristics in [true, false] {
                let options = Options {
                    neighbourhood_search: heuristics,
                    ..Options::default()
                };
                match solve(&random.program(), &options) {
                    Ok(s) => {
                        assert_eq!(s.status, Status::Optimal, "trial {trial}");
                        assert_eq!(s.objective, best, "trial {trial}");
                        assert!(s.max_row_violation < 1e-7, "trial {trial}");
                        solved += 1;
                    }
                    Err(MipError::Infeasible) => assert!(best.is_infinite(), "trial {trial}"),
                    Err(e) => panic!("trial {trial}: {e}"),
                }
            }
        }
        assert!(solved > 300);
    }

    /// The rounds of Gomory cuts keep every integer point of the program.
    #[test]
    fn cuts_keep_every_integer_point() {
        let mut rng = Rng(91);
        let mut added = 0;
        for trial in 0..200 {
            let random = Random::profile(&mut rng, 6 + trial % 8, 1 + trial % 3);
            let program = random.program();
            let mut lp = BoundedSimplex::new(
                program.rows,
                program.columns,
                program.cost,
                program.col_lo,
                program.col_hi,
                program.row_lo,
                program.row_hi,
            )
            .unwrap();
            lp.perturb_costs();
            if lp.solve(None).is_err() {
                continue;
            }
            let mut budget = Budget {
                node_limit: None,
                deadline: None,
                nodes: 0,
                pivots: 0,
            };
            let points = random.points();
            match cut_rounds(&program, &mut lp, false, &mut budget).unwrap() {
                Cuts::None => {}
                Cuts::Infeasible => assert!(points.is_empty(), "trial {trial}"),
                Cuts::Added(with_cuts) => {
                    let cut = with_cuts.0.program(&program);
                    let m = cut.rows;
                    for (x, _) in &points {
                        for i in program.rows..m {
                            let a: f64 = (0..x.len()).map(|j| cut.columns[j * m + i] * x[j]).sum();
                            assert!(a >= cut.row_lo[i] - 1e-9, "trial {trial}: cut {i} at {a}");
                        }
                    }
                    added += m - program.rows;
                }
            }
        }
        assert!(added > 50);
    }

    /// Every column the root's reduced costs force at a threshold takes the
    /// forced value in every integer point of at least that value.
    #[test]
    fn reduced_cost_fixing_keeps_every_good_point() {
        let mut rng = Rng(13);
        let mut fixed = 0;
        for trial in 0..200 {
            let random = Random::profile(&mut rng, 6 + trial % 8, 1 + trial % 3);
            let program = random.program();
            let points = random.points();
            let Some(best) = points.iter().map(|p| p.1).reduce(f64::max) else {
                continue;
            };
            let mut lp = BoundedSimplex::new(
                program.rows,
                program.columns,
                program.cost,
                program.col_lo,
                program.col_hi,
                program.row_lo,
                program.row_hi,
            )
            .unwrap();
            lp.perturb_costs();
            lp.solve(None).unwrap();
            let bound = lp.dual_bound();
            assert!(bound.value >= best - 1e-9, "trial {trial}");
            for threshold in [best, best - 1.0] {
                for j in 0..program.cost.len() {
                    let (lo, hi) = lp.bounds(j);
                    if let Some(b) = forced_value(&bound, lo, hi, j, bound.value - threshold) {
                        fixed += 1;
                        for (x, v) in &points {
                            if *v >= threshold {
                                assert_eq!(x[j], b, "trial {trial}: column {j}");
                            }
                        }
                    }
                }
            }
        }
        assert!(fixed > 100);
    }

    /// A point the local search returns satisfies the rows and the cutoff.
    #[test]
    fn local_search_returns_solutions() {
        let mut rng = Rng(29);
        let mut found = 0;
        for trial in 0..200 {
            let random = Random::profile(&mut rng, 8 + trial % 7, 1 + trial % 3);
            let program = random.program();
            let points = random.points();
            let Some(best) = points.iter().map(|p| p.1).reduce(f64::max) else {
                continue;
            };
            let mut lp = BoundedSimplex::new(
                program.rows,
                program.columns,
                program.cost,
                program.col_lo,
                program.col_hi,
                program.row_lo,
                program.row_hi,
            )
            .unwrap();
            lp.perturb_costs();
            lp.solve(None).unwrap();
            let shape = shape(&program);
            let tolerance: Vec<f64> = (0..program.rows).map(|i| lp.row_tolerance(i)).collect();
            let frozen = vec![false; program.cost.len()];
            for cutoff in [best, best - 1.0] {
                let start = lp.values();
                if let Some((x, v)) =
                    local_search(&program, &shape, &start, &frozen, cutoff, INF, &tolerance)
                {
                    assert!(v >= cutoff && v <= best, "trial {trial}");
                    assert!(row_violation(&program, &x) < 1e-7, "trial {trial}");
                    assert!(x.iter().all(|v| *v == v.round()), "trial {trial}");
                    found += 1;
                }
            }
        }
        assert!(found > 50);
    }

    /// Solves the relaxation of `program` and returns the repair search's
    /// solution of value in `[cutoff, upper]`, with `upper` the floored dual
    /// bound.
    fn repair_from_root(
        program: &MixedBinaryProgram<'_>,
        cutoff_below: f64,
    ) -> Option<(Vec<f64>, f64, f64)> {
        let mut lp = BoundedSimplex::new(
            program.rows,
            program.columns,
            program.cost,
            program.col_lo,
            program.col_hi,
            program.row_lo,
            program.row_hi,
        )
        .unwrap();
        lp.perturb_costs();
        lp.solve(None).ok()?;
        let shape = shape(program);
        let bound = lp.dual_bound();
        let upper = floor_to(bound.value, shape.step);
        let tolerance: Vec<f64> = (0..program.rows).map(|i| lp.row_tolerance(i)).collect();
        let frozen = vec![false; program.cost.len()];
        let cutoff = upper - cutoff_below;
        repair_search(
            program,
            &shape,
            &lp.values(),
            &bound.reduced_costs,
            &frozen,
            cutoff,
            upper,
            &tolerance,
        )
        .map(|(x, v)| (x, v, cutoff))
    }

    /// Every point the repair search returns is an integer point of the
    /// program, checked against the list of all of them, with its value in
    /// the range asked for.
    #[test]
    fn repair_search_returns_solutions_of_the_program() {
        let mut rng = Rng(311);
        let mut found = 0;
        let mut at_best = 0;
        for trial in 0..400 {
            let random = if trial % 2 == 0 {
                Random::profile(&mut rng, 8 + trial % 7, 1 + trial % 3)
            } else {
                Random::cardinality(&mut rng, 8 + trial % 8, 1 + trial % 3)
            };
            let program = random.program();
            let points = random.points();
            let best = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
            for below in [0.0, 2.0] {
                let Some((x, v, cutoff)) = repair_from_root(&program, below) else {
                    continue;
                };
                assert!(
                    v >= cutoff && v <= best,
                    "trial {trial}: {v} outside [{cutoff}, {best}]"
                );
                let integral: Vec<f64> = x.iter().map(|v| v.round()).collect();
                assert_eq!(x, integral, "trial {trial}");
                assert!(
                    points.iter().any(|(p, value)| *p == x && *value == v),
                    "trial {trial}: {x:?} is not a solution"
                );
                found += 1;
                if v == best {
                    at_best += 1;
                }
            }
        }
        assert!(found > 500, "{found} found");
        assert!(at_best > 500, "{at_best} at the optimum");
    }

    /// On a program of the shape of cardinality matching with many units the
    /// relaxation's point needs exchanges, and the repair search finds a
    /// solution at the floored bound of the relaxation, which is optimal.
    #[test]
    fn repair_search_closes_a_large_matching_program() {
        let mut rng = Rng(5);
        let random = Random::cardinality(&mut rng, 4000, 2);
        let program = random.program();
        let (x, v, cutoff) = repair_from_root(&program, 0.0).expect("a solution");
        assert_eq!(v, cutoff);
        assert!(row_violation(&program, &x) <= 1e-9 * 4000.0);
        let s = solve(&program, &Options::default()).unwrap();
        assert_eq!(s.status, Status::Optimal);
        assert_eq!(s.objective, v);
    }

    /// With its objective on the size column, as cardinality matching passes
    /// it, the solver searches by levels two apart and agrees with exhaustive
    /// search.
    #[test]
    fn cardinality_programs_agree_with_exhaustive_search() {
        let mut rng = Rng(808);
        let mut solved = 0;
        for trial in 0..300 {
            let random = Random::cardinality(&mut rng, 6 + trial % 9, 1 + trial % 3);
            assert_eq!(shape(&random.program()).step, Some(2.0));
            let best = random
                .points()
                .iter()
                .map(|p| p.1)
                .fold(f64::NEG_INFINITY, f64::max);
            for heuristics in [true, false] {
                let options = Options {
                    neighbourhood_search: heuristics,
                    ..Options::default()
                };
                match solve(&random.program(), &options) {
                    Ok(s) => {
                        assert_eq!(s.status, Status::Optimal, "trial {trial}");
                        assert_eq!(s.objective, best, "trial {trial}");
                        assert!(s.max_row_violation < 1e-7, "trial {trial}");
                        solved += 1;
                    }
                    Err(MipError::Infeasible) => assert!(best.is_infinite(), "trial {trial}"),
                    Err(e) => panic!("trial {trial}: {e}"),
                }
            }
        }
        assert!(solved > 550, "{solved} solved");
    }
}

//! Optimisers for the small problems of statistical estimation. The only
//! dependency is `rayon`, and it is optional.
//!
//! - [`bfgs`]: BFGS with the line search of Moré and Thuente, for smooth
//!   functions with a gradient.
//! - [`nelder_mead`]: the Nelder-Mead simplex method, for functions with values
//!   only.
//! - [`simplex`]: a dense primal simplex method for linear programs
//!   `max c'x  s.t.  A x = g, x ≥ 0`, with warm starts when only `c` changes.
//! - [`bounded_simplex`]: a bounded-variable dual simplex method for programs
//!   with few rows, inequality rows and up to a million bounded columns.
//! - [`branch_and_bound`]: branch and bound for programs with binary columns,
//!   on the bounded simplex.
//! - [`min_cost_flow`]: minimum-cost flow with integer data by the network
//!   simplex method, for assignment and matching problems of thousands of
//!   nodes.
//! - [`multistart`]: independent restarts of a fit, run in parallel under the
//!   `rayon` feature, with a selection that does not depend on thread count.
//!
//! Each module describes its algorithm, stopping rules and defaults, and lists
//! its references.

pub mod bfgs;
pub mod bounded_simplex;
pub mod branch_and_bound;
pub mod min_cost_flow;
pub mod multistart;
pub mod nelder_mead;
pub mod simplex;

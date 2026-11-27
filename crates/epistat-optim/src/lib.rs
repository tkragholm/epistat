//! Optimisers for the small problems of statistical estimation. The only
//! dependency is `rayon`, and it is optional.
//!
//! - [`bfgs`]: BFGS with the line search of Moré and Thuente, for smooth
//!   functions with a gradient.
//! - [`brent`]: Brent's one-dimensional minimiser on an interval, with the
//!   tolerance of R's `optimize`.
//! - [`nelder_mead`]: the Nelder-Mead simplex method, for functions with values
//!   only.
//! - [`simplex`]: a dense primal simplex method for linear programs
//!   `max c'x  s.t.  A x = g, x ≥ 0`, with warm starts when only `c` changes.
//! - [`bounded_simplex`]: a bounded-variable dual simplex method for programs
//!   with few rows, inequality rows and up to a million bounded columns.
//! - [`branch_and_bound`]: branch and bound for programs with binary columns,
//!   on the bounded simplex.
//! - [`frank_wolfe`]: the Frank-Wolfe method with an exact line search for
//!   ridge-penalised least squares on the probability simplex, the weights of
//!   synthetic control and synthetic difference-in-differences.
//! - [`min_cost_flow`]: minimum-cost flow with integer data by the network
//!   simplex method, for assignment and matching problems of thousands of
//!   nodes.
//! - [`root`]: roots in a bracket, by bisection and by the ITP method of
//!   Oliveira and Takahashi.
//! - [`multistart`]: independent restarts of a fit, run in parallel under the
//!   `rayon` feature, with a selection that does not depend on thread count.
//!
//! Each module describes its algorithm, stopping rules and defaults, and lists
//! its references. Every solver returns the crate's one [`Error`].
#![warn(missing_docs)]

mod error;

pub mod bfgs;
pub mod bounded_simplex;
pub mod branch_and_bound;
pub mod brent;
pub mod frank_wolfe;
pub mod min_cost_flow;
pub mod multistart;
pub mod nelder_mead;
pub mod root;
pub mod simplex;

pub use error::Error;

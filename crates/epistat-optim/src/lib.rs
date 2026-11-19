//! Optimisers for the small problems of statistical estimation, without
//! dependencies.
//!
//! - [`bfgs`]: BFGS with the line search of Moré and Thuente, for smooth
//!   functions with a gradient.
//! - [`nelder_mead`]: the Nelder-Mead simplex method, for functions with values
//!   only.
//! - [`simplex`]: a dense primal simplex method for linear programs
//!   `max c'x  s.t.  A x = g, x ≥ 0`, with warm starts when only `c` changes.
//!
//! - [`multistart`]: independent restarts of a fit, run in parallel under the
//!   `rayon` feature, with a selection that does not depend on thread count.
//!
//! Each module describes its algorithm, stopping rules and defaults, and lists
//! its references.

pub mod bfgs;
pub mod multistart;
pub mod nelder_mead;
pub mod simplex;

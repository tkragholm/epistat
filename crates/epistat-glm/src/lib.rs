//! Generalised linear models fitted by iteratively reweighted least squares,
//! with R's `glm.fit` as the reference: its starting values, working
//! response and weights, step halving when the deviance is not finite or the
//! mean leaves its range, and its convergence rule
//! `|dev - dev_old| / (|dev| + 0.1) < epsilon`.
//!
//! Each weighted least-squares step solves the normal equations with
//! [`epistat_linalg`], so results do not depend on the number of threads.

mod family;
mod fit;

pub use family::{Family, Link};
pub use fit::{Fit, Glm, GlmError};

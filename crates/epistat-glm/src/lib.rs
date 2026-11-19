//! Generalised linear models fitted by iteratively reweighted least squares,
//! as R's `glm.fit` does.
//!
//! [`Glm::fit`] takes a design matrix, a response and optional prior weights,
//! offset and starting coefficients, and returns a [`Fit`]. The families and
//! links, with their formulas, are described in the documentation of [`Family`]
//! and [`Link`]. The fit is checked against R's `glm()`, including the `NA` that
//! R reports for an aliased column, which is `None` here. The [`fit`] module
//! explains the method, with its formulas, edge cases and references.

mod family;
pub mod fit;

pub use family::{Family, Link};
pub use fit::{Fit, Glm, GlmError};

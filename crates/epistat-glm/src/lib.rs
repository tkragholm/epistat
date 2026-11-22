//! Generalised linear models fitted by iteratively reweighted least squares,
//! as R's `glm.fit` does.
//!
//! [`Glm::fit`] takes a design matrix, a response and optional prior weights,
//! offset and starting coefficients, and returns a [`Fit`]. The families and
//! links, with their formulas, are described in the documentation of [`Family`]
//! and [`Link`]. The fit is checked against R's `glm()`, including the `NA` that
//! R reports for an aliased column, which is `None` here. The [`fit`] module
//! explains the method, with its formulas, edge cases and references.
//!
//! [`multinom`] fits the multinomial logit, the model of the generalised
//! propensity score, as `nnet::multinom()` does.
//!
//! [`elastic_net`] fits the Gaussian and binomial elastic net by coordinate
//! descent as glmnet does, and [`park`] holds the modified Park test of a fitted
//! variance function and the classic Park test.
#![warn(missing_docs)]

mod columns;
pub mod elastic_net;
mod family;
pub mod fit;
pub mod multinom;
pub mod park;

pub use elastic_net::{ElasticNet, ElasticNetError, ElasticNetFit, ResponseScaling};
pub use family::{Family, Link, expit};
pub use fit::{Fit, Glm, GlmError};
pub use multinom::{Multinom, MultinomError, MultinomFit};

//! Linear and generalised linear mixed models with one grouping factor, as
//! `lme4` fits them, with results that do not depend on the number of
//! threads.
//!
//! - [`Lmm`] fits a linear mixed model with a random intercept and
//!   optionally one random slope, by REML or ML, as `lme4::lmer`. With one
//!   grouping factor every matrix of `lme4`'s formulation is block diagonal,
//!   so each evaluation of the profiled criterion costs `O(G q p²)` for `G`
//!   clusters, after sums over the rows taken once, and its gradient is
//!   closed form. The method is in [`lmm`].
//! - [`Glmm`] fits a binomial (logit) or Poisson (log) model with a random
//!   intercept by the Laplace approximation or adaptive Gauss-Hermite
//!   quadrature, as `lme4::glmer` with `nAGQ = 1` or more, with a closed-form
//!   gradient of the approximated likelihood. The method is in [`glmm`].
//! - [`quadrature`] holds the Gauss-Hermite rule for the standard normal
//!   density.
//!
//! Both report the fixed effects with their covariance, the variance
//! components, the log likelihood, the conditional modes of the random
//! effects (BLUPs) with their conditional variances, and the intraclass
//! correlation where it is defined. [`Lmm::evaluate`] and [`Glmm::evaluate`]
//! give the same quantities at fixed variance parameters, for profiles and for
//! checks of the approximation.
//!
//! Only one grouping factor can be given, so nested and crossed random
//! effects are not fitted: they need the sparse Cholesky factor of the whole
//! random-effects system that `lme4` uses.
#![warn(missing_docs)]

mod error;
pub mod glmm;
mod groups;
pub mod lmm;
pub mod quadrature;
mod small;

pub use error::MixedError;
pub use glmm::{Glmm, GlmmFit};
pub use lmm::{Convergence, Criterion, Lmm, LmmFit};

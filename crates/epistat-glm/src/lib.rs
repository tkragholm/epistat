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
//! [`negbin`] fits the negative binomial model with `θ` estimated by maximum
//! likelihood as `MASS::glm.nb` does. [`robust`] gives sandwich covariances of a
//! fit and risk ratios and differences by modified Poisson regression, and
//! [`within`] the linear within-family estimator with clustered variance.
//!
//! [`elastic_net`] fits the Gaussian and binomial elastic net by coordinate
//! descent as glmnet does, and [`park`] holds the modified Park test of a fitted
//! variance function and the classic Park test. [`attributable`] gives the
//! attributable fraction of an exposure from a logistic fit with a sandwich
//! standard error, as `AFglm` of the R package AF.
#![warn(missing_docs)]

pub mod attributable;
mod columns;
pub mod elastic_net;
mod family;
pub mod fit;
pub mod multinom;
pub mod negbin;
pub mod park;
pub mod robust;
pub mod within;

pub use elastic_net::{ElasticNet, ElasticNetError, ElasticNetFit, ResponseScaling};
pub use family::{Family, Link, expit};
pub use fit::{Fit, Glm, GlmError};
pub use multinom::{Multinom, MultinomError, MultinomFit};
pub use negbin::{NegBin, NegBinFit, ThetaRule};
pub use robust::{RiskFit, RiskScale, SandwichType, risk_regression, robust_covariance};
pub use within::{DfRule, Within, WithinFit, between_within};

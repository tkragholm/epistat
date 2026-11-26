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
//! [`gee`] fits generalised estimating equations for clustered outcomes as
//! `geepack::geeglm` and `gee::gee` do, and [`ordinal`] fits cumulative link
//! models for ordered outcomes as `MASS::polr` and `ordinal::clm` do, with the
//! Brant test of proportional odds.
//!
//! [`absorb`] removes any number of fixed-effect factors from columns by
//! preconditioned conjugate gradients or by `fixest`'s accelerated
//! alternating projections, and [`feols`] fits linear models on
//! top of it with the covariances of `fixest::feols`.
//!
//! [`rows`] fits the model of [`Glm::fit`] to rows formed block by block, for
//! designs too large to hold, with the same bits.
//!
//! [`elastic_net`] fits the Gaussian and binomial elastic net by coordinate
//! descent as glmnet does, and [`park`] holds the modified Park test of a fitted
//! variance function and the classic Park test. [`attributable`] gives the
//! attributable fraction of an exposure from a logistic fit with a sandwich
//! standard error, as `AFglm` of the R package AF.
#![warn(missing_docs)]

pub mod absorb;
pub mod attributable;
mod columns;
pub mod elastic_net;
mod family;
pub mod feols;
pub mod fit;
pub mod gee;
pub mod multinom;
pub mod negbin;
pub mod ordinal;
pub mod park;
pub mod robust;
pub mod rows;
pub mod within;

pub use absorb::{
    Absorb, AbsorbError, AbsorbMethod, Demeaned, FixedEffectValues, FixedEffects, Singletons,
};
pub use elastic_net::{ElasticNet, ElasticNetError, ElasticNetFit, ResponseScaling};
pub use family::{Family, Link, expit};
pub use feols::{Covariance, Feols, FeolsFit, GDf, KFixef, Ssc, Vcov};
pub use fit::{Fit, Glm, GlmError};
pub use gee::{Association, Correlation, Gee, GeeError, GeeFit, RobustVariance};
pub use multinom::{Multinom, MultinomError, MultinomFit};
pub use negbin::{NegBin, NegBinFit, ThetaRule};
pub use ordinal::{BrantTest, Ordinal, OrdinalError, OrdinalFit, OrdinalLink, brant};
pub use robust::{RiskFit, RiskScale, SandwichType, risk_regression, robust_covariance};
pub use within::{DfRule, Within, WithinFit, between_within};

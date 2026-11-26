//! Generalised linear models fitted by iteratively reweighted least squares,
//! as R's `glm.fit` does.
//!
//! [`Glm::fit`] takes a [`GlmData`], a design matrix, a response and optional
//! prior weights, offset and starting coefficients checked once, and returns a
//! [`Fit`]. [`Fit::full_rank`] gives every coefficient of a fit that converged
//! with no aliased column, and [`Glm::linear_predictor`] and [`Glm::predict`]
//! carry a fit to new rows. The families and
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
//! fit and risk ratios and differences by modified Poisson regression.
//!
//! [`gee`] fits generalised estimating equations for clustered outcomes as
//! `geepack::geeglm` and `gee::gee` do, and [`ordinal`] fits cumulative link
//! models for ordered outcomes as `MASS::polr` and `ordinal::clm` do, with the
//! Brant test of proportional odds.
//!
//! [`absorb`] removes any number of fixed-effect factors from columns by
//! preconditioned conjugate gradients or by `fixest`'s accelerated
//! alternating projections, and [`feols`] fits linear models on
//! top of it with the covariances of `fixest::feols`. The within-family
//! (sibling) estimator is [`Feols`] with one factor, and [`between_within`]
//! splits columns into family means and deviations.
//!
//! [`rows`] fits the model of [`Glm::fit`] to rows formed block by block, for
//! designs too large to hold, with the same bits.
//!
//! [`elastic_net`] fits the Gaussian and binomial elastic net by coordinate
//! descent as glmnet does, and [`park`] holds the modified Park test of a fitted
//! variance function and the classic Park test. [`attributable`] gives the
//! attributable fraction of an exposure from a logistic fit with a sandwich
//! standard error, as `AFglm` of the R package AF.
//!
//! # Conventions
//!
//! Each model takes its data as one borrowed view built by a builder that
//! checks it once: [`GlmData`] for [`Glm`], [`NegBin`] and [`ElasticNet`],
//! [`GeeData`], [`OrdinalData`] and [`MultinomData`]. Every covariance is an
//! [`epistat_core::Covariance`], which knows its dimension. Every fallible
//! function returns the one [`Error`], whose kind and stable code
//! [`epistat_core::Classify`] gives; [`GlmError`], [`ElasticNetError`] and
//! [`MultinomError`] are its deprecated names from 0.2. Results are owned and
//! `#[non_exhaustive]`, and [`expit`] is that of `epistat_stats::logistic`.
#![warn(missing_docs)]

pub mod absorb;
pub mod attributable;
mod between;
mod columns;
mod data;
pub mod elastic_net;
mod error;
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
#[cfg(test)]
mod test_support;

pub use absorb::{Absorb, AbsorbMethod, Demeaned, FixedEffectValues, FixedEffects, Singletons};
pub use between::between_within;
pub use data::{
    GeeData, GeeDataBuilder, GlmData, GlmDataBuilder, MultinomData, MultinomDataBuilder,
    OrdinalData, OrdinalDataBuilder,
};
pub use elastic_net::{ElasticNet, ElasticNetFit, ResponseScaling};
pub use epistat_stats::logistic::expit;
pub use error::Error;
pub use family::{Family, Link};
pub use feols::{Feols, FeolsCovariance, FeolsFit, GDf, KFixef, Ssc, Vcov};
pub use fit::{Fit, Glm};
pub use gee::{Association, Correlation, Gee, GeeFit, RobustVariance};
pub use multinom::{Multinom, MultinomFit};
pub use negbin::{NegBin, NegBinFit, ThetaRule};
pub use ordinal::{BrantTest, Ordinal, OrdinalFit, OrdinalLink, brant};
pub use robust::{RiskFit, RiskScale, SandwichType, risk_regression, robust_covariance};

/// The error type of epistat-glm 0.2.
#[deprecated(since = "0.3.0", note = "use epistat_glm::Error")]
pub type GlmError = Error;

/// The error type of the elastic net in epistat-glm 0.2.
#[deprecated(since = "0.3.0", note = "use epistat_glm::Error")]
pub type ElasticNetError = Error;

/// The error type of the multinomial logit in epistat-glm 0.2.
#[deprecated(since = "0.3.0", note = "use epistat_glm::Error")]
pub type MultinomError = Error;

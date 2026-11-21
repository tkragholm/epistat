//! Messages for the error types that carry no message of their own.
//!
//! Each says what is wrong and, where there is one, how to fix it.

use crate::matching::{EstimationError, MahalanobisError};
use crate::types::CriteriaValidationError;
use std::error::Error;
use std::fmt::{Display, Formatter, Result};

impl Display for CriteriaValidationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::NegativeBirthDateWindow => {
                f.write_str("the birth date window is negative; use a window of 0 days or more")
            }
            Self::ZeroMatchRatio => f.write_str("the match ratio is 0; use a ratio of 1 or more"),
        }
    }
}

impl Error for CriteriaValidationError {}

impl Display for MahalanobisError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::ZeroDimension => f.write_str("the distance has no covariates; give at least one"),
            Self::InvalidInverseCovarianceLength => f.write_str(
                "the inverse covariance matrix is not square in the number of covariates",
            ),
        }
    }
}

impl Error for MahalanobisError {}

impl Display for EstimationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::EmptyInput => f.write_str(
                "one of the two groups is empty, or too small for the distance; \
                 both groups need units",
            ),
            Self::MissingValuesNotAllowed { covariate } => write!(
                f,
                "the covariate {covariate} has missing values; \
                 remove them or choose the imputing missing-value policy"
            ),
            Self::NonFiniteNumericCovariate {
                record_id,
                covariate,
            } => write!(
                f,
                "the covariate {covariate} is not finite for unit {record_id}; \
                 give it a finite value or mark it missing"
            ),
            Self::MixedCovariateType { covariate } => write!(
                f,
                "the covariate {covariate} is numeric for some units and text for others; \
                 give it one type"
            ),
            Self::NoObservedValuesForCovariate { covariate } => {
                write!(f, "the covariate {covariate} has no observed value")
            }
            Self::NoCovariatesAvailable => {
                f.write_str("no covariate varies; name covariates that differ between units")
            }
            Self::InvalidElasticNetConfig => f.write_str(
                "the elastic net settings are out of range; \
                 alpha must lie in [0, 1] and lambda must not be negative",
            ),
            Self::InvalidGlmConfig => f.write_str(
                "the model settings are out of range; \
                 the iteration limit must be at least 1 and the tolerance positive",
            ),
            Self::InvalidCrossValidation => f.write_str(
                "the cross-validation settings are out of range, \
                 or a training set lacks one of the two groups; use at least 3 folds",
            ),
            Self::InvalidSamplingWeight { record_id } => write!(
                f,
                "the sampling weight of unit {record_id} is missing, negative or not finite; \
                 every unit needs a finite weight of 0 or more"
            ),
            Self::ReestimateWithoutDiscard => f.write_str(
                "refitting the propensity score needs a common-support rule; \
                 choose which group to discard outside the overlap",
            ),
            Self::NoCommonSupport => f.write_str(
                "the propensity scores of the two groups do not overlap, \
                 so no common-support rule can be applied",
            ),
            Self::GlmFitFailed(error) => {
                write!(f, "the propensity model could not be fitted: {error}")
            }
            Self::ElasticNetFitFailed(error) => {
                write!(f, "the elastic net could not be fitted: {error}")
            }
            Self::CbpsNeedsIntercept => f.write_str(
                "the covariate balancing propensity score needs an intercept column; \
                 include the intercept in the encoding",
            ),
            Self::CbpsNotFullRank => f.write_str(
                "the covariates are not of full rank, or there are no more units than columns; \
                 drop collinear covariates",
            ),
            Self::InvalidCbpsConfig => f.write_str(
                "the covariate balancing settings are out of range: \
                 the iteration limit must be at least 1, the tolerance positive, \
                 and both groups need weight",
            ),
            Self::CbpsFitFailed(error) => write!(
                f,
                "the covariate balancing propensity score could not be fitted: {error}"
            ),
            Self::LinearSystemSolveFailed => f.write_str(
                "a linear system in the fit could not be solved; \
                 check the covariates for perfect collinearity",
            ),
            Self::InvalidCovarianceMatrix => f.write_str(
                "the covariance matrix of the covariates is not valid; \
                 check for constant or perfectly collinear covariates",
            ),
            Self::MahalanobisConfigBuildFailed => {
                f.write_str("the distance could not be built from the covariates")
            }
            Self::ZeroWithinGroupVariance { column } => write!(
                f,
                "the column {column} does not vary within either group, \
                 so it cannot be scaled by its within-group standard deviation; \
                 drop it or do not scale"
            ),
        }
    }
}

impl Error for EstimationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::GlmFitFailed(error) => Some(error),
            Self::ElasticNetFitFailed(error) => Some(error),
            _ => None,
        }
    }
}

//! The one error of the crate.
//!
//! Invalid input that the shared checks describe, such as a column of the
//! wrong length or a number of data sets of 0, is [`Error::Input`] with the
//! shared [`InputError`]. The input errors that name a column of the data,
//! such as a method that does not apply to its type, have variants of their
//! own. The rest are failures of a regression or a draw on accepted input.
//! [`Classify`] gives a binding the kind and a stable code.

use epistat_core::{Classify, ErrorKind, InputError};
use epistat_glm::Error as GlmError;
use epistat_inference::Error as InferenceError;

/// Why an imputation could not be made.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The input cannot be used: a column, mask, method list or predictor
    /// list of the wrong length, no rows or columns, a number of data sets or
    /// iterations of 0, an observed continuous value or a survival time that
    /// is not finite, a status other than 0 and 1.
    #[error(transparent)]
    Input(#[from] InputError),
    /// An observed binary value is not 0 or 1, or a code lies outside the
    /// levels of its column.
    #[error("column {column}, row {row}: the value is not valid for the column's type")]
    Value {
        /// The column.
        column: usize,
        /// The row.
        row: usize,
    },
    /// A categorical column has fewer than two levels.
    #[error("column {column} has {levels} levels, at least 2 are needed")]
    Levels {
        /// The column.
        column: usize,
        /// Its number of levels.
        levels: usize,
    },
    /// A categorical column that is imputed or used as a predictor has a level
    /// that never occurs among its observed values.
    #[error("column {column}: level {level} does not occur among the observed values")]
    EmptyLevel {
        /// The column.
        column: usize,
        /// The level.
        level: usize,
    },
    /// A column without missing cells has a method.
    #[error("column {column} has no missing cells but has a method")]
    MethodOnComplete {
        /// The column.
        column: usize,
    },
    /// An incomplete column has no method.
    #[error("column {column} has missing cells but no method")]
    NoMethod {
        /// The column.
        column: usize,
    },
    /// Predictive mean matching is asked for with no donor.
    #[error("column {column}: predictive mean matching needs at least one donor")]
    NoDonors {
        /// The column.
        column: usize,
    },
    /// The method does not apply to the column's type, such as a logistic
    /// regression of a continuous column.
    #[error("column {column}: the method does not apply to the column's type")]
    WrongType {
        /// The column.
        column: usize,
    },
    /// A column has no observed value.
    #[error("column {column} has no observed value")]
    NothingObserved {
        /// The column.
        column: usize,
    },
    /// A predictor list names a column that does not exist, the column itself,
    /// or the same column twice.
    #[error("column {column}: predictor {predictor} is not valid")]
    Predictor {
        /// The imputed column.
        column: usize,
        /// The offending predictor.
        predictor: usize,
    },
    /// The visit order does not list each incomplete column exactly once.
    #[error("the visit order must list each incomplete column once")]
    VisitOrder,
    /// A column has too few observed rows for its regression.
    #[error("column {column}: {observed} observed rows for {parameters} parameters")]
    TooFewObserved {
        /// The column.
        column: usize,
        /// Its observed rows.
        observed: usize,
        /// The parameters of its model, with the intercept.
        parameters: usize,
    },
    /// A passive function returned a column of the wrong length or a value
    /// that is not finite at a missing cell.
    #[error("column {column}: the passive function returned an invalid column")]
    Passive {
        /// The column.
        column: usize,
    },
    /// The covariance of the coefficients has no Cholesky factor.
    #[error("column {column}: the covariance of the coefficients is not positive definite")]
    Covariance {
        /// The column.
        column: usize,
    },
    /// A regression did not converge.
    #[error("column {column}: the regression did not converge")]
    NotConverged {
        /// The column.
        column: usize,
    },
    /// A predicted value or a draw was not finite.
    #[error("column {column}: a predicted value is not finite")]
    NotFinite {
        /// The column.
        column: usize,
    },
    /// An analysis function passed to the pooling returned an error.
    #[error("the analysis of data set {set} failed: {source}")]
    Analysis {
        /// The data set.
        set: usize,
        /// The error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// A regression could not be fitted.
    #[error(transparent)]
    Glm(#[from] GlmError),
    /// A multinomial regression could not be fitted.
    #[error(transparent)]
    Multinom(GlmError),
    /// A random draw was out of range.
    #[error(transparent)]
    Draw(#[from] epistat_rng::Error),
    /// The estimates could not be pooled.
    #[error(transparent)]
    Pooling(#[from] InferenceError),
}

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(error) => error.kind(),
            Self::Glm(error) | Self::Multinom(error) => error.kind(),
            Self::Draw(error) => error.kind(),
            Self::Pooling(error) => error.kind(),
            Self::Value { .. }
            | Self::Levels { .. }
            | Self::EmptyLevel { .. }
            | Self::MethodOnComplete { .. }
            | Self::NoMethod { .. }
            | Self::NoDonors { .. }
            | Self::WrongType { .. }
            | Self::NothingObserved { .. }
            | Self::Predictor { .. }
            | Self::VisitOrder
            | Self::TooFewObserved { .. }
            | Self::Passive { .. } => ErrorKind::Input,
            Self::Covariance { .. }
            | Self::NotConverged { .. }
            | Self::NotFinite { .. }
            | Self::Analysis { .. } => ErrorKind::Numerical,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(error) => error.code(),
            Self::Glm(error) | Self::Multinom(error) => error.code(),
            Self::Draw(error) => error.code(),
            Self::Pooling(error) => error.code(),
            Self::Value { .. } => "impute.value",
            Self::Levels { .. } => "impute.levels",
            Self::EmptyLevel { .. } => "impute.empty_level",
            Self::MethodOnComplete { .. } => "impute.method_on_complete",
            Self::NoMethod { .. } => "impute.no_method",
            Self::NoDonors { .. } => "impute.no_donors",
            Self::WrongType { .. } => "impute.wrong_type",
            Self::NothingObserved { .. } => "impute.nothing_observed",
            Self::Predictor { .. } => "impute.predictor",
            Self::VisitOrder => "impute.visit_order",
            Self::TooFewObserved { .. } => "impute.too_few_observed",
            Self::Passive { .. } => "impute.passive",
            Self::Covariance { .. } => "impute.covariance",
            Self::NotConverged { .. } => "impute.not_converged",
            Self::NotFinite { .. } => "impute.not_finite",
            Self::Analysis { .. } => "impute.analysis",
        }
    }
}

/// The result of an imputation step.
pub type Result<T> = std::result::Result<T, Error>;

/// A length check, as [`epistat_core::check::same_length`].
pub(crate) fn length(what: &'static str, expected: usize, found: usize) -> Result<()> {
    Ok(epistat_core::check::same_length(what, expected, found)?)
}

/// A count that must be at least 1.
pub(crate) fn at_least_one(what: &'static str, value: usize) -> Result<()> {
    if value >= 1 {
        Ok(())
    } else {
        Err(Error::Input(InputError::Domain {
            what,
            value: 0.0,
            domain: "at least 1",
            row: None,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_distinct_and_kinds_follow_the_variant() {
        let column = 1;
        let own = [
            Error::Value { column, row: 2 },
            Error::Levels { column, levels: 1 },
            Error::EmptyLevel { column, level: 0 },
            Error::MethodOnComplete { column },
            Error::NoMethod { column },
            Error::NoDonors { column },
            Error::WrongType { column },
            Error::NothingObserved { column },
            Error::Predictor {
                column,
                predictor: 1,
            },
            Error::VisitOrder,
            Error::TooFewObserved {
                column,
                observed: 2,
                parameters: 3,
            },
            Error::Passive { column },
            Error::Covariance { column },
            Error::NotConverged { column },
            Error::NotFinite { column },
            Error::Analysis {
                set: 0,
                source: "failed".into(),
            },
        ];
        epistat_testkit::assert_codes_unique("impute", &own);
        for error in &own[..12] {
            assert_eq!(error.kind(), ErrorKind::Input, "{error:?}");
        }
        for error in &own[12..] {
            assert_eq!(error.kind(), ErrorKind::Numerical, "{error:?}");
        }
        assert_eq!(
            Error::from(InputError::Empty { what: "columns" }).code(),
            "input.empty"
        );
        assert_eq!(Error::Multinom(GlmError::Singular).code(), "glm.singular");
        assert_eq!(
            at_least_one("m", 0).map_err(|e| e.code()),
            Err("input.domain")
        );
        assert_eq!(length("z", 2, 3).map_err(|e| e.code()), Err("input.length"));
    }
}

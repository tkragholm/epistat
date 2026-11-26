use epistat_glm::{GlmError, MultinomError};
use epistat_inference::InferenceError;

/// Why an imputation could not be made.
#[derive(Debug, thiserror::Error)]
pub enum ImputeError {
    /// The number of data sets or of iterations is zero.
    #[error("{what} must be at least 1")]
    Count {
        /// `m` or `max_iterations`.
        what: &'static str,
    },
    /// A column, mask, method list or predictor list has the wrong length.
    #[error("{what} has {found} entries where {expected} were expected")]
    Length {
        /// The name of the input.
        what: &'static str,
        /// The length it needed.
        expected: usize,
        /// The length it had.
        found: usize,
    },
    /// An observed value of a continuous column is not finite, a binary value
    /// is not 0 or 1, or a code lies outside the levels of its column.
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
    /// An incomplete column has no method, or a complete column has one.
    #[error("column {column}: {reason}")]
    Method {
        /// The column.
        column: usize,
        /// What is wrong.
        reason: &'static str,
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
    /// A passive function returned a column of the wrong length or a value
    /// that is not finite.
    #[error("column {column}: the passive function returned an invalid column")]
    Passive {
        /// The column.
        column: usize,
    },
    /// A predicted value or a draw was not finite.
    #[error("column {column}: a predicted value is not finite")]
    NotFinite {
        /// The column.
        column: usize,
    },
    /// A column of this type cannot be imputed by this method.
    #[error("column {column}: the method does not apply to the column's type")]
    Unsupported {
        /// The column.
        column: usize,
    },
    /// The survival times or event indicators are not valid.
    #[error("{0}")]
    Survival(&'static str),
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
    Multinom(#[from] MultinomError),
    /// A random draw was out of range.
    #[error(transparent)]
    Draw(#[from] epistat_rng::Error),
    /// The estimates could not be pooled.
    #[error(transparent)]
    Pooling(#[from] InferenceError),
}

/// The result of an imputation step.
pub type Result<T> = std::result::Result<T, ImputeError>;

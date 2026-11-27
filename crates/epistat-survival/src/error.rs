//! The one error of the crate.
//!
//! Every estimator returns [`Error`]. A caller matches on its variants, and a
//! binding reads the [`ErrorKind`] and the stable code that [`Classify`]
//! gives.
//!
//! Input the estimator cannot use is of kind [`ErrorKind::Input`] and comes in
//! two variants. [`Error::Input`] carries the shared [`InputError`] of the
//! checks every epistat crate makes: a column of the wrong length, a value
//! that is not finite, an empty or unsorted column. [`Error::Invalid`] carries
//! a sentence for the refusals the shared checks have no variant for, such as
//! a cause that is not a code of the status or follow-up that leaves a rate
//! table. The documentation of each function calls both an input error.
//!
//! The other variants are failures of the computation on accepted input, an
//! option the method does not offer, and the errors of the crates survival
//! builds on, kept with their source.

use epistat_core::{Classify, ErrorKind, InputError};

/// Why an estimator refused its input or could not finish.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// An argument failed one of the shared checks: a wrong length, a value
    /// that is not finite, no rows, an order that is not ascending.
    #[error(transparent)]
    Input(#[from] InputError),
    /// The input cannot be used for a reason the shared checks do not name: a
    /// code outside its range, a setting without a meaning, or data that
    /// contradicts itself. The message names the argument.
    #[error("{0}")]
    Invalid(String),
    /// The options ask for something the method does not do, such as a
    /// robust variance of the exact partial likelihood.
    #[error("not available: {0}")]
    Unsupported(String),
    /// An iteration did not meet its criterion within its limit.
    #[error("no convergence in {iterations} iterations")]
    NotConverged {
        /// The iterations run.
        iterations: usize,
    },
    /// Column `column` of a design is a combination of the columns before
    /// it, where the method needs every coefficient.
    #[error("design column {column} is aliased")]
    Aliased {
        /// The first aliased column.
        column: usize,
    },
    /// An information or covariance matrix that must be inverted is
    /// singular.
    #[error("the information matrix is singular")]
    Singular,
    /// A quantity the computation needs is not defined on this input, such as
    /// a risk score that overflows or a risk set whose weights sum to 0.
    #[error("undefined: {0}")]
    Undefined(String),
    /// A nuisance model that the estimator fits, named by `model`, could not
    /// be fitted as the estimator needs it.
    #[error("{model}: {source}")]
    Model {
        /// The model, such as `"propensity model"`.
        model: String,
        /// Why it failed.
        source: epistat_glm::Error,
    },
    /// A generalised linear model could not be fitted.
    #[error(transparent)]
    Glm(#[from] epistat_glm::Error),
    /// A linear system could not be solved or a matrix not formed.
    #[error(transparent)]
    Linalg(#[from] epistat_linalg::Error),
    /// A sandwich, an influence variance or a cluster sum could not be formed.
    #[error(transparent)]
    Inference(#[from] epistat_inference::Error),
    /// The fit at a trial value of the frailty variance, tried by the search
    /// for theta, failed.
    #[error("the frailty fit at theta = {theta} failed: {source}")]
    Frailty {
        /// The trial variance.
        theta: f64,
        /// Why the fit failed.
        source: Box<Error>,
    },
    /// A root or a minimum could not be found.
    #[error(transparent)]
    Optim(#[from] epistat_optim::Error),
}

impl Error {
    /// An [`Error::Invalid`] with `message`.
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }

    /// An [`Error::Unsupported`] with `message`.
    pub(crate) fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported(message.into())
    }

    /// An [`Error::Undefined`] with `message`.
    pub(crate) fn undefined(message: impl Into<String>) -> Self {
        Self::Undefined(message.into())
    }

    /// A closure that wraps a glm error as the failure of `model`.
    pub(crate) fn model(model: impl Into<String>) -> impl FnOnce(epistat_glm::Error) -> Self {
        let model = model.into();
        move |source| Self::Model { model, source }
    }
}

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(error) => error.kind(),
            Self::Invalid(_) => ErrorKind::Input,
            Self::Unsupported(_) => ErrorKind::Unsupported,
            Self::NotConverged { .. }
            | Self::Aliased { .. }
            | Self::Singular
            | Self::Undefined(_) => ErrorKind::Numerical,
            Self::Model { source, .. } | Self::Glm(source) => source.kind(),
            Self::Linalg(error) => error.kind(),
            Self::Inference(error) => error.kind(),
            Self::Frailty { source, .. } => source.kind(),
            Self::Optim(error) => error.kind(),
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(error) => error.code(),
            Self::Invalid(_) => "survival.invalid",
            Self::Unsupported(_) => "survival.unsupported",
            Self::NotConverged { .. } => "survival.not_converged",
            Self::Aliased { .. } => "survival.aliased",
            Self::Singular => "survival.singular",
            Self::Undefined(_) => "survival.undefined",
            Self::Model { source, .. } | Self::Glm(source) => source.code(),
            Self::Linalg(error) => error.code(),
            Self::Inference(error) => error.code(),
            Self::Frailty { source, .. } => source.code(),
            Self::Optim(error) => error.code(),
        }
    }
}

/// The result of an estimator.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn own_codes_are_unique() {
        epistat_testkit::assert_codes_unique(
            "survival",
            &[
                Error::invalid("x"),
                Error::unsupported("x"),
                Error::NotConverged { iterations: 1 },
                Error::Aliased { column: 0 },
                Error::Singular,
                Error::undefined("x"),
            ],
        );
    }

    #[test]
    fn wrapped_errors_keep_their_kind_and_code() {
        let length = Error::from(InputError::Empty { what: "time" });
        assert_eq!(length.kind(), ErrorKind::Input);
        assert_eq!(length.code(), "input.empty");
        let model = Error::model("propensity model")(epistat_glm::Error::Aliased { column: 2 });
        assert_eq!(model.kind(), ErrorKind::Numerical);
        assert_eq!(model.code(), "glm.aliased");
        assert_eq!(
            model.to_string(),
            "propensity model: design column 2 is aliased"
        );
        assert_eq!(
            Error::invalid("cause 3 is not a code").kind(),
            ErrorKind::Input
        );
    }
}

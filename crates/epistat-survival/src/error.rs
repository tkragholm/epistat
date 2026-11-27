//! The one error of the crate.
//!
//! Every estimator returns [`Error`]. A caller matches on its variants, and a
//! binding reads the [`ErrorKind`] and the stable code that [`Classify`]
//! gives.
//!
//! Input the estimator cannot use is of kind [`ErrorKind::Input`].
//! [`Error::Input`] carries the shared [`InputError`] of the checks every
//! epistat crate makes: a column of the wrong length, a value that is not
//! finite, an empty or unsorted column. The refusals the shared checks have no
//! variant for are grouped by what is wrong, each with its own code and a
//! sentence that names the argument:
//!
//! | variant | code | the input |
//! |---|---|---|
//! | [`Error::Mismatch`] | `survival.mismatch` | arguments that disagree in size or shape |
//! | [`Error::OutOfRange`] | `survival.out_of_range` | a code, index or cause outside its range |
//! | [`Error::Missing`] | `survival.missing` | a column or argument the call needs is not given |
//! | [`Error::Options`] | `survival.options` | an option outside its range, or options that do not go together |
//! | [`Error::Value`] | `survival.value` | a value outside its domain |
//! | [`Error::Order`] | `survival.order` | rows, intervals or cut points not in the order or layout needed |
//! | [`Error::Inconsistent`] | `survival.inconsistent` | data that contradict themselves |
//! | [`Error::NoInformation`] | `survival.no_information` | data that leave nothing to estimate |
//! | [`Error::Fit`] | `survival.fit` | a fit that lacks what the call needs or is not of these data |
//! | [`Error::NotApplicable`] | `survival.not_applicable` | data of a form the method does not take |
//!
//! The documentation of each function calls all of these an input error.
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
    /// One of the shared checks failed on `count` rows of a column, the first
    /// of them named by `error`. The kind and the code are those of `error`,
    /// and the message gives the count, so a caller sees how much of the
    /// column to repair.
    #[error("{}", counted(error, *count))]
    Counted {
        /// The failure at the first offending row.
        error: InputError,
        /// The number of offending rows.
        count: usize,
    },
    /// Arguments that must agree in size or shape do not, in a way a single
    /// length does not describe: a design with another number of columns than
    /// the fit, a matrix of predictions that is not rows by times.
    #[error("{0}")]
    Mismatch(String),
    /// A code, an index or a cause lies outside the range the data, the fit
    /// or the options define, or names something that is not there: a cause
    /// above the number of event codes, a column outside the design, an id
    /// not among the subjects.
    #[error("{0}")]
    OutOfRange(String),
    /// A column or an argument the call needs is not given, such as the start
    /// times of a multi-state estimate or the cause when the status has
    /// several codes.
    #[error("{0}")]
    Missing(String),
    /// An option lies outside its range, or two options do not go together,
    /// such as an iteration limit of 0 or a degree of freedom for a
    /// distribution without one.
    #[error("{0}")]
    Options(String),
    /// A value of the data lies outside its domain where the shared checks
    /// cannot name it, such as a stop time not above its start or a weight
    /// other than 0 and 1 where only those select rows.
    #[error("{0}")]
    Value(String),
    /// Rows, intervals or cut points are not in the order or the layout the
    /// method needs: the rows of a subject apart, intervals that overlap,
    /// knots that do not increase.
    #[error("{0}")]
    Order(String),
    /// The data contradict themselves, such as a subject with two
    /// observations at one time or a cluster that changes within a subject.
    #[error("{0}")]
    Inconsistent(String),
    /// The data are well formed and leave the method nothing to estimate: no
    /// event, no row with a positive weight, a group without subjects.
    #[error("{0}")]
    NoInformation(String),
    /// A fit passed to a later call lacks what the call needs, such as a
    /// robust covariance or an estimate, or is not a fit of the data given.
    #[error("{0}")]
    Fit(String),
    /// The data have a form the method does not take, such as delayed entry
    /// for the Ederer I estimator or several event codes where the method
    /// needs a binary status.
    #[error("{0}")]
    NotApplicable(String),
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
    /// A step of the computation broke an invariant it relies on. This is a
    /// fault of the crate, not of the input, and is worth reporting.
    #[error("internal: {0}")]
    Internal(String),
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
    /// An [`Error::Mismatch`] with `message`.
    pub(crate) fn mismatch(message: impl Into<String>) -> Self {
        Self::Mismatch(message.into())
    }

    /// An [`Error::OutOfRange`] with `message`.
    pub(crate) fn out_of_range(message: impl Into<String>) -> Self {
        Self::OutOfRange(message.into())
    }

    /// An [`Error::Missing`] with `message`.
    pub(crate) fn missing(message: impl Into<String>) -> Self {
        Self::Missing(message.into())
    }

    /// An [`Error::Options`] with `message`.
    pub(crate) fn options(message: impl Into<String>) -> Self {
        Self::Options(message.into())
    }

    /// An [`Error::Value`] with `message`.
    pub(crate) fn value(message: impl Into<String>) -> Self {
        Self::Value(message.into())
    }

    /// An [`Error::Order`] with `message`.
    pub(crate) fn order(message: impl Into<String>) -> Self {
        Self::Order(message.into())
    }

    /// An [`Error::Inconsistent`] with `message`.
    pub(crate) fn inconsistent(message: impl Into<String>) -> Self {
        Self::Inconsistent(message.into())
    }

    /// An [`Error::NoInformation`] with `message`.
    pub(crate) fn no_information(message: impl Into<String>) -> Self {
        Self::NoInformation(message.into())
    }

    /// An [`Error::Fit`] with `message`.
    pub(crate) fn fit(message: impl Into<String>) -> Self {
        Self::Fit(message.into())
    }

    /// An [`Error::NotApplicable`] with `message`.
    pub(crate) fn not_applicable(message: impl Into<String>) -> Self {
        Self::NotApplicable(message.into())
    }

    /// An [`Error::Internal`] with `message`.
    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
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
            Self::Input(error) | Self::Counted { error, .. } => error.kind(),
            Self::Mismatch(_)
            | Self::OutOfRange(_)
            | Self::Missing(_)
            | Self::Options(_)
            | Self::Value(_)
            | Self::Order(_)
            | Self::Inconsistent(_)
            | Self::NoInformation(_)
            | Self::Fit(_)
            | Self::NotApplicable(_) => ErrorKind::Input,
            Self::Unsupported(_) => ErrorKind::Unsupported,
            Self::NotConverged { .. }
            | Self::Aliased { .. }
            | Self::Singular
            | Self::Undefined(_)
            | Self::Internal(_) => ErrorKind::Numerical,
            Self::Model { source, .. } | Self::Glm(source) => source.kind(),
            Self::Linalg(error) => error.kind(),
            Self::Inference(error) => error.kind(),
            Self::Frailty { source, .. } => source.kind(),
            Self::Optim(error) => error.kind(),
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(error) | Self::Counted { error, .. } => error.code(),
            Self::Mismatch(_) => "survival.mismatch",
            Self::OutOfRange(_) => "survival.out_of_range",
            Self::Missing(_) => "survival.missing",
            Self::Options(_) => "survival.options",
            Self::Value(_) => "survival.value",
            Self::Order(_) => "survival.order",
            Self::Inconsistent(_) => "survival.inconsistent",
            Self::NoInformation(_) => "survival.no_information",
            Self::Fit(_) => "survival.fit",
            Self::NotApplicable(_) => "survival.not_applicable",
            Self::Unsupported(_) => "survival.unsupported",
            Self::NotConverged { .. } => "survival.not_converged",
            Self::Aliased { .. } => "survival.aliased",
            Self::Singular => "survival.singular",
            Self::Undefined(_) => "survival.undefined",
            Self::Internal(_) => "survival.internal",
            Self::Model { source, .. } | Self::Glm(source) => source.code(),
            Self::Linalg(error) => error.code(),
            Self::Inference(error) => error.code(),
            Self::Frailty { source, .. } => source.code(),
            Self::Optim(error) => error.code(),
        }
    }
}

/// The message of [`Error::Counted`]: the column, the number of its rows that
/// fail and the first of them.
fn counted(error: &InputError, count: usize) -> String {
    match *error {
        InputError::NotFinite { what, row, .. } => format!(
            "{what}: {count} values are not finite; drop or resolve them before building, \
             first at row {row}"
        ),
        InputError::Domain {
            what,
            row: Some(row),
            ..
        } => format!("{what}: {count} values are negative, first at row {row}"),
        ref other => format!("{other} ({count} rows)"),
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
                Error::mismatch("x"),
                Error::out_of_range("x"),
                Error::missing("x"),
                Error::options("x"),
                Error::value("x"),
                Error::order("x"),
                Error::inconsistent("x"),
                Error::no_information("x"),
                Error::fit("x"),
                Error::not_applicable("x"),
                Error::internal("x"),
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
        let counted = Error::Counted {
            error: InputError::NotFinite {
                what: "stop",
                row: 1,
                column: None,
            },
            count: 2,
        };
        assert_eq!(counted.code(), "input.not_finite");
        assert_eq!(
            counted.to_string(),
            "stop: 2 values are not finite; drop or resolve them before building, first at row 1"
        );
        assert_eq!(
            Error::out_of_range("cause 3 is not a code").kind(),
            ErrorKind::Input
        );
    }
}

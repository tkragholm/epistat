//! The one error of the crate.
//!
//! Every model, the fixed-effect absorption, the sandwich covariances, the
//! attributable fraction and the Park tests return [`Error`]. Invalid input is
//! [`Error::Input`] with the shared [`InputError`], so a column of the wrong
//! length or a negative weight is reported as in every other epistat crate.
//! The few input errors whose payload the shared type cannot carry, such as
//! the row of a response outside the family's support, have variants of their
//! own. The other variants are failures of the computation on accepted input.
//! [`Classify`] gives a binding the kind and a stable code.

use epistat_core::{Classify, ErrorKind, InputError, check};
use epistat_linalg::MatRef;

/// Why a model could not be fitted or a quantity of a fit could not be
/// formed.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// The input cannot be used: an argument of the wrong length, a value
    /// that is not finite, a weight that is negative, a control outside its
    /// range.
    #[error(transparent)]
    Input(#[from] InputError),
    /// A response lies outside the support of the family, or a class label
    /// outside the classes.
    #[error("response {row} is {value}, outside the model's support")]
    Response {
        /// The row of the response.
        row: usize,
        /// Its value.
        value: f64,
    },
    /// Codes of a factor, of a cluster variable or of a group are not usable,
    /// or an outcome has too few categories.
    #[error("{0}")]
    Invalid(&'static str),
    /// A cluster of a GEE has two rows at one wave.
    #[error("cluster {cluster} has two rows at wave {wave}")]
    DuplicateWave {
        /// The cluster label.
        cluster: u32,
        /// The wave.
        wave: u32,
    },
    /// A column of an ordinal model's design is constant and not zero: the
    /// thresholds are the intercepts of the model.
    #[error("column {column} is constant, and the thresholds already are the intercepts")]
    Intercept {
        /// The column.
        column: usize,
    },
    /// A class or category of the outcome has no row with positive weight.
    #[error("class {class} has no row with positive weight")]
    EmptyClass {
        /// The class.
        class: usize,
    },
    /// The response does not vary, so glmnet's scaling is undefined.
    #[error("the response has no weighted variation to scale by")]
    ConstantResponse,
    /// The model does not do what the options ask: a link or family it is not
    /// defined for, a covariance it does not have.
    #[error("not available: {what}")]
    Unsupported {
        /// What was asked for.
        what: &'static str,
    },
    /// The first iterate left the model's domain, with no earlier one to halve
    /// toward.
    #[error("no valid coefficients were found; supply starting values")]
    NoValidStart,
    /// Halving a step did not bring the fit back into the domain.
    #[error("the step could not be corrected by halving")]
    StepHalving,
    /// The iteration did not meet its criterion: a fit asked to be strict
    /// that did not converge, or the demeaning of column `column`.
    #[error("no convergence in {iterations} iterations")]
    NotConverged {
        /// The iterations run.
        iterations: usize,
        /// The demeaned column, for the absorption of fixed effects.
        column: Option<usize>,
    },
    /// Column `column` of the design is a combination of the columns before
    /// it, where a fit must have every coefficient.
    #[error("design column {column} is aliased")]
    Aliased {
        /// The first aliased column.
        column: usize,
    },
    /// The information matrix is singular.
    #[error("the information matrix is singular")]
    Singular,
    /// The information matrix became singular with fitted probabilities at 0
    /// or 1: the covariates separate the classes and the likelihood has no
    /// maximum.
    #[error("the classes are separated by the covariates")]
    Separation,
    /// The estimated unstructured correlation matrix of a GEE is not positive
    /// definite.
    #[error("the estimated unstructured correlation matrix is not positive definite")]
    NotPositiveDefinite,
    /// The estimated correlation of a GEE is outside the range for which the
    /// working correlation matrix is positive definite.
    #[error(
        "the estimated correlation {0} gives a working correlation that is not positive definite"
    )]
    CorrelationRange(f64),
    /// A quantity the computation needs does not exist for this input, such
    /// as a dispersion without residual degrees of freedom or a correlation
    /// without pairs of rows.
    #[error("undefined: {0}")]
    Undefined(&'static str),
    /// The design could not be reduced to its identifiable columns.
    #[error(transparent)]
    Design(epistat_linalg::Error),
    /// A linear system could not be solved.
    #[error(transparent)]
    Solve(#[from] epistat_linalg::Error),
    /// A sandwich variance or cluster sum could not be formed.
    #[error(transparent)]
    Inference(#[from] epistat_inference::Error),
}

/// The error type of epistat-glm 0.2.
#[deprecated(since = "0.3.0", note = "use epistat_glm::Error")]
pub type GlmError = Error;

/// The error type of the elastic net in epistat-glm 0.2.
#[deprecated(since = "0.3.0", note = "use epistat_glm::Error")]
pub type ElasticNetError = Error;

/// The error type of the multinomial logit in epistat-glm 0.2.
#[deprecated(since = "0.3.0", note = "use epistat_glm::Error")]
pub type MultinomError = Error;

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(error) => error.kind(),
            Self::Design(error) | Self::Solve(error) => error.kind(),
            Self::Inference(error) => error.kind(),
            Self::Response { .. }
            | Self::Invalid(_)
            | Self::DuplicateWave { .. }
            | Self::Intercept { .. }
            | Self::EmptyClass { .. }
            | Self::ConstantResponse => ErrorKind::Input,
            Self::Unsupported { .. } => ErrorKind::Unsupported,
            Self::NoValidStart
            | Self::StepHalving
            | Self::NotConverged { .. }
            | Self::Aliased { .. }
            | Self::Singular
            | Self::Separation
            | Self::NotPositiveDefinite
            | Self::CorrelationRange(_)
            | Self::Undefined(_) => ErrorKind::Numerical,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(error) => error.code(),
            Self::Design(error) | Self::Solve(error) => error.code(),
            Self::Inference(error) => error.code(),
            Self::Response { .. } => "glm.response",
            Self::Invalid(_) => "glm.invalid",
            Self::DuplicateWave { .. } => "glm.duplicate_wave",
            Self::Intercept { .. } => "glm.intercept",
            Self::EmptyClass { .. } => "glm.empty_class",
            Self::ConstantResponse => "glm.constant_response",
            Self::Unsupported { .. } => "glm.unsupported",
            Self::NoValidStart => "glm.no_valid_start",
            Self::StepHalving => "glm.step_halving",
            Self::NotConverged { .. } => "glm.not_converged",
            Self::Aliased { .. } => "glm.aliased",
            Self::Singular => "glm.singular",
            Self::Separation => "glm.separation",
            Self::NotPositiveDefinite => "glm.not_positive_definite",
            Self::CorrelationRange(_) => "glm.correlation_range",
            Self::Undefined(_) => "glm.undefined",
        }
    }
}

/// `found` equals `expected`, or [`InputError::Length`].
pub(crate) fn length(what: &'static str, expected: usize, found: usize) -> Result<(), Error> {
    Ok(check::same_length(what, expected, found)?)
}

/// Every value is finite, or [`InputError::NotFinite`] at the first that is
/// not.
pub(crate) fn finite(what: &'static str, values: &[f64]) -> Result<(), Error> {
    Ok(check::finite(what, values)?)
}

/// Every entry of `x` is finite, or [`InputError::NotFinite`] with the first
/// row that holds an entry that is not.
pub(crate) fn finite_matrix(what: &'static str, x: MatRef<'_, f64>) -> Result<(), Error> {
    let mut first: Option<usize> = None;
    for j in 0..x.ncols() {
        let column = x.col(j);
        let limit = first.unwrap_or(x.nrows());
        if let Some(row) = (0..limit).find(|&i| !column[i].is_finite()) {
            first = Some(row);
        }
    }
    first.map_or(Ok(()), |row| Err(InputError::NotFinite { what, row }.into()))
}

/// Every weight is finite and not negative, or the error of the first that is
/// not, as [`weight_error`] gives it.
pub(crate) fn weights(what: &'static str, values: &[f64]) -> Result<(), Error> {
    values
        .iter()
        .position(|&w| !(w.is_finite() && w >= 0.0))
        .map_or(Ok(()), |row| Err(weight_error(what, row, values[row])))
}

/// The error of a weight `value` at `row` that is negative or not finite:
/// [`InputError::NotFinite`] for NaN or an infinity, [`InputError::Domain`]
/// for a negative weight.
pub(crate) fn weight_error(what: &'static str, row: usize, value: f64) -> Error {
    if value.is_finite() {
        InputError::Domain {
            what,
            value,
            domain: "[0, inf)",
        }
        .into()
    } else {
        InputError::NotFinite { what, row }.into()
    }
}

/// A convergence tolerance is positive and finite.
pub(crate) fn tolerance(what: &'static str, value: f64) -> Result<(), Error> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(InputError::Domain {
            what,
            value,
            domain: "(0, inf)",
        }
        .into())
    }
}

/// An iteration limit is at least 1.
pub(crate) fn iterations(what: &'static str, value: usize) -> Result<(), Error> {
    if value >= 1 {
        Ok(())
    } else {
        Err(InputError::Domain {
            what,
            value: 0.0,
            domain: "at least 1",
        }
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_distinct_and_kinds_follow_the_variant() {
        let own = [
            Error::Response {
                row: 0,
                value: -1.0,
            },
            Error::Invalid("codes"),
            Error::DuplicateWave {
                cluster: 1,
                wave: 2,
            },
            Error::Intercept { column: 0 },
            Error::EmptyClass { class: 1 },
            Error::ConstantResponse,
            Error::Unsupported { what: "a link" },
            Error::NoValidStart,
            Error::StepHalving,
            Error::NotConverged {
                iterations: 25,
                column: None,
            },
            Error::Aliased { column: 2 },
            Error::Singular,
            Error::Separation,
            Error::NotPositiveDefinite,
            Error::CorrelationRange(1.5),
            Error::Undefined("the dispersion"),
        ];
        epistat_testkit::assert_codes_unique("glm", &own);
        for error in &own[..6] {
            assert_eq!(error.kind(), ErrorKind::Input, "{error:?}");
        }
        assert_eq!(own[6].kind(), ErrorKind::Unsupported);
        for error in &own[7..] {
            assert_eq!(error.kind(), ErrorKind::Numerical, "{error:?}");
        }
        let wrapped = Error::from(InputError::Empty { what: "y" });
        assert_eq!(wrapped.code(), "input.empty");
        assert_eq!(
            Error::Solve(epistat_linalg::Error::Singular).code(),
            "linalg.singular"
        );
    }

    #[test]
    fn weights_name_the_first_bad_row() {
        assert!(weights("weights", &[1.0, 0.0, 2.0]).is_ok());
        assert_eq!(
            weights("weights", &[1.0, -2.0, f64::NAN]),
            Err(Error::Input(InputError::Domain {
                what: "weights",
                value: -2.0,
                domain: "[0, inf)"
            }))
        );
        assert_eq!(
            weights("weights", &[1.0, f64::INFINITY]),
            Err(Error::Input(InputError::NotFinite {
                what: "weights",
                row: 1
            }))
        );
    }
}

//! The one error of the crate.
//!
//! Invalid input is [`Error::Input`] with the shared [`InputError`], so a
//! column of the wrong length or a negative weight is reported as in every
//! other epistat crate. The input errors whose payload the shared type cannot
//! carry, such as the row of a response outside the family's support, have
//! variants of their own. The other variants are failures of the computation
//! on accepted input. [`Classify`] gives a binding the kind and a stable code.

use epistat_core::{Classify, ErrorKind, InputError};

/// Why a mixed model could not be fitted or evaluated.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// The input cannot be used: a column of the wrong length, a value that
    /// is not finite, a weight that is negative (or, for the linear model,
    /// not positive), a control or a parameter outside its range.
    #[error(transparent)]
    Input(#[from] InputError),
    /// A response lies outside the support of the family.
    #[error("response {row} is {value}, outside the family's support")]
    Response {
        /// The row.
        row: usize,
        /// The response.
        value: f64,
    },
    /// The design has no more rows than columns, so the fixed effects and
    /// the residual variance are not identified.
    #[error("{rows} rows for {coefficients} coefficients, more rows are needed")]
    TooFewRows {
        /// The rows of the design.
        rows: usize,
        /// Its columns.
        coefficients: usize,
    },
    /// The columns of the design are linearly dependent, so the fixed
    /// effects are not identified.
    #[error("the design matrix does not have full column rank")]
    RankDeficient,
    /// The model does not fit what was asked: a family and link other than
    /// the binomial with the logit and the Poisson with the log, or a random
    /// slope in a generalised linear mixed model.
    #[error("not available: {what}")]
    Unsupported {
        /// What was asked for.
        what: &'static str,
    },
    /// The conditional mode of a cluster could not be found.
    #[error("the conditional mode of cluster {cluster} did not converge")]
    Mode {
        /// The cluster, as its position among the increasing labels.
        cluster: usize,
    },
    /// The criterion is not finite at the start of the optimisation.
    #[error("the criterion is not finite at the starting values")]
    NotFiniteStart,
    /// The starting fit without random effects failed.
    #[error(transparent)]
    Start(#[from] epistat_glm::Error),
    /// A small dense system could not be formed.
    #[error(transparent)]
    Linalg(#[from] epistat_linalg::Error),
}

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(error) => error.kind(),
            Self::Start(error) => error.kind(),
            Self::Linalg(error) => error.kind(),
            Self::Response { .. } | Self::TooFewRows { .. } => ErrorKind::Input,
            Self::Unsupported { .. } => ErrorKind::Unsupported,
            Self::RankDeficient | Self::Mode { .. } | Self::NotFiniteStart => ErrorKind::Numerical,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(error) => error.code(),
            Self::Start(error) => error.code(),
            Self::Linalg(error) => error.code(),
            Self::Response { .. } => "mixed.response",
            Self::TooFewRows { .. } => "mixed.too_few_rows",
            Self::RankDeficient => "mixed.rank_deficient",
            Self::Unsupported { .. } => "mixed.unsupported",
            Self::Mode { .. } => "mixed.mode",
            Self::NotFiniteStart => "mixed.not_finite_start",
        }
    }
}

/// A convergence tolerance is positive and finite.
pub(crate) fn tolerance(value: f64) -> Result<(), Error> {
    Ok(epistat_core::check::positive("tolerance", value)?)
}

/// An iteration limit is at least 1.
pub(crate) const fn iterations(value: usize) -> Result<(), Error> {
    if value >= 1 {
        Ok(())
    } else {
        Err(Error::Input(InputError::Domain {
            what: "max_iterations",
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
        let own = [
            Error::Response {
                row: 0,
                value: -1.0,
            },
            Error::TooFewRows {
                rows: 2,
                coefficients: 2,
            },
            Error::Unsupported { what: "a link" },
            Error::RankDeficient,
            Error::Mode { cluster: 3 },
            Error::NotFiniteStart,
        ];
        epistat_testkit::assert_codes_unique("mixed", &own);
        for error in &own[..2] {
            assert_eq!(error.kind(), ErrorKind::Input, "{error:?}");
        }
        assert_eq!(own[2].kind(), ErrorKind::Unsupported);
        for error in &own[3..] {
            assert_eq!(error.kind(), ErrorKind::Numerical, "{error:?}");
        }
        assert_eq!(
            Error::from(InputError::Empty { what: "x" }).code(),
            "input.empty"
        );
        assert_eq!(
            Error::Linalg(epistat_linalg::Error::Singular).code(),
            "linalg.singular"
        );
        assert_eq!(tolerance(0.0).map_err(|e| e.code()), Err("input.domain"));
        assert_eq!(iterations(0).map_err(|e| e.code()), Err("input.domain"));
    }
}

//! The errors of the crate.
//!
//! [`Error`] is the error of the data views and of the estimators that take
//! them. A [`StaggeredPanel`] that does not build returns [`Error::Input`]
//! with the shared [`InputError`], as every epistat crate reports invalid
//! input. The other variants say why accepted data do not support an
//! estimate, which option cannot be used with them, or which lower step
//! failed, with that crate's error as the source. [`Classify`] gives a
//! binding the kind and a stable code.
//!
//! [`StaggeredPanel`]: crate::StaggeredPanel

use std::fmt;

use epistat_core::{Classify, ErrorKind, InputError};

/// Why the input of an estimator could not be used.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// A column of the wrong length, a value that is not finite, a negative
    /// weight, rows out of order, or a unit-level column that changes within
    /// a unit.
    #[error(transparent)]
    Input(#[from] InputError),
    /// No row carries a treatment indicator, so the regression has no
    /// treatment term.
    #[error("no row carries a treatment indicator")]
    NoTreatment,
    /// Never-treated controls were asked for and no unit is never treated.
    #[error("no unit is never treated in the panel")]
    NoNeverTreated,
    /// The periods span more than 2^32 values, or a period relative to
    /// treatment does not fit in 64 bits.
    #[error("the periods span more than 2^32 values")]
    Periods,
    /// The bins of an event study are empty, leave out the reference
    /// period, or have a width that is not positive.
    #[error("the bins need lo <= hi, the reference inside them, or a positive width")]
    Bins,
    /// A requested horizon is negative.
    #[error("horizon {0} is negative")]
    Horizon(i64),
    /// A requested pre-trend period is not negative.
    #[error("pre-trend period {0} is not negative")]
    Pretrend(i64),
    /// A requested reference period is not negative.
    #[error("reference period {0} is not negative")]
    Reference(i64),
    /// A column of target weights is not zero on an untreated row.
    #[error("target weight column {column} is not zero on the untreated row {row}")]
    TargetOnUntreated {
        /// The column.
        column: usize,
        /// The row.
        row: usize,
    },
    /// Some unit lacks a period where the method needs a balanced panel.
    #[error("the panel is not balanced")]
    Unbalanced,
    /// Fewer than two timing groups.
    #[error("the decomposition needs at least two timing groups")]
    Groups,
    /// The estimator does not do what the options or the data ask for, such
    /// as prior weights in the Goodman-Bacon decomposition.
    #[error("not available: {what}")]
    Unsupported {
        /// What was asked for.
        what: &'static str,
    },
    /// A regression, an absorption of fixed effects or a covariance of the
    /// fit failed.
    #[error(transparent)]
    Glm(#[from] epistat_glm::Error),
    /// An influence function could not be formed.
    #[error(transparent)]
    Inference(#[from] epistat_inference::Error),
    /// After the sample rules of dCDH no period has both switchers and
    /// controls (Design Restriction 1 of de Chaisemartin and
    /// D'Haultfoeuille fails), so no effect can be estimated.
    #[error("no treatment effect can be estimated")]
    NoEffect,
    /// After the sample rules of dCDH a period inside the panel has no rows.
    #[error("a period inside the panel has no rows after the sample rules")]
    EmptyPeriod,
    /// The controls of dCDH cannot be used for a baseline treatment.
    #[error("controls at baseline treatment {baseline}: {reason}")]
    Controls {
        /// The baseline treatment.
        baseline: f64,
        /// What is wrong.
        reason: &'static str,
    },
    /// The regressors of a regression are collinear, as the covariates of
    /// dCDH's heterogeneity regression with its cells.
    #[error("collinear: {what}")]
    Collinear {
        /// The regression.
        what: &'static str,
    },
}

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(error) => error.kind(),
            Self::Glm(error) => error.kind(),
            Self::Inference(error) => error.kind(),
            Self::Collinear { .. } => ErrorKind::Numerical,
            Self::NoEffect
            | Self::EmptyPeriod
            | Self::Controls { .. }
            | Self::NoTreatment
            | Self::NoNeverTreated
            | Self::Periods
            | Self::Bins
            | Self::Horizon(_)
            | Self::Pretrend(_)
            | Self::Reference(_)
            | Self::TargetOnUntreated { .. }
            | Self::Unbalanced
            | Self::Groups => ErrorKind::Input,
            Self::Unsupported { .. } => ErrorKind::Unsupported,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(error) => error.code(),
            Self::Glm(error) => error.code(),
            Self::Inference(error) => error.code(),
            Self::NoEffect => "did.no_effect",
            Self::EmptyPeriod => "did.empty_period",
            Self::Controls { .. } => "did.controls",
            Self::Collinear { .. } => "did.collinear",
            Self::NoTreatment => "did.no_treatment",
            Self::NoNeverTreated => "did.no_never_treated",
            Self::Periods => "did.periods",
            Self::Bins => "did.bins",
            Self::Horizon(_) => "did.horizon",
            Self::Pretrend(_) => "did.pretrend",
            Self::Reference(_) => "did.reference",
            Self::TargetOnUntreated { .. } => "did.target_on_untreated",
            Self::Unbalanced => "did.unbalanced",
            Self::Groups => "did.groups",
            Self::Unsupported { .. } => "did.unsupported",
        }
    }
}
/// Errors from internal estimation steps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum InternalDidError {
    /// An estimation step failed, with a description.
    Estimation(String),
    /// An iterative procedure did not converge, with a description.
    Convergence(String),
}

impl fmt::Display for InternalDidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Estimation(message) => write!(f, "Estimation failed: {message}"),
            Self::Convergence(message) => write!(f, "Convergence error: {message}"),
        }
    }
}

impl std::error::Error for InternalDidError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_unique_and_kinds_follow_the_cause() {
        let errors = [
            Error::NoTreatment,
            Error::NoNeverTreated,
            Error::Periods,
            Error::Bins,
            Error::Horizon(-1),
            Error::Pretrend(0),
            Error::Reference(1),
            Error::TargetOnUntreated { column: 0, row: 3 },
            Error::Unbalanced,
            Error::Groups,
            Error::Unsupported { what: "weights" },
            Error::NoEffect,
            Error::EmptyPeriod,
            Error::Controls {
                baseline: 0.0,
                reason: "collinear",
            },
            Error::Collinear { what: "covariates" },
        ];
        epistat_testkit::assert_codes_unique("did", &errors);
        assert_eq!(Error::Bins.kind(), ErrorKind::Input);
        assert_eq!(
            Error::Unsupported { what: "weights" }.kind(),
            ErrorKind::Unsupported
        );
        let glm = Error::from(epistat_glm::Error::Singular);
        assert_eq!(
            (glm.kind(), glm.code()),
            (ErrorKind::Numerical, "glm.singular")
        );
    }
}

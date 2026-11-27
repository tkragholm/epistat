//! The errors of the crate.
//!
//! [`Error`] is the error of the data views and the estimators on them: a
//! [`StaggeredPanel`] that does not build returns [`Error::Input`] with the
//! shared [`InputError`], as every epistat crate reports invalid input.
//! [`Classify`] gives a binding the kind and a stable code.
//!
//! [`StaggeredPanel`]: crate::StaggeredPanel

use std::fmt;

use epistat_core::{Classify, ErrorKind, InputError};

use crate::methods::att_gt::SkipReason;

/// Why the input of an estimator could not be used.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// A column of the wrong length, a value that is not finite, a negative
    /// weight, rows out of order, or a unit-level column that changes within
    /// a unit.
    #[error(transparent)]
    Input(#[from] InputError),
    /// An error of the shared inference routines, with its source.
    #[error(transparent)]
    Inference(#[from] epistat_inference::Error),
    /// The combination of options asks for something the method does not do.
    #[error("not supported: {what}")]
    Unsupported {
        /// What was asked for.
        what: &'static str,
    },
    /// No ATT(g,t) cell could be estimated.
    #[error("no ATT(g,t) cell could be estimated")]
    NoEstimableCells,
    /// The comparison group needs never-treated units and the sample has
    /// none.
    #[error("the comparison group needs never-treated units and the sample has none")]
    NoNeverTreated,
    /// The never-treated comparison group has fewer rows per period than the
    /// number of covariates plus 5, which `did::att_gt` refuses.
    #[error(
        "the never-treated group is too small to serve as a reliable comparison \
         ({per_period} rows per period, fewer than {required}, the number of covariates \
         plus 5); the not-yet-treated comparison group adds the units not yet treated"
    )]
    SmallNeverTreatedGroup {
        /// Never-treated rows divided by the number of periods.
        per_period: f64,
        /// The number of covariates plus 5.
        required: usize,
    },
    /// A cell could not be fitted and failed cells are not skipped.
    #[error("the ATT(g,t) cell (group {group}, time {time}, base {base}) failed: {reason}")]
    CellFailed {
        /// The cell's cohort.
        group: i64,
        /// The cell's later period.
        time: i64,
        /// The cell's base period.
        base: i64,
        /// Why the cell failed.
        reason: SkipReason,
    },
    /// The efficient estimator found a cohort without pre-treatment periods.
    #[error("cohort {group} has no pre-treatment period")]
    NoPrePeriods {
        /// The cohort.
        group: i64,
    },
    /// The efficient estimator found a cohort without post-treatment periods.
    #[error("cohort {group} has no post-treatment period")]
    NoPostPeriods {
        /// The cohort.
        group: i64,
    },
    /// An aggregation, band, pre-test or event study was asked of an ATT(g,t)
    /// whose influence function was not kept.
    #[error("the ATT(g,t) influence function was not kept, so it cannot be aggregated")]
    NoInfluence,
    /// No post-treatment cell, or for a fixed-weight event study no cell,
    /// survives the restrictions of a summary.
    #[error("no post-treatment cell survives the requested aggregation")]
    NoPostTreatmentCells,
    /// No pre-treatment cell has a standard error to test.
    #[error("no pre-treatment cell with a standard error to test")]
    NoPreTreatmentCells,
    /// A system to solve was singular.
    #[error("a system to solve is singular")]
    Singular,
    /// A result was not finite.
    #[error("a result is not finite")]
    NotFinite,
}

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(error) => error.kind(),
            Self::Inference(error) => error.kind(),
            Self::Unsupported { .. } | Self::NoInfluence => ErrorKind::Unsupported,
            Self::NoEstimableCells
            | Self::NoNeverTreated
            | Self::SmallNeverTreatedGroup { .. }
            | Self::NoPrePeriods { .. }
            | Self::NoPostPeriods { .. }
            | Self::NoPostTreatmentCells
            | Self::NoPreTreatmentCells => ErrorKind::Input,
            Self::CellFailed { .. } | Self::Singular | Self::NotFinite => ErrorKind::Numerical,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(error) => error.code(),
            Self::Inference(error) => error.code(),
            Self::Unsupported { .. } => "did.unsupported",
            Self::NoEstimableCells => "did.no_estimable_cells",
            Self::NoNeverTreated => "did.no_never_treated",
            Self::SmallNeverTreatedGroup { .. } => "did.small_never_treated",
            Self::CellFailed { .. } => "did.cell_failed",
            Self::NoPrePeriods { .. } => "did.no_pre_periods",
            Self::NoPostPeriods { .. } => "did.no_post_periods",
            Self::NoInfluence => "did.no_influence",
            Self::NoPostTreatmentCells => "did.no_post_treatment_cells",
            Self::NoPreTreatmentCells => "did.no_pre_treatment_cells",
            Self::Singular => "did.singular",
            Self::NotFinite => "did.not_finite",
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

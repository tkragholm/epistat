use std::fmt;

/// Errors from internal estimation steps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InternalDidError {
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

/// Errors raised while building or validating a panel dataset.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum DatasetError {
    /// A field is invalid, given by its name and the reason.
    #[error("invalid field '{0}': {1}")]
    InvalidField(String, String),

    /// The panel is unbalanced.
    #[error("unbalanced panel: units missing observations in periods {missing:?}")]
    UnbalancedPanel {
        /// All distinct periods of the dataset, in increasing order. The unit that
        /// failed the check has fewer rows than this has entries.
        missing: Vec<i32>,
    },

    /// A unit has an invalid treatment timing.
    #[error("invalid treatment timing for unit {unit}: {reason}")]
    InvalidTreatmentTiming {
        /// Identifier of the unit.
        unit: String,
        /// Why the timing is invalid.
        reason: String,
    },

    /// A treated cohort has no baseline period.
    #[error("no baseline period found for cohort {cohort}")]
    NoBaselinePeriod {
        /// Cohort without a baseline period.
        cohort: i32,
    },

    /// An outcome is not finite, for a unit in a period.
    #[error("missing outcome for unit {unit} in period {period}")]
    MissingOutcome {
        /// Identifier of the unit.
        unit: String,
        /// Period without an outcome.
        period: i32,
    },

    /// The number of cluster labels differs from the number of rows.
    #[error("cluster label mismatch: expected {expected} labels, got {actual}")]
    ClusterMismatch {
        /// Number of cluster labels expected, one per row.
        expected: usize,
        /// Number of cluster labels given.
        actual: usize,
    },
}

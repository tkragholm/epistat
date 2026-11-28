//! The error of the crate.
//!
//! [`Error`] says why an event study, a target or the options could not be
//! used ([`Error::Input`], with the shared [`InputError`] that every epistat
//! crate reports), which combination of restriction and method the analysis
//! does not do ([`Error::Unsupported`]), or why the computation failed on
//! accepted input. [`Classify`] gives a binding the kind and a stable code.

use epistat_core::{Classify, ErrorKind, InputError};

/// Why a sensitivity analysis or a pre-trend power calculation failed.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// An event study, a target or an option that cannot be used: a target
    /// of the wrong length or with no non-zero weight, a period that is not
    /// in the event study, a sensitivity value that is negative or not
    /// finite, a covariance entry that is undefined, too few pre-treatment
    /// coefficients.
    #[error(transparent)]
    Input(#[from] InputError),
    /// The restriction, its variant and the method do not go together, such
    /// as the fixed-length interval under a relative-magnitude restriction.
    #[error("not available: {what}")]
    Unsupported {
        /// What was asked for.
        what: &'static str,
    },
    /// A covariance block that must be positive definite is not.
    #[error("the covariance of the {what} is not positive definite")]
    NotPositiveDefinite {
        /// The block.
        what: &'static str,
    },
    /// No slope in the search interval gives the pre-test the requested
    /// power.
    #[error("no slope in [0, {upper}] has power {target}; the power there is {power_at_upper}")]
    NoSlope {
        /// The requested power.
        target: f64,
        /// The upper end of the search interval.
        upper: f64,
        /// The power at the upper end.
        power_at_upper: f64,
    },
    /// The pre-test passes with probability zero under the hypothesised
    /// violation, so the means after passing it are undefined.
    #[error("the pre-test passes with probability zero")]
    PassProbabilityZero,
    /// A linear or conic program of the analysis did not solve.
    #[error("the {program} program did not solve: {status}")]
    Solver {
        /// The program.
        program: &'static str,
        /// How the solver ended.
        status: SolverStatus,
    },
    /// A critical value or quantile of the analysis could not be found.
    #[error("the {what} could not be computed")]
    CriticalValue {
        /// The critical value.
        what: &'static str,
    },
    /// A variance, direction or basis that must be non-degenerate is not,
    /// such as a target whose variance is zero.
    #[error("degenerate: {what}")]
    Degenerate {
        /// What is degenerate.
        what: &'static str,
    },
    /// The conditional test accepted no value of its grid, so the confidence
    /// set is empty at that grid.
    #[error("the conditional test accepted no value of the grid")]
    NoAcceptedValues,
    /// Internal objects of the analysis disagree; a defect of the crate.
    #[error("internal inconsistency: {what}")]
    Internal {
        /// What disagrees.
        what: &'static str,
    },
    /// A simulated critical value failed.
    #[error(transparent)]
    Inference(#[from] epistat_inference::Error),
    /// A program or optimisation of epistat-optim failed.
    #[error(transparent)]
    Optim(#[from] epistat_optim::Error),
}

/// How a linear or conic program ended without a solution.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverStatus {
    /// The solver could not be set up for the program.
    Setup,
    /// The program is infeasible.
    Infeasible,
    /// The program is unbounded.
    Unbounded,
    /// The solver stopped without a solution: too many iterations,
    /// insufficient progress or a numerical error.
    NotConverged,
}

impl std::fmt::Display for SolverStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Setup => "the solver could not be set up",
            Self::Infeasible => "infeasible",
            Self::Unbounded => "unbounded",
            Self::NotConverged => "stopped without a solution",
        })
    }
}

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(_) => ErrorKind::Input,
            Self::Unsupported { .. } => ErrorKind::Unsupported,
            Self::Inference(error) => error.kind(),
            Self::Optim(error) => error.kind(),
            Self::NotPositiveDefinite { .. }
            | Self::NoSlope { .. }
            | Self::PassProbabilityZero
            | Self::Solver { .. }
            | Self::CriticalValue { .. }
            | Self::Degenerate { .. }
            | Self::NoAcceptedValues
            | Self::Internal { .. } => ErrorKind::Numerical,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(error) => error.code(),
            Self::Unsupported { .. } => "honestdid.unsupported",
            Self::NotPositiveDefinite { .. } => "honestdid.not_positive_definite",
            Self::NoSlope { .. } => "honestdid.no_slope",
            Self::PassProbabilityZero => "honestdid.pass_probability_zero",
            Self::Solver { .. } => "honestdid.solver",
            Self::CriticalValue { .. } => "honestdid.critical_value",
            Self::Degenerate { .. } => "honestdid.degenerate",
            Self::NoAcceptedValues => "honestdid.no_accepted_values",
            Self::Internal { .. } => "honestdid.internal",
            Self::Inference(error) => error.code(),
            Self::Optim(error) => error.code(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_distinct_and_kinds_follow_the_variant() {
        let errors = [
            Error::Unsupported { what: "x" },
            Error::NotPositiveDefinite { what: "x" },
            Error::NoSlope {
                target: 0.5,
                upper: 1.0,
                power_at_upper: 0.1,
            },
            Error::PassProbabilityZero,
            Error::Solver {
                program: "x",
                status: SolverStatus::Infeasible,
            },
            Error::CriticalValue { what: "x" },
            Error::Degenerate { what: "x" },
            Error::NoAcceptedValues,
            Error::Internal { what: "x" },
        ];
        epistat_testkit::assert_codes_unique("honestdid", &errors);
        assert_eq!(errors[0].kind(), ErrorKind::Unsupported);
        assert_eq!(errors[4].kind(), ErrorKind::Numerical);
        let input = Error::from(InputError::Empty { what: "x" });
        assert_eq!(input.kind(), ErrorKind::Input);
        assert_eq!(input.code(), "input.empty");
        let optim = Error::from(epistat_optim::Error::Infeasible);
        assert_eq!(optim.code(), epistat_optim::Error::Infeasible.code());
    }
}

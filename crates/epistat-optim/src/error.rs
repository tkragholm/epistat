//! The one error of the crate's solvers.

use std::fmt;

use epistat_core::{Classify, ErrorKind, InputError};

use crate::branch_and_bound::Status;

/// Why a solver refused its input or stopped without a solution.
///
/// Every solver of the crate returns this type and uses the variants that
/// apply to it; each solver's documentation lists them.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The input cannot be used: a wrong length, a value that is not finite,
    /// a bound above its upper bound, an interval that is empty.
    Input(InputError),
    /// The ends of a bracket do not have values of opposite sign, or one of
    /// them is NaN, so the bracket need not hold a root.
    NoSignChange,
    /// A cost, capacity or supply is too large for 64-bit arithmetic, or
    /// there are more nodes or arcs than the solver indexes.
    Overflow,
    /// No point satisfies the constraints.
    Infeasible,
    /// The objective increases without bound.
    Unbounded,
    /// A non-basic column of a bounded program has a reduced cost its bounds
    /// cannot absorb, so the starting basis is not dual feasible. The program
    /// is unbounded or infeasible.
    DualInfeasible,
    /// A basis could not be factorised.
    Singular,
    /// The pivot budget ran out.
    IterationLimit,
    /// The deadline passed.
    TimeLimit,
    /// A limit ended branch and bound before any integer solution was found.
    NoSolution {
        /// How the search ended.
        status: Status,
        /// The best upper bound proved.
        bound: f64,
    },
    /// An integer solution broke a row by more than rounding explains.
    Numerical,
    /// A search direction of BFGS is not a descent direction, as when a
    /// starting inverse Hessian is not positive definite.
    NotDescent,
    /// A function returned a value at `x` that a search cannot use: NaN, or
    /// `-inf` where a minimum is sought.
    NonFinite {
        /// The point at which the value was returned.
        x: f64,
    },
    /// An objective value of Nelder-Mead at a trial point cannot be ordered,
    /// such as NaN.
    Unordered,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(error) => error.fmt(f),
            Self::NoSignChange => {
                f.write_str("the function does not change sign between the ends of the bracket")
            }
            Self::Overflow => f.write_str("a cost, capacity or supply is too large to solve"),
            Self::Infeasible => f.write_str("the problem has no feasible point"),
            Self::Unbounded => f.write_str("the objective is unbounded"),
            Self::DualInfeasible => f.write_str("the starting basis is not dual feasible"),
            Self::Singular => f.write_str("a basis could not be factorised"),
            Self::IterationLimit => f.write_str("the pivot budget ran out"),
            Self::TimeLimit => f.write_str("the deadline passed"),
            Self::NoSolution { status, bound } => {
                write!(f, "no integer solution after {status:?}, bound {bound}")
            }
            Self::Numerical => {
                f.write_str("an integer solution broke a row by more than rounding explains")
            }
            Self::NotDescent => f.write_str("the search direction is not a descent direction"),
            Self::NonFinite { x } => write!(f, "the function is not usable at {x}"),
            Self::Unordered => f.write_str("the objective returned a value that cannot be ordered"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Input(error) => Some(error),
            _ => None,
        }
    }
}

impl From<InputError> for Error {
    fn from(error: InputError) -> Self {
        Self::Input(error)
    }
}

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(error) => error.kind(),
            Self::NoSignChange | Self::Overflow => ErrorKind::Input,
            _ => ErrorKind::Numerical,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(error) => error.code(),
            Self::NoSignChange => "optim.no_sign_change",
            Self::Overflow => "optim.overflow",
            Self::Infeasible => "optim.infeasible",
            Self::Unbounded => "optim.unbounded",
            Self::DualInfeasible => "optim.dual_infeasible",
            Self::Singular => "optim.singular",
            Self::IterationLimit => "optim.iteration_limit",
            Self::TimeLimit => "optim.time_limit",
            Self::NoSolution { .. } => "optim.no_solution",
            Self::Numerical => "optim.numerical",
            Self::NotDescent => "optim.not_descent",
            Self::NonFinite { .. } => "optim.non_finite",
            Self::Unordered => "optim.unordered",
        }
    }
}

/// The InputError of a scalar outside its domain.
pub(crate) const fn domain(what: &'static str, value: f64, domain: &'static str) -> Error {
    Error::Input(InputError::Domain {
        what,
        value,
        domain,
        row: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_distinct_and_kinds_follow_the_cause() {
        let all = [
            Error::NoSignChange,
            Error::Overflow,
            Error::Infeasible,
            Error::Unbounded,
            Error::DualInfeasible,
            Error::Singular,
            Error::IterationLimit,
            Error::TimeLimit,
            Error::NoSolution {
                status: Status::NodeLimit,
                bound: 1.0,
            },
            Error::Numerical,
            Error::NotDescent,
            Error::NonFinite { x: 0.5 },
            Error::Unordered,
        ];
        let mut codes: Vec<&str> = all.iter().map(Classify::code).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), all.len());
        assert!(codes.iter().all(|c| c.starts_with("optim.")));
        assert_eq!(Error::NoSignChange.kind(), ErrorKind::Input);
        assert_eq!(Error::Infeasible.kind(), ErrorKind::Numerical);
        let input = domain("tol", -1.0, "(0, inf)");
        assert_eq!(
            (input.kind(), input.code()),
            (ErrorKind::Input, "input.domain")
        );
    }
}

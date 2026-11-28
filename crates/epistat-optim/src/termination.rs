//! Why an iterative solver stopped, and the check that it converged.
//!
//! BFGS, Nelder-Mead, Brent's minimiser, bisection, ITP and Frank-Wolfe
//! return `Ok` when they stop at their iteration limit, as R's `optim`,
//! `optimize` and `uniroot` return with a flag or a warning. Their results
//! carry `converged` and a [`Termination`] that says which rule ended the
//! search, and `require_converged()` turns a result that did not converge
//! into [`Error::NotConverged`](crate::Error::NotConverged) for a caller that
//! wants an error instead.

/// The rule that ended an iterative search.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Termination {
    /// The gradient norm fell below its tolerance (BFGS). Converged.
    GradientTolerance,
    /// The cost changed by less than its tolerance in a step (BFGS,
    /// Frank-Wolfe). Converged.
    CostTolerance,
    /// The standard deviation of the costs at the vertices fell below its
    /// tolerance (Nelder-Mead). Converged.
    SpreadTolerance,
    /// The bracket around the point narrowed to the tolerance (Brent,
    /// bisection, ITP). Converged.
    BracketTolerance,
    /// The function is exactly zero at the point (ITP). Converged.
    ExactRoot,
    /// A step gave a value or gradient that is not finite, or raised the
    /// cost, and the search ended at the point before it (BFGS). Not
    /// converged.
    StepRejected,
    /// The best cost is minus infinity (Nelder-Mead). Not converged.
    UnboundedBelow,
    /// The iteration limit was reached first. Not converged.
    MaxIterations,
}

impl Termination {
    /// Whether the rule is one of the convergence tests.
    #[must_use]
    pub const fn is_converged(self) -> bool {
        matches!(
            self,
            Self::GradientTolerance
                | Self::CostTolerance
                | Self::SpreadTolerance
                | Self::BracketTolerance
                | Self::ExactRoot
        )
    }
}

impl core::fmt::Display for Termination {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::GradientTolerance => "the gradient norm fell below its tolerance",
            Self::CostTolerance => "the cost stopped changing beyond its tolerance",
            Self::SpreadTolerance => "the costs at the vertices fell within their tolerance",
            Self::BracketTolerance => "the bracket narrowed to the tolerance",
            Self::ExactRoot => "the function is zero at the point",
            Self::StepRejected => "a step was not finite or raised the cost",
            Self::UnboundedBelow => "the cost is unbounded below",
            Self::MaxIterations => "the iteration limit was reached",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_tolerances_converge() {
        for t in [
            Termination::GradientTolerance,
            Termination::CostTolerance,
            Termination::SpreadTolerance,
            Termination::BracketTolerance,
            Termination::ExactRoot,
        ] {
            assert!(t.is_converged(), "{t}");
        }
        for t in [
            Termination::StepRejected,
            Termination::UnboundedBelow,
            Termination::MaxIterations,
        ] {
            assert!(!t.is_converged(), "{t}");
        }
    }
}

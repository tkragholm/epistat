//! The error type of the crate.
//!
//! Every check runs before the estimate it guards, so an error means nothing
//! was estimated. An argument whose length or value is wrong is
//! [`Error::Input`] with the shared [`InputError`], as in every epistat crate;
//! a rule that involves several arguments, or no single value, is
//! [`Error::Invalid`]. The other variants are failures of the computation on
//! input that was accepted, named by the [`Model`] or step that failed. An
//! error of a crate below keeps its type and code. [`Classify`] gives a binding
//! the kind and a stable code.

use core::fmt;

use epistat_core::{Classify, ErrorKind, InputError};

/// The model or step of an estimator that an error names.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Model {
    /// The propensity model of the treatment.
    Propensity,
    /// The outcome model.
    Outcome,
    /// The censoring model.
    Censoring,
    /// The model of the event's hazard.
    Hazard,
    /// The model of a competing event.
    Competing,
    /// The fluctuation of a targeted estimator.
    Fluctuation,
    /// The one-step path of a targeted estimator.
    OneStep,
    /// The model of the g-formula's time-varying variable with this index.
    Variable(usize),
    /// The balancing weights of CBPS or entropy balancing.
    Balance,
    /// The weighted regression of the outcome on the treatment.
    Regression,
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Propensity => f.write_str("the propensity model"),
            Self::Outcome => f.write_str("the outcome model"),
            Self::Censoring => f.write_str("the censoring model"),
            Self::Hazard => f.write_str("the hazard model"),
            Self::Competing => f.write_str("the competing-event model"),
            Self::Fluctuation => f.write_str("the fluctuation"),
            Self::OneStep => f.write_str("the one-step path"),
            Self::Variable(j) => write!(f, "the model of variable {j}"),
            Self::Balance => f.write_str("the balancing weights"),
            Self::Regression => f.write_str("the weighted regression"),
        }
    }
}

/// Why an estimator refused its input or could not finish.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An argument cannot be used: a wrong length, an empty argument, a value
    /// that is not finite or outside its domain. Nothing was estimated.
    #[error(transparent)]
    Input(#[from] InputError),
    /// The input breaks a rule of the method that involves several arguments
    /// or no single value, such as a treatment arm without persons or two
    /// rules for one variable. `index` is the row, variable or intervention
    /// the rule concerns, where there is one. Nothing was estimated.
    #[error("{what}{}: {rule}", index.map(|i| format!(" {i}")).unwrap_or_default())]
    Invalid {
        /// The argument, or what `index` counts.
        what: &'static str,
        /// The row, variable or intervention concerned.
        index: Option<usize>,
        /// The rule it breaks.
        rule: &'static str,
    },
    /// A term of a model's formula names a variable that does not exist, a
    /// lag of 0, a current value simulated after the modelled variable, or an
    /// indicator of a value that is not finite; or the formula has no terms.
    #[error("{model}: {rule}{}", variable.map(|v| format!(" (variable {v})")).unwrap_or_default())]
    Term {
        /// The model whose formula holds the term.
        model: Model,
        /// The variable the term names, where it names one.
        variable: Option<usize>,
        /// What is wrong with the term.
        rule: &'static str,
    },
    /// The data leave a model no rows to fit, as when every person has an
    /// event before the periods it models.
    #[error("{model}: no rows to fit")]
    NoRows {
        /// The model.
        model: Model,
    },
    /// A fitted probability of person `row` is numerically 0 or 1, as a
    /// logistic fit reaches only under separation.
    #[error("{model}: person {row} has a fitted probability numerically 0 or 1")]
    Separation {
        /// The model.
        model: Model,
        /// The person, as its row.
        row: usize,
    },
    /// A column of a model's design is a linear combination of the others.
    #[error("{model}: design column {column} is aliased")]
    Aliased {
        /// The model.
        model: Model,
        /// The column.
        column: usize,
    },
    /// A linear system of a model or step is singular.
    #[error("{model}: a linear system is singular")]
    Singular {
        /// The model or step.
        model: Model,
    },
    /// No weights meet the conditions the method asks of them, for example
    /// target moments outside the range of the group.
    #[error("{model}: {why}")]
    Infeasible {
        /// The model or step.
        model: Model,
        /// Why the conditions cannot be met.
        why: &'static str,
    },
    /// An iteration did not reach its tolerance.
    #[error("{model}: no convergence{}", iterations.map(|n| format!(" in {n} iterations")).unwrap_or_default())]
    NotConverged {
        /// The model or step.
        model: Model,
        /// The iterations run, where the iteration counts them.
        iterations: Option<usize>,
    },
    /// The fit of a model failed in epistat-glm.
    #[error("{model}: {source}")]
    Fit {
        /// The model.
        model: Model,
        /// The error of the fit.
        source: epistat_glm::Error,
    },
    /// A matrix product or factorisation of a model failed in epistat-linalg.
    #[error("{model}: {source}")]
    Linalg {
        /// The model or step.
        model: Model,
        /// The error of the algebra.
        source: epistat_linalg::Error,
    },
    /// An influence function or a variance was refused by epistat-inference.
    #[error(transparent)]
    Inference(#[from] epistat_inference::Error),
}

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(e) => e.kind(),
            Self::Fit { source, .. } => source.kind(),
            Self::Linalg { source, .. } => source.kind(),
            Self::Inference(e) => e.kind(),
            Self::Invalid { .. } | Self::Term { .. } | Self::NoRows { .. } => ErrorKind::Input,
            Self::Separation { .. }
            | Self::Aliased { .. }
            | Self::Singular { .. }
            | Self::Infeasible { .. }
            | Self::NotConverged { .. } => ErrorKind::Numerical,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(e) => e.code(),
            Self::Fit { source, .. } => source.code(),
            Self::Linalg { source, .. } => source.code(),
            Self::Inference(e) => e.code(),
            Self::Invalid { .. } => "causal.invalid",
            Self::NoRows { .. } => "causal.no_rows",
            Self::Term { .. } => "causal.term",
            Self::Separation { .. } => "causal.separation",
            Self::Aliased { .. } => "causal.aliased",
            Self::Singular { .. } => "causal.singular",
            Self::Infeasible { .. } => "causal.infeasible",
            Self::NotConverged { .. } => "causal.not_converged",
        }
    }
}

impl Error {
    /// The error of a rule `rule` broken by `what`.
    pub(crate) const fn invalid(what: &'static str, rule: &'static str) -> Self {
        Self::Invalid {
            what,
            index: None,
            rule,
        }
    }

    /// The error of a rule `rule` broken by `what` number `index`.
    pub(crate) const fn invalid_at(what: &'static str, index: usize, rule: &'static str) -> Self {
        Self::Invalid {
            what,
            index: Some(index),
            rule,
        }
    }

    /// The error of `model` whose fit failed with `source`, with its
    /// non-convergence and its aliased columns as this crate's variants.
    pub(crate) fn fit(model: Model, source: epistat_glm::Error) -> Self {
        match source {
            epistat_glm::Error::NotConverged { iterations, .. } => Self::NotConverged {
                model,
                iterations: Some(iterations),
            },
            epistat_glm::Error::Aliased { column } => Self::Aliased { model, column },
            source => Self::Fit { model, source },
        }
    }

    /// The error of `model` whose algebra failed with `source`.
    pub(crate) const fn linalg(model: Model, source: epistat_linalg::Error) -> Self {
        Self::Linalg { model, source }
    }
}

/// An argument `what` whose value lies outside `domain`.
pub(crate) const fn domain(what: &'static str, value: f64, domain: &'static str) -> Error {
    Error::Input(InputError::Domain {
        what,
        value,
        domain,
        row: None,
    })
}

/// An argument `what` whose value at `row` lies outside `domain`.
pub(crate) const fn domain_at(
    what: &'static str,
    value: f64,
    domain: &'static str,
    row: usize,
) -> Error {
    Error::Input(InputError::Domain {
        what,
        value,
        domain,
        row: Some(row),
    })
}

/// An argument `what` with no rows where at least one is needed.
pub(crate) const fn empty(what: &'static str) -> Error {
    Error::Input(InputError::Empty { what })
}

/// The result of an estimator.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_unique_and_kinds_follow_the_variant() {
        let own = [
            Error::invalid("treatment", "both groups need persons"),
            Error::NoRows {
                model: Model::Outcome,
            },
            Error::Term {
                model: Model::Outcome,
                variable: Some(1),
                rule: "a lag must be at least 1",
            },
            Error::Separation {
                model: Model::Propensity,
                row: 3,
            },
            Error::Aliased {
                model: Model::Outcome,
                column: 1,
            },
            Error::Singular {
                model: Model::Balance,
            },
            Error::Infeasible {
                model: Model::Balance,
                why: "the target lies outside the group",
            },
            Error::NotConverged {
                model: Model::Fluctuation,
                iterations: Some(25),
            },
        ];
        epistat_testkit::assert_codes_unique("causal", &own);
        assert_eq!(own[0].kind(), ErrorKind::Input);
        assert_eq!(own[1].kind(), ErrorKind::Input);
        assert_eq!(own[2].kind(), ErrorKind::Input);
        for error in &own[3..] {
            assert_eq!(error.kind(), ErrorKind::Numerical, "{error:?}");
        }
        assert_eq!(empty("treatment").code(), "input.empty");
        assert_eq!(
            Error::invalid_at("variable", 2, "needs two classes").to_string(),
            "variable 2: needs two classes"
        );
        assert_eq!(
            Error::NotConverged {
                model: Model::Variable(1),
                iterations: None
            }
            .to_string(),
            "the model of variable 1: no convergence"
        );
    }

    #[test]
    fn a_fit_keeps_its_convergence_and_aliasing_typed() {
        let aliased = Error::fit(Model::Outcome, epistat_glm::Error::Aliased { column: 2 });
        assert_eq!(aliased.code(), "causal.aliased");
    }
}

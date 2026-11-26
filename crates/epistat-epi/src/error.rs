//! The error of every fallible function in the crate.

use epistat_core::{Classify, ErrorKind, InputError};

use crate::bias::BiasError;
use crate::evalue::EValueError;
use crate::mantel_haenszel::MhError;
use crate::rates::RateError;
use crate::rosenbaum::SensitivityError;
use crate::signed_rank::SignedRankError;
use crate::sir::SirError;

/// Why a function of this crate could not give a result.
///
/// Invalid input is [`Error::Input`] where one argument and its value say
/// what is wrong, and [`Error::Invalid`] where the rule involves several
/// arguments or no single value, such as a table without a count in a row.
/// The other variants are results the method does not define on input it
/// accepted.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum Error {
    /// An argument's value cannot be used.
    #[error(transparent)]
    Input(#[from] InputError),
    /// The input breaks a rule of the method.
    #[error("{what}: {rule}")]
    Invalid {
        /// The argument or arguments.
        what: &'static str,
        /// The rule they break.
        rule: &'static str,
    },
    /// A table corrected for bias has a negative cell: the assumed bias
    /// parameters are incompatible with the observed table.
    #[error("the parameters give a negative cell in the corrected table")]
    NegativeCell,
    /// Every draw of a probabilistic bias analysis was discarded.
    #[error("all {discarded} draws were discarded: the parameters are incompatible with the table")]
    NoValidDraws {
        /// The number of draws discarded.
        discarded: usize,
    },
    /// No matched set carries information, such as pairs without a non-zero
    /// difference.
    #[error("no matched set carries information")]
    NoInformation,
    /// The variance of a test statistic is zero.
    #[error("the variance of the statistic is zero")]
    ZeroVariance,
    /// The scale of the scores is zero, so the scores are undefined.
    #[error("the scale of the scores is zero")]
    ZeroScale,
    /// A stratum has no expected events, so its ratio is undefined.
    #[error("a stratum has no expected events")]
    ZeroExpected,
    /// The exact distribution asked for is too large to tabulate.
    #[error("the exact distribution is too large to tabulate")]
    TooLarge,
}

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(e) => e.kind(),
            Self::Invalid { .. } => ErrorKind::Input,
            Self::TooLarge => ErrorKind::Unsupported,
            Self::NegativeCell
            | Self::NoValidDraws { .. }
            | Self::NoInformation
            | Self::ZeroVariance
            | Self::ZeroScale
            | Self::ZeroExpected => ErrorKind::Numerical,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(e) => e.code(),
            Self::Invalid { .. } => "epi.invalid",
            Self::NegativeCell => "epi.negative_cell",
            Self::NoValidDraws { .. } => "epi.no_valid_draws",
            Self::NoInformation => "epi.no_information",
            Self::ZeroVariance => "epi.zero_variance",
            Self::ZeroScale => "epi.zero_scale",
            Self::ZeroExpected => "epi.zero_expected",
            Self::TooLarge => "epi.too_large",
        }
    }
}

const fn invalid(what: &'static str, rule: &'static str) -> Error {
    Error::Invalid { what, rule }
}

const fn domain(what: &'static str, value: f64, domain: &'static str) -> Error {
    Error::Input(InputError::Domain {
        what,
        value,
        domain,
    })
}

const LENGTH: Error = invalid("inputs", "differ in length");
const STRATA: Error = invalid("strata", "there are none");

impl From<MhError> for Error {
    fn from(e: MhError) -> Self {
        match e {
            MhError::Empty => STRATA,
            MhError::Cell => invalid(
                "cells",
                "counts and person-time must be finite and not negative, and events need person-time",
            ),
            MhError::EmptyStratum => invalid("stratum", "has no persons or no person-time"),
            MhError::Length => LENGTH,
        }
    }
}

impl From<RateError> for Error {
    fn from(e: RateError) -> Self {
        match e {
            RateError::Length => LENGTH,
            RateError::Events => invalid("events", "must be finite and not negative"),
            RateError::Time => invalid("person-time", "must be finite and not negative"),
            RateError::Weights => invalid(
                "weights",
                "must be finite, not negative and sum to more than 0",
            ),
            RateError::Scale => invalid("scale", "must be positive and finite"),
            RateError::NotInteger => invalid("events", "the exact test needs integer counts"),
            RateError::Empty => STRATA,
        }
    }
}

impl From<SirError> for Error {
    fn from(e: SirError) -> Self {
        match e {
            SirError::Observed => invalid("observed", "must be finite and not negative"),
            SirError::Expected => invalid("expected", "must be finite and not negative"),
            SirError::Time => invalid(
                "person-time",
                "must be finite and not negative, and positive where there are events",
            ),
            SirError::Length => LENGTH,
            SirError::ZeroExpected => Self::ZeroExpected,
            SirError::Score => invalid("scores", "must be finite"),
        }
    }
}

impl From<EValueError> for Error {
    fn from(e: EValueError) -> Self {
        match e {
            EValueError::NotPositive { what, value } => domain(what, value, "(0, inf)"),
            EValueError::NotFinite { what } => invalid(what, "must be finite"),
            EValueError::NegativeStandardError => invalid("standard error", "cannot be negative"),
            EValueError::LimitsReversed => invalid("limits", "the lower is above the upper"),
            EValueError::EstimateOutsideInterval => {
                invalid("estimate", "lies outside its interval")
            }
            EValueError::BelowOne { what, value } => domain(what, value, "[1, inf)"),
            EValueError::Table(rule) => invalid("table", rule),
            EValueError::NegativeDifference => {
                invalid("risk difference", "must be positive; recode the exposure")
            }
            EValueError::TrueNotBelowEstimate => {
                invalid("true value", "must be below the estimate")
            }
        }
    }
}

impl From<SignedRankError> for Error {
    fn from(e: SignedRankError) -> Self {
        match e {
            SignedRankError::NotFinite => invalid("values", "must be finite"),
            SignedRankError::Probability(p) => domain("probability", p, "[0, 1]"),
            SignedRankError::TooLarge => Self::TooLarge,
        }
    }
}

impl From<BiasError> for Error {
    fn from(e: BiasError) -> Self {
        match e {
            BiasError::Table(rule) => invalid("table", rule),
            BiasError::Parameter(rule) => invalid("bias parameter", rule),
            BiasError::NegativeCell => Self::NegativeCell,
            BiasError::NoValidDraws { discarded } => Self::NoValidDraws { discarded },
        }
    }
}

impl From<SensitivityError> for Error {
    fn from(e: SensitivityError) -> Self {
        match e {
            SensitivityError::Gamma(gamma) => domain("gamma", gamma, "[1, inf)"),
            SensitivityError::NotFinite => invalid("responses", "must be finite"),
            SensitivityError::NoInformation => Self::NoInformation,
            SensitivityError::SetTooSmall => invalid("matched set", "has fewer than two units"),
            SensitivityError::ZeroScale => Self::ZeroScale,
            SensitivityError::Parameter(what) => invalid(what, "is out of range"),
            SensitivityError::ZeroVariance => Self::ZeroVariance,
            SensitivityError::SignedRank(e) => e.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_have_kinds_and_codes() {
        let level = Error::from(MhError::Empty);
        assert_eq!(level.kind(), ErrorKind::Input);
        assert_eq!(level.code(), "epi.invalid");
        assert_eq!(level.to_string(), "strata: there are none");
        let gamma = Error::from(SensitivityError::Gamma(0.5));
        assert_eq!(gamma.code(), "input.domain");
        assert_eq!(Error::NoInformation.kind(), ErrorKind::Numerical);
        assert_eq!(Error::TooLarge.kind(), ErrorKind::Unsupported);
        let codes = [
            Error::NegativeCell,
            Error::NoValidDraws { discarded: 1 },
            Error::NoInformation,
            Error::ZeroVariance,
            Error::ZeroScale,
            Error::ZeroExpected,
            Error::TooLarge,
            level,
        ]
        .map(|e| e.code());
        let mut sorted = codes.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), codes.len());
    }
}

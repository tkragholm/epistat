//! The error of every fallible function in the crate.

use epistat_core::{Classify, ErrorKind, InputError, check};

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
        row: None,
    })
}

/// A count or person-time `value`, finite and not negative: NaN or an infinity
/// is [`InputError::NotFinite`] and a negative value [`InputError::Domain`],
/// at `row` of its column when it has one.
pub(crate) const fn count(what: &'static str, value: f64, row: Option<usize>) -> Result<(), Error> {
    if !value.is_finite() {
        return Err(Error::Input(InputError::NotFinite {
            what,
            row: match row {
                Some(row) => row,
                None => 0,
            },
            column: None,
        }));
    }
    if value < 0.0 {
        return Err(Error::Input(InputError::Domain {
            what,
            value,
            domain: "[0, inf)",
            row,
        }));
    }
    Ok(())
}

/// Every value of the column `what` a count or person-time, as [`count`].
pub(crate) fn counts(what: &'static str, values: &[f64]) -> Result<(), Error> {
    values
        .iter()
        .enumerate()
        .try_for_each(|(row, &value)| count(what, value, Some(row)))
}

/// A number `value` that must be finite, or [`InputError::NotFinite`].
pub(crate) fn finite_number(what: &'static str, value: f64) -> Result<(), Error> {
    Ok(check::finite(what, &[value])?)
}

/// Every value of matched set `row` of `what` finite, or
/// [`InputError::NotFinite`] at the set and its position in it.
pub(crate) fn finite_set(what: &'static str, row: usize, set: &[f64]) -> Result<(), Error> {
    match set.iter().position(|v| !v.is_finite()) {
        Some(column) => Err(Error::Input(InputError::NotFinite {
            what,
            row,
            column: Some(column),
        })),
        None => Ok(()),
    }
}

/// Columns of one length with at least one row: the first column is the
/// reference, and every other one has its length or [`InputError::Length`].
pub(crate) fn columns(columns: &[(&'static str, usize)]) -> Result<(), Error> {
    let Some(&(first, rows)) = columns.first() else {
        return Ok(());
    };
    for &(what, found) in &columns[1..] {
        check::same_length(what, rows, found)?;
    }
    Ok(check::non_empty(first, rows)?)
}

impl From<MhError> for Error {
    fn from(e: MhError) -> Self {
        match e {
            MhError::Cell => invalid("cells", "events need person-time"),
            MhError::EmptyStratum => invalid("stratum", "has no persons or no person-time"),
        }
    }
}

impl From<RateError> for Error {
    fn from(e: RateError) -> Self {
        match e {
            RateError::Events => invalid("events", "the exact interval is undefined"),
            RateError::Weights => invalid("weights", "sum to 0"),
            RateError::NotInteger => invalid("events", "the exact test needs integer counts"),
        }
    }
}

impl From<SirError> for Error {
    fn from(e: SirError) -> Self {
        match e {
            SirError::Expected => invalid("expected", "the interval is undefined"),
            SirError::ZeroExpected => Self::ZeroExpected,
        }
    }
}

impl From<EValueError> for Error {
    fn from(e: EValueError) -> Self {
        match e {
            EValueError::NotPositive { what, value } => domain(what, value, "(0, inf)"),
            EValueError::NotFinite { what } => Self::Input(InputError::NotFinite {
                what,
                row: 0,
                column: None,
            }),
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
            EValueError::RatioOnIdentityScale => invalid(
                "estimate",
                "a ratio must be on Scale::Log; a number on the identity scale may be a log ratio, \
                 so exponentiate it first (Coefficients::exp)",
            ),
            EValueError::DifferenceOnLogScale => {
                invalid("estimate", "a mean difference must be on Scale::Identity")
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
            SensitivityError::Empty(what) => Self::Input(InputError::Empty { what }),
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
        let level = Error::from(MhError::EmptyStratum);
        assert_eq!(level.kind(), ErrorKind::Input);
        assert_eq!(level.code(), "epi.invalid");
        assert_eq!(
            level.to_string(),
            "stratum: has no persons or no person-time"
        );
        let gamma = Error::from(SensitivityError::Gamma(0.5));
        assert_eq!(gamma.code(), "input.domain");
        assert_eq!(Error::NoInformation.kind(), ErrorKind::Numerical);
        assert_eq!(Error::TooLarge.kind(), ErrorKind::Unsupported);
        epistat_testkit::assert_codes_unique(
            "epi",
            &[
                Error::NegativeCell,
                Error::NoValidDraws { discarded: 1 },
                Error::NoInformation,
                Error::ZeroVariance,
                Error::ZeroScale,
                Error::ZeroExpected,
                Error::TooLarge,
                level,
            ],
        );
    }
}

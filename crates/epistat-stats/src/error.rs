//! The error of every fallible function in the crate.

use epistat_core::{Classify, ErrorKind, InputError};

/// Why a function of this crate could not give a result.
///
/// Invalid input is [`Error::Input`], with the same variants and messages as
/// in every epistat crate. The distribution functions of [`crate::dist`] do
/// not fail: they return NaN for a parameter outside the domain.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum Error {
    /// The input cannot be used.
    #[error(transparent)]
    Input(#[from] InputError),
    /// A cluster of a k-means partition lost all its points, where R's
    /// `kmeans()` stops with "empty cluster".
    #[error("cluster {cluster} is empty")]
    EmptyCluster {
        /// The cluster.
        cluster: usize,
    },
    /// The rounding of a fixed-point sum to `places` decimals could take more
    /// than `1e-12` of the magnitude of its terms, which are too small for
    /// the decimals kept.
    #[error(
        "the terms are too small for a sum to {places} decimals: its rounding of up to {bound:e} \
         is more than 1e-12 of their magnitude {magnitude:e}"
    )]
    FixedPointRounding {
        /// The decimals each term is rounded to.
        places: u32,
        /// The largest error the rounding can make in the sum.
        bound: f64,
        /// The sum of the magnitudes of the terms, rounded as they are.
        magnitude: f64,
    },
    /// A term or the total of a fixed-point sum to `places` decimals has a
    /// magnitude of `10^(38 - places)` or more.
    #[error("a term or the total is too large for a sum to {places} decimals")]
    FixedPointOverflow {
        /// The decimals each term is rounded to.
        places: u32,
    },
    /// `stats.non_positive_divisor`: the divisor `what` is zero or negative,
    /// as the divisor `W - sum(w^2) / W` of a weighted variance is when one
    /// row carries all the weight.
    #[error("{what} is not positive")]
    NonPositiveDivisor {
        /// The divisor.
        what: &'static str,
    },
}

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(e) => e.kind(),
            Self::EmptyCluster { .. }
            | Self::FixedPointRounding { .. }
            | Self::FixedPointOverflow { .. }
            | Self::NonPositiveDivisor { .. } => ErrorKind::Numerical,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(e) => e.code(),
            Self::EmptyCluster { .. } => "stats.empty_cluster",
            Self::FixedPointRounding { .. } => "stats.fixed_point_rounding",
            Self::FixedPointOverflow { .. } => "stats.fixed_point_overflow",
            Self::NonPositiveDivisor { .. } => "stats.non_positive_divisor",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_have_kinds_and_codes() {
        let input = Error::from(InputError::Empty { what: "points" });
        assert_eq!(input.kind(), ErrorKind::Input);
        assert_eq!(input.code(), "input.empty");
        assert_eq!(input.to_string(), "points is empty");
        let empty = Error::EmptyCluster { cluster: 2 };
        assert_eq!(empty.kind(), ErrorKind::Numerical);
        assert_eq!(empty.code(), "stats.empty_cluster");
        let rounding = Error::FixedPointRounding {
            places: 15,
            bound: 1.5e-15,
            magnitude: 6e-16,
        };
        assert_eq!(rounding.kind(), ErrorKind::Numerical);
        assert_eq!(rounding.code(), "stats.fixed_point_rounding");
        let overflow = Error::FixedPointOverflow { places: 15 };
        assert_eq!(overflow.code(), "stats.fixed_point_overflow");
        let divisor = Error::NonPositiveDivisor {
            what: "the divisor",
        };
        assert_eq!(divisor.code(), "stats.non_positive_divisor");
        assert_eq!(divisor.to_string(), "the divisor is not positive");
        epistat_testkit::assert_codes_unique("stats", &[empty, rounding, overflow, divisor]);
    }
}

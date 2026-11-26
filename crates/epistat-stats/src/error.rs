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
}

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(e) => e.kind(),
            Self::EmptyCluster { .. } => ErrorKind::Numerical,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(e) => e.code(),
            Self::EmptyCluster { .. } => "stats.empty_cluster",
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
    }
}

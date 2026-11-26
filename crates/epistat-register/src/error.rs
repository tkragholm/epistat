//! The error type of the crate.

/// Why a function refused its input or a price index lacked a year.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum RegisterError {
    /// Input the function cannot take: columns of different lengths, a key
    /// code outside its range, an interval that ends before it starts or more
    /// rows than a `u32` row number holds.
    #[error("{0}")]
    Input(String),
    /// The reference year has no price index.
    #[error("Reference price year {0} is missing from the CPI table.")]
    MissingReferenceYear(i32),
    /// One or more years have no price index, sorted and without duplicates.
    #[error("Missing CPI adjustment factors for years: {}", join_years(.0))]
    MissingYears(Vec<i32>),
    /// A fit that could not be computed from valid input.
    #[error("{0}")]
    Numerical(String),
}

impl RegisterError {
    pub(crate) fn input(message: impl Into<String>) -> Self {
        Self::Input(message.into())
    }

    pub(crate) fn numerical(message: impl Into<String>) -> Self {
        Self::Numerical(message.into())
    }
}

fn join_years(years: &[i32]) -> String {
    years
        .iter()
        .map(i32::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

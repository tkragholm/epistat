//! The [`Covariance`] results of the estimators, from the row-major values
//! they form.
//!
//! An entry that is not finite is undefined and becomes `None`, as an entry
//! the estimator leaves undefined already is. Every finite entry keeps its
//! bits.

use epistat_core::Covariance;

use crate::error::Result;

/// The `dim x dim` matrix of the row-major `values`.
///
/// # Errors
///
/// [`InputError::Length`](epistat_core::InputError::Length) unless there are
/// `dim * dim` values.
pub(crate) fn from_options(dim: usize, values: Vec<Option<f64>>) -> Result<Covariance> {
    let values = values
        .into_iter()
        .map(|value| value.filter(|v| v.is_finite()))
        .collect();
    Ok(Covariance::new(dim, values)?)
}

/// The `dim x dim` matrix of the row-major `values`, each defined where it
/// is finite.
///
/// # Errors
///
/// As [`from_options`].
pub(crate) fn from_values(dim: usize, values: &[f64]) -> Result<Covariance> {
    Ok(Covariance::new(
        dim,
        values.iter().map(|&v| v.is_finite().then_some(v)).collect(),
    )?)
}

/// The `dim x dim` matrix with every entry undefined.
///
/// # Errors
///
/// [`InputError::TooLarge`](epistat_core::InputError::TooLarge) when
/// `dim * dim` overflows.
pub(crate) fn undefined(dim: usize) -> Result<Covariance> {
    let size = dim
        .checked_mul(dim)
        .ok_or(epistat_core::InputError::TooLarge {
            what: "covariance",
            limit: usize::MAX as u64,
            found: u64::MAX,
        })?;
    Ok(Covariance::new(dim, vec![None; size])?)
}

/// The `count` matrices of `dim x dim` in the row-major `values`, one block
/// of `dim * dim` after the other.
///
/// # Errors
///
/// [`InputError::Length`](epistat_core::InputError::Length) unless there are
/// `count * dim * dim` values.
pub(crate) fn blocks(count: usize, dim: usize, values: &[f64]) -> Result<Vec<Covariance>> {
    let size = dim * dim;
    epistat_core::check::same_length("covariance", count * size, values.len())?;
    (0..count)
        .map(|t| from_values(dim, &values[t * size..(t + 1) * size]))
        .collect()
}

/// [`blocks`] of values that may be undefined.
///
/// # Errors
///
/// As [`blocks`].
pub(crate) fn option_blocks(
    count: usize,
    dim: usize,
    values: &[Option<f64>],
) -> Result<Vec<Covariance>> {
    let size = dim * dim;
    epistat_core::check::same_length("covariance", count * size, values.len())?;
    (0..count)
        .map(|t| from_options(dim, values[t * size..(t + 1) * size].to_vec()))
        .collect()
}

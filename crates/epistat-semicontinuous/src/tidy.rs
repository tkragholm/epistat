//! The coefficient tables of the fits, built in one place.

use epistat_core::{Coefficients, Covariance, EffectScale, Level, VarianceKind};

use crate::Error;

/// What a fit hands to [`table`]: its coefficients, the covariance it
/// computed and of which kind, the scale of its coefficients and, for a
/// model-based covariance whose dispersion is estimated, the residual degrees
/// of freedom of the t reference R's `summary` uses.
pub(crate) struct TableParts<'a> {
    pub coefficients: &'a [f64],
    pub covariance: Option<&'a Covariance>,
    pub computed: Option<VarianceKind>,
    pub effect: EffectScale,
    pub t_df: Option<f64>,
}

/// The coefficient table of `parts` at `level`, with positional names.
///
/// # Errors
///
/// [`Error::NoCovariance`] when `asked` is not the covariance the fit
/// computed.
pub(crate) fn table(
    parts: &TableParts<'_>,
    level: Level,
    asked: VarianceKind,
) -> Result<Coefficients, Error> {
    let covariance = match (parts.covariance, parts.computed) {
        (Some(covariance), Some(computed)) if computed == asked => covariance.clone(),
        _ => return Err(Error::NoCovariance { asked }),
    };
    let estimates: Vec<Option<f64>> = parts.coefficients.iter().copied().map(Some).collect();
    Ok(epistat_stats::coefficients(
        Coefficients::positional_names(estimates.len()),
        &estimates,
        covariance,
        level,
        asked,
        parts.effect,
        parts.t_df,
    )?)
}

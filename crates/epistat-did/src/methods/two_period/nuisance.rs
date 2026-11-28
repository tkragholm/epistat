//! The nuisance models of the panel doubly robust estimator, as
//! [`estimate`](super::estimate) fits them for its ATT.
//!
//! [`fit`] returns both models with their standard errors, [`propensity`] the
//! propensity score alone, and [`independent_columns`] the columns of the
//! design that a fit can identify.
//!
//! # Pruning
//!
//! The doubly robust fit regresses the outcome on the design within the
//! comparison arm and models treatment on it across both arms. Either model is
//! singular when a column is constant, or a combination of the others, inside
//! the rows that model reads: a rare indicator held only by treated units is
//! all zero in the comparison arm. So the estimator keeps a column when it is
//! linearly independent of the intercept and the columns kept before it
//! within each arm, in the column order of the design. A column is dependent
//! when the share of its weighted sum of squares left after projecting out
//! the kept columns is at most the tolerance; the estimators use `1e-8`. R's
//! `did` returns `NA` for such a cell instead.

pub use crate::methods::drdid::panel::{NuisanceFit, Propensity};

use epistat_core::Level;

use super::{NuisanceOptions, TwoPeriodData, domain};
use crate::error::Error;
use crate::methods::drdid::panel::{
    PanelFlatInput, fit_drdid_panel_nuisance, fit_drdid_panel_propensity_coefficients,
};

/// The rows on which a design column has to be identified to be kept.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Pruning {
    /// Within each arm separately, the rule the panel estimator applies to
    /// every fit.
    #[default]
    PerArm,
    /// On all rows together, a weaker test: a column that passes it can still
    /// fail within one arm, and the estimator then drops it as well.
    Pooled,
}

/// The panel's two nuisance models, fitted as [`super::estimate`] fits them
/// for the doubly robust ATT: the same pruning of the design, the weighted
/// logistic propensity score by maximum likelihood, and the weighted least
/// squares outcome regression among the comparators with `options.ridge` on
/// its diagonal. The standard errors are computed here and play no part in
/// the ATT; neither accounts for clusters.
///
/// # Errors
///
/// [`Error::Input`] for an option outside its domain, [`Error::Unsupported`]
/// for a repeated cross-section, [`Error::NoTreated`] and
/// [`Error::NoComparison`] for an arm without rows, and [`Error::Singular`]
/// for a fit, or an information matrix behind the standard errors, that
/// cannot be solved.
pub fn fit(data: &TwoPeriodData<'_>, options: &NuisanceOptions) -> Result<NuisanceFit, Error> {
    options.validate()?;
    with_input(data, |input| {
        fit_drdid_panel_nuisance(input, options.config(Level::default(), None)).map_err(Error::from)
    })
}

/// The propensity score of [`fit`] alone, with the same pruning and the same
/// iterations, so its coefficients, iteration count and convergence flag
/// equal [`fit`]'s bit for bit whenever that succeeds. It skips the standard
/// errors and the outcome regression, so a design for which those are
/// singular succeeds here.
///
/// # Errors
///
/// As [`fit`], except that only the propensity iterations can be singular.
pub fn propensity(
    data: &TwoPeriodData<'_>,
    options: &NuisanceOptions,
) -> Result<Propensity, Error> {
    options.validate()?;
    with_input(data, |input| {
        fit_drdid_panel_propensity_coefficients(input, options.config(Level::default(), None))
            .map_err(Error::from)
    })
}

/// One flag per column of the estimators' design of `data`, the intercept
/// first, true where the column is kept under `rule` at `tolerance`. The
/// intercept is always kept. The weights of the Gram matrix are the data's prior weights.
///
/// # Errors
///
/// [`Error::Input`] for a tolerance that is negative or not finite.
pub fn independent_columns(
    data: &TwoPeriodData<'_>,
    rule: Pruning,
    tolerance: f64,
) -> Result<Vec<bool>, Error> {
    if !(tolerance.is_finite() && tolerance >= 0.0) {
        return Err(domain("tolerance", tolerance, "[0, inf)"));
    }
    let (design, feature_count) = data.design_flat();
    let weights = data.weight_column();
    let group: Vec<usize> = match rule {
        Pruning::PerArm => data.treated().iter().map(|&t| usize::from(t)).collect(),
        Pruning::Pooled => vec![0; data.len()],
    };
    let pruned =
        crate::methods::drdid::prune_design(&design, feature_count, &group, &weights, tolerance)
            .map_err(Error::from)?;
    Ok(pruned.kept)
}

fn with_input<T>(
    data: &TwoPeriodData<'_>,
    run: impl FnOnce(PanelFlatInput<'_>) -> Result<T, Error>,
) -> Result<T, Error> {
    let super::Design::Panel { change } = data.design() else {
        return Err(Error::Unsupported {
            what: "the panel nuisance models on a repeated cross-section",
        });
    };
    let weights = data.weight_column();
    let (design, feature_count) = data.design_flat();
    run(PanelFlatInput {
        treated: data.treated(),
        delta_outcome: change,
        weight: &weights,
        design_matrix_flat: &design,
        feature_count,
    })
}

//! Two-way fixed-effects difference-in-differences: the static regression and
//! the event study with relative-period indicators, as
//! `fixest::feols(y ~ treat | id + t)` and
//! `fixest::feols(y ~ i(rel, ref = -1) | id + t)` fit them.
//!
//! # Method
//!
//! For unit `i` in period `t`, first treated in period `g_i` (never, for the
//! never treated), the static regression is
//!
//! ```text
//! y_it = alpha_i + lambda_t + beta D_it + x_it' gamma + e_it,      D_it = 1[t >= g_i]
//! ```
//!
//! and the event study replaces `beta D_it` by one indicator per relative
//! period `r = t - g_i` of the treated units,
//!
//! ```text
//! y_it = alpha_i + lambda_t + sum_{r != r_0} beta_r 1[t - g_i = r] + x_it' gamma + e_it
//! ```
//!
//! with the reference period `r_0` left out (by default -1, the period before
//! treatment) and with the never treated in no indicator. With binned end
//! points `(lo, hi)` every relative period below `lo` counts as `lo` and every
//! one above `hi` as `hi`, so the end coefficients pool the periods beyond
//! them. An indicator that is collinear with the fixed effects and the others,
//! as one more period is when no unit is never treated, is dropped and its
//! coefficient is `None`, as `fixest` drops it.
//!
//! The unit and period effects are absorbed by [`epistat_glm::absorb`] and the
//! regression is fitted by [`epistat_glm::feols`], so the cost is linear in the
//! rows and no unit dummies are formed. The covariance is clustered by unit,
//! or by the caller's clusters, with the small-sample correction of `fixest`
//! (`G/(G - 1) (n - 1)/(n - K)`, with the unit effects nested in unit
//! clusters not counted in `K`).
//!
//! # Influence
//!
//! The influence function of a coefficient for unit `u` is
//!
//! ```text
//! psi_u = N_u e_k' B sum_{rows of u} w_it e_it x~_it,      B = (X~' W X~)^-1
//! ```
//!
//! with `N_u` the number of units in the fit and `x~` the demeaned
//! regressors. The `psi_u` sum to zero, and
//! [`did_cluster_sums`](crate::did_cluster_sums) and
//! [`did_covariance`](crate::did_covariance) over unit clusters give
//! `B (sum_u s_u s_u') B`, the clustered covariance without the small-sample
//! factors, as for the other estimators of the crate.
//!
//! # Interpretation
//!
//! With staggered adoption and effects that change over time or differ
//! between cohorts, the static coefficient is a weighted mean of all 2x2
//! comparisons, some with already treated units as controls and weights that
//! can be negative \[1\]; [`bacon_decomposition`](crate::bacon_decomposition)
//! shows those comparisons. The event-study coefficients are contaminated by
//! the effects at other relative periods in the same way \[2\]. They are
//! reported as the comparison readers expect next to the heterogeneity-robust
//! estimators.
//!
//! # References
//!
//! 1. Goodman-Bacon A. Difference-in-differences with variation in treatment
//!    timing. J Econom. 2021;225(2):254-77.
//!    doi:10.1016/j.jeconom.2021.03.014
//! 2. Sun L, Abraham S. Estimating dynamic treatment effects in event studies
//!    with heterogeneous treatment effects. J Econom. 2021;225(2):175-99.
//!    doi:10.1016/j.jeconom.2020.09.006
//! 3. Bergé L. The R package `fixest` 0.14.2: `feols` and `i()`.

use std::collections::BTreeSet;

use epistat_glm::{Absorb, AbsorbError, Feols, FixedEffects, Singletons, Ssc, Vcov};
use epistat_linalg::Mat;

use crate::methods::staggered::StaggeredPanel;

/// The treatment terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwfeSpec {
    /// One indicator of being treated.
    Static,
    /// One indicator per relative period.
    EventStudy {
        /// The relative period left out.
        reference: i32,
        /// Relative periods below the first are counted as the first, above
        /// the second as the second.
        bins: Option<(i32, i32)>,
    },
}

/// The controls of a two-way fixed-effects fit.
#[derive(Clone, Copy, Debug)]
pub struct TwfeConfig<'a> {
    /// The treatment terms.
    pub spec: TwfeSpec,
    /// Cluster codes, one per row; the units when `None`.
    pub clusters: Option<&'a [u32]>,
    /// The demeaning.
    pub absorb: Absorb,
    /// The small-sample correction of the covariance.
    pub ssc: Ssc,
}

impl TwfeConfig<'_> {
    /// The event study with reference period -1, no bins, unit clusters and
    /// the defaults of `fixest`.
    #[must_use]
    pub const fn event_study() -> Self {
        Self {
            spec: TwfeSpec::EventStudy {
                reference: -1,
                bins: None,
            },
            clusters: None,
            absorb: Absorb::new(),
            ssc: Ssc::new(),
        }
    }

    /// The static regression with unit clusters and the defaults of
    /// `fixest`.
    #[must_use]
    pub const fn static_twfe() -> Self {
        Self {
            spec: TwfeSpec::Static,
            ..Self::event_study()
        }
    }
}

/// Why a two-way fixed-effects fit failed.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TwfeError {
    /// An input has the wrong length.
    #[error("{what} has {found} entries where {expected} were expected")]
    Length {
        /// The input.
        what: &'static str,
        /// The length it needed.
        expected: usize,
        /// The length it had.
        found: usize,
    },
    /// A unit has rows with different first treated periods.
    #[error("unit {unit} has rows with different first treated periods")]
    Cohort {
        /// The unit.
        unit: u32,
    },
    /// No treated row, so there is no treatment term.
    #[error("no row carries a treatment indicator")]
    NoTreatment,
    /// The bins are empty or do not contain the reference period.
    #[error("the bins must satisfy lo <= reference <= hi")]
    Bins,
    /// The panel has more than 2^32 periods.
    #[error("the periods span more than 2^32 values")]
    Periods,
    /// The fit failed.
    #[error(transparent)]
    Fit(#[from] AbsorbError),
}

/// One treatment coefficient.
#[derive(Clone, Debug, PartialEq)]
pub struct TwfeTerm {
    /// The relative period, `None` for the static indicator.
    pub relative_period: Option<i32>,
    /// The coefficient, `None` when the indicator is collinear.
    pub estimate: Option<f64>,
    /// Its clustered standard error.
    pub std_error: Option<f64>,
}

/// A two-way fixed-effects fit.
#[derive(Clone, Debug)]
pub struct TwfeResult {
    /// The treatment coefficients, in increasing relative period.
    pub terms: Vec<TwfeTerm>,
    /// The coefficients of the covariates.
    pub covariate_coefficients: Vec<Option<f64>>,
    /// The clustered covariance of the treatment terms and then the
    /// covariates, row-major, `None` for dropped terms.
    pub covariance: Vec<Option<f64>>,
    /// The `K` of the small-sample correction.
    pub df_k: usize,
    /// The number of clusters.
    pub n_clusters: usize,
    /// The influence function of each coefficient (terms, then covariates),
    /// one value per unit of [`TwfeResult::units`]; `None` for dropped terms.
    pub influence: Vec<Option<Vec<f64>>>,
    /// The units in the fit, in increasing code.
    pub units: Vec<u32>,
    /// The rows in the fit.
    pub n_rows: usize,
    /// The rows removed as singletons.
    pub removed_singletons: usize,
}

/// Fits the two-way fixed-effects regression of `config.spec` to `panel`.
///
/// # Errors
///
/// [`TwfeError`] for inputs of the wrong length, cohorts that vary within a
/// unit, no treated row, unusable bins, and the errors of the fit.
pub fn estimate_twfe(
    panel: &StaggeredPanel<'_>,
    config: &TwfeConfig<'_>,
) -> Result<TwfeResult, TwfeError> {
    let n = panel.unit.len();
    for (what, found) in [
        ("time", panel.time.len()),
        ("first_treated", panel.first_treated.len()),
        ("outcome", panel.outcome.len()),
        ("weights", panel.weights.map_or(n, <[f64]>::len)),
        ("covariates", panel.covariates.map_or(n, |x| x.nrows())),
        ("clusters", config.clusters.map_or(n, <[u32]>::len)),
    ] {
        if found != n {
            return Err(TwfeError::Length {
                what,
                expected: n,
                found,
            });
        }
    }
    check_cohorts(panel)?;
    let t_min = panel.time.iter().copied().min().unwrap_or(0);
    let time: Vec<u32> = panel
        .time
        .iter()
        .map(|&t| u32::try_from(i64::from(t) - i64::from(t_min)).map_err(|_| TwfeError::Periods))
        .collect::<Result<_, _>>()?;

    // The treatment indicators.
    let relative = |i: usize| -> Option<i32> {
        let g = panel.first_treated[i]?;
        let r = panel.time[i].saturating_sub(g);
        Some(match config.spec {
            TwfeSpec::EventStudy {
                bins: Some((lo, hi)),
                ..
            } => r.clamp(lo, hi),
            _ => r,
        })
    };
    let periods: Vec<Option<i32>> = match config.spec {
        TwfeSpec::Static => vec![None],
        TwfeSpec::EventStudy { reference, bins } => {
            if let Some((lo, hi)) = bins
                && !(lo <= reference && reference <= hi)
            {
                return Err(TwfeError::Bins);
            }
            let set: BTreeSet<i32> = (0..n)
                .filter_map(relative)
                .filter(|&r| r != reference)
                .collect();
            set.into_iter().map(Some).collect()
        }
    };
    if periods.is_empty() {
        return Err(TwfeError::NoTreatment);
    }
    let k_terms = periods.len();
    let p_cov = panel.covariates.map_or(0, |x| x.ncols());
    let mut x = Mat::<f64>::zeros(n, k_terms + p_cov);
    match config.spec {
        TwfeSpec::Static => {
            let column = x.col_as_slice_mut(0);
            for (i, v) in column.iter_mut().enumerate() {
                if panel.first_treated[i].is_some_and(|g| panel.time[i] >= g) {
                    *v = 1.0;
                }
            }
            if column.iter().all(|&v| v == 0.0) {
                return Err(TwfeError::NoTreatment);
            }
        }
        TwfeSpec::EventStudy { .. } => {
            let position: Vec<i32> = periods.iter().map(|p| p.unwrap_or(0)).collect();
            for i in 0..n {
                if let Some(r) = relative(i)
                    && let Ok(j) = position.binary_search(&r)
                {
                    x[(i, j)] = 1.0;
                }
            }
        }
    }
    if let Some(cov) = panel.covariates {
        for j in 0..p_cov {
            let column = x.col_as_slice_mut(k_terms + j);
            for (i, v) in column.iter_mut().enumerate() {
                *v = cov[(i, j)];
            }
        }
    }

    let fe = FixedEffects::new(&[panel.unit, &time], panel.weights, Singletons::Remove)?;
    let fit = Feols {
        absorb: config.absorb,
    }
    .fit(panel.outcome, x.as_ref(), &fe)?;
    let clusters = config.clusters.unwrap_or(panel.unit);
    let ways = [clusters];
    let cov = fit.vcov(Vcov::Cluster(&ways), config.ssc)?;
    let p = k_terms + p_cov;
    let se = |j: usize| cov.matrix[j * p + j].map(|v| v.max(0.0).sqrt());
    let terms = periods
        .iter()
        .enumerate()
        .map(|(j, &r)| TwfeTerm {
            relative_period: r,
            estimate: fit.coefficients[j],
            std_error: se(j),
        })
        .collect();

    // Influence by unit.
    let unit_codes = fe.codes(0);
    let units = fe.labels(0).to_vec();
    let n_units = units.len();
    let sums = fit.influence_sums(unit_codes, n_units)?;
    let mut influence = vec![None; p];
    for (a, &j) in fit.kept_columns().iter().enumerate() {
        influence[j] = Some(
            (0..n_units)
                .map(|u| n_units as f64 * sums[(u, a)])
                .collect(),
        );
    }
    Ok(TwfeResult {
        terms,
        covariate_coefficients: fit.coefficients[k_terms..].to_vec(),
        covariance: cov.matrix,
        df_k: cov.df_k,
        n_clusters: cov.min_clusters.unwrap_or(0),
        influence,
        units,
        n_rows: fe.n_rows(),
        removed_singletons: fe.removed_singletons(),
    })
}

/// Checks that each unit has one first treated period.
fn check_cohorts(panel: &StaggeredPanel<'_>) -> Result<(), TwfeError> {
    let levels = panel
        .unit
        .iter()
        .copied()
        .max()
        .map_or(0, |m| m as usize + 1);
    let mut seen: Vec<Option<Option<i32>>> = vec![None; levels];
    for (&u, &g) in panel.unit.iter().zip(panel.first_treated) {
        match seen[u as usize] {
            None => seen[u as usize] = Some(g),
            Some(h) if h != g => return Err(TwfeError::Cohort { unit: u }),
            Some(_) => {}
        }
    }
    Ok(())
}

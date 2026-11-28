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
//! rows and no unit dummies are formed. The covariance is clustered by the
//! panel's clusters, or by unit when it has none, with the small-sample
//! correction of `fixest` (`G/(G - 1) (n - 1)/(n - K)`, with the unit effects
//! nested in unit clusters not counted in `K`). The intervals and p-values use
//! the t distribution with `G - 1` degrees of freedom, as `fixest` does.
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
//! regressors. The `psi_u` sum to zero. [`Influence::variance`] over the
//! panel's clusters gives `G/(G - 1) B (sum_g s_g s_g') B`, the reported
//! covariance without its factor `(n - 1)/(n - K)`, under the
//! [`ClusterRule`](epistat_inference::ClusterRule) of the factor `G/(G - 1)`
//! of [`TwfeOptions::ssc`].
//!
//! # Interpretation
//!
//! With staggered adoption and effects that change over time or differ
//! between cohorts, the static coefficient is a weighted mean of all 2x2
//! comparisons, some with already treated units as controls and weights that
//! can be negative \[1\]; [`bacon::decompose`](crate::bacon::decompose)
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

use epistat_core::{Covariance, Estimate, EventStudy, Level};
use epistat_glm::{Absorb, Feols, FixedEffects, Singletons, Ssc, Vcov};
use epistat_inference::Influence;
use epistat_linalg::Mat;

use crate::error::Error;
use crate::methods::fe_output::{self, Reference};
use crate::methods::staggered::StaggeredPanel;

/// The treatment terms.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwfeSpec {
    /// One indicator of being treated, `feols(y ~ treat | id + t)`.
    Static,
    /// One indicator per relative period, `feols(y ~ i(rel, ref = reference) |
    /// id + t)`.
    EventStudy {
        /// The relative period left out.
        reference: i64,
        /// Relative periods below the first are counted as the first, above
        /// the second as the second.
        bins: Option<(i64, i64)>,
    },
}

impl Default for TwfeSpec {
    /// The event study with reference period -1 and no bins.
    fn default() -> Self {
        Self::EventStudy {
            reference: -1,
            bins: None,
        }
    }
}

/// The options of [`fit`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TwfeOptions {
    /// The treatment terms; by default the event study with reference period
    /// -1 and no bins.
    pub spec: TwfeSpec,
    /// The demeaning, `fixef.tol` and `fixef.iter` of `feols`.
    pub absorb: Absorb,
    /// The small-sample correction of the covariance, `ssc` of `feols`.
    pub ssc: Ssc,
    /// The level of the intervals.
    pub level: Level,
    #[doc(hidden)]
    pub _extensible: epistat_core::Extensible,
}

/// One treatment coefficient.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct TwfeTerm {
    /// The relative period, `None` for the static indicator.
    pub relative_period: Option<i64>,
    /// The coefficient with its clustered standard error, interval and
    /// p-value, `None` when the indicator is collinear.
    pub estimate: Option<Estimate>,
}

/// A two-way fixed-effects fit.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct TwfeFit {
    /// The treatment coefficients, in increasing relative period.
    pub terms: Vec<TwfeTerm>,
    /// The relative period left out of the event study, `None` for the
    /// static regression.
    pub reference: Option<i64>,
    /// The coefficients of the covariates, `None` when aliased.
    pub covariate_coefficients: Vec<Option<Estimate>>,
    /// The clustered covariance of the treatment terms and then the
    /// covariates, `None` for dropped terms.
    pub covariance: Covariance,
    /// The `K` of the small-sample correction.
    pub df_k: usize,
    /// The number of clusters.
    pub n_clusters: usize,
    /// The influence functions of the coefficients that were estimated, the
    /// terms and then the covariates, one column each in that order, with
    /// one value per unit in the fit in increasing code; `None` when no
    /// coefficient was estimated. [`TwfeFit::influence_of`] finds the column
    /// of a coefficient.
    pub influence: Option<Influence>,
    /// The units in the fit.
    pub n_units: usize,
    /// The rows in the fit.
    pub n_rows: usize,
    /// The rows removed as singletons.
    pub removed_singletons: usize,
}

impl TwfeFit {
    /// The influence function of coefficient `j`, counting the terms and
    /// then the covariates; `None` for a dropped coefficient or one out of
    /// range.
    #[must_use]
    pub fn influence_of(&self, j: usize) -> Option<&[f64]> {
        let defined = self
            .terms
            .iter()
            .map(|t| t.estimate.is_some())
            .chain(self.covariate_coefficients.iter().map(Option::is_some));
        fe_output::column_of(self.influence.as_ref(), defined, j)
    }

    /// The event study of the relative-period coefficients that were
    /// estimated, with their block of the covariance, the input of HonestDiD
    /// and of the pre-trend power.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for the static regression, and [`Error::Input`]
    /// when no coefficient was estimated or a binned period carries the
    /// reference's label.
    pub fn event_study(&self) -> Result<EventStudy, Error> {
        let Some(reference) = self.reference else {
            return Err(Error::Unsupported {
                what: "an event study of the static regression",
            });
        };
        let estimates: Vec<Option<f64>> = self
            .terms
            .iter()
            .map(|t| t.estimate.map(|e| e.estimate))
            .collect();
        let periods: Vec<i64> = self
            .terms
            .iter()
            .map(|t| t.relative_period.unwrap_or(reference))
            .collect();
        fe_output::event_study(&estimates, &periods, &self.covariance, reference)
    }
}

/// Fits the two-way fixed-effects regression of `options.spec` to `panel`,
/// with the covariance clustered by the panel's clusters, or by unit when it
/// has none.
///
/// # Errors
///
/// [`Error::NoTreatment`] for no treated row, [`Error::Bins`] for bins that do
/// not contain the reference, [`Error::Periods`] for periods that span more
/// than 2^32 values, and [`Error::Glm`] for the errors of the fit.
pub fn fit(panel: &StaggeredPanel<'_>, options: &TwfeOptions) -> Result<TwfeFit, Error> {
    let n = panel.len();
    let period = panel.period();
    let first_treated = panel.first_treated();
    let time = panel.period_codes().ok_or(Error::Periods)?;

    // The treatment indicators.
    let relative = |i: usize| -> Option<i64> {
        let g = first_treated[i]?;
        let r = period[i].saturating_sub(g);
        Some(match options.spec {
            TwfeSpec::EventStudy {
                bins: Some((lo, hi)),
                ..
            } => r.clamp(lo, hi),
            _ => r,
        })
    };
    let periods: Vec<Option<i64>> = match options.spec {
        TwfeSpec::Static => vec![None],
        TwfeSpec::EventStudy { reference, bins } => {
            if let Some((lo, hi)) = bins
                && !(lo <= reference && reference <= hi)
            {
                return Err(Error::Bins);
            }
            let set: BTreeSet<i64> = (0..n)
                .filter_map(relative)
                .filter(|&r| r != reference)
                .collect();
            set.into_iter().map(Some).collect()
        }
    };
    if periods.is_empty() {
        return Err(Error::NoTreatment);
    }
    let k_terms = periods.len();
    let p_cov = panel.covariates().map_or(0, |x| x.ncols());
    let mut x = Mat::<f64>::zeros(n, k_terms + p_cov);
    match options.spec {
        TwfeSpec::Static => {
            let column = x.col_as_slice_mut(0);
            for (i, v) in column.iter_mut().enumerate() {
                if first_treated[i].is_some_and(|g| period[i] >= g) {
                    *v = 1.0;
                }
            }
            if column.iter().all(|&v| v == 0.0) {
                return Err(Error::NoTreatment);
            }
        }
        TwfeSpec::EventStudy { .. } => {
            let position: Vec<i64> = periods.iter().map(|p| p.unwrap_or(0)).collect();
            for i in 0..n {
                if let Some(r) = relative(i)
                    && let Ok(j) = position.binary_search(&r)
                {
                    x[(i, j)] = 1.0;
                }
            }
        }
    }
    if let Some(cov) = panel.covariates() {
        for j in 0..p_cov {
            let column = x.col_as_slice_mut(k_terms + j);
            for (i, v) in column.iter_mut().enumerate() {
                *v = cov[(i, j)];
            }
        }
    }

    let fe = FixedEffects::new(
        &[panel.panel_unit()?, &time],
        panel.weights(),
        Singletons::Remove,
    )?;
    let fit = Feols {
        absorb: options.absorb,
    }
    .fit(x.as_ref(), panel.outcome(), &fe)?;
    let clusters = panel.clusters().unwrap_or(panel.panel_unit()?);
    let ways = [clusters];
    let cov = fit.vcov(&fe, Vcov::Cluster(&ways), options.ssc)?;
    let p = k_terms + p_cov;
    let reference = Reference::fixest_linear(&cov, fe.n_rows());
    let coefficient = |j: usize| -> Option<Estimate> {
        let se = cov.matrix.as_slice()[j * p + j].map(|v| v.max(0.0).sqrt());
        fit.coefficients[j].map(|b| fe_output::estimate(b, se, options.level, reference))
    };
    let terms = periods
        .iter()
        .enumerate()
        .map(|(j, &r)| TwfeTerm {
            relative_period: r,
            estimate: coefficient(j),
        })
        .collect();
    let covariate_coefficients = (k_terms..p).map(coefficient).collect();

    // Influence by unit.
    let unit_codes = fe.codes(0);
    let units = fe.labels(0).to_vec();
    let n_units = units.len();
    let sums = fit.influence_sums(&fe, unit_codes, n_units)?;
    let mut columns = vec![None; p];
    for (a, &j) in fit.kept_columns().iter().enumerate() {
        columns[j] = Some(
            (0..n_units)
                .map(|u| n_units as f64 * sums[(u, a)])
                .collect(),
        );
    }
    let influence =
        fe_output::influence(columns, units, fe_output::fixest_rule(options.ssc.g_adj))?;
    let reference = match options.spec {
        TwfeSpec::Static => None,
        TwfeSpec::EventStudy { reference, .. } => Some(reference),
    };
    Ok(TwfeFit {
        terms,
        reference,
        covariate_coefficients,
        covariance: cov.matrix,
        df_k: cov.df_k,
        n_clusters: cov.min_clusters.unwrap_or(0),
        influence,
        n_units,
        n_rows: fe.n_rows(),
        removed_singletons: fe.removed_singletons(),
    })
}

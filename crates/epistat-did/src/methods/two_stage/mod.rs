//! The two-stage difference-in-differences estimator of Gardner \[1\], as
//! `did2s::did2s` 1.2.1 \[2\] computes it with indicator second stages.
//!
//! # Algorithm
//!
//! 1. The first stage is that of the imputation estimator
//!    ([`crate::imputation`]): unit and period effects and covariates fitted
//!    by weighted least squares on the untreated rows of positive weight, with
//!    levels of one row kept. Rows whose untreated outcome cannot be imputed
//!    are dropped.
//! 2. The outcome is residualised on every remaining row,
//!    `y~_it = y_it - alpha^_i - lambda^_t - x_it' delta^`.
//! 3. The second stage regresses `y~` on the treatment indicators by weighted
//!    least squares over all rows, without an intercept: being treated
//!    ([`TwoStageSpec::Static`], `i(treat, ref = 0)`), or one indicator per
//!    period relative to treatment of the units ever treated, with the
//!    reference periods and the never treated left out
//!    ([`TwoStageSpec::EventStudy`], `i(rel, ref = c(-1, Inf))`). The
//!    indicators are exclusive, so each coefficient is the weighted mean of
//!    `y~` over its rows; an indicator without rows of positive weight has no
//!    coefficient.
//!
//! For the static effect and for the horizons `k >= 0` the coefficients are
//! the imputation estimates with the prior weights as target weights \[3\];
//! the tests check this equality.
//!
//! # Variance
//!
//! The coefficients `beta2` solve the moment conditions of the two stages
//! jointly, and the GMM variance \[1\] has the influence of row `i`
//!
//! ```text
//! psi_i = B2 [ w_i x2_i e2_i - 1[i untreated] w_i Gamma' z_i u_i ],
//! B2 = (X2' W X2)^-1,      Gamma = (Z0' W0 Z0)^-1 Z' W X2,
//! ```
//!
//! with `e2` the second-stage residual, `u` the first-stage residual and `z`
//! the first stage's covariates and dummies. The covariance is
//! `sum_c (sum_{i in c} psi_i)(sum_{i in c} psi_i)'` over the clusters,
//! without small-sample factors, as in the package. The products
//! `z_i' Gamma` on the untreated rows are the projections of the module
//! `first_stage` of [`crate::imputation`] with the column `x2k` on the
//! untreated rows and `w x2k` on the treated rows, so no dummies are formed.
//! Clusters must contain whole units; they are the units by default.
//!
//! # Influence
//!
//! The influence function of a coefficient for unit `u` is `N sum_t
//! psi_ut`, `N` the units in the estimation; it sums to zero over the units
//! (the two stages' normal equations), so
//! [`did_cluster_sums`](crate::did_cluster_sums) over the clusters gives the
//! covariance above.
//!
//! # References
//!
//! 1. Gardner J. Two-stage differences in differences. Working paper, 2021.
//!    arXiv:2207.05943
//! 2. Butts K, Gardner J. did2s: Two-stage difference-in-differences. R J.
//!    2022;14(3):162-73. The R package `did2s` 1.2.1: `did2s` and
//!    `did2s_estimate`.
//! 3. Borusyak K, Jaravel X, Spiess J. Revisiting event-study designs: robust
//!    and efficient estimation. Rev Econ Stud. 2024;91(6):3253-85.
//!    doi:10.1093/restud/rdae007

use std::collections::BTreeSet;

use epistat_glm::{Absorb, Feols};

use crate::methods::imputation::first_stage::{FirstStage, Projection, Projector};
use crate::methods::imputation::{
    ImputationError, Prepared, UnitIndex, first_stage_fixed_effects, outer_covariance,
};
use crate::methods::staggered::StaggeredPanel;

/// The second-stage indicators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwoStageSpec<'a> {
    /// One indicator of being treated.
    Static,
    /// One indicator per period relative to treatment of the units ever
    /// treated, but the reference periods.
    EventStudy {
        /// The relative periods left out, each negative.
        references: &'a [i32],
    },
}

/// The controls of the two-stage estimator.
#[derive(Clone, Copy, Debug)]
pub struct TwoStageConfig<'a> {
    /// The second stage.
    pub spec: TwoStageSpec<'a>,
    /// Cluster codes, one per row and constant within units; the units when
    /// `None`.
    pub clusters: Option<&'a [u32]>,
    /// The demeaning of the first stage and of the variance's projections.
    pub absorb: Absorb,
    /// Whether to return the projections `z_i' Gamma` of each term.
    pub keep_projections: bool,
}

impl TwoStageConfig<'_> {
    /// The static effect, clustered by unit.
    #[must_use]
    pub const fn static_effect() -> Self {
        Self {
            spec: TwoStageSpec::Static,
            clusters: None,
            absorb: Absorb::new(),
            keep_projections: false,
        }
    }

    /// The event study with reference period -1, clustered by unit.
    #[must_use]
    pub const fn event_study() -> Self {
        Self {
            spec: TwoStageSpec::EventStudy { references: &[-1] },
            ..Self::static_effect()
        }
    }
}

/// One second-stage coefficient.
#[derive(Clone, Debug, PartialEq)]
pub struct TwoStageTerm {
    /// The relative period, `None` for the static indicator.
    pub relative_period: Option<i32>,
    /// The coefficient, `None` when the indicator has no rows of positive
    /// weight.
    pub estimate: Option<f64>,
    /// Its corrected clustered standard error.
    pub std_error: Option<f64>,
}

/// The two-stage estimates.
#[derive(Clone, Debug)]
pub struct TwoStageResult {
    /// The coefficients, in increasing relative period.
    pub terms: Vec<TwoStageTerm>,
    /// Their corrected clustered covariance, row-major.
    pub covariance: Vec<Option<f64>>,
    /// The influence function of each coefficient, one value per unit of
    /// [`TwoStageResult::units`].
    pub influence: Vec<Option<Vec<f64>>>,
    /// The units in the estimation, in increasing code.
    pub units: Vec<u32>,
    /// The number of clusters.
    pub n_clusters: usize,
    /// The coefficients of the first stage's covariates, `None` if aliased.
    pub covariate_coefficients: Vec<Option<f64>>,
    /// The rows in the estimation.
    pub n_rows: usize,
    /// The untreated rows in the first stage.
    pub n_untreated: usize,
    /// The rows dropped because their untreated outcome cannot be imputed.
    pub dropped_rows: Vec<usize>,
    /// The residualised outcome `y - y^(0)` of every row, `None` on dropped
    /// rows.
    pub residualised: Vec<Option<f64>>,
    /// With [`TwoStageConfig::keep_projections`], `z_i' Gamma` of each term
    /// on the untreated rows of the first stage, one value per input row and
    /// zero elsewhere.
    pub projections: Option<Vec<Vec<f64>>>,
}

/// Estimates Gardner's two-stage estimator. See the module documentation.
///
/// # Errors
///
/// [`ImputationError`] for inputs of the wrong length or not finite, cohorts
/// or clusters that vary within a unit, reference periods that are not
/// negative, and the errors of the first stage.
pub fn estimate_did2s(
    panel: &StaggeredPanel<'_>,
    config: &TwoStageConfig<'_>,
) -> Result<TwoStageResult, ImputationError> {
    let n = panel.unit.len();
    if let TwoStageSpec::EventStudy { references } = config.spec
        && let Some(&bad) = references.iter().find(|&&r| r >= 0)
    {
        return Err(ImputationError::Reference(bad));
    }
    let prepared = Prepared::new(panel, config.clusters)?;
    let fe = first_stage_fixed_effects(panel, &prepared)?;
    let feols = Feols {
        absorb: config.absorb,
    };
    let x = prepared.covariates.as_ref();
    let fit = feols.fit(x, panel.outcome, &fe)?;
    let stage = FirstStage::new(
        panel.unit,
        &prepared.time,
        prepared.treated.clone(),
        panel.outcome,
        prepared.weight.clone(),
        x,
        &fe,
        &fit,
    )?;
    let used: Vec<bool> = stage.adjusted.iter().map(Option::is_some).collect();
    let dropped_rows: Vec<usize> = (0..n).filter(|&i| !used[i]).collect();
    let treated_rows: Vec<usize> = (0..n).filter(|&i| used[i] && stage.treated[i]).collect();
    let index = UnitIndex::new(panel.unit, &used, config.clusters);
    let n_units = index.units.len();

    // The indicator of each row: its term, or None.
    let periods: Vec<Option<i32>> = match config.spec {
        TwoStageSpec::Static => vec![None],
        TwoStageSpec::EventStudy { references } => {
            let set: BTreeSet<i32> = (0..n)
                .filter(|&i| used[i] && stage.weight[i] > 0.0)
                .filter_map(|i| prepared.event[i])
                .filter(|e| !references.contains(e))
                .collect();
            set.into_iter().map(Some).collect()
        }
    };
    let term_of = |i: usize| -> Option<usize> {
        match config.spec {
            TwoStageSpec::Static => stage.treated[i].then_some(0),
            TwoStageSpec::EventStudy { .. } => {
                let e = prepared.event[i]?;
                periods.binary_search(&Some(e)).ok()
            }
        }
    };
    let k_terms = periods.len();
    let mut weight_sum = vec![0.0; k_terms];
    let mut weighted_y = vec![0.0; k_terms];
    let mut row_term: Vec<Option<usize>> = vec![None; n];
    for i in 0..n {
        let Some(adjusted) = stage.adjusted[i] else {
            continue;
        };
        if let Some(k) = term_of(i) {
            row_term[i] = Some(k);
            weight_sum[k] += stage.weight[i];
            weighted_y[k] += stage.weight[i] * adjusted;
        }
    }
    let beta: Vec<Option<f64>> = (0..k_terms)
        .map(|k| (weight_sum[k] > 0.0).then(|| weighted_y[k] / weight_sum[k]))
        .collect();

    let projector = Projector::new(&fe, &fit, x, &[panel.unit, &prepared.time]);
    let fit_rows = fe.rows();
    let fit_weights = fe.weights();
    let mut sums: Vec<Option<Vec<f64>>> = Vec::with_capacity(k_terms);
    let mut influence = Vec::with_capacity(k_terms);
    let mut kept_projections = config.keep_projections.then(Vec::new);
    for k in 0..k_terms {
        let Some(b) = beta[k] else {
            sums.push(None);
            influence.push(None);
            if let Some(p) = kept_projections.as_mut() {
                p.push(vec![0.0; n]);
            }
            continue;
        };
        let mut base = vec![0.0; n];
        let mut treated = vec![0.0; n];
        for i in 0..n {
            if row_term[i] == Some(k) {
                if stage.treated[i] {
                    treated[i] = stage.weight[i];
                } else {
                    base[i] = 1.0;
                }
            }
        }
        let h = projector.project(
            &Projection {
                base: Some(&base),
                treated: &treated,
            },
            &treated_rows,
            &config.absorb,
        )?;
        let scale = 1.0 / weight_sum[k];
        let mut by_unit = vec![0.0; n_units];
        for i in 0..n {
            if row_term[i] == Some(k) {
                let e2 = stage.adjusted[i].unwrap_or(0.0) - b;
                let u = index.position[panel.unit[i] as usize] as usize;
                by_unit[u] += scale * stage.weight[i] * e2;
            }
        }
        for (j, &i) in fit_rows.iter().enumerate() {
            let w = fit_weights.map_or(1.0, |w| w[j]);
            let u = index.position[panel.unit[i] as usize] as usize;
            by_unit[u] -= scale * w * h[j] * stage.adjusted[i].unwrap_or(0.0);
        }
        sums.push(Some(index.cluster_sums(&by_unit)));
        influence.push(Some(by_unit.iter().map(|s| s * n_units as f64).collect()));
        if let Some(p) = kept_projections.as_mut() {
            let mut full = vec![0.0; n];
            for (j, &i) in fit_rows.iter().enumerate() {
                full[i] = h[j];
            }
            p.push(full);
        }
    }
    let covariance = outer_covariance(&sums);
    let terms = periods
        .iter()
        .enumerate()
        .map(|(k, &r)| TwoStageTerm {
            relative_period: r,
            estimate: beta[k],
            std_error: covariance[k * k_terms + k].map(|v| v.max(0.0).sqrt()),
        })
        .collect();
    Ok(TwoStageResult {
        terms,
        covariance,
        influence,
        units: index.units,
        n_clusters: index.clusters.count(),
        covariate_coefficients: stage.coefficients.clone(),
        n_rows: n - dropped_rows.len(),
        n_untreated: fe.n_rows(),
        dropped_rows,
        residualised: stage.adjusted,
        projections: kept_projections,
    })
}

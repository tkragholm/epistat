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
//! without small-sample factors, as in the package. The intervals and
//! p-values use the t distribution with `n - K` degrees of freedom, `n` the
//! rows of positive weight in the estimation and `K` the coefficients, as
//! `fixest` forms them from the package's covariance. The products
//! `z_i' Gamma` on the untreated rows are the projections of the module
//! `first_stage` of [`crate::imputation`] with the column `x2k` on the
//! untreated rows and `w x2k` on the treated rows, so no dummies are formed.
//! Clusters must contain whole units; they are the units by default.
//!
//! # Influence
//!
//! The influence function of a coefficient for unit `u` is `N sum_t
//! psi_ut`, `N` the units in the estimation; it sums to zero over the units
//! (the two stages' normal equations), so [`Influence::variance`] over the
//! clusters gives the covariance above, under its rule
//! [`ClusterRule::NoSmallSample`](epistat_inference::ClusterRule).
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

use epistat_core::{Covariance, Estimate, Level};
use epistat_glm::{Absorb, Feols};
use epistat_inference::{ClusterRule, Influence};

use crate::error::Error;
use crate::methods::fe_output::{self, Reference};
use crate::methods::imputation::first_stage::{FirstStage, Projection, Projector};
use crate::methods::imputation::{
    Prepared, UnitIndex, first_stage_fixed_effects, outer_covariance,
};
use crate::methods::staggered::StaggeredPanel;

/// The second-stage indicators.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum TwoStageSpec {
    /// One indicator of being treated, `second_stage = ~ i(treat, ref =
    /// FALSE)`.
    #[default]
    Static,
    /// One indicator per period relative to treatment of the units ever
    /// treated, but the reference periods, `~ i(rel, ref = c(-1, Inf))`.
    EventStudy {
        /// The relative periods left out, each negative.
        references: Vec<i64>,
    },
}

/// The options of [`estimate`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TwoStageOptions {
    /// The second stage; by default the static effect.
    pub spec: TwoStageSpec,
    /// The demeaning of the first stage and of the variance's projections.
    pub absorb: Absorb,
    /// Whether to return the projections `z_i' Gamma` of each term.
    pub keep_projections: bool,
    /// The level of the intervals.
    pub level: Level,
}

/// One second-stage coefficient.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct TwoStageTerm {
    /// The relative period, `None` for the static indicator.
    pub relative_period: Option<i64>,
    /// The coefficient with its corrected clustered standard error, interval
    /// and p-value, `None` when the indicator has no rows of positive weight.
    pub estimate: Option<Estimate>,
}

/// The two-stage estimates.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct TwoStage {
    /// The coefficients, in increasing relative period.
    pub terms: Vec<TwoStageTerm>,
    /// Their corrected clustered covariance, `None` where a coefficient is
    /// undefined.
    pub covariance: Covariance,
    /// The influence functions of the coefficients that are defined, one
    /// column each in their order with one value per unit in the estimation
    /// in increasing code; `None` when none is. [`TwoStage::influence_of`]
    /// finds the column of a coefficient.
    pub influence: Option<Influence>,
    /// The units in the estimation.
    pub n_units: usize,
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
    /// With [`TwoStageOptions::keep_projections`], `z_i' Gamma` of each term
    /// on the untreated rows of the first stage, one value per input row and
    /// zero elsewhere.
    pub projections: Option<Vec<Vec<f64>>>,
}

impl TwoStage {
    /// The influence function of coefficient `j`; `None` for an undefined
    /// coefficient or one out of range.
    #[must_use]
    pub fn influence_of(&self, j: usize) -> Option<&[f64]> {
        let defined = self.terms.iter().map(|t| t.estimate.is_some());
        fe_output::column_of(self.influence.as_ref(), defined, j)
    }
}

/// Estimates Gardner's two-stage estimator, with standard errors clustered by
/// the panel's clusters, or by unit when it has none. See the module
/// documentation.
///
/// # Errors
///
/// [`Error::Reference`] for a reference period that is not negative,
/// [`Error::Periods`] for periods out of range, and [`Error::Glm`] for the
/// errors of the first stage.
#[expect(
    clippy::too_many_lines,
    reason = "one pass from the first stage to the second and its variance"
)]
pub fn estimate(panel: &StaggeredPanel<'_>, options: &TwoStageOptions) -> Result<TwoStage, Error> {
    let n = panel.len();
    if let TwoStageSpec::EventStudy { references } = &options.spec
        && let Some(&bad) = references.iter().find(|&&r| r >= 0)
    {
        return Err(Error::Reference(bad));
    }
    let prepared = Prepared::new(panel)?;
    let fe = first_stage_fixed_effects(panel, &prepared)?;
    let feols = Feols {
        absorb: options.absorb,
    };
    let x = prepared.covariates.as_ref();
    let fit = feols.fit(x, panel.outcome(), &fe)?;
    let stage = FirstStage::new(
        panel.unit(),
        &prepared.time,
        prepared.treated.clone(),
        panel.outcome(),
        prepared.weight.clone(),
        x,
        &fe,
        &fit,
    )?;
    let used: Vec<bool> = stage.adjusted.iter().map(Option::is_some).collect();
    let dropped_rows: Vec<usize> = (0..n).filter(|&i| !used[i]).collect();
    let treated_rows: Vec<usize> = (0..n).filter(|&i| used[i] && stage.treated[i]).collect();
    let index = UnitIndex::new(panel.unit(), &used, panel.clusters());
    let n_units = index.units.len();

    // The indicator of each row: its term, or None.
    let periods: Vec<Option<i64>> = match &options.spec {
        TwoStageSpec::Static => vec![None],
        TwoStageSpec::EventStudy { references } => {
            let set: BTreeSet<i64> = (0..n)
                .filter(|&i| used[i] && stage.weight[i] > 0.0)
                .filter_map(|i| prepared.event[i])
                .filter(|e| !references.contains(e))
                .collect();
            set.into_iter().map(Some).collect()
        }
    };
    let term_of = |i: usize| -> Option<usize> {
        match options.spec {
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

    let projector = Projector::new(&fe, &fit, x, &[panel.unit(), &prepared.time]);
    let fit_rows = fe.rows();
    let fit_weights = fe.weights();
    let mut sums: Vec<Option<Vec<f64>>> = Vec::with_capacity(k_terms);
    let mut influence = Vec::with_capacity(k_terms);
    let mut kept_projections = options.keep_projections.then(Vec::new);
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
            &options.absorb,
        )?;
        let scale = 1.0 / weight_sum[k];
        let mut by_unit = vec![0.0; n_units];
        for i in 0..n {
            if row_term[i] == Some(k) {
                let e2 = stage.adjusted[i].unwrap_or(0.0) - b;
                let u = index.position[panel.unit()[i] as usize] as usize;
                by_unit[u] += scale * stage.weight[i] * e2;
            }
        }
        for (j, &i) in fit_rows.iter().enumerate() {
            let w = fit_weights.map_or(1.0, |w| w[j]);
            let u = index.position[panel.unit()[i] as usize] as usize;
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
    let covariance = outer_covariance(&sums)?;
    // fixest's residual degrees of freedom of the second stage: its rows of
    // positive weight less its coefficients.
    let n_fit = (0..n)
        .filter(|&i| stage.adjusted[i].is_some() && stage.weight[i] > 0.0)
        .count();
    let k_fit = beta.iter().flatten().count();
    let reference = Reference::StudentT(n_fit.saturating_sub(k_fit).max(1) as f64);
    let terms = periods
        .iter()
        .enumerate()
        .map(|(k, &r)| TwoStageTerm {
            relative_period: r,
            estimate: beta[k].map(|b| {
                let se = covariance.get(k, k).map(|v| v.max(0.0).sqrt());
                fe_output::estimate(b, se, options.level, reference)
            }),
        })
        .collect();
    Ok(TwoStage {
        terms,
        covariance,
        influence: fe_output::influence(influence, index.units, ClusterRule::NoSmallSample)?,
        n_units,
        n_clusters: index.clusters.count(),
        covariate_coefficients: stage.coefficients.clone(),
        n_rows: n - dropped_rows.len(),
        n_untreated: fe.n_rows(),
        dropped_rows,
        residualised: stage.adjusted,
        projections: kept_projections,
    })
}

//! The imputation estimator of Borusyak, Jaravel and Spiess \[1\] for
//! staggered adoption, as `didimputation::did_imputation` 0.5.1 \[3\]
//! computes it.
//!
//! # Algorithm
//!
//! Unit `i` is first treated in period `g_i` (never, for the never treated),
//! and row `(i, t)` is treated when `t >= g_i`. The untreated outcomes follow
//!
//! ```text
//! Y_it(0) = alpha_i + lambda_t + x_it' delta + e_it
//! ```
//!
//! 1. The unit and period effects and `delta` are fitted by weighted least
//!    squares on the untreated rows of positive weight: the effects are
//!    absorbed by [`epistat_glm::absorb`] and the fit is
//!    [`epistat_glm::feols`], so the cost is linear in the rows and no dummies
//!    are formed. Levels with one row are kept (`fixef.rm = "none"`).
//! 2. The untreated outcome of every row is imputed, `Y^_it(0) = alpha^_i +
//!    lambda^_t + x_it' delta^`, and `tau^_it = Y_it - Y^_it(0)`. A row whose
//!    unit or period has no untreated row of positive weight cannot be
//!    imputed and is dropped, as the package drops it. So is a row whose unit
//!    and period lie in different connected components of the untreated rows,
//!    where the sum of the two effects is not identified; the package imputes
//!    such a row with whatever normalisation `fixest::fixef` chose.
//! 3. An estimand is a weighted mean of the `tau^_it` over the treated rows
//!    that remain, `tau^_w = sum w_it tau^_it`, with weights `w_it`
//!    proportional to the prior weight times a target: being treated (the
//!    static effect), `t - g_i = h` for a horizon `h`, or the caller's column,
//!    scaled to sum to one.
//!
//! # Variance
//!
//! The estimator is linear in the outcomes, `tau^_w = sum_it v_it Y_it`, with
//! `v_it = w_it` on the treated rows and
//!
//! ```text
//! v_0 = -W0 Z0 (Z0' W0 Z0)^-1 Z1' w_1
//! ```
//!
//! on the untreated rows, `Z` the covariates and the unit and period dummies,
//! `W0` the prior weights of the untreated rows (equation 6 of \[1\]). The
//! fitted values `Z0 (Z0' W0 Z0)^-1 Z1' w_1` are found without forming the
//! dummies, by the projection of the module `first_stage`. The conservative
//! variance of \[1\] (theorem 3, equation 8, clustered) is
//!
//! ```text
//! V = sum_c ( sum_{it in c} v_it e~_it )^2
//! ```
//!
//! with `e~_it` the first-stage residual on the untreated rows and
//! `tau^_it - tau_bar_{g t}` on the treated rows, where the auxiliary model of
//! the effects is the mean by cohort and period (equation 10),
//!
//! ```text
//! tau_bar_{g t} = sum_{cohort g, period t} v_it^2 tau^_it / sum v_it^2,
//! ```
//!
//! zero where the sum of `v^2` is zero. There is no small-sample factor, as in
//! the package, and no leave-out correction, which the package does not
//! implement. Clusters must contain whole units; they are the units by
//! default.
//!
//! # Pre-trend coefficients
//!
//! The test of \[1\] (section 4.4) fits, on the untreated rows only, the
//! first-stage model with indicators of the periods relative to treatment of
//! the units treated later, `1[t - g_i = k]`. As in the package, which writes
//! the indicators with `fixest::i`, the smallest relative period present (the
//! never treated, when there are any) is the reference, the rest are fitted
//! with all ([`Pretrends::All`]) or with those requested
//! ([`Pretrends::Periods`]), an indicator that is collinear with the others and
//! the fixed effects is dropped, and the covariance is `fixest`'s clustered
//! one with its default small-sample correction.
//!
//! # Influence
//!
//! The influence function of an estimand for unit `u` is `N sum_{t} v_ut
//! e~_ut`, with `N` the units in the estimation, so that the sums of the
//! influence within clusters, divided by `N`, are the terms of `V`. Its rule
//! is [`ClusterRule::NoSmallSample`](epistat_inference::ClusterRule), as the
//! package has no small-sample factor. [`Influence::variance`] centres the
//! influence, which changes the variance by `N` times the squared mean; the
//! mean is zero when the weights are constant within each cohort and period
//! (no prior weights, or weights that are constant within cells).
//!
//! # Intervals
//!
//! The intervals and p-values of the estimands and of the pre-trend
//! coefficients use the normal distribution at [`ImputationOptions::level`],
//! as the package forms `estimate +- 1.96 se`; at the default level the
//! quantile is `1.959964`, not the package's rounded `1.96`.
//!
//! # Weights in the variance
//!
//! With prior weights that vary, `did_imputation` 0.5.1 computes `v_0` as
//! `-W0 Z0 (Z0' W0^2 Z0)^-1 Z1' W1 w_1`, which is not the derivative of its
//! own estimate; with constant weights the two agree. This module uses the
//! formula above; the reference script records both (`tests/data/
//! generate_imputation_reference.R`).
//!
//! # References
//!
//! 1. Borusyak K, Jaravel X, Spiess J. Revisiting event-study designs: robust
//!    and efficient estimation. Rev Econ Stud. 2024;91(6):3253-85.
//!    doi:10.1093/restud/rdae007
//! 2. Gardner J. Two-stage differences in differences. Working paper, 2021.
//!    arXiv:2207.05943
//! 3. Butts K. The R package `didimputation` 0.5.1: `did_imputation` and
//!    `se_inner`.

pub(crate) mod first_stage;

use std::collections::{BTreeSet, HashMap};

use epistat_core::{Covariance, Estimate, Level, check};
use epistat_glm::{Absorb, Feols, FixedEffects, Singletons, Ssc, Vcov};
use epistat_inference::{ClusterRule, Clusters, Influence};
use epistat_linalg::Mat;

use first_stage::{FirstStage, Projection, Projector};

use crate::error::Error;
use crate::methods::fe_output::{self, Reference};
use crate::methods::staggered::StaggeredPanel;

/// What the imputed effects are averaged into.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub enum ImputationTarget {
    /// One mean over every treated row, the package's default.
    #[default]
    Static,
    /// One mean per horizon `t - g >= 0`: the given horizons, or with `None`
    /// every horizon of the panel (`horizon = TRUE`).
    Horizons(Option<Vec<i64>>),
    /// One mean per column of the caller's weights, each one value per row
    /// of the panel, read on the treated rows and zero on the untreated ones.
    Weights(Vec<Vec<f64>>),
}

/// Which pre-trend coefficients to fit.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pretrends {
    /// Every relative period before treatment but the reference
    /// (`pretrends = TRUE`).
    All,
    /// These relative periods, each negative.
    Periods(Vec<i64>),
}

/// The options of [`estimate`], with the defaults of `did_imputation`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImputationOptions {
    /// The estimands; by default the static effect.
    pub target: ImputationTarget,
    /// The pre-trend coefficients, or `None`, the default.
    pub pretrends: Option<Pretrends>,
    /// The demeaning of the first stage and of the variance's projections.
    pub absorb: Absorb,
    /// Whether to return the weights `v_it` of every estimand.
    pub keep_weights: bool,
    /// The level of the intervals.
    pub level: Level,
}

/// What an estimand averages.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImputationTerm {
    /// Every treated row.
    Treated,
    /// The rows at this horizon.
    Horizon(i64),
    /// The caller's weight column with this index.
    Weights(usize),
}

/// One estimand.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct ImputationEffect {
    /// What it averages.
    pub term: ImputationTerm,
    /// The estimate with its conservative clustered standard error, interval
    /// and p-value, `None` when no imputed row carries weight.
    pub estimate: Option<Estimate>,
    /// The treated rows with positive weight in the estimand.
    pub n_rows: usize,
}

/// One pre-trend coefficient.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct PretrendTerm {
    /// The period relative to treatment.
    pub relative_period: i64,
    /// The coefficient with its clustered standard error, interval and
    /// p-value, `None` when the indicator is absent or collinear.
    pub estimate: Option<Estimate>,
}

/// The pre-trend regression.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ImputationPretrends {
    /// The coefficients, in increasing relative period.
    pub terms: Vec<PretrendTerm>,
    /// Their clustered covariance with `fixest`'s default small-sample
    /// correction, `None` for absent or dropped terms.
    pub covariance: Covariance,
    /// The influence functions of the coefficients that were estimated, one
    /// column each in their order with one value per unit in the estimation
    /// in increasing code, under `fixest`'s factor `G/(G - 1)`; `None` when
    /// none was. [`ImputationPretrends::influence_of`] finds the column of a
    /// coefficient.
    pub influence: Option<Influence>,
}

impl ImputationPretrends {
    /// The influence function of coefficient `j`; `None` for an absent or
    /// dropped coefficient or one out of range.
    #[must_use]
    pub fn influence_of(&self, j: usize) -> Option<&[f64]> {
        let defined = self.terms.iter().map(|t| t.estimate.is_some());
        fe_output::column_of(self.influence.as_ref(), defined, j)
    }
}

/// The imputation estimates.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Imputation {
    /// The estimands, in the order of the target (horizons increasing).
    pub effects: Vec<ImputationEffect>,
    /// The covariance of the estimands, `sum_c s_c s_c'`, `None` where an
    /// estimand is undefined.
    pub covariance: Covariance,
    /// The influence functions of the estimands that are defined, one column
    /// each in their order with one value per unit in the estimation in
    /// increasing code; `None` when none is. [`Imputation::influence_of`]
    /// finds the column of an estimand.
    pub influence: Option<Influence>,
    /// The units in the estimation.
    pub n_units: usize,
    /// The number of clusters.
    pub n_clusters: usize,
    /// The pre-trend regression, when asked for.
    pub pretrends: Option<ImputationPretrends>,
    /// The coefficients of the first stage's covariates, `None` if aliased.
    pub covariate_coefficients: Vec<Option<f64>>,
    /// The rows in the estimation.
    pub n_rows: usize,
    /// The untreated rows in the first stage.
    pub n_untreated: usize,
    /// The rows dropped because their untreated outcome cannot be imputed.
    pub dropped_rows: Vec<usize>,
    /// `y - y^(0)` on every row: the imputed effect on treated rows, the
    /// first-stage residual on untreated rows, `None` on dropped rows.
    pub residualised: Vec<Option<f64>>,
    /// With [`ImputationOptions::keep_weights`], the weights `v_it` of each
    /// estimand, one per input row, zero on rows outside the estimation.
    pub weights: Option<Vec<Vec<f64>>>,
}

impl Imputation {
    /// The influence function of estimand `j`; `None` for an undefined
    /// estimand or one out of range.
    #[must_use]
    pub fn influence_of(&self, j: usize) -> Option<&[f64]> {
        let defined = self.effects.iter().map(|e| e.estimate.is_some());
        fe_output::column_of(self.influence.as_ref(), defined, j)
    }
}

/// The checked panel and the first stage's inputs.
pub(crate) struct Prepared {
    /// Period codes from 0.
    pub(crate) time: Vec<u32>,
    /// `t - g` for rows of units that are ever treated.
    pub(crate) event: Vec<Option<i64>>,
    pub(crate) treated: Vec<bool>,
    pub(crate) weight: Vec<f64>,
    /// The weights of the first stage: the prior weight on untreated rows,
    /// zero on treated ones.
    pub(crate) fit_weight: Vec<f64>,
    pub(crate) covariates: Mat<f64>,
}

impl Prepared {
    pub(crate) fn new(panel: &StaggeredPanel<'_>) -> Result<Self, Error> {
        let n = panel.len();
        let period = panel.period();
        let covariates = panel
            .covariates()
            .map_or_else(|| Mat::<f64>::zeros(n, 0), |x| x.to_owned());
        let time = panel.period_codes().ok_or(Error::Periods)?;
        let event: Vec<Option<i64>> = period
            .iter()
            .zip(panel.first_treated())
            .map(|(&t, g)| match g {
                None => Ok(None),
                Some(g) => t.checked_sub(*g).map(Some).ok_or(Error::Periods),
            })
            .collect::<Result<_, _>>()?;
        let treated: Vec<bool> = event.iter().map(|e| e.is_some_and(|e| e >= 0)).collect();
        let weight: Vec<f64> = panel
            .weights()
            .map_or_else(|| vec![1.0; n], <[f64]>::to_vec);
        let fit_weight: Vec<f64> = weight
            .iter()
            .zip(&treated)
            .map(|(&w, &d)| if d { 0.0 } else { w })
            .collect();
        Ok(Self {
            time,
            event,
            treated,
            weight,
            fit_weight,
            covariates,
        })
    }
}

/// The units of the estimation and the clusters over them.
pub(crate) struct UnitIndex {
    /// The units, in increasing code.
    pub(crate) units: Vec<u32>,
    /// The position in `units` of each caller code.
    pub(crate) position: Vec<u32>,
    pub(crate) clusters: Clusters,
}

impl UnitIndex {
    pub(crate) fn new(
        unit: &[u32],
        used: &[bool],
        clusters: Option<&[u32]>,
    ) -> Result<Self, Error> {
        let levels = unit.iter().copied().max().map_or(0, |m| m as usize + 1);
        let mut present = vec![false; levels];
        let mut cluster_of = vec![0u32; levels];
        for (i, &u) in unit.iter().enumerate() {
            if used[i] {
                present[u as usize] = true;
                cluster_of[u as usize] = clusters.map_or(u, |c| c[i]);
            }
        }
        let mut position = vec![u32::MAX; levels];
        let mut units = Vec::new();
        let mut labels = Vec::new();
        for (code, &p) in present.iter().enumerate() {
            if p {
                position[code] = u32::try_from(units.len()).unwrap_or(u32::MAX);
                let code = u32::try_from(code).unwrap_or(u32::MAX);
                units.push(code);
                labels.push(cluster_of[code as usize]);
            }
        }
        Ok(Self {
            units,
            position,
            clusters: Clusters::from_labels_first_seen(&labels)?,
        })
    }

    /// The sums by cluster of per-unit values.
    pub(crate) fn cluster_sums(&self, by_unit: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.clusters.count()];
        for (value, &c) in by_unit.iter().zip(self.clusters.of()) {
            out[c as usize] += value;
        }
        out
    }
}

/// The covariance `sum_c s_c s_c'` of columns of cluster sums, `None` where
/// a column is undefined.
pub(crate) fn outer_covariance(sums: &[Option<Vec<f64>>]) -> Result<Covariance, Error> {
    let k = sums.len();
    let mut out = vec![None; k * k];
    for a in 0..k {
        for b in 0..k {
            if let (Some(x), Some(y)) = (&sums[a], &sums[b]) {
                let mut total = 0.0;
                for (p, q) in x.iter().zip(y) {
                    total += p * q;
                }
                out[a * k + b] = Some(total);
            }
        }
    }
    Ok(Covariance::new(k, out)?)
}

/// Fits the first stage of a staggered panel: the fixed effects of the
/// untreated rows, the fit and the imputed outcomes.
pub(crate) fn first_stage_fixed_effects(
    panel: &StaggeredPanel<'_>,
    prepared: &Prepared,
) -> Result<FixedEffects, Error> {
    Ok(FixedEffects::new(
        &[panel.panel_unit()?, &prepared.time],
        Some(&prepared.fit_weight),
        Singletons::Keep,
    )?)
}

/// Estimates the imputation estimator of Borusyak, Jaravel and Spiess, with
/// standard errors clustered by the panel's clusters, or by unit when it has
/// none. See the module documentation.
///
/// # Errors
///
/// [`Error::Input`] for target weights of the wrong length or not finite,
/// [`Error::Horizon`] for a negative horizon, [`Error::Pretrend`] for a
/// pre-trend period that is not negative, [`Error::TargetOnUntreated`] for
/// target weights on untreated rows, [`Error::Periods`] for periods out of
/// range, and [`Error::Glm`] for the errors of the first stage.
#[expect(
    clippy::too_many_lines,
    reason = "one pass from the first stage to the estimands and their variance"
)]
pub fn estimate(
    panel: &StaggeredPanel<'_>,
    options: &ImputationOptions,
) -> Result<Imputation, Error> {
    let n = panel.len();
    let prepared = Prepared::new(panel)?;
    let fe = first_stage_fixed_effects(panel, &prepared)?;
    let feols = Feols {
        absorb: options.absorb,
    };
    let x = prepared.covariates.as_ref();
    let fit = feols.fit(x, panel.outcome(), &fe)?;
    let stage = FirstStage::new(
        panel.panel_unit()?,
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
    let index = UnitIndex::new(panel.panel_unit()?, &used, panel.clusters())?;
    let n_units = index.units.len();

    // The estimands.
    let (terms, columns): (Vec<ImputationTerm>, Vec<Column<'_>>) = match &options.target {
        ImputationTarget::Static => (vec![ImputationTerm::Treated], vec![Column::Treated]),
        ImputationTarget::Horizons(given) => {
            let horizons: BTreeSet<i64> = if let Some(h) = given {
                if let Some(&bad) = h.iter().find(|&&h| h < 0) {
                    return Err(Error::Horizon(bad));
                }
                h.iter().copied().collect()
            } else {
                prepared
                    .event
                    .iter()
                    .filter_map(|e| e.filter(|&e| e >= 0))
                    .collect()
            };
            (
                horizons
                    .iter()
                    .map(|&h| ImputationTerm::Horizon(h))
                    .collect(),
                horizons.iter().map(|&h| Column::Horizon(h)).collect(),
            )
        }
        ImputationTarget::Weights(given) => {
            for (j, column) in given.iter().enumerate() {
                check::same_length("target weights", n, column.len())?;
                check::finite("target weights", column)?;
                if let Some(row) = (0..n).find(|&i| column[i] != 0.0 && !prepared.treated[i]) {
                    return Err(Error::TargetOnUntreated { column: j, row });
                }
            }
            (
                (0..given.len()).map(ImputationTerm::Weights).collect(),
                given.iter().map(|c| Column::Weights(c)).collect(),
            )
        }
    };

    // Cohort-by-period cells of the treated rows, for tau_bar.
    let mut cell_of: HashMap<(i64, i64), usize> = HashMap::new();
    let cells: Vec<usize> = treated_rows
        .iter()
        .map(|&i| {
            let key = (panel.first_treated()[i].unwrap_or(0), panel.period()[i]);
            let next = cell_of.len();
            *cell_of.entry(key).or_insert(next)
        })
        .collect();
    let n_cells = cell_of.len();

    let projector = Projector::new(&fe, &fit, x, &[panel.panel_unit()?, &prepared.time]);
    let fit_rows = fe.rows();
    let fit_weights = fe.weights();
    let mut effects = Vec::with_capacity(terms.len());
    let mut sums: Vec<Option<Vec<f64>>> = Vec::with_capacity(terms.len());
    let mut influence = Vec::with_capacity(terms.len());
    let mut kept_weights = options.keep_weights.then(Vec::new);
    for (term, column) in terms.iter().zip(&columns) {
        let mut target = vec![0.0; n];
        let mut total = 0.0;
        let mut n_rows = 0;
        for &i in &treated_rows {
            let raw = match column {
                Column::Treated => 1.0,
                Column::Horizon(h) => {
                    if prepared.event[i] == Some(*h) {
                        1.0
                    } else {
                        0.0
                    }
                }
                Column::Weights(c) => c[i],
            };
            let value = raw * stage.weight[i];
            if value != 0.0 {
                n_rows += 1;
            }
            target[i] = value;
            total += value;
        }
        if total == 0.0 {
            effects.push(ImputationEffect {
                term: *term,
                estimate: None,
                n_rows,
            });
            sums.push(None);
            influence.push(None);
            if let Some(k) = kept_weights.as_mut() {
                k.push(vec![0.0; n]);
            }
            continue;
        }
        let mut estimate = 0.0;
        for &i in &treated_rows {
            target[i] /= total;
            estimate += target[i] * stage.adjusted[i].unwrap_or(0.0);
        }
        let h = projector.project(
            &Projection {
                base: None,
                treated: &target,
            },
            &treated_rows,
            &options.absorb,
        )?;
        // v on the untreated rows of the fit; the target on the treated rows.
        let mut v = target;
        for (k, &i) in fit_rows.iter().enumerate() {
            v[i] = -fit_weights.map_or(1.0, |w| w[k]) * h[k];
        }
        let mut num = vec![0.0; n_cells];
        let mut den = vec![0.0; n_cells];
        for (&i, &c) in treated_rows.iter().zip(&cells) {
            let v2 = v[i] * v[i];
            num[c] += v2 * stage.adjusted[i].unwrap_or(0.0);
            den[c] += v2;
        }
        let mut by_unit = vec![0.0; n_units];
        for i in 0..n {
            if v[i] == 0.0 {
                continue;
            }
            let Some(adjusted) = stage.adjusted[i] else {
                continue;
            };
            let u = index.position[panel.panel_unit()?[i] as usize] as usize;
            by_unit[u] += v[i] * adjusted;
        }
        // The cell means enter the treated rows only.
        for (&i, &c) in treated_rows.iter().zip(&cells) {
            if v[i] != 0.0 && den[c] > 0.0 {
                let u = index.position[panel.panel_unit()?[i] as usize] as usize;
                by_unit[u] -= v[i] * (num[c] / den[c]);
            }
        }
        let cluster = index.cluster_sums(&by_unit);
        let se = cluster.iter().map(|s| s * s).sum::<f64>().sqrt();
        effects.push(ImputationEffect {
            term: *term,
            estimate: Some(fe_output::estimate(
                estimate,
                Some(se),
                options.level,
                Reference::Normal,
            )),
            n_rows,
        });
        sums.push(Some(cluster));
        influence.push(Some(by_unit.iter().map(|s| s * n_units as f64).collect()));
        if let Some(k) = kept_weights.as_mut() {
            k.push(v);
        }
    }
    let covariance = outer_covariance(&sums)?;

    let pretrends = match &options.pretrends {
        None => None,
        Some(spec) => Some(pretrend_regression(
            panel,
            &prepared,
            &fe,
            &feols,
            &index,
            spec,
            options.level,
        )?),
    };

    Ok(Imputation {
        effects,
        covariance,
        influence: fe_output::influence(influence, index.units, ClusterRule::NoSmallSample)?,
        n_units,
        n_clusters: index.clusters.count(),
        pretrends,
        covariate_coefficients: stage.coefficients.clone(),
        n_rows: n - dropped_rows.len(),
        n_untreated: fe.n_rows(),
        dropped_rows,
        residualised: stage.adjusted,
        weights: kept_weights,
    })
}

/// The raw target of an estimand.
enum Column<'a> {
    Treated,
    Horizon(i64),
    Weights(&'a [f64]),
}

/// The regression of the outcome on relative-period indicators on the
/// untreated rows of the first stage.
fn pretrend_regression(
    panel: &StaggeredPanel<'_>,
    prepared: &Prepared,
    fe: &FixedEffects,
    feols: &Feols,
    index: &UnitIndex,
    spec: &Pretrends,
    level: Level,
) -> Result<ImputationPretrends, Error> {
    let n = panel.len();
    let fit_rows = fe.rows();
    // The relative periods of the untreated rows, the never treated first;
    // the first is the reference of fixest::i.
    let levels: BTreeSet<Option<i64>> = fit_rows.iter().map(|&i| prepared.event[i]).collect();
    let free: Vec<i64> = levels.iter().skip(1).filter_map(|&e| e).collect();
    let periods: Vec<i64> = match spec {
        Pretrends::All => free.clone(),
        Pretrends::Periods(p) => {
            if let Some(&bad) = p.iter().find(|&&k| k >= 0) {
                return Err(Error::Pretrend(bad));
            }
            let set: BTreeSet<i64> = p.iter().copied().collect();
            set.into_iter().collect()
        }
    };
    let fitted: Vec<i64> = periods
        .iter()
        .copied()
        .filter(|k| free.binary_search(k).is_ok())
        .collect();
    let k_terms = fitted.len();
    let p_cov = prepared.covariates.ncols();
    let mut x = Mat::<f64>::zeros(n, k_terms + p_cov);
    for &i in fit_rows {
        if let Some(e) = prepared.event[i]
            && let Ok(j) = fitted.binary_search(&e)
        {
            x[(i, j)] = 1.0;
        }
    }
    for j in 0..p_cov {
        for i in 0..n {
            x[(i, k_terms + j)] = prepared.covariates[(i, j)];
        }
    }
    let fit = feols.fit(x.as_ref(), panel.outcome(), fe)?;
    let codes = panel.clusters().unwrap_or(panel.panel_unit()?);
    let ways = [codes];
    let cov = fit.vcov(fe, Vcov::Cluster(&ways), Ssc::new())?;
    let p = k_terms + p_cov;
    let rows_influence = fit.influence(fe)?;
    let n_units = index.units.len();
    let mut column_influence: Vec<Option<Vec<f64>>> = vec![None; p];
    for (a, &j) in fit.kept_columns().iter().enumerate() {
        let column = rows_influence.col_as_slice(a);
        let mut sums = vec![0.0; n_units];
        for (k, &i) in fit_rows.iter().enumerate() {
            sums[index.position[panel.panel_unit()?[i] as usize] as usize] += column[k];
        }
        for s in &mut sums {
            *s *= n_units as f64;
        }
        column_influence[j] = Some(sums);
    }
    let m = periods.len();
    let mut terms = Vec::with_capacity(m);
    let mut position = Vec::with_capacity(m);
    for &k in &periods {
        let j = fitted.binary_search(&k).ok();
        position.push(j);
        let estimate = j.and_then(|j| {
            let se = cov.matrix.as_slice()[j * p + j].map(|v| v.max(0.0).sqrt());
            fit.coefficients[j].map(|b| fe_output::estimate(b, se, level, Reference::Normal))
        });
        terms.push(PretrendTerm {
            relative_period: k,
            estimate,
        });
    }
    let mut covariance = vec![None; m * m];
    for a in 0..m {
        for b in 0..m {
            if let (Some(i), Some(j)) = (position[a], position[b]) {
                covariance[a * m + b] = cov.matrix.as_slice()[i * p + j];
            }
        }
    }
    let influence = position
        .iter()
        .map(|j| j.and_then(|j| column_influence[j].clone()))
        .collect();
    Ok(ImputationPretrends {
        terms,
        covariance: Covariance::new(m, covariance)?,
        influence: fe_output::influence(influence, index.units.clone(), ClusterRule::SmallSample)?,
    })
}

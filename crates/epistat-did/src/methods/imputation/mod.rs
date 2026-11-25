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
//! influence within clusters, divided by `N`, are the terms of `V`.
//! [`did_cluster_sums`](crate::did_cluster_sums) centres the influence first,
//! which changes the variance by `N` times the squared mean; the mean is zero
//! when the weights are constant within each cohort and period (no prior
//! weights, or weights that are constant within cells).
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

use epistat_glm::{Absorb, AbsorbError, Feols, FixedEffects, Singletons, Ssc, Vcov};
use epistat_inference::Clusters;
use epistat_linalg::Mat;

use first_stage::{FirstStage, Projection, Projector};

use crate::methods::staggered::StaggeredPanel;

/// What the imputed effects are averaged into.
#[derive(Clone, Copy, Debug)]
pub enum ImputationTarget<'a> {
    /// One mean over every treated row.
    Static,
    /// One mean per horizon `t - g >= 0`: the given horizons, or with `None`
    /// every horizon of the panel.
    Horizons(Option<&'a [i32]>),
    /// One mean per column of the caller's weights, one value per row, read on
    /// the treated rows and zero on the untreated ones.
    Weights(&'a [&'a [f64]]),
}

/// Which pre-trend coefficients to fit.
#[derive(Clone, Copy, Debug)]
pub enum Pretrends<'a> {
    /// Every relative period before treatment but the reference.
    All,
    /// These relative periods, each negative.
    Periods(&'a [i32]),
}

/// The controls of the imputation estimator.
#[derive(Clone, Copy, Debug)]
pub struct ImputationConfig<'a> {
    /// The estimands.
    pub target: ImputationTarget<'a>,
    /// The pre-trend coefficients, or `None`.
    pub pretrends: Option<Pretrends<'a>>,
    /// Cluster codes, one per row and constant within units; the units when
    /// `None`.
    pub clusters: Option<&'a [u32]>,
    /// The demeaning of the first stage and of the variance's projections.
    pub absorb: Absorb,
    /// Whether to return the weights `v_it` of every estimand.
    pub keep_weights: bool,
}

impl ImputationConfig<'_> {
    /// The static effect, clustered by unit, without pre-trends.
    #[must_use]
    pub const fn static_effect() -> Self {
        Self {
            target: ImputationTarget::Static,
            pretrends: None,
            clusters: None,
            absorb: Absorb::new(),
            keep_weights: false,
        }
    }

    /// The effect at every horizon, clustered by unit, without pre-trends.
    #[must_use]
    pub const fn event_study() -> Self {
        Self {
            target: ImputationTarget::Horizons(None),
            ..Self::static_effect()
        }
    }
}

/// Why the imputation or two-stage estimator failed.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ImputationError {
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
    /// A value is not finite.
    #[error("{what} is not finite on row {row}")]
    NotFinite {
        /// The input.
        what: &'static str,
        /// The row.
        row: usize,
    },
    /// A unit has rows with different first treated periods.
    #[error("unit {unit} has rows with different first treated periods")]
    Cohort {
        /// The unit.
        unit: u32,
    },
    /// A unit has rows in different clusters.
    #[error("unit {unit} has rows in different clusters")]
    Clusters {
        /// The unit.
        unit: u32,
    },
    /// A requested horizon is negative.
    #[error("horizon {0} is negative")]
    Horizon(i32),
    /// A requested pre-trend period is not negative.
    #[error("pre-trend period {0} is not negative")]
    Pretrend(i32),
    /// A requested reference period is not negative.
    #[error("reference period {0} is not negative")]
    Reference(i32),
    /// A column of target weights is not zero on an untreated row.
    #[error("target weight column {column} is not zero on the untreated row {row}")]
    TargetOnUntreated {
        /// The column.
        column: usize,
        /// The row.
        row: usize,
    },
    /// A period or a period relative to treatment does not fit in 32 bits.
    #[error("the periods span more than 2^31 values")]
    Periods,
    /// The first stage failed.
    #[error(transparent)]
    Fit(#[from] AbsorbError),
}

/// What an estimate averages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImputationTerm {
    /// Every treated row.
    Treated,
    /// The rows at this horizon.
    Horizon(i32),
    /// The caller's weight column with this index.
    Weights(usize),
}

/// One estimand.
#[derive(Clone, Debug, PartialEq)]
pub struct ImputationEstimate {
    /// What it averages.
    pub term: ImputationTerm,
    /// The estimate, `None` when no imputed row carries weight.
    pub estimate: Option<f64>,
    /// The conservative clustered standard error.
    pub std_error: Option<f64>,
    /// The treated rows with positive weight in the estimand.
    pub n_rows: usize,
}

/// One pre-trend coefficient.
#[derive(Clone, Debug, PartialEq)]
pub struct PretrendTerm {
    /// The period relative to treatment.
    pub relative_period: i32,
    /// The coefficient, `None` when the indicator is absent or collinear.
    pub estimate: Option<f64>,
    /// Its clustered standard error.
    pub std_error: Option<f64>,
}

/// The pre-trend regression.
#[derive(Clone, Debug)]
pub struct PretrendResult {
    /// The coefficients, in increasing relative period.
    pub terms: Vec<PretrendTerm>,
    /// Their clustered covariance with `fixest`'s default small-sample
    /// correction, row-major, `None` for absent or dropped terms.
    pub covariance: Vec<Option<f64>>,
    /// The influence function of each coefficient, one value per unit of
    /// [`ImputationResult::units`].
    pub influence: Vec<Option<Vec<f64>>>,
}

/// The imputation estimates.
#[derive(Clone, Debug)]
pub struct ImputationResult {
    /// The estimands, in the order of the target (horizons increasing).
    pub estimates: Vec<ImputationEstimate>,
    /// The covariance of the estimands, `sum_c s_c s_c'`, row-major.
    pub covariance: Vec<Option<f64>>,
    /// The influence function of each estimand, one value per unit of
    /// [`ImputationResult::units`].
    pub influence: Vec<Option<Vec<f64>>>,
    /// The units in the estimation, in increasing code.
    pub units: Vec<u32>,
    /// The number of clusters.
    pub n_clusters: usize,
    /// The pre-trend regression, when asked for.
    pub pretrends: Option<PretrendResult>,
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
    /// With [`ImputationConfig::keep_weights`], the weights `v_it` of each
    /// estimand, one per input row, zero on rows outside the estimation.
    pub weights: Option<Vec<Vec<f64>>>,
}

/// The checked panel and the first stage's inputs.
pub(crate) struct Prepared {
    /// Period codes from 0.
    pub(crate) time: Vec<u32>,
    /// `t - g` for rows of units that are ever treated.
    pub(crate) event: Vec<Option<i32>>,
    pub(crate) treated: Vec<bool>,
    pub(crate) weight: Vec<f64>,
    /// The weights of the first stage: the prior weight on untreated rows,
    /// zero on treated ones.
    pub(crate) fit_weight: Vec<f64>,
    pub(crate) covariates: Mat<f64>,
}

impl Prepared {
    pub(crate) fn new(
        panel: &StaggeredPanel<'_>,
        clusters: Option<&[u32]>,
    ) -> Result<Self, ImputationError> {
        let n = panel.unit.len();
        for (what, found) in [
            ("time", panel.time.len()),
            ("first_treated", panel.first_treated.len()),
            ("outcome", panel.outcome.len()),
            ("weights", panel.weights.map_or(n, <[f64]>::len)),
            ("covariates", panel.covariates.map_or(n, |x| x.nrows())),
            ("clusters", clusters.map_or(n, <[u32]>::len)),
        ] {
            if found != n {
                return Err(ImputationError::Length {
                    what,
                    expected: n,
                    found,
                });
            }
        }
        if let Some(row) = panel.outcome.iter().position(|v| !v.is_finite()) {
            return Err(ImputationError::NotFinite {
                what: "outcome",
                row,
            });
        }
        if let Some(w) = panel.weights
            && let Some(row) = w.iter().position(|v| !(v.is_finite() && *v >= 0.0))
        {
            return Err(ImputationError::NotFinite {
                what: "weights",
                row,
            });
        }
        let p = panel.covariates.map_or(0, |x| x.ncols());
        let covariates = match panel.covariates {
            Some(x) => {
                for j in 0..p {
                    for i in 0..n {
                        if !x[(i, j)].is_finite() {
                            return Err(ImputationError::NotFinite {
                                what: "covariates",
                                row: i,
                            });
                        }
                    }
                }
                x.to_owned()
            }
            None => Mat::<f64>::zeros(n, 0),
        };
        check_constant_within_units(panel.unit, panel.first_treated)
            .map_err(|unit| ImputationError::Cohort { unit })?;
        if let Some(c) = clusters {
            check_constant_within_units(panel.unit, c)
                .map_err(|unit| ImputationError::Clusters { unit })?;
        }
        let t_min = panel.time.iter().copied().min().unwrap_or(0);
        let time: Vec<u32> = panel
            .time
            .iter()
            .map(|&t| {
                u32::try_from(i64::from(t) - i64::from(t_min)).map_err(|_| ImputationError::Periods)
            })
            .collect::<Result<_, _>>()?;
        let event: Vec<Option<i32>> = panel
            .time
            .iter()
            .zip(panel.first_treated)
            .map(|(&t, g)| match g {
                None => Ok(None),
                Some(g) => t.checked_sub(*g).map(Some).ok_or(ImputationError::Periods),
            })
            .collect::<Result<_, _>>()?;
        let treated: Vec<bool> = event.iter().map(|e| e.is_some_and(|e| e >= 0)).collect();
        let weight: Vec<f64> = panel.weights.map_or_else(|| vec![1.0; n], <[f64]>::to_vec);
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

/// `Err(unit)` for the first unit whose rows carry different values.
fn check_constant_within_units<T: Copy + PartialEq>(unit: &[u32], value: &[T]) -> Result<(), u32> {
    let levels = unit.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut seen: Vec<Option<T>> = vec![None; levels];
    for (&u, &v) in unit.iter().zip(value) {
        match seen[u as usize] {
            None => seen[u as usize] = Some(v),
            Some(w) if w != v => return Err(u),
            Some(_) => {}
        }
    }
    Ok(())
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
    pub(crate) fn new(unit: &[u32], used: &[bool], clusters: Option<&[u32]>) -> Self {
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
        Self {
            units,
            position,
            clusters: Clusters::from_labels(&labels),
        }
    }

    /// The sums by cluster of per-unit values.
    pub(crate) fn cluster_sums(&self, by_unit: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.clusters.count()];
        for (value, &c) in by_unit.iter().zip(self.clusters.of()) {
            out[c] += value;
        }
        out
    }
}

/// The covariance `sum_c s_c s_c'` of columns of cluster sums, `None` where
/// a column is undefined.
pub(crate) fn outer_covariance(sums: &[Option<Vec<f64>>]) -> Vec<Option<f64>> {
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
    out
}

/// Fits the first stage of a staggered panel: the fixed effects of the
/// untreated rows, the fit and the imputed outcomes.
pub(crate) fn first_stage_fixed_effects(
    panel: &StaggeredPanel<'_>,
    prepared: &Prepared,
) -> Result<FixedEffects, ImputationError> {
    Ok(FixedEffects::new(
        &[panel.unit, &prepared.time],
        Some(&prepared.fit_weight),
        Singletons::Keep,
    )?)
}

/// Estimates the imputation estimator of Borusyak, Jaravel and Spiess. See
/// the module documentation.
///
/// # Errors
///
/// [`ImputationError`] for inputs of the wrong length or not finite, cohorts
/// or clusters that vary within a unit, negative horizons, pre-trend periods
/// that are not negative, target weights on untreated rows, and the errors of
/// the first stage.
pub fn estimate_did_imputation(
    panel: &StaggeredPanel<'_>,
    config: &ImputationConfig<'_>,
) -> Result<ImputationResult, ImputationError> {
    let n = panel.unit.len();
    let prepared = Prepared::new(panel, config.clusters)?;
    let fe = first_stage_fixed_effects(panel, &prepared)?;
    let feols = Feols {
        absorb: config.absorb,
    };
    let x = prepared.covariates.as_ref();
    let fit = feols.fit(panel.outcome, x, &fe)?;
    let stage = FirstStage::new(
        panel.unit,
        &prepared.time,
        prepared.treated.clone(),
        panel.outcome,
        prepared.weight.clone(),
        x,
        &fit,
    );
    let used: Vec<bool> = stage.adjusted.iter().map(Option::is_some).collect();
    let dropped_rows: Vec<usize> = (0..n).filter(|&i| !used[i]).collect();
    let treated_rows: Vec<usize> = (0..n).filter(|&i| used[i] && stage.treated[i]).collect();
    let index = UnitIndex::new(panel.unit, &used, config.clusters);
    let n_units = index.units.len();

    // The estimands.
    let (terms, columns): (Vec<ImputationTerm>, Vec<Column<'_>>) = match config.target {
        ImputationTarget::Static => (vec![ImputationTerm::Treated], vec![Column::Treated]),
        ImputationTarget::Horizons(given) => {
            let horizons: BTreeSet<i32> = if let Some(h) = given {
                if let Some(&bad) = h.iter().find(|&&h| h < 0) {
                    return Err(ImputationError::Horizon(bad));
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
                if column.len() != n {
                    return Err(ImputationError::Length {
                        what: "target weights",
                        expected: n,
                        found: column.len(),
                    });
                }
                for (i, &v) in column.iter().enumerate() {
                    if !v.is_finite() {
                        return Err(ImputationError::NotFinite {
                            what: "target weights",
                            row: i,
                        });
                    }
                    if v != 0.0 && !prepared.treated[i] {
                        return Err(ImputationError::TargetOnUntreated { column: j, row: i });
                    }
                }
            }
            (
                (0..given.len()).map(ImputationTerm::Weights).collect(),
                given.iter().map(|c| Column::Weights(c)).collect(),
            )
        }
    };

    // Cohort-by-period cells of the treated rows, for tau_bar.
    let mut cell_of: HashMap<(i32, i32), usize> = HashMap::new();
    let cells: Vec<usize> = treated_rows
        .iter()
        .map(|&i| {
            let key = (panel.first_treated[i].unwrap_or(0), panel.time[i]);
            let next = cell_of.len();
            *cell_of.entry(key).or_insert(next)
        })
        .collect();
    let n_cells = cell_of.len();

    let projector = Projector::new(&fe, &fit, x, panel.unit, &prepared.time);
    let fit_rows = fe.rows();
    let fit_weights = fe.weights();
    let mut estimates = Vec::with_capacity(terms.len());
    let mut sums: Vec<Option<Vec<f64>>> = Vec::with_capacity(terms.len());
    let mut influence = Vec::with_capacity(terms.len());
    let mut kept_weights = config.keep_weights.then(Vec::new);
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
            estimates.push(ImputationEstimate {
                term: *term,
                estimate: None,
                std_error: None,
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
            &config.absorb,
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
            let u = index.position[panel.unit[i] as usize] as usize;
            by_unit[u] += v[i] * adjusted;
        }
        // The cell means enter the treated rows only.
        for (&i, &c) in treated_rows.iter().zip(&cells) {
            if v[i] != 0.0 && den[c] > 0.0 {
                let u = index.position[panel.unit[i] as usize] as usize;
                by_unit[u] -= v[i] * (num[c] / den[c]);
            }
        }
        let cluster = index.cluster_sums(&by_unit);
        let se = cluster.iter().map(|s| s * s).sum::<f64>().sqrt();
        estimates.push(ImputationEstimate {
            term: *term,
            estimate: Some(estimate),
            std_error: Some(se),
            n_rows,
        });
        sums.push(Some(cluster));
        influence.push(Some(by_unit.iter().map(|s| s * n_units as f64).collect()));
        if let Some(k) = kept_weights.as_mut() {
            k.push(v);
        }
    }
    let covariance = outer_covariance(&sums);

    let pretrends = match config.pretrends {
        None => None,
        Some(spec) => Some(pretrend_regression(
            panel,
            &prepared,
            &fe,
            &feols,
            &index,
            spec,
            config.clusters,
        )?),
    };

    Ok(ImputationResult {
        estimates,
        covariance,
        influence,
        units: index.units,
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
    Horizon(i32),
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
    spec: Pretrends<'_>,
    clusters: Option<&[u32]>,
) -> Result<PretrendResult, ImputationError> {
    let n = panel.unit.len();
    let fit_rows = fe.rows();
    // The relative periods of the untreated rows, the never treated first;
    // the first is the reference of fixest::i.
    let levels: BTreeSet<Option<i32>> = fit_rows.iter().map(|&i| prepared.event[i]).collect();
    let free: Vec<i32> = levels.iter().skip(1).filter_map(|&e| e).collect();
    let periods: Vec<i32> = match spec {
        Pretrends::All => free.clone(),
        Pretrends::Periods(p) => {
            if let Some(&bad) = p.iter().find(|&&k| k >= 0) {
                return Err(ImputationError::Pretrend(bad));
            }
            let set: BTreeSet<i32> = p.iter().copied().collect();
            set.into_iter().collect()
        }
    };
    let fitted: Vec<i32> = periods
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
    let fit = feols.fit(panel.outcome, x.as_ref(), fe)?;
    let codes = clusters.unwrap_or(panel.unit);
    let ways = [codes];
    let cov = fit.vcov(Vcov::Cluster(&ways), Ssc::new())?;
    let p = k_terms + p_cov;
    let rows_influence = fit.influence();
    let n_units = index.units.len();
    let mut column_influence: Vec<Option<Vec<f64>>> = vec![None; p];
    for (a, &j) in fit.kept_columns().iter().enumerate() {
        let column = rows_influence.col_as_slice(a);
        let mut sums = vec![0.0; n_units];
        for (k, &i) in fit_rows.iter().enumerate() {
            sums[index.position[panel.unit[i] as usize] as usize] += column[k];
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
        let estimate = j.and_then(|j| fit.coefficients[j]);
        let std_error = j.and_then(|j| cov.matrix[j * p + j].map(|v| v.max(0.0).sqrt()));
        terms.push(PretrendTerm {
            relative_period: k,
            estimate,
            std_error,
        });
    }
    let mut covariance = vec![None; m * m];
    for a in 0..m {
        for b in 0..m {
            if let (Some(i), Some(j)) = (position[a], position[b]) {
                covariance[a * m + b] = cov.matrix[i * p + j];
            }
        }
    }
    let influence = position
        .iter()
        .map(|j| j.and_then(|j| column_influence[j].clone()))
        .collect();
    Ok(PretrendResult {
        terms,
        covariance,
        influence,
    })
}

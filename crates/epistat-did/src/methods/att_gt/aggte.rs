//! Aggregation of `ATT(g,t)` into the four summaries `did::aggte` produces.
//!
//! The cohort, calendar and overall summaries of [`super::aggregation`] are
//! built on this module: under treated-count weighting they are its group,
//! calendar and simple summaries, and under their other weightings they use its
//! cell selection and standard error with fixed weights.
//!
//! # The weights are estimated, and that shows up in the standard error
//!
//! Three of the four summaries weight cells by cohort share `pg = P(G = g)`,
//! which is estimated from the same sample. Treating it as fixed understates the
//! variance: measured against `did` 2.5.1 on `mpdta` at event time 0, the
//! fixed-weight standard error is 0.0114607 and the correct one is 0.0114942.
//! [`weight_influence`] is the correction, ported from `did:::wif`.
//!
//! The group summary is the exception: within a cohort the cells are averaged
//! with equal weights, which are not estimated, so no correction applies.
//!
//! # Standard errors and the bootstrap
//!
//! Every aggregate has an influence function `psi_i`, the weighted sum of the
//! cells' influence functions plus the weight correction. Its analytic standard
//! error is `sqrt(sum_g s_g^2)` with `s_g` the sum of the centred `psi_i` in
//! cluster `g`, divided by `n` ([`crate::inference::did_clusters`]).
//!
//! With [`AggteBootstrap`] the standard errors come from the multiplier
//! bootstrap of `did::mboot`. Replicate `b` draws a Rademacher weight `w_gb`
//! per cluster (per unit without clusters) and forms
//! `T_b = sqrt(G) * mean_g(w_gb * S_g)` with `S_g` the uncentred cluster sum of
//! the influence function. For each point of a path, `sigma` is the
//! interquartile range of its `T_b` divided by `qnorm(.75) - qnorm(.25)`, and
//! its standard error is `sigma * sqrt(G) / n`. The uniform critical value is
//! the type 1 quantile at `1 - alpha` of `max_j |T_bj| / sigma_j` over the
//! points of the path, so the band `att_j +- crit * se_j` covers the whole path
//! with probability about `1 - alpha`. The standard error of the overall effect
//! has its own bootstrap on the overall influence function.
//!
//! The analytic and the bootstrap standard errors differ when the draws are not
//! normal, since the interquartile range ignores the tails: for a cell with few
//! units the bootstrap value can be several per cent above the analytic one.
//! That is a property of `did::mboot`, which this reproduces.
//!
//! # `max_e`, `min_e` and `balance_e`
//!
//! `did::aggte` applies `max_e` to the simple, group and dynamic summaries,
//! `min_e` and `balance_e` to the dynamic one only, and none of them to the
//! calendar one. See [`AggteConfig`] for the unit in which `max_e` is counted.
//!
//! # References
//!
//! Callaway B, Sant'Anna PHC. Difference-in-differences with multiple time
//! periods. J Econom 2021;225(2):200-30. R package `did` 2.5.1, functions
//! `aggte`, `mboot`, `getSE` and `wif`.

use std::collections::{BTreeMap, BTreeSet};

use epistat_core::{Bootstrap, Level};
use epistat_inference::Clusters;
use epistat_inference::simultaneous::multiplier_bands;

use crate::inference::did_clusters::standard_error;
use crate::inference::{normal_interval, validate_confidence_level, z_score_for_confidence};
use crate::types::{AttGtEstimate, InferenceConfig};
use crate::util::usize_to_f64;

/// Which summary to produce, matching `aggte(type = ...)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggteType {
    /// One number: every post-treatment cell, weighted by cohort share.
    Simple,
    /// By event time, then averaged over non-negative event times.
    Dynamic,
    /// By cohort, then averaged over cohorts by share.
    Group,
    /// By calendar period, then averaged over periods.
    Calendar,
}

/// Configuration for [`aggregate_att_gt`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AggteConfig {
    /// Which summary to produce.
    pub aggregation: AggteType,
    /// Restrict the dynamic path to a FIXED cohort composition: keep only
    /// cohorts observed for at least this many post-treatment periods, and only
    /// event times up to it. `did::aggte(balance_e = )`.
    ///
    /// This is the answer to "did the effect change with duration, or is the
    /// long-horizon estimate simply made of different cohorts?" Without it the
    /// two are not separable. Ignored for the other three types, as in R.
    pub balance_e: Option<i32>,
    /// Drop event times below this, inclusive bound. `did::aggte(min_e = )`.
    ///
    /// Applied AFTER `balance_e`, and it narrows the overall number as well as
    /// the path: the dynamic overall is the mean over retained non-negative event
    /// times, so trimming the path moves it. Dynamic only, as in R.
    pub min_e: Option<i32>,
    /// Drop event times above this, inclusive bound. `did::aggte(max_e = )`.
    ///
    /// For [`AggteType::Dynamic`] the bound is in the units of the period
    /// variable. For [`AggteType::Simple`] and [`AggteType::Group`] only the
    /// post-treatment cells at most `max_e` periods after the cohort's first
    /// treated period enter, as `did` 2.5.1 selects them, and the distance is
    /// counted in steps of the sorted list of periods and cohorts that occur in
    /// the estimates, where `did` counts in its own period list. The two agree
    /// when the periods are consecutive. Ignored by [`AggteType::Calendar`], as
    /// in `did`.
    pub max_e: Option<i32>,
    /// Confidence level settings for the intervals.
    pub confidence_level: InferenceConfig,
    /// Multiplier bootstrap for the standard errors and the uniform band, as
    /// `did::aggte(bstrap = TRUE, cband = TRUE)`. `None` gives analytic
    /// standard errors and pointwise intervals.
    pub bootstrap: Option<AggteBootstrap>,
}

/// Settings for the multiplier bootstrap of [`aggregate_att_gt`].
///
/// One Rademacher weight is drawn per unit, or per cluster when
/// [`UnitPanel::clusters`] is set, and replicate `r` draws from
/// `replicate_seed(seed, r)`, so the result depends on the seed and not on the
/// number of threads. The path of an aggregate (event times, cohorts or
/// periods) is bootstrapped jointly, and the overall effect separately from the
/// same seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AggteBootstrap {
    /// Number of replicates, `biters` in `did`. At least 1.
    pub replicates: usize,
    /// Seed of the draws.
    pub seed: u64,
    /// Replace the analytic standard errors by the bootstrap ones, as
    /// `bstrap = TRUE`.
    pub standard_errors: bool,
    /// Use the uniform critical value of the path for its intervals, as
    /// `cband = TRUE`. It is never below the pointwise normal value. When no
    /// finite critical value exists the intervals stay pointwise and
    /// [`AggteResult::uniform_band`] is `false`.
    pub uniform_band: bool,
}

impl Default for AggteConfig {
    fn default() -> Self {
        Self {
            aggregation: AggteType::Dynamic,
            balance_e: None,
            min_e: None,
            max_e: None,
            confidence_level: InferenceConfig::default(),
            bootstrap: None,
        }
    }
}

/// One aggregated point, plus the influence function behind it.
#[derive(Debug, Clone, PartialEq)]
pub struct AggteEstimate {
    /// Event time, cohort or calendar period, depending on the type.
    pub key: i32,
    /// Estimated average treatment effect on the treated.
    pub att: f64,
    /// Standard error of the estimate.
    pub se: f64,
    /// Lower bound of the confidence interval.
    pub ci_low: f64,
    /// Upper bound of the confidence interval.
    pub ci_high: f64,
    /// Influence function of the aggregated estimate, one value per unit.
    pub influence: Vec<f64>,
}

/// The result of an aggregation: the headline number and, where the type has
/// one, the path behind it.
#[derive(Debug, Clone, PartialEq)]
pub struct AggteResult {
    /// Overall aggregated effect.
    pub overall_att: f64,
    /// Standard error of the overall effect.
    pub overall_se: f64,
    /// Lower bound of the overall confidence interval.
    pub overall_ci_low: f64,
    /// Upper bound of the overall confidence interval.
    pub overall_ci_high: f64,
    /// Influence function of the overall effect, one value per unit.
    pub overall_influence: Vec<f64>,
    /// Empty for [`AggteType::Simple`], which has no path.
    pub by_key: Vec<AggteEstimate>,
    /// The multiplier of a path point's standard error in its interval: the
    /// pointwise normal quantile, or the uniform critical value when
    /// [`AggteResult::uniform_band`] is set.
    pub critical_value: f64,
    /// Whether the intervals of the path use a uniform critical value.
    pub uniform_band: bool,
}

/// Errors from aggregation.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum AggteError {
    /// No ATT(g,t) estimates were given.
    #[error("aggregation requires at least one ATT(g,t) estimate")]
    EmptyInput,
    /// The number of influence vectors differs from the number of estimates.
    #[error("influence matrix has {influence} vectors for {estimates} estimates")]
    ShapeMismatch {
        /// Number of influence vectors.
        influence: usize,
        /// Number of ATT(g,t) estimates.
        estimates: usize,
    },
    /// The influence vectors differ in length.
    #[error("influence vectors must all have the unit count; got {actual}, expected {expected}")]
    RaggedInfluence {
        /// Number of units, taken from the first vector.
        expected: usize,
        /// Length of the offending vector.
        actual: usize,
    },
    /// The number of `unit_groups` entries differs from the influence vector length.
    #[error("unit_groups has {actual} entries but the influence vectors have {expected}")]
    UnitGroupMismatch {
        /// Length of the influence vectors.
        expected: usize,
        /// Number of entries in `unit_groups`.
        actual: usize,
    },
    /// No post-treatment cells remain after the requested restrictions.
    #[error("no post-treatment cells survive the requested aggregation")]
    NoPostTreatmentCells,
    /// The confidence level is not strictly between 0 and 1.
    #[error("confidence_level must be finite and in (0, 1)")]
    InvalidConfidenceLevel,
    /// The bootstrap has no replicates, or asks for neither standard errors nor
    /// a uniform band.
    #[error("bootstrap needs at least one replicate and standard errors or a uniform band")]
    InvalidBootstrap,
    /// An aggregated influence function contains a value that is not finite.
    #[error("an aggregated influence function is not finite")]
    NonFiniteInfluence,
}

/// The per-unit inputs the weights are estimated from.
#[derive(Debug, Clone)]
pub struct UnitPanel {
    /// Each unit's cohort, `None` for never-treated. Indexed like the influence
    /// vectors: position in the sorted distinct unit ids.
    pub groups: Vec<Option<i32>>,
    /// Sampling weights, one per unit. `did` calls these `weights.ind` and
    /// defaults them to 1.
    pub weights: Vec<f64>,
    /// Cluster label per unit, `None` to treat each unit as its own cluster.
    ///
    /// Set this when units are not independent. In a matched family design the
    /// two parents of one child are two units sharing every shock to that
    /// household, so the family is the cluster and the unit is not.
    pub clusters: Option<Vec<i64>>,
}

impl UnitPanel {
    /// Unweighted, unclustered panel.
    #[must_use]
    pub fn unweighted(groups: Vec<Option<i32>>) -> Self {
        let weights = vec![1.0; groups.len()];
        Self {
            groups,
            weights,
            clusters: None,
        }
    }

    /// The same panel with a cluster label attached to each unit.
    #[must_use]
    pub fn clustered_by(mut self, clusters: Vec<i64>) -> Self {
        self.clusters = Some(clusters);
        self
    }

    /// The clusters of the units: the labels in increasing order, or one
    /// cluster per unit without labels.
    ///
    /// One cluster per unit is the identity case: [`standard_error`] then
    /// reduces to the unclustered formula, and
    /// `clustering_by_unit_reproduces_the_unclustered_standard_error` pins that.
    fn cluster_index(&self) -> Clusters {
        match self.clusters {
            None => Clusters::singletons(self.groups.len()),
            Some(ref labels) => Clusters::from_sorted_labels(labels),
        }
    }
}

/// Cohort share `pg = sum_i w_i * 1(G_i = g) / n`, as `did` computes it.
///
/// Note the denominator is the FULL unit count, never-treated included. It
/// cancels out of the normalised weights, but not out of [`weight_influence`],
/// so it has to be the same `n` in both.
fn cohort_share(panel: &UnitPanel, group: i32) -> f64 {
    let n = usize_to_f64(panel.groups.len());
    let total: f64 = panel
        .groups
        .iter()
        .zip(&panel.weights)
        .filter(|(unit_group, _)| **unit_group == Some(group))
        .map(|(_, weight)| *weight)
        .sum();
    total / n
}

/// The influence function of the estimated weights, ported from `did:::wif`.
///
/// Returns a `n_units x n_cells` matrix in row-major order. Without this term
/// the standard error of any cohort-share-weighted summary is too small,
/// because it treats a quantity estimated from the sample as if it were known.
fn weight_influence(panel: &UnitPanel, groups: &[i32], shares: &[f64]) -> Vec<Vec<f64>> {
    let share_total: f64 = shares.iter().sum();
    let n_units = panel.groups.len();

    // `if2` needs the row sum of the centred indicators, so it is built once.
    let row_sums = (0..n_units)
        .map(|unit| {
            groups
                .iter()
                .zip(shares)
                .map(|(group, share)| indicator(panel, unit, *group) - share)
                .sum::<f64>()
        })
        .collect::<Vec<f64>>();

    (0..n_units)
        .map(|unit| {
            groups
                .iter()
                .zip(shares)
                .map(|(group, share)| {
                    let if1 = (indicator(panel, unit, *group) - share) / share_total;
                    let if2 = row_sums[unit] * share / (share_total * share_total);
                    if1 - if2
                })
                .collect()
        })
        .collect()
}

fn indicator(panel: &UnitPanel, unit: usize, group: i32) -> f64 {
    if panel.groups[unit] == Some(group) {
        panel.weights[unit]
    } else {
        0.0
    }
}

/// Combine selected cells with cohort-share weights, correcting for the fact
/// that those weights were estimated.
fn combine_by_share(
    cells: &[usize],
    estimates: &[AttGtEstimate],
    influence: &[Vec<f64>],
    panel: &UnitPanel,
) -> (f64, Vec<f64>) {
    let groups = cells
        .iter()
        .map(|&c| estimates[c].group)
        .collect::<Vec<_>>();
    let shares = groups
        .iter()
        .map(|&g| cohort_share(panel, g))
        .collect::<Vec<_>>();
    let share_total: f64 = shares.iter().sum();
    let weights = shares.iter().map(|s| s / share_total).collect::<Vec<_>>();

    let att = cells
        .iter()
        .zip(&weights)
        .map(|(&c, w)| w * estimates[c].att)
        .sum::<f64>();

    let wif = weight_influence(panel, &groups, &shares);
    let combined = (0..panel.groups.len())
        .map(|unit| {
            let weighted = cells
                .iter()
                .zip(&weights)
                .map(|(&c, w)| w * influence[c][unit])
                .sum::<f64>();
            let correction = wif[unit]
                .iter()
                .zip(cells)
                .map(|(w, &c)| w * estimates[c].att)
                .sum::<f64>();
            weighted + correction
        })
        .collect();

    (att, combined)
}

/// Combine selected cells with equal weights. No correction: `1/k` is not
/// estimated from the sample.
fn combine_equally(
    cells: &[usize],
    estimates: &[AttGtEstimate],
    influence: &[Vec<f64>],
    n_units: usize,
) -> (f64, Vec<f64>) {
    let weights = vec![1.0 / usize_to_f64(cells.len()); cells.len()];
    combine_fixed(cells, &weights, estimates, influence, n_units)
}

/// Combine selected cells with the given weights, which are taken as known, so
/// the influence function is the weighted sum of the cells' and nothing more.
pub(super) fn combine_fixed(
    cells: &[usize],
    weights: &[f64],
    estimates: &[AttGtEstimate],
    influence: &[Vec<f64>],
    n_units: usize,
) -> (f64, Vec<f64>) {
    let att = cells
        .iter()
        .zip(weights)
        .map(|(&c, w)| w * estimates[c].att)
        .sum();
    let combined = (0..n_units)
        .map(|unit| {
            cells
                .iter()
                .zip(weights)
                .map(|(&c, w)| w * influence[c][unit])
                .sum()
        })
        .collect();
    (att, combined)
}

/// Average already-aggregated points with equal weights.
fn average_points(points: &[&AggteEstimate], n_units: usize) -> (f64, Vec<f64>) {
    let weight = 1.0 / usize_to_f64(points.len());
    let att = points.iter().map(|p| weight * p.att).sum();
    let influence = (0..n_units)
        .map(|unit| points.iter().map(|p| weight * p.influence[unit]).sum())
        .collect();
    (att, influence)
}

fn finish(
    key: i32,
    att: f64,
    influence: Vec<f64>,
    confidence_level: f64,
    clusters: &Clusters,
) -> AggteEstimate {
    let se = standard_error(&influence, clusters);
    let (ci_low, ci_high) = normal_interval(att, se, confidence_level);
    AggteEstimate {
        key,
        att,
        se,
        ci_low,
        ci_high,
        influence,
    }
}

/// Checks that the estimates, their influence vectors and the panel agree in
/// shape, and returns the panel's clusters.
pub(super) fn checked_cluster_index(
    estimates: &[AttGtEstimate],
    influence: &[Vec<f64>],
    panel: &UnitPanel,
) -> Result<Clusters, AggteError> {
    if estimates.is_empty() {
        return Err(AggteError::EmptyInput);
    }
    if influence.len() != estimates.len() {
        return Err(AggteError::ShapeMismatch {
            influence: influence.len(),
            estimates: estimates.len(),
        });
    }
    let n_units = panel.groups.len();
    if panel.weights.len() != n_units {
        return Err(AggteError::UnitGroupMismatch {
            expected: n_units,
            actual: panel.weights.len(),
        });
    }
    if let Some(bad) = influence.iter().find(|vector| vector.len() != n_units) {
        return Err(AggteError::RaggedInfluence {
            expected: n_units,
            actual: bad.len(),
        });
    }
    let clusters = panel.cluster_index();
    if clusters.len() != n_units {
        return Err(AggteError::UnitGroupMismatch {
            expected: n_units,
            actual: clusters.len(),
        });
    }
    Ok(clusters)
}

/// Aggregate `ATT(g,t)` the way `did::aggte` does.
///
/// `influence` must be unit-indexed and on the full-sample scale, which is what
/// [`super::panel_pairs::estimate_att_gt_dr_panel_with_influence`] returns.
///
/// Standard errors are analytic, from the combined influence function including
/// the influence of the estimated cohort shares, unless
/// [`AggteConfig::bootstrap`] asks for the multiplier bootstrap. With
/// [`AggteBootstrap::standard_errors`] the standard error of every point and of
/// the overall effect is the bootstrap one, which uses the interquartile range
/// of the draws as `did::mboot` does and so differs from the analytic value
/// where the draws are not normal. With [`AggteBootstrap::uniform_band`] the
/// intervals of the path use the critical value of the maximum over the path.
///
/// # Errors
/// See [`AggteError`].
pub fn aggregate_att_gt(
    estimates: &[AttGtEstimate],
    influence: &[Vec<f64>],
    panel: &UnitPanel,
    config: AggteConfig,
) -> Result<AggteResult, AggteError> {
    let confidence_level = config.confidence_level.confidence_level;
    if !validate_confidence_level(confidence_level) {
        return Err(AggteError::InvalidConfidenceLevel);
    }
    if let Some(bootstrap) = config.bootstrap
        && (bootstrap.replicates == 0 || !(bootstrap.standard_errors || bootstrap.uniform_band))
    {
        return Err(AggteError::InvalidBootstrap);
    }
    let result = aggregate_point_estimates(estimates, influence, panel, config)?;
    match config.bootstrap {
        None => Ok(result),
        Some(bootstrap) => apply_bootstrap(result, panel, bootstrap, confidence_level),
    }
}

/// Position of each period or cohort in the sorted list of those that occur in
/// the estimates, which is how `did` measures `max_e` for the simple and group
/// summaries.
fn period_ranks(estimates: &[AttGtEstimate]) -> BTreeMap<i32, i32> {
    let periods = estimates
        .iter()
        .flat_map(|estimate| [estimate.time, estimate.group])
        .collect::<BTreeSet<i32>>();
    periods
        .into_iter()
        .enumerate()
        .map(|(rank, period)| (period, i32::try_from(rank).unwrap_or(i32::MAX)))
        .collect()
}

/// The post-treatment cells that `max_e` leaves in the simple and group
/// summaries.
fn post_cells_within(estimates: &[AttGtEstimate], max_e: Option<i32>) -> Vec<usize> {
    let ranks = max_e.map(|_| period_ranks(estimates));
    (0..estimates.len())
        .filter(|&i| estimates[i].time >= estimates[i].group)
        .filter(|&i| match (max_e, ranks.as_ref()) {
            (Some(limit), Some(ranks)) => {
                ranks[&estimates[i].time] - ranks[&estimates[i].group] <= limit
            }
            _ => true,
        })
        .collect()
}

#[expect(
    clippy::too_many_lines,
    reason = "the four aggregation types differ in which cells they select and \
              how they weight them; splitting them apart hides the comparison"
)]
fn aggregate_point_estimates(
    estimates: &[AttGtEstimate],
    influence: &[Vec<f64>],
    panel: &UnitPanel,
    config: AggteConfig,
) -> Result<AggteResult, AggteError> {
    let clusters = checked_cluster_index(estimates, influence, panel)?;
    let n_units = panel.groups.len();
    let confidence_level = config.confidence_level.confidence_level;
    let critical_value = z_score_for_confidence(confidence_level);
    let post = post_cells_within(estimates, config.max_e);

    match config.aggregation {
        AggteType::Simple => {
            if post.is_empty() {
                return Err(AggteError::NoPostTreatmentCells);
            }
            let (att, influence_vector) = combine_by_share(&post, estimates, influence, panel);
            let summary = finish(0, att, influence_vector, confidence_level, &clusters);
            Ok(AggteResult {
                overall_att: summary.att,
                overall_se: summary.se,
                overall_ci_low: summary.ci_low,
                overall_ci_high: summary.ci_high,
                overall_influence: summary.influence,
                by_key: Vec::new(),
                critical_value,
                uniform_band: false,
            })
        }
        AggteType::Dynamic => {
            // balance_e keeps only cohorts with at least `b` post-treatment
            // periods observed, and then shows event times up to `b`. Both halves
            // are needed: without the first the composition still moves, without
            // the second the path runs past where every kept cohort is present.
            let max_time = estimates.iter().map(|e| e.time).max().unwrap_or_default();
            let kept_groups = config.balance_e.map(|b| {
                estimates
                    .iter()
                    .map(|e| e.group)
                    .filter(|g| max_time - g >= b)
                    .collect::<BTreeSet<i32>>()
            });

            let mut buckets = BTreeMap::<i32, Vec<usize>>::new();
            for (index, estimate) in estimates.iter().enumerate() {
                if kept_groups
                    .as_ref()
                    .is_some_and(|keep| !keep.contains(&estimate.group))
                {
                    continue;
                }
                if config.balance_e.is_some_and(|b| estimate.event_time > b) {
                    continue;
                }
                if config.min_e.is_some_and(|e| estimate.event_time < e)
                    || config.max_e.is_some_and(|e| estimate.event_time > e)
                {
                    continue;
                }
                buckets.entry(estimate.event_time).or_default().push(index);
            }

            let by_key = buckets
                .into_iter()
                .map(|(event_time, cells)| {
                    let (att, vector) = combine_by_share(&cells, estimates, influence, panel);
                    finish(event_time, att, vector, confidence_level, &clusters)
                })
                .collect::<Vec<_>>();

            let positive = by_key.iter().filter(|p| p.key >= 0).collect::<Vec<_>>();
            if positive.is_empty() {
                return Err(AggteError::NoPostTreatmentCells);
            }
            let (overall_att, overall_influence) = average_points(&positive, n_units);
            let overall = finish(
                0,
                overall_att,
                overall_influence,
                confidence_level,
                &clusters,
            );
            Ok(AggteResult {
                overall_att: overall.att,
                overall_se: overall.se,
                overall_ci_low: overall.ci_low,
                overall_ci_high: overall.ci_high,
                overall_influence: overall.influence,
                by_key,
                critical_value,
                uniform_band: false,
            })
        }
        AggteType::Group => {
            if post.is_empty() {
                return Err(AggteError::NoPostTreatmentCells);
            }
            let mut buckets = BTreeMap::<i32, Vec<usize>>::new();
            for &index in &post {
                buckets
                    .entry(estimates[index].group)
                    .or_default()
                    .push(index);
            }
            // Equal weights WITHIN a cohort: an ATT(g,t) for one cohort is
            // averaged over its own post periods, and 1/k is not estimated.
            let by_key = buckets
                .into_iter()
                .map(|(group, cells)| {
                    let (att, vector) = combine_equally(&cells, estimates, influence, n_units);
                    finish(group, att, vector, confidence_level, &clusters)
                })
                .collect::<Vec<_>>();

            // Cohort shares ACROSS cohorts, so the correction applies here.
            let shares = by_key
                .iter()
                .map(|point| cohort_share(panel, point.key))
                .collect::<Vec<_>>();
            let share_total: f64 = shares.iter().sum();
            let weights = shares.iter().map(|s| s / share_total).collect::<Vec<_>>();
            let groups = by_key.iter().map(|point| point.key).collect::<Vec<_>>();
            let atts = by_key.iter().map(|point| point.att).collect::<Vec<_>>();
            let wif = weight_influence(panel, &groups, &shares);

            let overall_att = by_key
                .iter()
                .zip(&weights)
                .map(|(point, w)| w * point.att)
                .sum::<f64>();
            let overall_influence = (0..n_units)
                .map(|unit| {
                    let weighted = by_key
                        .iter()
                        .zip(&weights)
                        .map(|(point, w)| w * point.influence[unit])
                        .sum::<f64>();
                    let correction = wif[unit]
                        .iter()
                        .zip(&atts)
                        .map(|(w, att)| w * att)
                        .sum::<f64>();
                    weighted + correction
                })
                .collect();
            let overall = finish(
                0,
                overall_att,
                overall_influence,
                confidence_level,
                &clusters,
            );
            Ok(AggteResult {
                overall_att: overall.att,
                overall_se: overall.se,
                overall_ci_low: overall.ci_low,
                overall_ci_high: overall.ci_high,
                overall_influence: overall.influence,
                by_key,
                critical_value,
                uniform_band: false,
            })
        }
        AggteType::Calendar => {
            let post = post_cells_within(estimates, None);
            if post.is_empty() {
                return Err(AggteError::NoPostTreatmentCells);
            }
            let mut buckets = BTreeMap::<i32, Vec<usize>>::new();
            for &index in &post {
                buckets
                    .entry(estimates[index].time)
                    .or_default()
                    .push(index);
            }
            let by_key = buckets
                .into_iter()
                .map(|(time, cells)| {
                    let (att, vector) = combine_by_share(&cells, estimates, influence, panel);
                    finish(time, att, vector, confidence_level, &clusters)
                })
                .collect::<Vec<_>>();

            let all = by_key.iter().collect::<Vec<_>>();
            let (overall_att, overall_influence) = average_points(&all, n_units);
            let overall = finish(
                0,
                overall_att,
                overall_influence,
                confidence_level,
                &clusters,
            );
            Ok(AggteResult {
                overall_att: overall.att,
                overall_se: overall.se,
                overall_ci_low: overall.ci_low,
                overall_ci_high: overall.ci_high,
                overall_influence: overall.influence,
                by_key,
                critical_value,
                uniform_band: false,
            })
        }
    }
}

/// Replaces the analytic standard errors and the pointwise critical value by the
/// bootstrap ones, as `did::aggte` does for `bstrap = TRUE` and `cband = TRUE`.
///
/// The path is bootstrapped as one matrix, so its standard errors and critical
/// value come from the same draws. The overall effect has a separate bootstrap
/// from the same seed, so for a path whose overall effect is a fixed linear
/// combination of its points the draw of the overall effect is that combination
/// of the draws of the points.
fn apply_bootstrap(
    mut result: AggteResult,
    panel: &UnitPanel,
    bootstrap: AggteBootstrap,
    confidence_level: f64,
) -> Result<AggteResult, AggteError> {
    let alpha = 1.0 - confidence_level;
    let pointwise = z_score_for_confidence(confidence_level);
    let finite = result
        .by_key
        .iter()
        .map(|point| &point.influence)
        .chain(std::iter::once(&result.overall_influence))
        .all(|vector| vector.iter().all(|value| value.is_finite()));
    if !finite {
        return Err(AggteError::NonFiniteInfluence);
    }
    let clusters = panel.clusters.as_deref().map(Clusters::from_sorted_labels);
    // An alpha of 0 or 1 is an error of every band call below, as an invalid
    // level is.
    let level = Level::alpha(alpha).map_err(epistat_inference::Error::from);

    if !result.by_key.is_empty() {
        let columns = result
            .by_key
            .iter()
            .map(|point| point.influence.as_slice())
            .collect::<Vec<_>>();
        let bands = level.clone().and_then(|level| {
            multiplier_bands(
                &columns,
                clusters.as_ref(),
                level,
                Bootstrap::new(bootstrap.replicates, bootstrap.seed),
            )
        });
        // An error here means that no draw gave a finite studentised maximum, so
        // there are no bootstrap standard errors either.
        let standard_errors = bands.as_ref().ok().map(|bands| bands.se.clone());
        if bootstrap.standard_errors {
            for (index, point) in result.by_key.iter_mut().enumerate() {
                point.se = standard_errors
                    .as_ref()
                    .and_then(|se| se[index])
                    .unwrap_or(f64::NAN);
            }
        }
        if let (true, Ok(bands)) = (bootstrap.uniform_band, bands.as_ref())
            && bands.critical.is_finite()
            && bands.critical >= pointwise
        {
            result.critical_value = bands.critical;
            result.uniform_band = true;
        }
        for point in &mut result.by_key {
            let margin = result.critical_value * point.se;
            point.ci_low = point.att - margin;
            point.ci_high = point.att + margin;
        }
    }

    if bootstrap.standard_errors {
        let bands = level.and_then(|level| {
            multiplier_bands(
                &[result.overall_influence.as_slice()],
                clusters.as_ref(),
                level,
                Bootstrap::new(bootstrap.replicates, bootstrap.seed),
            )
        });
        result.overall_se = bands.ok().and_then(|bands| bands.se[0]).unwrap_or(f64::NAN);
        let margin = pointwise * result.overall_se;
        result.overall_ci_low = result.overall_att - margin;
        result.overall_ci_high = result.overall_att + margin;
    }
    Ok(result)
}

/// Per-cell standard errors and intervals for `ATT(g,t)`, clustered.
///
/// # Why this exists
///
/// The `se` on an [`AttGtEstimate`] comes from the pair estimator, which sees
/// one `(g, t)` cell and has no idea that two of its units are the two parents of
/// one child. It is therefore an unclustered standard error, and in a design
/// where units share a household it is the wrong one.
///
/// `did` reports the clustered version whenever `clustervars` is set, under both
/// `bstrap = TRUE` and `bstrap = FALSE`. Under `FALSE` it is analytic, and this
/// is that: measured against `did` 2.5.1 on mpdta with counties clustered into
/// states, `sqrt(sum_c s_c^2) / n` where `s_c` is the cluster's summed influence.
/// No `G / (G - 1)` correction, which is the same convention as the aggregation
/// above and NOT the one
/// [`crate::inference::clustered_variance_from_index`] uses.
///
/// The effect is not marginal. On that data the first cell goes from 0.0221
/// unclustered to 0.0102 clustered, and another from 0.0312 to 0.0489. It moves
/// in both directions, so there is no safe direction to round in.
///
/// Pass a [`UnitPanel`] whose `clusters` are set. With `clusters: None` every
/// unit is its own cluster and the result reproduces the input `se`, which
/// `clustered_per_cell_reduces_to_the_pair_estimator` pins.
///
/// # Errors
/// [`AggteError::ShapeMismatch`] if the influence matrix does not match the
/// estimates, [`AggteError::RaggedInfluence`] if the vectors are not all the unit
/// count, [`AggteError::UnitGroupMismatch`] if the panel is a different size.
pub fn att_gt_clustered_standard_errors(
    estimates: &[AttGtEstimate],
    influence: &[Vec<f64>],
    panel: &UnitPanel,
    inference: InferenceConfig,
) -> Result<Vec<AttGtEstimate>, AggteError> {
    if estimates.is_empty() {
        return Err(AggteError::EmptyInput);
    }
    if influence.len() != estimates.len() {
        return Err(AggteError::ShapeMismatch {
            influence: influence.len(),
            estimates: estimates.len(),
        });
    }
    let n_units = panel.groups.len();
    if let Some(bad) = influence.iter().find(|vector| vector.len() != n_units) {
        return Err(AggteError::RaggedInfluence {
            expected: n_units,
            actual: bad.len(),
        });
    }
    let clusters = panel.cluster_index();
    if clusters.len() != n_units {
        return Err(AggteError::UnitGroupMismatch {
            expected: n_units,
            actual: clusters.len(),
        });
    }

    Ok(estimates
        .iter()
        .zip(influence)
        .map(|(estimate, vector)| {
            let se = standard_error(vector, &clusters);
            let (ci_low, ci_high) = normal_interval(estimate.att, se, inference.confidence_level);
            AttGtEstimate {
                se,
                ci_low,
                ci_high,
                ..*estimate
            }
        })
        .collect())
}

/// A [`UnitPanel`] for the repeated-cross-section route, indexed by input ROW.
///
/// [`super::panel_pairs::unit_panel`] is indexed by unit, because the panel
/// route's influence vectors are. The RC route's are indexed by input row, one
/// entry per observation, so its panel has to be too. Passing the wrong one is
/// caught by the length check in [`aggregate_att_gt`] rather than being absorbed
/// silently, but the two are easy to confuse, hence two named constructors
/// instead of one that guesses.
///
/// Cohort shares come out the same either way on a balanced panel, since every
/// unit contributes the same number of rows. On an unbalanced one they do not,
/// and the row-level share is the right one here: it is what `did` computes with
/// `panel = FALSE`, where a row IS the observation.
#[must_use]
pub fn row_panel(observations: &[crate::types::AttGtDrObservation]) -> UnitPanel {
    UnitPanel {
        groups: observations
            .iter()
            .map(|row| row.first_treated_time)
            .collect(),
        weights: observations.iter().map(|row| row.weight).collect(),
        clusters: None,
    }
}

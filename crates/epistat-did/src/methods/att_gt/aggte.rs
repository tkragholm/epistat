//! Aggregation of ATT(g,t) into the four summaries `did::aggte` produces.
//!
//! [`AttGt::aggregate`] forms the summary [`AggteOptions::aggregation`]
//! names, with the weights [`AggteOptions::weights`] names.
//!
//! # The weights are estimated, and that shows up in the standard error
//!
//! Under [`AggteWeights::Estimated`], `did::aggte`'s, three of the four
//! summaries weight cells by cohort share `pg = P(G = g)`,
//!
//! ```text
//! pg = sum_i w_i 1(G_i = g) / n,
//! ```
//!
//! which is estimated from the same sample. Treating it as fixed understates
//! the variance: measured against `did` 2.5.1 on `mpdta` at event time 0, the
//! fixed-weight standard error is 0.0114607 and the correct one is 0.0114942.
//! The correction, ported from `did:::wif`, adds to the influence function of
//! a combination `sum_k a_k ATT_k` with `a_k = pg_k / sum_l pg_l`
//!
//! ```text
//! wif_ik = (w_i 1(G_i = g_k) - pg_k) / sum_l pg_l
//!          - pg_k sum_l (w_i 1(G_i = g_l) - pg_l) / (sum_l pg_l)^2,
//! psi_i  = sum_k a_k psi_ik + sum_k wif_ik ATT_k.
//! ```
//!
//! The group summary averages a cohort's cells with equal weights, which are
//! not estimated, so no correction applies within a cohort; the overall
//! effect weights the cohorts by share and has it. The event-study overall is
//! the mean of the non-negative event times, and the calendar overall the
//! mean of the periods.
//!
//! The fixed weights, [`AggteWeights::Equal`], [`AggteWeights::TreatedCount`]
//! and [`AggteWeights::TotalWeight`], normalise the cells' weights within a
//! key and take them as known. The event-study summary then aggregates every
//! cell of an event time, combined in one rounding per multiply-add; the
//! cohort, calendar and simple summaries aggregate the post-treatment cells.
//! Only the simple summary has an overall effect.
//!
//! # Standard errors and the bootstrap
//!
//! Every aggregate has an influence function `psi_i`. Its analytic standard
//! error is `sqrt(sum_g s_g^2)` with `s_g` the sum of the centred `psi_i` in
//! cluster `g`, divided by `n`, as `did` forms it, with each unit its own
//! cluster when the panel has no clusters
//! ([`crate::inference::did_clusters`]). The one exception is the
//! fixed-weight event-study summary without clusters, whose standard error is
//! `sqrt(mean((psi_i - mean(psi))^2) / n)`, the same quantity up to the order
//! of the sums.
//!
//! With [`Inference::Bootstrap`] or [`Inference::UniformBand`] in
//! [`AggteOptions::inference`] the standard errors come from the
//! multiplier bootstrap of `did::mboot`. Replicate `b` draws a Rademacher
//! weight `w_gb` per cluster (per unit without clusters) and forms
//! `T_b = sqrt(G) * mean_g(w_gb * S_g)` with `S_g` the uncentred cluster sum
//! of the influence function. For each point of a path, `sigma` is the
//! interquartile range of its `T_b` divided by `qnorm(.75) - qnorm(.25)`, and
//! its standard error is `sigma * sqrt(G) / n`. With [`Inference::UniformBand`] the
//! critical value is the type 1 quantile at `1 - alpha` of
//! `max_j |T_bj| / sigma_j` over the points of the path, so the band
//! `att_j +- crit * se_j` covers the whole path with probability about
//! `1 - alpha`. Those intervals are
//! [simultaneous](epistat_core::Coverage::Simultaneous), with the critical
//! value, and their points have no p-value, since a pointwise p-value would
//! contradict a band that holds for the whole path. The overall effect has
//! its own bootstrap on its influence function, from the same seed, and a
//! pointwise interval.
//!
//! The analytic and the bootstrap standard errors differ when the draws are
//! not normal, since the interquartile range ignores the tails: for a cell
//! with few units the bootstrap value can be several per cent above the
//! analytic one. That is a property of `did::mboot`, which this reproduces.
//!
//! # `max_e`, `min_e` and `balance_e`
//!
//! `did::aggte` applies `max_e` to the simple, group and dynamic summaries,
//! `min_e` and `balance_e` to the dynamic one only, and none of them to the
//! calendar one; the fixed weights follow the same rule. See
//! [`AggteOptions::max_e`] for the unit in which `max_e` is counted.
//!
//! # References
//!
//! Callaway B, Sant'Anna PHC. Difference-in-differences with multiple time
//! periods. J Econom 2021;225(2):200-30. R package `did` 2.5.1, functions
//! `aggte`, `mboot`, `getSE` and `wif`.

use std::collections::{BTreeMap, BTreeSet};

use epistat_core::{Bootstrap, Covariance, Estimate, EventStudy, InputError, Interval, Level};
use epistat_inference::simultaneous::multiplier_bands;
use epistat_inference::{ClusterRule, Clusters, Influence};

use crate::error::Error;
use crate::inference::did_clusters::{covariance, standard_error};
use crate::inference::{estimate_from_parts, finite, normal_interval, z_score_for_confidence};
use crate::util::usize_to_f64;

use super::bands::by_unit;
use super::{AttGt, BasePeriod, aggregation, cell_estimate};

/// Which summary to produce, `aggte(type = ...)`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum AggteType {
    /// One number: every post-treatment cell, `"simple"`.
    Simple,
    /// By event time, then averaged over the non-negative event times,
    /// `"dynamic"`: the event study.
    Dynamic,
    /// By cohort, then averaged over cohorts by share, `"group"`, `aggte`'s
    /// default.
    #[default]
    Group,
    /// By calendar period, then averaged over periods, `"calendar"`.
    Calendar,
}

/// How the cells are weighted within a summary.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum AggteWeights {
    /// By cohort share, estimated from the sample's weights, with the
    /// influence of that estimate in the standard error: `did::aggte`.
    #[default]
    Estimated,
    /// Equal weights within a key, taken as known.
    Equal,
    /// By each cell's number of treated units or rows, taken as known.
    TreatedCount,
    /// By each cell's sum of weights, taken as known.
    TotalWeight,
}

/// Pointwise intervals or a band over the whole path, `aggte(cband = )`.
/// Read only beside the deprecated [`AggteOptions::bootstrap`];
/// [`Inference`] holds both choices.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Band {
    /// Pointwise normal intervals, `cband = FALSE`.
    #[default]
    Pointwise,
    /// The uniform critical value of the path from the multiplier bootstrap,
    /// `cband = TRUE`, as [`Inference::UniformBand`].
    Uniform,
}

/// How the standard errors and intervals of a summary are formed, `aggte`'s
/// `bstrap` and `cband`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Inference {
    /// Analytic standard errors from the influence function and pointwise
    /// normal intervals, `bstrap = FALSE`.
    #[default]
    Analytic,
    /// Standard errors from the multiplier bootstrap with `biters`
    /// replicates and pointwise intervals, `bstrap = TRUE, cband = FALSE`.
    Bootstrap(Bootstrap),
    /// The bootstrap standard errors and the uniform band over the path,
    /// `bstrap = TRUE, cband = TRUE`, R's default and what it prints. The
    /// critical value is never below the pointwise normal value. When no
    /// finite critical value exists the intervals stay pointwise and
    /// [`Aggte::uniform_band`] is `false`.
    UniformBand(Bootstrap),
}

impl Inference {
    /// The bootstrap of the two bootstrap variants.
    #[must_use]
    pub const fn bootstrap(self) -> Option<Bootstrap> {
        match self {
            Self::Analytic => None,
            Self::Bootstrap(bootstrap) | Self::UniformBand(bootstrap) => Some(bootstrap),
        }
    }
}

/// The options of [`AttGt::aggregate`], with `did::aggte`'s defaults except
/// that the inference is analytic and pointwise: R bootstraps and prints a
/// uniform band, which needs a seed, and none is chosen for the caller. Set
/// `inference: Inference::UniformBand(Bootstrap::new(1_000, seed))` for R's
/// printed table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AggteOptions {
    /// Which summary to produce, `type`.
    pub aggregation: AggteType,
    /// How the cells are weighted.
    pub weights: AggteWeights,
    /// Restrict the event study to a fixed composition of cohorts, those
    /// observed for at least this many post-treatment periods, and to event
    /// times up to it, `balance_e`. Without it the effect at a long horizon is
    /// estimated from other cohorts than the effect at a short one. Dynamic
    /// only, as in `did`.
    pub balance_e: Option<i64>,
    /// Drop event times below this inclusive bound, `min_e`, after
    /// `balance_e`. It narrows the overall effect as well as the path, since
    /// the event-study overall is the mean over the retained non-negative
    /// event times. Dynamic only, as in `did`.
    pub min_e: Option<i64>,
    /// Drop event times above this inclusive bound, `max_e`.
    ///
    /// For [`AggteType::Dynamic`] the bound is in the units of the period.
    /// For [`AggteType::Simple`] and [`AggteType::Group`] only the
    /// post-treatment cells at most `max_e` periods after the cohort's first
    /// treated period enter, as `did` 2.5.1 selects them, and the distance is
    /// counted in steps of the sorted list of periods and cohorts that occur
    /// in the cells, where `did` counts in its own period list. The two agree
    /// when the periods are consecutive. Ignored by [`AggteType::Calendar`],
    /// as in `did`.
    pub max_e: Option<i64>,
    /// The confidence level of the intervals, `1 - alp`.
    pub level: Level,
    /// Analytic or bootstrap standard errors, and pointwise intervals or a
    /// uniform band, `bstrap` and `cband`.
    pub inference: Inference,
    /// The multiplier bootstrap for the standard errors.
    #[deprecated(
        since = "0.1.0",
        note = "use `inference: Inference::Bootstrap(..)` or `Inference::UniformBand(..)`; \
                this field is removed in the next release"
    )]
    pub bootstrap: Option<Bootstrap>,
    /// Pointwise intervals or a uniform band.
    #[deprecated(
        since = "0.1.0",
        note = "use `inference: Inference::UniformBand(..)`; this field is removed in the \
                next release"
    )]
    pub band: Band,
    #[doc(hidden)]
    pub _extensible: epistat_core::Extensible,
}

impl Default for AggteOptions {
    #[expect(
        deprecated,
        reason = "the deprecated fields keep their defaults for one release"
    )]
    fn default() -> Self {
        Self {
            _extensible: epistat_core::Extensible::NEW,
            aggregation: AggteType::default(),
            weights: AggteWeights::default(),
            balance_e: None,
            min_e: None,
            max_e: None,
            level: Level::default(),
            inference: Inference::Analytic,
            bootstrap: None,
            band: Band::Pointwise,
        }
    }
}

impl AggteOptions {
    /// The inference these options ask for, from [`inference`](Self::inference)
    /// or the deprecated `bootstrap` and `band`.
    fn resolved_inference(&self) -> Result<Inference, Error> {
        #[expect(
            deprecated,
            reason = "the deprecated fields are honoured for one release"
        )]
        let (bootstrap, band) = (self.bootstrap, self.band);
        let inference = match (self.inference, bootstrap, band) {
            (inference, None, Band::Pointwise) => inference,
            (Inference::Analytic, Some(bootstrap), Band::Pointwise) => {
                Inference::Bootstrap(bootstrap)
            }
            (Inference::Analytic, Some(bootstrap), _) => Inference::UniformBand(bootstrap),
            (Inference::Analytic, None, _) => {
                return Err(Error::Unsupported {
                    what: "a uniform band without a bootstrap; set `inference: \
                           Inference::UniformBand(Bootstrap::new(1_000, seed))`",
                });
            }
            _ => {
                return Err(Error::Unsupported {
                    what: "both `inference` and the deprecated `bootstrap` or `band`; set \
                           `inference` alone",
                });
            }
        };
        if let Some(bootstrap) = inference.bootstrap()
            && bootstrap.replicates == 0
        {
            return Err(InputError::Domain {
                what: "bootstrap replicates",
                value: 0.0,
                domain: "[1, inf)",
                row: None,
            }
            .into());
        }
        Ok(inference)
    }
}

/// The options of [`AttGt::event_study`]: the restrictions of the event-time
/// path and the weights of its cells, `aggte(type = "dynamic")`'s
/// `balance_e`, `min_e` and `max_e`. The event study carries the
/// coefficients and their analytic covariance, so it has no level and no
/// bootstrap.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EventStudyOptions {
    /// As [`AggteOptions::balance_e`].
    pub balance_e: Option<i64>,
    /// As [`AggteOptions::min_e`].
    pub min_e: Option<i64>,
    /// As [`AggteOptions::max_e`].
    pub max_e: Option<i64>,
    /// How the cells are weighted within an event time.
    pub weights: AggteWeights,
    #[doc(hidden)]
    pub _extensible: epistat_core::Extensible,
}

/// One point of a summary's path.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AggtePoint {
    /// The event time, cohort or period, as the summary has it.
    pub key: i64,
    /// The estimate with its standard error and interval. Under a uniform
    /// band ([`Aggte::uniform_band`]) the interval is simultaneous, with the
    /// band's critical value, and there is no p-value.
    pub estimate: Estimate,
    /// The number of cells that entered it.
    pub n_cells: usize,
    /// The sum of those cells' total weights.
    pub total_weight: f64,
}

/// The overall effect of a summary.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AggteOverall {
    /// The estimate with its standard error and pointwise interval.
    pub estimate: Estimate,
    /// The number of cells behind it.
    pub n_cells: usize,
    /// The sum of those cells' total weights.
    pub total_weight: f64,
}

/// A summary of ATT(g,t): its path, its overall effect and their influence
/// function.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct Aggte {
    /// The summary.
    pub aggregation: AggteType,
    /// Its weights.
    pub weights: AggteWeights,
    /// The path by event time, cohort or period; empty for
    /// [`AggteType::Simple`].
    pub points: Vec<AggtePoint>,
    /// The overall effect, with a pointwise interval. Under fixed weights
    /// only the simple summary has one.
    pub overall: Option<AggteOverall>,
    /// The multiplier of a point's standard error in its interval: the
    /// pointwise normal quantile, or the uniform critical value when
    /// [`Aggte::uniform_band`] is set.
    pub critical_value: f64,
    /// Whether the intervals of the path use a uniform critical value, in
    /// which case they are simultaneous and the points have no p-value.
    pub uniform_band: bool,
    influence: Influence,
    clusters: Option<Vec<u32>>,
    level: Level,
    anticipation: i64,
    base_period: BasePeriod,
}

impl Aggte {
    /// The influence function of the points, in order, followed by that of
    /// the overall effect when there is one, over the units of the
    /// ATT(g,t)'s influence function.
    #[must_use]
    pub const fn influence(&self) -> &Influence {
        &self.influence
    }

    /// The point of the path at `key`.
    #[must_use]
    pub fn point(&self, key: i64) -> Option<&AggtePoint> {
        self.points.iter().find(|point| point.key == key)
    }

    /// The linear combination `sum w_k theta_k` of points of the path, each
    /// given by its key and weight, with its standard error from the combined
    /// influence function, clustered as the summary's, and a normal interval
    /// at the summary's level. A window average of event times 2 and 3 is
    /// `combine(&[(2, 0.5), (3, 0.5)])`.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for no weights, a weight that is not finite, or a key
    /// the path does not have.
    pub fn combine(&self, weights: &[(i64, f64)]) -> Result<Estimate, Error> {
        epistat_core::check::non_empty("weights", weights.len())?;
        let mut columns = vec![0.0; self.influence.n_columns()];
        let mut estimate = 0.0;
        for &(key, weight) in weights {
            if !weight.is_finite() {
                return Err(InputError::NotFinite {
                    what: "weights",
                    row: 0,
                    column: None,
                }
                .into());
            }
            let index = self
                .points
                .iter()
                .position(|point| point.key == key)
                .ok_or(InputError::Domain {
                    what: "key",
                    value: key as f64,
                    domain: "the keys of the path",
                    row: None,
                })?;
            columns[index] += weight;
            estimate += weight * self.points[index].estimate.estimate;
        }
        let combined = self.influence.combine(&columns)?;
        let se = standard_error(combined.column(0), &self.numbered_clusters()?);
        let ci = normal_interval(estimate, se, self.level.confidence_value());
        Ok(cell_estimate(estimate, se, ci, self.level))
    }

    /// The event study of a dynamic summary under a universal base period:
    /// the points' estimates by event time, their covariance clustered as
    /// `did` clusters it, and the reference period `-1 - anticipation`.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for another summary or a varying base period,
    /// and [`Error::Input`] when the path holds the reference period.
    pub fn event_study(&self) -> Result<EventStudy, Error> {
        if self.aggregation != AggteType::Dynamic {
            return Err(Error::Unsupported {
                what: "an event study of a summary other than the dynamic one; aggregate with \
                       `aggregation: AggteType::Dynamic`",
            });
        }
        if self.base_period != BasePeriod::Universal {
            return Err(VARYING_BASE);
        }
        let k = self.points.len();
        let columns: Vec<&[f64]> = (0..k).map(|j| self.influence.column(j)).collect();
        let values = covariance(&columns, &self.numbered_clusters()?);
        let covariance = Covariance::new(k, values.into_iter().map(Some).collect())?;
        Ok(EventStudy::new(
            self.points
                .iter()
                .map(|point| point.estimate.estimate)
                .collect(),
            covariance,
            self.points.iter().map(|point| point.key).collect(),
            -1 - self.anticipation,
        )?)
    }

    fn numbered_clusters(&self) -> Result<Clusters, Error> {
        Ok(match &self.clusters {
            Some(codes) => Clusters::from_codes(codes),
            None => Clusters::singletons(self.influence.n_units())?,
        })
    }
}

/// The refusal of an event study under a varying base period, whose
/// pre-treatment cells are not measured against one reference.
pub(super) const VARYING_BASE: Error = Error::Unsupported {
    what: "an event study under a varying base period, `did`'s default; set `base_period: \
           BasePeriod::Universal` in the AttGtOptions of att_gt::estimate",
};

/// A fitted cell as the aggregations read it.
#[derive(Debug, Clone, Copy)]
pub(super) struct Cell {
    pub group: i64,
    pub time: i64,
    pub event_time: i64,
    pub att: f64,
    pub treated_n: usize,
    pub total_weight: f64,
}

/// One aggregated point before it becomes an [`AggtePoint`]; a standard error
/// or interval end that is not finite is `None`.
#[derive(Debug, Clone)]
pub(super) struct Point {
    pub key: i64,
    pub att: f64,
    pub se: Option<f64>,
    pub interval: (Option<f64>, Option<f64>),
    pub influence: Vec<f64>,
    pub n_cells: usize,
    pub total_weight: f64,
}

impl Point {
    /// The point at `key` with the normal interval at `confidence_level`.
    pub(super) fn normal(
        key: i64,
        (att, influence): (f64, Vec<f64>),
        se: f64,
        confidence_level: f64,
        n_cells: usize,
        total_weight: f64,
    ) -> Self {
        let (lower, upper) = normal_interval(att, se, confidence_level);
        Self {
            key,
            att,
            se: finite(se),
            interval: (finite(lower), finite(upper)),
            influence,
            n_cells,
            total_weight,
        }
    }

    /// Replaces the standard error, and the interval by `att -/+ critical se`.
    fn rescale(&mut self, se: Option<f64>, critical: f64) {
        self.se = se.and_then(finite);
        let margin = self.se.map(|se| critical * se);
        self.interval = (
            margin.and_then(|margin| finite(self.att - margin)),
            margin.and_then(|margin| finite(self.att + margin)),
        );
    }
}

/// A summary before it becomes an [`Aggte`].
struct Summary {
    points: Vec<Point>,
    overall: Option<Point>,
    critical_value: f64,
    uniform_band: bool,
}

/// The per-unit inputs the cohort shares are estimated from.
struct Units<'a> {
    groups: &'a [Option<i64>],
    weights: &'a [f64],
}

/// Cohort share `pg = sum_i w_i * 1(G_i = g) / n`, as `did` computes it.
///
/// The denominator is the full unit count, never-treated included. It cancels
/// out of the normalised weights, but not out of [`weight_influence`], so it
/// has to be the same `n` in both.
fn cohort_share(units: &Units<'_>, group: i64) -> f64 {
    let n = usize_to_f64(units.groups.len());
    let total: f64 = units
        .groups
        .iter()
        .zip(units.weights)
        .filter(|(unit_group, _)| **unit_group == Some(group))
        .map(|(_, weight)| *weight)
        .sum();
    total / n
}

/// The influence function of the estimated weights, ported from `did:::wif`,
/// `n_units x n_cells`.
fn weight_influence(units: &Units<'_>, groups: &[i64], shares: &[f64]) -> Vec<Vec<f64>> {
    let share_total: f64 = shares.iter().sum();
    let n_units = units.groups.len();

    // `if2` needs the row sum of the centred indicators, so it is built once.
    let row_sums = (0..n_units)
        .map(|unit| {
            groups
                .iter()
                .zip(shares)
                .map(|(group, share)| indicator(units, unit, *group) - share)
                .sum::<f64>()
        })
        .collect::<Vec<f64>>();

    (0..n_units)
        .map(|unit| {
            groups
                .iter()
                .zip(shares)
                .map(|(group, share)| {
                    let if1 = (indicator(units, unit, *group) - share) / share_total;
                    let if2 = row_sums[unit] * share / (share_total * share_total);
                    if1 - if2
                })
                .collect()
        })
        .collect()
}

fn indicator(units: &Units<'_>, unit: usize, group: i64) -> f64 {
    if units.groups[unit] == Some(group) {
        units.weights[unit]
    } else {
        0.0
    }
}

/// Combine selected cells with cohort-share weights, correcting for the fact
/// that those weights were estimated.
fn combine_by_share(
    selected: &[usize],
    cells: &[Cell],
    influence: &[&[f64]],
    units: &Units<'_>,
) -> (f64, Vec<f64>) {
    let groups = selected.iter().map(|&c| cells[c].group).collect::<Vec<_>>();
    let shares = groups
        .iter()
        .map(|&g| cohort_share(units, g))
        .collect::<Vec<_>>();
    let share_total: f64 = shares.iter().sum();
    let weights = shares.iter().map(|s| s / share_total).collect::<Vec<_>>();

    let att = selected
        .iter()
        .zip(&weights)
        .map(|(&c, w)| w * cells[c].att)
        .sum::<f64>();

    let wif = weight_influence(units, &groups, &shares);
    let combined = (0..units.groups.len())
        .map(|unit| {
            let weighted = selected
                .iter()
                .zip(&weights)
                .map(|(&c, w)| w * influence[c][unit])
                .sum::<f64>();
            let correction = wif[unit]
                .iter()
                .zip(selected)
                .map(|(w, &c)| w * cells[c].att)
                .sum::<f64>();
            weighted + correction
        })
        .collect();

    (att, combined)
}

/// Combine selected cells with equal weights. No correction: `1/k` is not
/// estimated from the sample.
fn combine_equally(
    selected: &[usize],
    cells: &[Cell],
    influence: &[&[f64]],
    n_units: usize,
) -> (f64, Vec<f64>) {
    let weights = vec![1.0 / usize_to_f64(selected.len()); selected.len()];
    combine_fixed(selected, &weights, cells, influence, n_units)
}

/// Combine selected cells with the given weights, which are taken as known,
/// so the influence function is the weighted sum of the cells' and nothing
/// more.
pub(super) fn combine_fixed(
    selected: &[usize],
    weights: &[f64],
    cells: &[Cell],
    influence: &[&[f64]],
    n_units: usize,
) -> (f64, Vec<f64>) {
    let att = selected
        .iter()
        .zip(weights)
        .map(|(&c, w)| w * cells[c].att)
        .sum();
    let combined = (0..n_units)
        .map(|unit| {
            selected
                .iter()
                .zip(weights)
                .map(|(&c, w)| w * influence[c][unit])
                .sum()
        })
        .collect();
    (att, combined)
}

/// Average already-aggregated points with equal weights.
fn average_points(points: &[&Point], n_units: usize) -> (f64, Vec<f64>) {
    let weight = 1.0 / usize_to_f64(points.len());
    let att = points.iter().map(|p| weight * p.att).sum();
    let influence = (0..n_units)
        .map(|unit| points.iter().map(|p| weight * p.influence[unit]).sum())
        .collect();
    (att, influence)
}

/// The point with its standard error and interval, from the cells
/// `selected` that entered it.
fn finish(
    key: i64,
    (att, influence): (f64, Vec<f64>),
    selected: &[usize],
    cells: &[Cell],
    confidence_level: f64,
    clusters: &Clusters,
) -> Point {
    let se = standard_error(&influence, clusters);
    Point::normal(
        key,
        (att, influence),
        se,
        confidence_level,
        selected.len(),
        selected.iter().map(|&c| cells[c].total_weight).sum(),
    )
}

/// Position of each period or cohort in the sorted list of those that occur
/// in the cells, which is how `did` measures `max_e` for the simple and group
/// summaries.
fn period_ranks(cells: &[Cell]) -> BTreeMap<i64, i64> {
    let periods = cells
        .iter()
        .flat_map(|cell| [cell.time, cell.group])
        .collect::<BTreeSet<i64>>();
    periods
        .into_iter()
        .enumerate()
        .map(|(rank, period)| (period, i64::try_from(rank).unwrap_or(i64::MAX)))
        .collect()
}

/// The post-treatment cells that `max_e` leaves in the simple and group
/// summaries.
fn post_cells_within(cells: &[Cell], max_e: Option<i64>) -> Vec<usize> {
    let ranks = max_e.map(|_| period_ranks(cells));
    (0..cells.len())
        .filter(|&i| cells[i].time >= cells[i].group)
        .filter(|&i| match (max_e, ranks.as_ref()) {
            (Some(limit), Some(ranks)) => ranks[&cells[i].time] - ranks[&cells[i].group] <= limit,
            _ => true,
        })
        .collect()
}

/// The cells of the event study that `balance_e`, `min_e` and `max_e` leave,
/// by event time.
///
/// `balance_e` keeps only cohorts with at least `b` post-treatment periods
/// observed, and then shows event times up to `b`. Both halves are needed:
/// without the first the composition still moves, without the second the
/// path runs past where every kept cohort is present. The last period is taken
/// over all cells, since it is the end of the data.
fn dynamic_cells(cells: &[Cell], options: &AggteOptions) -> BTreeMap<i64, Vec<usize>> {
    let max_time = cells.iter().map(|e| e.time).max().unwrap_or_default();
    let kept_groups = options.balance_e.map(|b| {
        cells
            .iter()
            .map(|e| e.group)
            .filter(|g| max_time - g >= b)
            .collect::<BTreeSet<i64>>()
    });

    let mut buckets = BTreeMap::<i64, Vec<usize>>::new();
    for (index, cell) in cells.iter().enumerate() {
        if kept_groups
            .as_ref()
            .is_some_and(|keep| !keep.contains(&cell.group))
        {
            continue;
        }
        if options.balance_e.is_some_and(|b| cell.event_time > b) {
            continue;
        }
        if options.min_e.is_some_and(|e| cell.event_time < e)
            || options.max_e.is_some_and(|e| cell.event_time > e)
        {
            continue;
        }
        buckets.entry(cell.event_time).or_default().push(index);
    }
    buckets
}

/// The post-treatment cells `post` by the key of the summary.
fn post_buckets(
    cells: &[Cell],
    post: &[usize],
    key: fn(&Cell) -> i64,
) -> BTreeMap<i64, Vec<usize>> {
    let mut buckets = BTreeMap::<i64, Vec<usize>>::new();
    for &index in post {
        buckets.entry(key(&cells[index])).or_default().push(index);
    }
    buckets
}

/// The summary under the estimated cohort shares, `did::aggte`.
#[expect(
    clippy::too_many_lines,
    reason = "the four aggregation types differ in which cells they select and \
              how they weight them; splitting them apart hides the comparison"
)]
fn by_share(
    cells: &[Cell],
    influence: &[&[f64]],
    units: &Units<'_>,
    clusters: &Clusters,
    options: &AggteOptions,
) -> Result<Summary, Error> {
    let n_units = units.groups.len();
    let confidence_level = options.level.confidence_value();
    let critical_value = z_score_for_confidence(confidence_level);
    let post = post_cells_within(cells, options.max_e);
    let summary = |points, overall| Summary {
        points,
        overall: Some(overall),
        critical_value,
        uniform_band: false,
    };

    match options.aggregation {
        AggteType::Simple => {
            if post.is_empty() {
                return Err(Error::NoPostTreatmentCells);
            }
            let combined = combine_by_share(&post, cells, influence, units);
            let overall = finish(0, combined, &post, cells, confidence_level, clusters);
            Ok(summary(Vec::new(), overall))
        }
        AggteType::Dynamic => {
            let buckets = dynamic_cells(cells, options);
            let points = buckets
                .iter()
                .map(|(&event_time, selected)| {
                    let combined = combine_by_share(selected, cells, influence, units);
                    finish(
                        event_time,
                        combined,
                        selected,
                        cells,
                        confidence_level,
                        clusters,
                    )
                })
                .collect::<Vec<_>>();
            let positive = points.iter().filter(|p| p.key >= 0).collect::<Vec<_>>();
            if positive.is_empty() {
                return Err(Error::NoPostTreatmentCells);
            }
            let behind: Vec<usize> = buckets
                .iter()
                .filter(|(key, _)| **key >= 0)
                .flat_map(|(_, selected)| selected.iter().copied())
                .collect();
            let combined = average_points(&positive, n_units);
            let overall = finish(0, combined, &behind, cells, confidence_level, clusters);
            Ok(summary(points, overall))
        }
        AggteType::Group => {
            if post.is_empty() {
                return Err(Error::NoPostTreatmentCells);
            }
            let buckets = post_buckets(cells, &post, |cell| cell.group);
            // Equal weights WITHIN a cohort: an ATT(g,t) for one cohort is
            // averaged over its own post periods, and 1/k is not estimated.
            let points = buckets
                .iter()
                .map(|(&group, selected)| {
                    let combined = combine_equally(selected, cells, influence, n_units);
                    finish(group, combined, selected, cells, confidence_level, clusters)
                })
                .collect::<Vec<_>>();

            // Cohort shares ACROSS cohorts, so the correction applies here.
            let shares = points
                .iter()
                .map(|point| cohort_share(units, point.key))
                .collect::<Vec<_>>();
            let share_total: f64 = shares.iter().sum();
            let weights = shares.iter().map(|s| s / share_total).collect::<Vec<_>>();
            let groups = points.iter().map(|point| point.key).collect::<Vec<_>>();
            let atts = points.iter().map(|point| point.att).collect::<Vec<_>>();
            let wif = weight_influence(units, &groups, &shares);

            let overall_att = points
                .iter()
                .zip(&weights)
                .map(|(point, w)| w * point.att)
                .sum::<f64>();
            let overall_influence = (0..n_units)
                .map(|unit| {
                    let weighted = points
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
                (overall_att, overall_influence),
                &post,
                cells,
                confidence_level,
                clusters,
            );
            Ok(summary(points, overall))
        }
        AggteType::Calendar => {
            let post = post_cells_within(cells, None);
            if post.is_empty() {
                return Err(Error::NoPostTreatmentCells);
            }
            let buckets = post_buckets(cells, &post, |cell| cell.time);
            let points = buckets
                .iter()
                .map(|(&time, selected)| {
                    let combined = combine_by_share(selected, cells, influence, units);
                    finish(time, combined, selected, cells, confidence_level, clusters)
                })
                .collect::<Vec<_>>();
            let all = points.iter().collect::<Vec<_>>();
            let combined = average_points(&all, n_units);
            let overall = finish(0, combined, &post, cells, confidence_level, clusters);
            Ok(summary(points, overall))
        }
    }
}

/// The summary under fixed weights.
fn by_fixed_weights(
    cells: &[Cell],
    influence: &[&[f64]],
    clusters: &Clusters,
    has_clusters: bool,
    options: &AggteOptions,
) -> Result<Summary, Error> {
    let confidence_level = options.level.confidence_value();
    let critical_value = z_score_for_confidence(confidence_level);
    let keyed = |buckets: BTreeMap<i64, Vec<usize>>| {
        aggregation::by_key(
            buckets,
            cells,
            influence,
            options.weights,
            clusters,
            confidence_level,
        )
    };
    let (points, overall) = match options.aggregation {
        AggteType::Dynamic => {
            let selected: Vec<usize> = dynamic_cells(cells, options)
                .into_values()
                .flatten()
                .collect::<BTreeSet<usize>>()
                .into_iter()
                .collect();
            let kept: Vec<Cell> = selected.iter().map(|&c| cells[c]).collect();
            let columns: Vec<&[f64]> = selected.iter().map(|&c| influence[c]).collect();
            let points = aggregation::by_event_time(
                &kept,
                &columns,
                options.weights,
                has_clusters.then_some(clusters),
                confidence_level,
            );
            (points, None)
        }
        AggteType::Simple => {
            let post = post_cells_within(cells, options.max_e);
            if post.is_empty() {
                return Err(Error::NoPostTreatmentCells);
            }
            let overall = keyed(BTreeMap::from([(0, post)])).pop();
            (Vec::new(), overall)
        }
        AggteType::Group => {
            let post = post_cells_within(cells, options.max_e);
            if post.is_empty() {
                return Err(Error::NoPostTreatmentCells);
            }
            (keyed(post_buckets(cells, &post, |cell| cell.group)), None)
        }
        AggteType::Calendar => {
            let post = post_cells_within(cells, None);
            if post.is_empty() {
                return Err(Error::NoPostTreatmentCells);
            }
            (keyed(post_buckets(cells, &post, |cell| cell.time)), None)
        }
    };
    Ok(Summary {
        points,
        overall,
        critical_value,
        uniform_band: false,
    })
}

/// Replaces the analytic standard errors and the pointwise critical value by
/// the bootstrap ones, as `did::aggte` does for `bstrap = TRUE` and
/// `cband = TRUE`.
///
/// The path is bootstrapped as one matrix, so its standard errors and
/// critical value come from the same draws. The overall effect has a separate
/// bootstrap from the same seed, so for a path whose overall effect is a
/// fixed linear combination of its points the draw of the overall effect is
/// that combination of the draws of the points.
fn apply_bootstrap(
    mut summary: Summary,
    clusters: Option<&Clusters>,
    bootstrap: Bootstrap,
    uniform_band: bool,
    confidence_level: f64,
) -> Result<Summary, Error> {
    let alpha = 1.0 - confidence_level;
    let pointwise = z_score_for_confidence(confidence_level);
    let finite = summary
        .points
        .iter()
        .chain(summary.overall.as_ref())
        .all(|point| point.influence.iter().all(|value| value.is_finite()));
    if !finite {
        return Err(Error::NotFinite);
    }
    // An alpha of 0 or 1 is an error of every band call below, as an invalid
    // level is.
    let level = Level::alpha(alpha).map_err(epistat_inference::Error::from);

    if !summary.points.is_empty() {
        let columns = summary
            .points
            .iter()
            .map(|point| point.influence.as_slice())
            .collect::<Vec<_>>();
        let bands = level.clone().and_then(|level| {
            multiplier_bands(by_unit(&columns)?.as_ref(), clusters, level, bootstrap)
        });
        // An error here means that no draw gave a finite studentised maximum,
        // so there are no bootstrap standard errors either.
        let standard_errors = bands.as_ref().ok().map(|bands| bands.se.clone());
        if let (true, Ok(bands)) = (uniform_band, bands.as_ref())
            && bands.critical.is_finite()
            && bands.critical >= pointwise
        {
            summary.critical_value = bands.critical;
            summary.uniform_band = true;
        }
        for (index, point) in summary.points.iter_mut().enumerate() {
            let se = standard_errors.as_ref().and_then(|se| se[index]);
            point.rescale(se, summary.critical_value);
        }
    }

    if let Some(overall) = summary.overall.as_mut() {
        let bands = level.and_then(|level| {
            let influence = by_unit(&[overall.influence.as_slice()])?;
            multiplier_bands(influence.as_ref(), clusters, level, bootstrap)
        });
        overall.rescale(bands.ok().and_then(|bands| bands.se[0]), pointwise);
    }
    Ok(summary)
}

/// [`AttGt::aggregate`].
pub(super) fn aggregate(fit: &AttGt, options: &AggteOptions) -> Result<Aggte, Error> {
    let influence = fit.kept_influence()?;
    let inference = options.resolved_inference()?;
    let cells: Vec<Cell> = fit
        .estimated()
        .map(|(cell, estimate)| Cell {
            group: cell.group,
            time: cell.time,
            event_time: cell.event_time,
            att: estimate.estimate,
            treated_n: cell.n_treated,
            total_weight: cell.total_weight,
        })
        .collect();
    let columns: Vec<&[f64]> = (0..influence.n_columns())
        .map(|j| influence.column(j))
        .collect();
    let units = Units {
        groups: &fit.sample.groups,
        weights: &fit.sample.weights,
    };
    let codes = fit.clusters.as_deref();
    let clusters = match codes {
        Some(codes) => Clusters::from_codes(codes),
        None => Clusters::singletons(units.groups.len())?,
    };

    let summary = match options.weights {
        AggteWeights::Estimated => by_share(&cells, &columns, &units, &clusters, options)?,
        _ => by_fixed_weights(&cells, &columns, &clusters, codes.is_some(), options)?,
    };
    let confidence_level = options.level.confidence_value();
    let summary = match inference.bootstrap() {
        None => summary,
        Some(bootstrap) => {
            let bootstrap_clusters = codes.map(Clusters::from_codes);
            apply_bootstrap(
                summary,
                bootstrap_clusters.as_ref(),
                bootstrap,
                matches!(inference, Inference::UniformBand(_)),
                confidence_level,
            )?
        }
    };
    into_aggte(fit, options, &summary, influence)
}

/// The public summary, with the points' and the overall effect's influence
/// functions in one [`Influence`].
fn into_aggte(
    fit: &AttGt,
    options: &AggteOptions,
    summary: &Summary,
    cells_influence: &Influence,
) -> Result<Aggte, Error> {
    let level = options.level;
    let estimate = |point: &Point| estimate_from_parts(point.att, point.se, point.interval, level);
    // A uniform band covers the whole path, so its intervals are simultaneous
    // and a pointwise p-value would contradict them.
    let path_estimate = |point: &Point| -> Result<Estimate, Error> {
        let pointwise = estimate(point);
        if !summary.uniform_band {
            return Ok(pointwise);
        }
        let interval = Interval::simultaneous(
            pointwise.lower(),
            pointwise.upper(),
            level,
            summary.critical_value,
        )?;
        Ok(pointwise.with_interval(Some(interval)).with_p_value(None))
    };
    let points: Vec<AggtePoint> = summary
        .points
        .iter()
        .map(|point| {
            Ok(AggtePoint {
                key: point.key,
                estimate: path_estimate(point)?,
                n_cells: point.n_cells,
                total_weight: point.total_weight,
            })
        })
        .collect::<Result<_, Error>>()?;
    let overall = summary.overall.as_ref().map(|point| AggteOverall {
        estimate: estimate(point),
        n_cells: point.n_cells,
        total_weight: point.total_weight,
    });
    let n_columns = summary.points.len() + usize::from(summary.overall.is_some());
    // Only a fixed-weight event study whose restrictions keep no cell gets here
    // without a point.
    if n_columns == 0 {
        return Err(Error::NoPostTreatmentCells);
    }
    let mut values = Vec::with_capacity(n_columns * cells_influence.n_units());
    for point in summary.points.iter().chain(summary.overall.as_ref()) {
        values.extend_from_slice(&point.influence);
    }
    let influence = cells_influence
        .with_values(values, n_columns, ClusterRule::NoSmallSample)
        .map_err(|_| Error::NotFinite)?;
    Ok(Aggte {
        aggregation: options.aggregation,
        weights: options.weights,
        points,
        overall,
        critical_value: summary.critical_value,
        uniform_band: summary.uniform_band,
        influence,
        clusters: fit.clusters.clone(),
        level,
        anticipation: fit.anticipation,
        base_period: fit.base_period,
    })
}

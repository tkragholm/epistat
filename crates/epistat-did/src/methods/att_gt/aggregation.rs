use std::collections::BTreeMap;

use super::aggte::{
    AggteConfig, AggteError, AggteType, UnitPanel, aggregate_att_gt, checked_cluster_index,
    combine_fixed,
};
use crate::inference::did_clusters::standard_error;
use crate::inference::{normal_interval, validate_confidence_level};
use crate::types::{
    AttGtAggregatedEstimate, AttGtAggregationConfig, AttGtAggregationError,
    AttGtAggregationWeighting, AttGtCalendarEstimate, AttGtCohortEstimate, AttGtEstimate,
    AttGtEventTimeEstimate, AttGtEventTimeInfluenceOutput,
};
use crate::util::usize_to_f64;

/// Aggregate ATT(g,t) estimates by event-time while preserving aligned
/// influence-function vectors for joint-process inference.
///
/// Every cell enters, pre-treatment ones included, with weights that are taken
/// as known, and the standard error is that of the combined influence function,
/// unclustered. It is not `did::aggte`, which estimates the cohort shares and
/// corrects for it: use [`aggregate_att_gt`](super::aggte::aggregate_att_gt)
/// for that.
///
/// # Errors
///
/// Returns [`AttGtAggregationError`] when estimates/configuration are invalid or
/// when the influence-function matrix is inconsistent with the estimate vector.
pub fn aggregate_att_gt_event_time_with_influence(
    estimates: &[AttGtEstimate],
    influence_functions: &[Vec<f64>],
    config: AttGtAggregationConfig,
) -> Result<AttGtEventTimeInfluenceOutput, AttGtAggregationError> {
    validate_aggregation_inputs(estimates, config)?;
    if influence_functions.len() != estimates.len() {
        return Err(AttGtAggregationError::EmptyInput);
    }
    let n = influence_functions.first().map_or(0, Vec::len);
    if n == 0 {
        return Err(AttGtAggregationError::EmptyInput);
    }
    if influence_functions
        .iter()
        .any(|influence| influence.len() != n)
    {
        return Err(AttGtAggregationError::EmptyInput);
    }

    let cells = estimates
        .iter()
        .zip(influence_functions)
        .map(|(estimate, influence)| WeightedCell {
            key: estimate.event_time,
            att: estimate.att,
            influence,
            raw_weight: raw_weight(estimate, config.weighting),
            total_weight: estimate.total_weight,
        })
        .collect::<Vec<_>>();
    let (summaries, aggregated_influence) =
        aggregate_weighted_cells(&cells, config.confidence_level.confidence_level);
    Ok(AttGtEventTimeInfluenceOutput {
        estimates: summaries
            .into_iter()
            .map(|(event_time, summary)| AttGtEventTimeEstimate {
                event_time,
                summary,
            })
            .collect(),
        influence_functions: aggregated_influence,
    })
}

/// One cell entering a fixed-weight aggregation.
pub(crate) struct WeightedCell<'a> {
    /// The key the cell is aggregated under: event time, cohort or period.
    pub key: i32,
    /// The cell's estimate.
    pub att: f64,
    /// The cell's influence function, one value per unit or row.
    pub influence: &'a [f64],
    /// The weight before normalisation within the key.
    pub raw_weight: f64,
    /// The sum of the observation weights the cell used.
    pub total_weight: f64,
}

/// Weighted mean of the cells under each key, with the weights taken as known.
///
/// The weights within a key are normalised to sum to 1. The standard error is
/// that of the combined influence function, so it counts the covariance between
/// cells that share units, and it is not clustered. The estimate and the
/// influence function are built cell by cell in input order with one rounding
/// per multiply-add. The second element holds the combined influence function
/// of each key, in key order.
///
/// Every cell is aggregated whatever its sign of event time, and the weights
/// carry no correction for being estimated; [`aggregate_att_gt`] is the
/// aggregation of `did::aggte`.
pub(crate) fn aggregate_weighted_cells(
    cells: &[WeightedCell<'_>],
    confidence_level: f64,
) -> (BTreeMap<i32, AttGtAggregatedEstimate>, Vec<Vec<f64>>) {
    let n = cells.first().map_or(0, |cell| cell.influence.len());
    let mut grouped = BTreeMap::<i32, Vec<&WeightedCell<'_>>>::new();
    for cell in cells {
        grouped.entry(cell.key).or_default().push(cell);
    }

    let mut summaries = BTreeMap::new();
    let mut influences = Vec::with_capacity(grouped.len());
    for (key, bucket) in grouped {
        let raw_total = bucket.iter().map(|cell| cell.raw_weight).sum::<f64>();
        let mut estimate = 0.0;
        let mut total_influence = vec![0.0; n];
        let mut total_weight = 0.0;
        for cell in &bucket {
            let normalized_weight = cell.raw_weight / raw_total;
            estimate = normalized_weight.mul_add(cell.att, estimate);
            total_weight += cell.total_weight;
            for (dst, src) in total_influence.iter_mut().zip(cell.influence) {
                *dst = normalized_weight.mul_add(*src, *dst);
            }
        }
        // From the COMBINED influence function, not from `sum(w^2 * se^2)`, which
        // is the variance of a weighted sum of independent terms. ATT(g,t) cells
        // share units, so the off-diagonal terms are not zero, and the
        // independent-terms variance is too small when the cells are positively
        // correlated, which is the ordinary case.
        let se = crate::inference::standard_error_from_influence(&total_influence);
        let (ci_low, ci_high) = normal_interval(estimate, se, confidence_level);
        summaries.insert(
            key,
            AttGtAggregatedEstimate {
                estimate,
                se,
                ci_low,
                ci_high,
                components: bucket.len(),
                total_weight,
            },
        );
        influences.push(total_influence);
    }
    (summaries, influences)
}

/// Aggregate post-treatment `ATT(g,t)` estimates by treatment cohort, as
/// `did::aggte(type = "group")` reports them per cohort.
///
/// Only cells with `t >= g` enter. Within a cohort the cells are weighted by
/// [`AttGtAggregationConfig::weighting`]: under `ByTreatedCount` and `Equal`
/// every cell has the same weight, which is `aggte`'s average over the
/// cohort's post-treatment periods, and under `ByTotalWeight` each cell's
/// weight is its `total_weight`. These weights are taken as known, so no
/// estimation correction applies. The standard error is that of the combined
/// influence function, centred and summed within the clusters of `panel`, as
/// [`aggregate_att_gt`](super::aggte::aggregate_att_gt) forms it.
///
/// `influence` holds one vector per estimate, indexed like `panel`:
/// [`unit_panel`](super::panel_pairs::unit_panel) for the panel route and
/// [`row_panel`](super::aggte::row_panel) for repeated cross-sections.
/// `components` and `total_weight` count the post-treatment cells that entered.
///
/// # Errors
///
/// [`AggteError`] when the shapes disagree, the confidence level is invalid or
/// no estimate is post-treatment.
pub fn aggregate_att_gt_by_cohort(
    estimates: &[AttGtEstimate],
    influence: &[Vec<f64>],
    panel: &UnitPanel,
    config: AttGtAggregationConfig,
) -> Result<Vec<AttGtCohortEstimate>, AggteError> {
    aggregate_post_treatment(estimates, influence, panel, config, Summary::Cohort).map(|grouped| {
        grouped
            .into_iter()
            .map(|(group, summary)| AttGtCohortEstimate { group, summary })
            .collect()
    })
}

/// Aggregate post-treatment `ATT(g,t)` estimates by calendar time, as
/// `did::aggte(type = "calendar")` reports them per period.
///
/// Only cells with `t >= g` enter. Under `ByTreatedCount` the cells of a
/// period are weighted by their cohort's share `P(G = g)`, estimated from the
/// sampling weights of `panel` as `did` does, and the influence of that
/// estimate is added as `did:::wif` adds it: this is
/// [`aggregate_att_gt`](super::aggte::aggregate_att_gt) with
/// [`AggteType::Calendar`]. Under `Equal` and `ByTotalWeight` the weights are
/// equal or the cells' `total_weight`, taken as known, with no correction. The
/// standard error, inputs and counts are as in [`aggregate_att_gt_by_cohort`].
///
/// # Errors
///
/// [`AggteError`] when the shapes disagree, the confidence level is invalid or
/// no estimate is post-treatment.
pub fn aggregate_att_gt_by_calendar_time(
    estimates: &[AttGtEstimate],
    influence: &[Vec<f64>],
    panel: &UnitPanel,
    config: AttGtAggregationConfig,
) -> Result<Vec<AttGtCalendarEstimate>, AggteError> {
    aggregate_post_treatment(estimates, influence, panel, config, Summary::Calendar).map(
        |grouped| {
            grouped
                .into_iter()
                .map(|(time, summary)| AttGtCalendarEstimate { time, summary })
                .collect()
        },
    )
}

/// Aggregate all post-treatment `ATT(g,t)` estimates into one number, as
/// `did::aggte(type = "simple")`.
///
/// Under `ByTreatedCount` each cell with `t >= g` is weighted by its cohort's
/// estimated share `P(G = g)`, with the influence of that estimate included:
/// this is [`aggregate_att_gt`](super::aggte::aggregate_att_gt) with
/// [`AggteType::Simple`]. Under `Equal` and `ByTotalWeight` the weights are
/// equal or the cells' `total_weight`, taken as known. The standard error,
/// inputs and counts are as in [`aggregate_att_gt_by_cohort`].
///
/// # Errors
///
/// [`AggteError`] when the shapes disagree, the confidence level is invalid or
/// no estimate is post-treatment.
pub fn aggregate_att_gt_overall(
    estimates: &[AttGtEstimate],
    influence: &[Vec<f64>],
    panel: &UnitPanel,
    config: AttGtAggregationConfig,
) -> Result<AttGtAggregatedEstimate, AggteError> {
    aggregate_post_treatment(estimates, influence, panel, config, Summary::Overall)?
        .into_values()
        .next()
        .ok_or(AggteError::NoPostTreatmentCells)
}

#[derive(Debug, Clone, Copy)]
enum Summary {
    Cohort,
    Calendar,
    Overall,
}

impl Summary {
    const fn key(self, estimate: &AttGtEstimate) -> i32 {
        match self {
            Self::Cohort => estimate.group,
            Self::Calendar => estimate.time,
            Self::Overall => 0,
        }
    }

    const fn aggte_type(self) -> AggteType {
        match self {
            Self::Cohort => AggteType::Group,
            Self::Calendar => AggteType::Calendar,
            Self::Overall => AggteType::Simple,
        }
    }
}

/// The post-treatment cells grouped by key, each key's estimate, standard
/// error and interval from `aggregate_att_gt` under treated-count weighting
/// and from fixed weights otherwise.
fn aggregate_post_treatment(
    estimates: &[AttGtEstimate],
    influence: &[Vec<f64>],
    panel: &UnitPanel,
    config: AttGtAggregationConfig,
    summary: Summary,
) -> Result<BTreeMap<i32, AttGtAggregatedEstimate>, AggteError> {
    let confidence_level = config.confidence_level.confidence_level;
    if !validate_confidence_level(confidence_level) {
        return Err(AggteError::InvalidConfidenceLevel);
    }
    let clusters = checked_cluster_index(estimates, influence, panel)?;
    let mut buckets = BTreeMap::<i32, Vec<usize>>::new();
    for (index, estimate) in estimates.iter().enumerate() {
        if estimate.time >= estimate.group {
            buckets
                .entry(summary.key(estimate))
                .or_default()
                .push(index);
        }
    }
    if buckets.is_empty() {
        return Err(AggteError::NoPostTreatmentCells);
    }

    let points: BTreeMap<i32, (f64, f64, f64, f64)> = match config.weighting {
        AttGtAggregationWeighting::ByTreatedCount => {
            let result = aggregate_att_gt(
                estimates,
                influence,
                panel,
                AggteConfig {
                    aggregation: summary.aggte_type(),
                    confidence_level: config.confidence_level,
                    ..AggteConfig::default()
                },
            )?;
            match summary {
                Summary::Overall => BTreeMap::from([(
                    0,
                    (
                        result.overall_att,
                        result.overall_se,
                        result.overall_ci_low,
                        result.overall_ci_high,
                    ),
                )]),
                Summary::Cohort | Summary::Calendar => result
                    .by_key
                    .into_iter()
                    .map(|point| {
                        (
                            point.key,
                            (point.att, point.se, point.ci_low, point.ci_high),
                        )
                    })
                    .collect(),
            }
        }
        AttGtAggregationWeighting::Equal | AttGtAggregationWeighting::ByTotalWeight => buckets
            .iter()
            .map(|(&key, cells)| {
                let raw = cells
                    .iter()
                    .map(|&c| raw_weight(&estimates[c], config.weighting))
                    .collect::<Vec<_>>();
                let raw_total = raw.iter().sum::<f64>();
                let weights = raw.iter().map(|w| w / raw_total).collect::<Vec<_>>();
                let (att, combined) =
                    combine_fixed(cells, &weights, estimates, influence, panel.groups.len());
                let se = standard_error(&combined, &clusters);
                let (ci_low, ci_high) = normal_interval(att, se, confidence_level);
                (key, (att, se, ci_low, ci_high))
            })
            .collect(),
    };

    buckets
        .into_iter()
        .map(|(key, cells)| {
            let &(estimate, se, ci_low, ci_high) =
                points.get(&key).ok_or(AggteError::NoPostTreatmentCells)?;
            Ok((
                key,
                AttGtAggregatedEstimate {
                    estimate,
                    se,
                    ci_low,
                    ci_high,
                    components: cells.len(),
                    total_weight: cells.iter().map(|&c| estimates[c].total_weight).sum(),
                },
            ))
        })
        .collect()
}

fn validate_aggregation_inputs(
    estimates: &[AttGtEstimate],
    config: AttGtAggregationConfig,
) -> Result<(), AttGtAggregationError> {
    if estimates.is_empty() {
        return Err(AttGtAggregationError::EmptyInput);
    }
    if !validate_confidence_level(config.confidence_level.confidence_level) {
        return Err(AttGtAggregationError::InvalidConfidenceLevel);
    }
    for estimate in estimates {
        if !estimate.att.is_finite() {
            return Err(AttGtAggregationError::InvalidEstimate);
        }
        if !estimate.se.is_finite() || estimate.se < 0.0 {
            return Err(AttGtAggregationError::InvalidSe);
        }
    }
    Ok(())
}

fn raw_weight(estimate: &AttGtEstimate, weighting: AttGtAggregationWeighting) -> f64 {
    match weighting {
        AttGtAggregationWeighting::Equal => 1.0,
        AttGtAggregationWeighting::ByTreatedCount => usize_to_f64(estimate.treated_n),
        AttGtAggregationWeighting::ByTotalWeight => estimate.total_weight,
    }
}

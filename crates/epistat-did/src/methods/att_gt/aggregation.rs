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

/// Aggregate ATT(g,t) estimates by event-time.
///
/// Every cell enters, pre-treatment ones included, and the standard error
/// combines the components' standard errors as if they were independent, which
/// understates it when cells share units.
/// [`aggregate_att_gt_event_time_with_influence`] forms it from the combined
/// influence function instead.
///
/// # Errors
///
/// Returns [`AttGtAggregationError`] when estimates/configuration are invalid.
pub fn aggregate_att_gt_event_time(
    estimates: &[AttGtEstimate],
    config: AttGtAggregationConfig,
) -> Result<Vec<AttGtEventTimeEstimate>, AttGtAggregationError> {
    aggregate_by_key(estimates, config, |est| est.event_time).map(|grouped| {
        grouped
            .into_iter()
            .map(|(event_time, summary)| AttGtEventTimeEstimate {
                event_time,
                summary,
            })
            .collect()
    })
}

/// Aggregate ATT(g,t) estimates by event-time while preserving aligned
/// influence-function vectors for joint-process inference.
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

    let mut grouped = BTreeMap::<i32, Vec<usize>>::new();
    for (idx, estimate) in estimates.iter().enumerate() {
        grouped.entry(estimate.event_time).or_default().push(idx);
    }

    let mut aggregated_estimates = Vec::with_capacity(grouped.len());
    let mut aggregated_influence = Vec::with_capacity(grouped.len());
    for (event_time, bucket) in grouped {
        let component_count = bucket.len();
        let raw_total = bucket
            .iter()
            .map(|idx| raw_weight(&estimates[*idx], config.weighting))
            .sum::<f64>();
        let mut estimate = 0.0;
        let mut total_influence = vec![0.0; n];
        let mut total_weight = 0.0;
        for idx in bucket {
            let component = &estimates[idx];
            let normalized_weight = raw_weight(component, config.weighting) / raw_total;
            estimate = normalized_weight.mul_add(component.att, estimate);
            total_weight += component.total_weight;
            for (dst, src) in total_influence.iter_mut().zip(&influence_functions[idx]) {
                *dst = normalized_weight.mul_add(*src, *dst);
            }
        }
        // From the COMBINED influence function, not from `sum(w^2 * se^2)`.
        // The latter is the variance of a weighted sum of INDEPENDENT terms, and
        // ATT(g,t) cells share units, so its off-diagonal terms are not zero. It
        // understated the standard error whenever the components were positively
        // correlated, which is the ordinary case. The influence vectors were
        // being carried alongside for exactly this and were not being used.
        let se = crate::inference::standard_error_from_influence(&total_influence);
        let (ci_low, ci_high) =
            normal_interval(estimate, se, config.confidence_level.confidence_level);
        aggregated_estimates.push(AttGtEventTimeEstimate {
            event_time,
            summary: AttGtAggregatedEstimate {
                estimate,
                se,
                ci_low,
                ci_high,
                components: component_count,
                total_weight,
            },
        });
        aggregated_influence.push(total_influence);
    }

    Ok(AttGtEventTimeInfluenceOutput {
        estimates: aggregated_estimates,
        influence_functions: aggregated_influence,
    })
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

fn aggregate_by_key<F>(
    estimates: &[AttGtEstimate],
    config: AttGtAggregationConfig,
    key_fn: F,
) -> Result<BTreeMap<i32, AttGtAggregatedEstimate>, AttGtAggregationError>
where
    F: Fn(&AttGtEstimate) -> i32,
{
    validate_aggregation_inputs(estimates, config)?;

    let mut grouped = BTreeMap::<i32, Vec<&AttGtEstimate>>::new();
    for estimate in estimates {
        grouped.entry(key_fn(estimate)).or_default().push(estimate);
    }

    Ok(grouped
        .into_iter()
        .map(|(key, bucket)| {
            let raw_total = bucket
                .iter()
                .map(|estimate| raw_weight(estimate, config.weighting))
                .sum::<f64>();
            let estimate = bucket
                .iter()
                .map(|entry| {
                    let w = raw_weight(entry, config.weighting) / raw_total;
                    w * entry.att
                })
                .sum::<f64>();
            let variance = bucket
                .iter()
                .map(|entry| {
                    let w = raw_weight(entry, config.weighting) / raw_total;
                    w * w * entry.se * entry.se
                })
                .sum::<f64>();
            let se = variance.sqrt();
            let (ci_low, ci_high) =
                normal_interval(estimate, se, config.confidence_level.confidence_level);
            (
                key,
                AttGtAggregatedEstimate {
                    estimate,
                    se,
                    ci_low,
                    ci_high,
                    components: bucket.len(),
                    total_weight: bucket.iter().map(|entry| entry.total_weight).sum::<f64>(),
                },
            )
        })
        .collect())
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

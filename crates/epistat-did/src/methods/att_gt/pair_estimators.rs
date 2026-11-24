//! Repeated cross-section `ATT(g,t)` with covariate adjustment,
//! `did::att_gt(panel = FALSE)`.
//!
//! Each cell takes the rows of its two periods, the cohort's as treated and
//! the comparison group's as controls, and fits a two-period repeated
//! cross-section estimator. The input is first pre-processed as `did` does it
//! (see [`crate::AttGtSample`]). Influence vectors have one entry per kept row,
//! indexed by [`crate::AttGtSample::ids`], and a cell fitted on `n_cell` of the
//! `n` rows has its vector scaled by `n / n_cell`, so that
//! `sqrt(sum(psi^2)) / n` is its standard error.

use std::collections::BTreeSet;

use crate::methods::drdid::ipw_or::{
    estimate_ipw_did_repeated_cross_section, estimate_or_did_repeated_cross_section,
};
use crate::methods::drdid::repeated::estimate_drdid_repeated_cross_section;
use crate::methods::drdid::repeated_efficient::estimate_drdid_repeated_efficient;
use crate::types::{
    AttGtConfig, AttGtDrConfig, AttGtDrObservation, AttGtError, AttGtEstimate,
    AttGtInfluenceOutput, DidCell, DrDidEstimate, DrDidRepeatedObservation, PrunedCell,
    SkippedCell, TimePeriod, TreatmentGroup,
};
use crate::util::usize_to_f64;

use super::preprocess::{Prepared, Route, cell_grid, preprocess};

/// A two-period repeated cross-section estimator.
type PairEstimator = fn(
    &[DrDidRepeatedObservation],
    crate::types::DrDidConfig,
) -> Result<DrDidEstimate, crate::types::DrDidError>;

/// `ATT(g,t)` through the traditional doubly robust repeated cross-section
/// estimator, `DRDID::drdid_rc1`.
pub fn estimate_att_gt_dr_with_influence(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<AttGtInfluenceOutput, AttGtError> {
    estimate_att_gt_pair_with_influence(
        observations,
        config,
        "dr",
        estimate_drdid_repeated_cross_section,
    )
}

pub fn estimate_att_gt_dr(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<Vec<AttGtEstimate>, AttGtError> {
    estimate_att_gt_dr_with_influence(observations, config).map(|out| out.estimates)
}

/// `ATT(g,t)` through the LOCALLY EFFICIENT repeated-cross-section estimator.
///
/// [`estimate_att_gt_dr_with_influence`](crate::estimate_att_gt_dr_with_influence) uses the traditional one,
/// `DRDID::drdid_rc1`. `did::att_gt(est_method = "dr", panel = FALSE)` uses this
/// one, `DRDID::drdid_rc`, so this is the route to take when the numbers have to
/// match `did` on repeated cross sections.
///
/// The two agree on the point estimate and disagree on the standard error, by
/// about 2.8% on `mpdta`.
///
/// # Errors
/// As [`estimate_att_gt_dr_with_influence`](crate::estimate_att_gt_dr_with_influence).
pub fn estimate_att_gt_dr_efficient_with_influence(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<AttGtInfluenceOutput, AttGtError> {
    estimate_att_gt_pair_with_influence(
        observations,
        config,
        "dr_efficient",
        estimate_drdid_repeated_efficient,
    )
}

/// `ATT(g,t)` by outcome regression, `DRDID::reg_did_rc` in every cell.
pub fn estimate_att_gt_or_with_influence(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<AttGtInfluenceOutput, AttGtError> {
    estimate_att_gt_pair_with_influence(
        observations,
        config,
        "or",
        estimate_or_did_repeated_cross_section,
    )
}

/// `ATT(g,t)` by inverse probability weighting, `DRDID::std_ipw_did_rc` in
/// every cell.
pub fn estimate_att_gt_ipw_with_influence(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<AttGtInfluenceOutput, AttGtError> {
    estimate_att_gt_pair_with_influence(
        observations,
        config,
        "ipw",
        estimate_ipw_did_repeated_cross_section,
    )
}

/// The design (intercept first), its column count and the arm of a cell's
/// rows, for the overlap check.
pub(super) fn overlap_design(rows: &[DrDidRepeatedObservation]) -> (Vec<f64>, usize, Vec<f64>) {
    let feature_count = rows.first().map_or(1, |row| row.covariates.len() + 1);
    let mut design = Vec::with_capacity(rows.len() * feature_count);
    for row in rows {
        design.push(1.0);
        design.extend_from_slice(&row.covariates);
    }
    let treated = rows
        .iter()
        .map(|row| f64::from(u8::from(row.treated)))
        .collect();
    (design, feature_count, treated)
}

/// The rows of cell `(group, time)` against `baseline_time`, with their
/// influence positions.
fn build_prepared_pair_rows(
    observations: &[AttGtDrObservation],
    prepared: &Prepared,
    group: i32,
    time: i32,
    baseline_time: i32,
    config: AttGtConfig,
    rows: &mut Vec<DrDidRepeatedObservation>,
    positions: &mut Vec<usize>,
) {
    rows.clear();
    positions.clear();
    for (k, &i) in prepared.rows.iter().enumerate() {
        let row = &observations[i];
        if row.time != baseline_time && row.time != time {
            continue;
        }
        let cohort = prepared.cohort[k];
        let arm = if cohort == Some(group) {
            TreatmentGroup::Treated
        } else if super::is_control_for_pair(
            cohort,
            row.comparison_cohort,
            group,
            // The later of the two periods, as in `did`; see
            // `panel_pairs::cell_role`.
            time.max(baseline_time),
            config.comparison_group,
            config.anticipation_periods,
        ) {
            TreatmentGroup::Control
        } else {
            continue;
        };
        rows.push(DrDidRepeatedObservation {
            weight: row.weight,
            covariates: row.covariates.clone(),
            ..DrDidRepeatedObservation::new(
                DidCell::from_parts(arm, TimePeriod::from_bool(row.time == time)),
                row.outcome,
            )
        });
        positions.push(prepared.position[k]);
    }
}

fn estimate_att_gt_pair_with_influence(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
    method: &'static str,
    pair_estimator: PairEstimator,
) -> Result<AttGtInfluenceOutput, AttGtError> {
    // `did` checks overlap for its doubly robust and weighting estimators.
    let uses_propensity = method != "or";
    let prepared = preprocess(observations, config.att_gt, Route::RepeatedCrossSection)?;
    let full_n = prepared.position_count();
    let mut estimates = Vec::new();
    let mut influence_functions = Vec::new();
    let mut skipped = Vec::new();
    let mut pruned = Vec::new();
    let mut overlap = Vec::new();
    let mut pair_rows = Vec::new();
    let mut pair_positions = Vec::new();

    for &group in &prepared.sample.cohorts {
        for (time, baseline_time) in cell_grid(&prepared.sample.periods, group, config.att_gt) {
            build_prepared_pair_rows(
                observations,
                &prepared,
                group,
                time,
                baseline_time,
                config.att_gt,
                &mut pair_rows,
                &mut pair_positions,
            );
            if let Some(cell) = missing_pair_cell(&pair_rows) {
                if config.att_gt.skip_incomplete_pairs {
                    skipped.push(SkippedCell {
                        group,
                        time,
                        baseline_time,
                        reason: cell,
                    });
                    continue;
                }
                return Err(AttGtError::MissingCell {
                    group,
                    time,
                    baseline_time,
                    cell,
                });
            }

            let (design, feature_count, treated) = overlap_design(&pair_rows);
            if let Err(cell) = super::panel_pairs::check_overlap(
                &design,
                feature_count,
                &treated,
                (group, time, baseline_time),
                config,
                uses_propensity,
                &mut overlap,
            ) {
                if config.att_gt.skip_incomplete_pairs {
                    skipped.push(SkippedCell {
                        group,
                        time,
                        baseline_time,
                        reason: cell,
                    });
                    continue;
                }
                return Err(AttGtError::MissingCell {
                    group,
                    time,
                    baseline_time,
                    cell,
                });
            }

            let fit = pair_estimator(&pair_rows, config.drdid).map_err(|_| {
                AttGtError::PairEstimationFailure {
                    method,
                    group,
                    time,
                }
            })?;
            if fit.influence_function.len() != pair_rows.len() {
                return Err(AttGtError::InfluenceLengthMismatch {
                    method,
                    group,
                    time,
                    expected: pair_rows.len(),
                    actual: fit.influence_function.len(),
                });
            }

            let scale = usize_to_f64(full_n) / usize_to_f64(pair_rows.len());
            let mut aligned = vec![0.0; full_n];
            for (local, &position) in pair_positions.iter().enumerate() {
                aligned[position] = fit.influence_function[local] * scale;
            }
            if fit.design_columns_dropped > 0 {
                pruned.push(PrunedCell {
                    group,
                    time,
                    baseline_time,
                    columns_dropped: fit.design_columns_dropped,
                });
            }
            estimates.push(AttGtEstimate {
                group,
                time,
                event_time: time - group,
                att: fit.att,
                se: fit.se,
                ci_low: fit.ci_low,
                ci_high: fit.ci_high,
                treated_n: fit.treated_n,
                control_n: fit.control_n,
                total_weight: fit.total_weight,
            });
            influence_functions.push(aligned);
        }
    }

    if estimates.is_empty() {
        return Err(AttGtError::NoEstimablePairs);
    }
    Ok(AttGtInfluenceOutput {
        estimates,
        influence_functions,
        skipped,
        pruned,
        sample: prepared.sample,
        overlap,
    })
}

pub fn prepare_att_gt_dr_inputs(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<(Vec<i32>, Vec<i32>), AttGtError> {
    super::validate_config(config.att_gt)?;
    if observations.is_empty() {
        return Err(AttGtError::EmptyInput);
    }

    let covariate_count = observations.first().map_or(0, |row| row.covariates.len());
    let mut times = BTreeSet::new();
    let mut treated_groups = BTreeSet::new();
    let mut has_never_treated = false;

    for row in observations {
        if row.covariates.len() != covariate_count {
            return Err(AttGtError::InconsistentCovariateCount {
                expected: covariate_count,
                actual: row.covariates.len(),
            });
        }
        if !row.weight.is_finite() || row.weight <= 0.0 {
            return Err(AttGtError::InvalidWeight { value: row.weight });
        }
        if !row.outcome.is_finite() {
            return Err(AttGtError::InvalidOutcome { value: row.outcome });
        }
        for covariate in &row.covariates {
            if !covariate.is_finite() {
                return Err(AttGtError::InvalidCovariate { value: *covariate });
            }
        }
        times.insert(row.time);
        if let Some(group) = row.first_treated_time {
            treated_groups.insert(group);
        } else {
            has_never_treated = true;
        }
    }

    if config.att_gt.comparison_group == crate::types::ComparisonGroup::NeverTreated
        && !has_never_treated
    {
        return Err(AttGtError::MissingNeverTreatedGroup);
    }

    Ok((
        times.into_iter().collect::<Vec<_>>(),
        treated_groups.into_iter().collect::<Vec<_>>(),
    ))
}

pub fn missing_pair_cell(rows: &[DrDidRepeatedObservation]) -> Option<&'static str> {
    let (treated_pre, treated_post, control_pre, control_post) = cell_counts(rows);
    [
        (treated_pre == 0, "treated_pre"),
        (treated_post == 0, "treated_post"),
        (control_pre == 0, "control_pre"),
        (control_post == 0, "control_post"),
    ]
    .into_iter()
    .find_map(|(is_missing, label)| if is_missing { Some(label) } else { None })
}

pub fn build_pair_rows(
    observations: &[AttGtDrObservation],
    group: i32,
    time: i32,
    baseline_time: i32,
    config: AttGtConfig,
) -> (Vec<DrDidRepeatedObservation>, Vec<usize>) {
    let mut rows = Vec::with_capacity(observations.len().min(4096));
    let mut indices = Vec::with_capacity(observations.len().min(4096));
    build_pair_rows_into(
        observations,
        group,
        time,
        baseline_time,
        config,
        &mut rows,
        &mut indices,
    );
    (rows, indices)
}

pub fn build_pair_rows_into(
    observations: &[AttGtDrObservation],
    group: i32,
    time: i32,
    baseline_time: i32,
    config: AttGtConfig,
    rows: &mut Vec<DrDidRepeatedObservation>,
    indices: &mut Vec<usize>,
) {
    rows.clear();
    indices.clear();
    for (idx, row) in observations.iter().enumerate() {
        if row.time != baseline_time && row.time != time {
            continue;
        }
        let entry = if row.first_treated_time == Some(group) {
            Some(DrDidRepeatedObservation {
                weight: row.weight,
                covariates: row.covariates.clone(),
                ..DrDidRepeatedObservation::new(
                    DidCell::from_parts(
                        TreatmentGroup::Treated,
                        TimePeriod::from_bool(row.time == time),
                    ),
                    row.outcome,
                )
            })
        } else if super::is_control_for_pair(
            row.first_treated_time,
            row.comparison_cohort,
            group,
            // The LATER of the two periods being compared, not `time`. Under a
            // universal base period a pre-treatment cell has baseline > time, and
            // a unit treated in between is already treated when the baseline is
            // read. See the note in `panel_pairs::collect_cell_units` for the
            // measured difference against did 2.5.1.
            time.max(baseline_time),
            config.comparison_group,
            config.anticipation_periods,
        ) {
            Some(DrDidRepeatedObservation {
                weight: row.weight,
                covariates: row.covariates.clone(),
                ..DrDidRepeatedObservation::new(
                    DidCell::from_parts(
                        TreatmentGroup::Control,
                        TimePeriod::from_bool(row.time == time),
                    ),
                    row.outcome,
                )
            })
        } else {
            None
        };

        if let Some(entry) = entry {
            rows.push(entry);
            indices.push(idx);
        }
    }
}

fn cell_counts(rows: &[DrDidRepeatedObservation]) -> (usize, usize, usize, usize) {
    let mut treated_pre = 0_usize;
    let mut treated_post = 0_usize;
    let mut control_pre = 0_usize;
    let mut control_post = 0_usize;

    for row in rows {
        match (row.treated, row.post_period) {
            (true, false) => treated_pre += 1,
            (true, true) => treated_post += 1,
            (false, false) => control_pre += 1,
            (false, true) => control_post += 1,
        }
    }

    (treated_pre, treated_post, control_pre, control_post)
}

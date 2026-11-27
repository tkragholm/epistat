//! Repeated cross-section `ATT(g,t)` with covariate adjustment,
//! `did::att_gt(panel = FALSE)`.
//!
//! Each cell takes the rows of its two periods, the cohort's as treated and
//! the comparison group's as controls, and fits a two-period repeated
//! cross-section estimator. The input is first pre-processed as `did` does it
//! (see [`AttGtSample`](super::AttGtSample)). Influence vectors have one entry
//! per kept row, indexed by [`AttGtSample::units`](super::AttGtSample::units),
//! and a cell fitted on `n_cell` of the `n` rows has its vector scaled by
//! `n / n_cell`, so that `sqrt(sum(psi^2)) / n` is its standard error.

use crate::error::Error;
use crate::methods::drdid::ipw_or::{
    estimate_ipw_did_repeated_cross_section, estimate_or_did_repeated_cross_section,
};
use crate::methods::drdid::repeated::estimate_drdid_repeated_cross_section;
use crate::methods::drdid::repeated_efficient::estimate_drdid_repeated_efficient;
use crate::types::{
    DidCell, DrDidConfig, DrDidError, DrDidEstimate, DrDidRepeatedObservation, TimePeriod,
    TreatmentGroup,
};
use crate::util::usize_to_f64;

use super::preprocess::{Prepared, Route, cell_grid, preprocess};
use super::rows::Rows;
use super::{CellFit, ComparisonGroup, Config, FitFailure, Output, Overlap, SkipReason};

/// A two-period repeated cross-section estimator.
type PairEstimator =
    fn(&[DrDidRepeatedObservation], DrDidConfig) -> Result<DrDidEstimate, DrDidError>;

/// The estimator of each cell on a repeated cross-section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Estimator {
    /// The locally efficient doubly robust estimator, `DRDID::drdid_rc`, as
    /// `did::att_gt(est_method = "dr", panel = FALSE)`.
    DoublyRobust,
    /// The traditional doubly robust estimator, `DRDID::drdid_rc1`.
    TraditionalDoublyRobust,
    /// `DRDID::std_ipw_did_rc`.
    InverseProbabilityWeighting,
    /// `DRDID::reg_did_rc`.
    OutcomeRegression,
}

impl Estimator {
    fn pair_estimator(self) -> PairEstimator {
        match self {
            Self::DoublyRobust => estimate_drdid_repeated_efficient,
            Self::TraditionalDoublyRobust => estimate_drdid_repeated_cross_section,
            Self::InverseProbabilityWeighting => estimate_ipw_did_repeated_cross_section,
            Self::OutcomeRegression => estimate_or_did_repeated_cross_section,
        }
    }
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

/// The repeated cross-section row of row `i` in a cell, as treated or
/// comparison, at the cell's later period or not.
fn pair_row(rows: &Rows<'_>, i: usize, treated: bool, time: i64) -> DrDidRepeatedObservation {
    DrDidRepeatedObservation {
        weight: rows.weight(i),
        covariates: rows.covariates_of(i),
        ..DrDidRepeatedObservation::new(
            DidCell::from_parts(
                if treated {
                    TreatmentGroup::Treated
                } else {
                    TreatmentGroup::Control
                },
                TimePeriod::from_bool(rows.period[i] == time),
            ),
            rows.outcome[i],
        )
    }
}

/// Whether row `i` of cohort `cohort` is in cell `(group, time)` against
/// `baseline_time`, and as treated (`Some(true)`) or comparison.
///
/// Not-yet-treated eligibility is judged at the LATER of the two periods, not
/// at `time`. Under a universal base period a pre-treatment cell has
/// `baseline > time`, and a unit treated in between is already treated when
/// the baseline is read, as in `did` 2.5.1.
fn pair_role(
    rows: &Rows<'_>,
    i: usize,
    cohort: Option<i64>,
    (group, time, baseline_time): (i64, i64, i64),
    comparison_group: ComparisonGroup,
    anticipation: i64,
) -> Option<bool> {
    let period = rows.period[i];
    if period != baseline_time && period != time {
        return None;
    }
    if cohort == Some(group) {
        return Some(true);
    }
    super::is_control_for_pair(
        cohort,
        rows.comparison_cohort(i),
        group,
        time.max(baseline_time),
        comparison_group,
        anticipation,
    )
    .then_some(false)
}

/// The rows of cell `(group, time)` against `baseline_time`, with their
/// influence positions.
fn build_prepared_pair_rows(
    rows: &Rows<'_>,
    prepared: &Prepared,
    cell: (i64, i64, i64),
    config: Config,
    out: &mut Vec<DrDidRepeatedObservation>,
    positions: &mut Vec<usize>,
) {
    out.clear();
    positions.clear();
    for (k, &i) in prepared.rows.iter().enumerate() {
        let Some(treated) = pair_role(
            rows,
            i,
            prepared.cohort[k],
            cell,
            config.comparison_group,
            config.anticipation,
        ) else {
            continue;
        };
        out.push(pair_row(rows, i, treated, cell.1));
        positions.push(prepared.position[k]);
    }
}

/// The rows of cell `(group, time)` against `baseline_time` among all rows,
/// without pre-processing, with their row numbers.
pub(super) fn build_pair_rows(
    rows: &Rows<'_>,
    cell: (i64, i64, i64),
    config: Config,
) -> (Vec<DrDidRepeatedObservation>, Vec<usize>) {
    let mut out = Vec::with_capacity(rows.len().min(4096));
    let mut indices = Vec::with_capacity(rows.len().min(4096));
    for i in 0..rows.len() {
        if let Some(treated) = pair_role(
            rows,
            i,
            rows.cohort[i],
            cell,
            config.comparison_group,
            config.anticipation,
        ) {
            out.push(pair_row(rows, i, treated, cell.1));
            indices.push(i);
        }
    }
    (out, indices)
}

/// The cell fit of one repeated cross-section cell and its row-aligned
/// influence vector over `full_n` positions.
fn estimate_rc_cell(
    pair_rows: &[DrDidRepeatedObservation],
    pair_positions: &[usize],
    full_n: usize,
    config: Config,
    estimator: Estimator,
    overlap: &mut Option<Overlap>,
) -> Result<(CellFit, Vec<f64>), SkipReason> {
    if let Some(reason) = missing_pair_cell(pair_rows) {
        return Err(reason);
    }
    let (design, feature_count, treated) = overlap_design(pair_rows);
    // `did` checks overlap for its doubly robust and weighting estimators.
    let uses_propensity = estimator != Estimator::OutcomeRegression;
    super::panel_pairs::check_overlap(
        &design,
        feature_count,
        &treated,
        config,
        uses_propensity,
        overlap,
    )?;
    let fit = (estimator.pair_estimator())(pair_rows, config.drdid)
        .map_err(|error| SkipReason::FitFailed(super::panel_pairs::fit_failure(&error)))?;
    if fit.influence_function.len() != pair_rows.len() {
        return Err(SkipReason::FitFailed(FitFailure::Other));
    }
    let scale = usize_to_f64(full_n) / usize_to_f64(pair_rows.len());
    let mut aligned = vec![0.0; full_n];
    for (local, &position) in pair_positions.iter().enumerate() {
        aligned[position] = fit.influence_function[local] * scale;
    }
    if !fit.att.is_finite() || aligned.iter().any(|value| !value.is_finite()) {
        return Err(SkipReason::NotFinite);
    }
    Ok((
        CellFit {
            att: fit.att,
            se: fit.se,
            ci_low: fit.ci_low,
            ci_high: fit.ci_high,
            treated_n: fit.treated_n,
            control_n: fit.control_n,
            total_weight: fit.total_weight,
            columns_dropped: fit.design_columns_dropped,
        },
        aligned,
    ))
}

/// Repeated cross-section `ATT(g,t)`, `did::att_gt(..., panel = FALSE)`,
/// with the influence vectors indexed by the kept rows.
///
/// # Errors
/// The errors of [`preprocess`], [`Error::CellFailed`] for a cell that cannot
/// be fitted when cells are not skipped, and [`Error::NoEstimableCells`] if
/// no cell is fitted.
pub(super) fn estimate(
    rows: &Rows<'_>,
    config: Config,
    estimator: Estimator,
) -> Result<Output, Error> {
    let prepared = preprocess(rows, config, Route::RepeatedCrossSection)?;
    let full_n = prepared.position_count();
    let mut attempts = Vec::new();
    let mut influence = Vec::new();
    let mut pair_rows = Vec::new();
    let mut pair_positions = Vec::new();

    for &group in &prepared.sample.cohorts {
        for (time, baseline_time) in cell_grid(&prepared.sample.periods, group, config) {
            let cell = (group, time, baseline_time);
            build_prepared_pair_rows(
                rows,
                &prepared,
                cell,
                config,
                &mut pair_rows,
                &mut pair_positions,
            );
            let mut overlap = None;
            let fitted = estimate_rc_cell(
                &pair_rows,
                &pair_positions,
                full_n,
                config,
                estimator,
                &mut overlap,
            );
            attempts.push(super::record(
                cell,
                fitted,
                overlap,
                &mut influence,
                config,
            )?);
        }
    }
    super::finish(attempts, influence, prepared.sample)
}

/// The first of the four arms of a repeated cross-section cell that has no
/// row.
pub(super) fn missing_pair_cell(rows: &[DrDidRepeatedObservation]) -> Option<SkipReason> {
    let (treated_pre, treated_post, control_pre, control_post) = cell_counts(rows);
    [
        (treated_pre == 0, SkipReason::NoTreatedBase),
        (treated_post == 0, SkipReason::NoTreatedLater),
        (control_pre == 0, SkipReason::NoComparisonBase),
        (control_post == 0, SkipReason::NoComparisonLater),
    ]
    .into_iter()
    .find_map(|(is_missing, reason)| is_missing.then_some(reason))
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

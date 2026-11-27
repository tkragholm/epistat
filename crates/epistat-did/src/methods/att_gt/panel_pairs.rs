//! Panel `ATT(g,t)` with covariate adjustment, `did::att_gt(panel = TRUE)`.
//!
//! [`super::pair_estimators`] treats the two periods of a cell as independent
//! samples, which is right for repeated cross-sections. On a panel they are
//! the same units observed twice, and each cell here is a two-period panel
//! estimator on the outcome changes of the units observed in both periods:
//! `DRDID::drdid_panel`, `std_ipw_did_panel` or `reg_did_panel`, as `did`
//! calls them for `est_method = "dr"`, `"ipw"` and `"reg"`.
//!
//! # Influence vectors
//!
//! One unit is one observation, so every cell's influence vector has one
//! entry per unit, indexed by position in the sorted codes of the units the
//! pre-processing kept ([`AttGtSample::units`](super::AttGtSample::units));
//! units absent from a cell carry zero. A cell fitted on `n_cell` of the `n` units has its vector
//! scaled by `n / n_cell`, so that, as in `did`,
//!
//! ```text
//! se = sqrt(sum(psi^2)) / n
//! ```
//!
//! for every cell, and the vectors of different cells are on one scale for the
//! aggregations, the bands and the `HonestDiD` covariance. A row-indexed
//! vector would be wrong by a factor of the number of periods.
//!
//! An unbalanced panel under
//! [`UnbalancedPanel::RepeatedCrossSection`](super::UnbalancedPanel) is fitted
//! by the repeated cross-section estimators on each cell's rows, and each
//! unit's rows are summed in its entry.

use crate::error::Error;
use crate::inference::standard_error_from_influence;
use crate::methods::drdid::ipw_or::{
    estimate_ipw_did_panel_flat, estimate_ipw_did_repeated_cross_section,
    estimate_or_did_panel_flat, estimate_or_did_repeated_cross_section,
};
use crate::methods::drdid::panel::{PanelFlatInput, estimate_drdid_panel_flat};
use crate::methods::drdid::repeated_efficient::estimate_drdid_repeated_efficient;
use crate::types::{
    DidCell, DrDidError, DrDidEstimate, DrDidRepeatedObservation, TimePeriod, TreatmentGroup,
};

use super::preprocess::{Prepared, Route, cell_grid, preprocess};
use super::rows::Rows;
use super::{CellFit, Config, FitFailure, Output, Overlap, SkipReason};

/// A fitted cell and its unit-aligned influence vector, or the reason it
/// could not be fitted.
type Fitted = Result<(CellFit, Vec<f64>), SkipReason>;

/// The estimator of each cell, as `did::att_gt`'s `est_method`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Estimator {
    /// `"dr"`: `DRDID::drdid_panel`, or `drdid_rc` on an unbalanced panel.
    DoublyRobust,
    /// `"ipw"`: `DRDID::std_ipw_did_panel`, or `std_ipw_did_rc`.
    InverseProbabilityWeighting,
    /// `"reg"`: `DRDID::reg_did_panel`, or `reg_did_rc`.
    OutcomeRegression,
}

impl Estimator {
    /// Whether the estimator fits a propensity score, which is when `did`
    /// checks overlap.
    pub(super) fn uses_propensity(self) -> bool {
        self != Self::OutcomeRegression
    }
}

/// One unit's contribution to a single `(g, t)` cell.
struct PanelUnit {
    /// Index into the sorted distinct-unit list, for influence alignment.
    unit_index: usize,
    treated: bool,
    baseline_outcome: Option<f64>,
    follow_up_outcome: Option<f64>,
    weight: f64,
    /// The row of the earlier of the cell's two periods, whose covariates
    /// the cell takes, as `did` does. That is the base period except in a
    /// pre-treatment cell under a universal base, whose base is later.
    covariate_row: usize,
}

/// Buffers the cell loop reuses, so the per-cell cost is the cell's own size.
///
/// `slots` is indexed by the unit position the influence vectors are aligned
/// on. `touched` lists the slots a cell wrote, so that resetting costs the
/// cell and not the whole panel.
struct CellScratch {
    slots: Vec<Option<PanelUnit>>,
    touched: Vec<usize>,
    units: Vec<PanelUnit>,
    treated: Vec<bool>,
    delta: Vec<f64>,
    weights: Vec<f64>,
    design: Vec<f64>,
}

impl CellScratch {
    fn new(unit_count: usize) -> Self {
        let mut slots = Vec::new();
        slots.resize_with(unit_count, || None);
        Self {
            slots,
            touched: Vec::new(),
            units: Vec::new(),
            treated: Vec::new(),
            delta: Vec::new(),
            weights: Vec::new(),
            design: Vec::new(),
        }
    }

    fn reset(&mut self) {
        for &index in &self.touched {
            self.slots[index] = None;
        }
        self.touched.clear();
        self.units.clear();
        self.treated.clear();
        self.delta.clear();
        self.weights.clear();
        self.design.clear();
    }
}

/// Whether row `i`, of cohort `cohort` after the pre-processing, is in the
/// cell `(group, time)` against `baseline_time`, and as treated (`Some(true)`)
/// or comparison (`Some(false)`).
///
/// Not-yet-treated eligibility is judged at the later of the two periods, as
/// in `did`: under a universal base a pre-treatment cell has its base after
/// `time`, and a unit treated between the two is treated at the base.
fn cell_role(
    rows: &Rows<'_>,
    i: usize,
    cohort: Option<i64>,
    (group, time, baseline_time): (i64, i64, i64),
    config: Config,
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
        config.comparison_group,
        config.anticipation,
    )
    .then_some(false)
}

/// Collect the units contributing to one `(group, time)` cell against
/// `baseline_time`: those observed at both periods. The key of a validated
/// panel has one row per unit and period, so each slot is written at most
/// once per period.
fn collect_cell_units(
    rows: &Rows<'_>,
    prepared: &Prepared,
    cell: (i64, i64, i64),
    config: Config,
    scratch: &mut CellScratch,
) {
    scratch.reset();
    let (_, time, baseline_time) = cell;
    let earlier = time.min(baseline_time);

    for (k, &i) in prepared.rows.iter().enumerate() {
        let Some(treated) = cell_role(rows, i, prepared.cohort[k], cell, config) else {
            continue;
        };
        let unit_index = prepared.position[k];
        let slot = &mut scratch.slots[unit_index];
        if slot.is_none() {
            scratch.touched.push(unit_index);
            *slot = Some(PanelUnit {
                unit_index,
                treated,
                baseline_outcome: None,
                follow_up_outcome: None,
                weight: rows.weight(i),
                covariate_row: i,
            });
        }
        let Some(entry) = slot.as_mut() else { continue };
        let period = rows.period[i];
        if period == baseline_time {
            entry.baseline_outcome = Some(rows.outcome[i]);
        } else {
            entry.follow_up_outcome = Some(rows.outcome[i]);
        }
        if period == earlier {
            entry.weight = rows.weight(i);
            entry.covariate_row = i;
        }
    }

    // Ascending unit position, which is ascending unit code: the design is
    // summed in this order.
    scratch.touched.sort_unstable();
    for index in 0..scratch.touched.len() {
        let slot = scratch.touched[index];
        if let Some(unit) = scratch.slots[slot].take()
            && unit.baseline_outcome.is_some()
            && unit.follow_up_outcome.is_some()
        {
            scratch.units.push(unit);
        }
    }
}

/// Why a two-period fit failed, one kind per error, so a caller counting
/// skipped cells can tell a non-finite outcome from a singular design.
pub(super) fn fit_failure(error: &DrDidError) -> FitFailure {
    use DrDidError as E;
    match error {
        E::EmptyInput => FitFailure::Empty,
        E::NoTreated => FitFailure::NoTreated,
        E::NoControl => FitFailure::NoComparison,
        E::InvalidWeight { .. } => FitFailure::InvalidWeight,
        E::InvalidOutcome { .. } => FitFailure::InvalidOutcome,
        E::InvalidCovariate { .. } => FitFailure::InvalidCovariate,
        E::SingularSystem => FitFailure::Singular,
        _ => FitFailure::Other,
    }
}

/// The cell's numbers from a two-period fit.
const fn cell_fit(fit: &DrDidEstimate, se: f64, ci: (f64, f64)) -> CellFit {
    CellFit {
        att: fit.att,
        se,
        ci_low: ci.0,
        ci_high: ci.1,
        treated_n: fit.treated_n,
        control_n: fit.control_n,
        total_weight: fit.total_weight,
        columns_dropped: fit.design_columns_dropped,
    }
}

/// The overlap diagnostic of a cell, and whether `did` would set the cell to
/// NA: under [`AttGtOptions::refuse_poor_overlap`](super::AttGtOptions), for
/// the doubly robust and weighting estimators, when the largest score is at
/// least 0.999 or the check's logistic fit fails.
pub(super) fn check_overlap(
    design: &[f64],
    feature_count: usize,
    treated: &[f64],
    config: Config,
    uses_propensity: bool,
    overlap: &mut Option<Overlap>,
) -> Result<(), SkipReason> {
    let max_propensity = super::overlap::max_propensity(design, feature_count, treated);
    *overlap = Some(Overlap { max_propensity });
    if !config.refuse_poor_overlap || !uses_propensity {
        return Ok(());
    }
    match max_propensity {
        None => Err(SkipReason::OverlapFitFailed),
        Some(score) if score >= super::overlap::OVERLAP_THRESHOLD => Err(SkipReason::PoorOverlap),
        Some(_) => Ok(()),
    }
}

/// Estimate one `(g, t)` cell and return the estimate plus a unit-aligned
/// influence vector of length `unit_count`.
fn estimate_panel_cell(
    rows: &Rows<'_>,
    scratch: &mut CellScratch,
    unit_count: usize,
    config: Config,
    estimator: Estimator,
    overlap: &mut Option<Overlap>,
) -> Fitted {
    let treated_count = scratch.units.iter().filter(|unit| unit.treated).count();
    if treated_count == 0 {
        return Err(SkipReason::NoTreated);
    }
    if treated_count == scratch.units.len() {
        return Err(SkipReason::NoComparison);
    }

    // The intercept is prepended here: `xformla = ~ x` means intercept plus x,
    // and the panel estimators take a finished design.
    let feature_count = rows.n_covariates() + 1;
    scratch.treated.reserve(scratch.units.len());
    scratch.delta.reserve(scratch.units.len());
    scratch.weights.reserve(scratch.units.len());
    scratch.design.reserve(scratch.units.len() * feature_count);
    for unit in &scratch.units {
        scratch.treated.push(unit.treated);
        // Both are Some by construction: collect_cell_units filters on it.
        scratch.delta.push(
            unit.follow_up_outcome.unwrap_or_default() - unit.baseline_outcome.unwrap_or_default(),
        );
        scratch.weights.push(unit.weight);
        scratch.design.push(1.0);
        rows.extend_covariates(unit.covariate_row, &mut scratch.design);
    }

    let treated: Vec<f64> = scratch
        .treated
        .iter()
        .map(|&d| f64::from(u8::from(d)))
        .collect();
    check_overlap(
        &scratch.design,
        feature_count,
        &treated,
        config,
        estimator.uses_propensity(),
        overlap,
    )?;

    let input = PanelFlatInput {
        treated: &scratch.treated,
        delta_outcome: &scratch.delta,
        weight: &scratch.weights,
        design_matrix_flat: &scratch.design,
        feature_count,
    };
    let fit = match estimator {
        Estimator::DoublyRobust => estimate_drdid_panel_flat(input, config.drdid),
        Estimator::InverseProbabilityWeighting => estimate_ipw_did_panel_flat(input, config.drdid),
        Estimator::OutcomeRegression => estimate_or_did_panel_flat(input, config.drdid),
    }
    .map_err(|error| SkipReason::FitFailed(fit_failure(&error)))?;
    if fit.influence_function.len() != scratch.units.len() {
        return Err(SkipReason::FitFailed(FitFailure::Other));
    }
    // A solve that ran to a non-finite number is a failed cell, not a row.
    if !fit.att.is_finite() || !fit.se.is_finite() {
        return Err(SkipReason::NotFinite);
    }

    // The cell's psi is normalised so that sqrt(sum(psi^2)) / n_cell is its
    // standard error. Embedded in the whole sample it is scaled by
    // n / n_cell, as `did` does (`if_x <- (n/n1) * att.inf.func`), so that
    // every cell's vector is on the scale of the full sample.
    let scale =
        crate::util::usize_to_f64(unit_count) / crate::util::usize_to_f64(scratch.units.len());
    let mut aligned = vec![0.0; unit_count];
    for (unit, value) in scratch.units.iter().zip(&fit.influence_function) {
        aligned[unit.unit_index] = value * scale;
    }
    Ok((cell_fit(&fit, fit.se, (fit.ci_low, fit.ci_high)), aligned))
}

/// Buffers of the unbalanced-panel route.
struct RepeatedScratch {
    rows: Vec<DrDidRepeatedObservation>,
    positions: Vec<usize>,
}

/// Estimate one cell of an unbalanced panel by the repeated cross-section
/// estimator on the rows of its two periods, with each unit's rows summed in
/// the influence function.
///
/// This is `did`'s `allow_unbalanced_panel = TRUE`: each row's influence is
/// scaled by `n / n_rows` and summed by unit, and the standard error is that
/// of the summed vector, which accounts for a unit observed in both periods.
fn estimate_repeated_cell(
    rows: &Rows<'_>,
    prepared: &Prepared,
    cell: (i64, i64, i64),
    config: Config,
    estimator: Estimator,
    scratch: &mut RepeatedScratch,
    overlap: &mut Option<Overlap>,
) -> Fitted {
    scratch.rows.clear();
    scratch.positions.clear();
    let (_, time, _) = cell;
    for (k, &i) in prepared.rows.iter().enumerate() {
        let Some(treated) = cell_role(rows, i, prepared.cohort[k], cell, config) else {
            continue;
        };
        scratch.rows.push(DrDidRepeatedObservation {
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
        });
        scratch.positions.push(prepared.position[k]);
    }
    if let Some(reason) = super::pair_estimators::missing_pair_cell(&scratch.rows) {
        return Err(reason);
    }
    let (design, feature_count, treated) = super::pair_estimators::overlap_design(&scratch.rows);
    check_overlap(
        &design,
        feature_count,
        &treated,
        config,
        estimator.uses_propensity(),
        overlap,
    )?;
    let fit = match estimator {
        Estimator::DoublyRobust => estimate_drdid_repeated_efficient(&scratch.rows, config.drdid),
        Estimator::InverseProbabilityWeighting => {
            estimate_ipw_did_repeated_cross_section(&scratch.rows, config.drdid)
        }
        Estimator::OutcomeRegression => {
            estimate_or_did_repeated_cross_section(&scratch.rows, config.drdid)
        }
    }
    .map_err(|error| SkipReason::FitFailed(fit_failure(&error)))?;
    if !fit.att.is_finite() || !fit.se.is_finite() {
        return Err(SkipReason::NotFinite);
    }
    let unit_count = prepared.position_count();
    let scale =
        crate::util::usize_to_f64(unit_count) / crate::util::usize_to_f64(scratch.rows.len());
    let mut folded = vec![0.0; unit_count];
    for (&unit, value) in scratch.positions.iter().zip(&fit.influence_function) {
        folded[unit] += value * scale;
    }
    let se = standard_error_from_influence(&folded);
    let ci = crate::methods::drdid::confidence_interval(fit.att, se, &folded, config.drdid);
    Ok((cell_fit(&fit, se, ci), folded))
}

/// Panel `ATT(g,t)`, `did::att_gt(..., panel = TRUE)`, with the influence
/// vectors indexed by the sorted unit codes of the sample.
///
/// The input is first pre-processed as `did` does it, and each cell is fitted
/// by the two-period panel estimator on the units observed at both its
/// periods, with the covariates and the weight of the earlier period's row.
/// Under [`UnbalancedPanel::RepeatedCrossSection`](super::UnbalancedPanel) an
/// unbalanced panel is fitted cell by cell with the repeated cross-section
/// estimators, as `did` fits it.
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
    let prepared = preprocess(rows, config, Route::Panel)?;
    let unit_count = prepared.position_count();

    let mut attempts = Vec::new();
    let mut influence = Vec::new();
    let mut scratch = CellScratch::new(if prepared.as_repeated { 0 } else { unit_count });
    let mut repeated = RepeatedScratch {
        rows: Vec::new(),
        positions: Vec::new(),
    };

    for &group in &prepared.sample.cohorts {
        for (time, baseline_time) in cell_grid(&prepared.sample.periods, group, config) {
            let cell = (group, time, baseline_time);
            let mut overlap = None;
            let fitted = if prepared.as_repeated {
                estimate_repeated_cell(
                    rows,
                    &prepared,
                    cell,
                    config,
                    estimator,
                    &mut repeated,
                    &mut overlap,
                )
            } else {
                collect_cell_units(rows, &prepared, cell, config, &mut scratch);
                estimate_panel_cell(
                    rows,
                    &mut scratch,
                    unit_count,
                    config,
                    estimator,
                    &mut overlap,
                )
            };
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

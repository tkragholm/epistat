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
//! entry per unit, indexed by position in the sorted ids of the units the
//! pre-processing kept ([`crate::AttGtSample::ids`]); units absent from a cell
//! carry zero. A cell fitted on `n_cell` of the `n` units has its vector
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
//! [`UnbalancedPanel::RepeatedCrossSection`](crate::UnbalancedPanel) is fitted
//! by the repeated cross-section estimators on each cell's rows, and each
//! unit's rows are summed in its entry.

use crate::inference::standard_error_from_influence;
use crate::methods::drdid::ipw_or::{
    estimate_ipw_did_panel_flat, estimate_ipw_did_repeated_cross_section,
    estimate_or_did_panel_flat, estimate_or_did_repeated_cross_section,
};
use crate::methods::drdid::panel::{PanelFlatInput, estimate_drdid_panel_flat};
use crate::methods::drdid::repeated_efficient::estimate_drdid_repeated_efficient;
use crate::types::{
    AttGtDrConfig, AttGtDrObservation, AttGtError, AttGtEstimate, AttGtInfluenceOutput,
    CellOverlap, DidCell, DrDidEstimate, DrDidRepeatedObservation, PrunedCell, SkippedCell,
    TimePeriod, TreatmentGroup,
};

use super::preprocess::{Prepared, Route, cell_grid, preprocess};

/// A fitted cell, its unit-aligned influence vector and the covariate columns
/// it dropped, or the reason it could not be fitted.
type CellFit = Result<(AttGtEstimate, Vec<f64>, usize), &'static str>;

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

/// One unit's contribution to a single `(g, t)` cell.
struct PanelUnit<'a> {
    /// Index into the sorted distinct-unit list, for influence alignment.
    unit_index: usize,
    treated: bool,
    baseline_outcome: Option<f64>,
    follow_up_outcome: Option<f64>,
    weight: f64,
    /// The covariates of the row of the earlier of the cell's two periods,
    /// borrowed from it, as `did` takes them. That is the base period except
    /// in a pre-treatment cell under a universal base, whose base is later.
    covariates: &'a [f64],
}

/// Buffers the cell loop reuses, so the per-cell cost is the cell's own size.
///
/// `slots` is indexed by the unit position the influence vectors are aligned
/// on. `touched` lists the slots a cell wrote, so that resetting costs the
/// cell and not the whole panel.
struct CellScratch<'a> {
    slots: Vec<Option<PanelUnit<'a>>>,
    touched: Vec<usize>,
    units: Vec<PanelUnit<'a>>,
    treated: Vec<bool>,
    delta: Vec<f64>,
    weights: Vec<f64>,
    design: Vec<f64>,
}

impl CellScratch<'_> {
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

/// Whether `row`, of cohort `cohort` after the pre-processing, is in the cell
/// `(group, time)` against `baseline_time`, and as treated (`Some(true)`) or
/// comparison (`Some(false)`).
///
/// Not-yet-treated eligibility is judged at the later of the two periods, as
/// in `did`: under a universal base a pre-treatment cell has its base after
/// `time`, and a unit treated between the two is treated at the base.
fn cell_role(
    row: &AttGtDrObservation,
    cohort: Option<i32>,
    group: i32,
    time: i32,
    baseline_time: i32,
    config: AttGtDrConfig,
) -> Option<bool> {
    if row.time != baseline_time && row.time != time {
        return None;
    }
    if cohort == Some(group) {
        return Some(true);
    }
    super::is_control_for_pair(
        cohort,
        row.comparison_cohort,
        group,
        time.max(baseline_time),
        config.att_gt.comparison_group,
        config.att_gt.anticipation_periods,
    )
    .then_some(false)
}

/// Collect the units contributing to one `(group, time)` cell against
/// `baseline_time`: those observed at both periods.
fn collect_cell_units<'a>(
    observations: &'a [AttGtDrObservation],
    prepared: &Prepared,
    group: i32,
    time: i32,
    baseline_time: i32,
    config: AttGtDrConfig,
    scratch: &mut CellScratch<'a>,
) -> Result<(), AttGtError> {
    scratch.reset();
    let mut duplicate: Option<i64> = None;
    let earlier = time.min(baseline_time);

    for (k, &i) in prepared.rows.iter().enumerate() {
        let row = &observations[i];
        let Some(treated) = cell_role(row, prepared.cohort[k], group, time, baseline_time, config)
        else {
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
                weight: row.weight,
                covariates: &[],
            });
        }
        let Some(entry) = slot.as_mut() else { continue };

        // A second row of a unit in one period is refused rather than
        // absorbed: a caller who repeats rows to weight a unit would otherwise
        // get one unit and no error. The weight field carries weights.
        let outcome_slot = if row.time == baseline_time {
            &mut entry.baseline_outcome
        } else {
            &mut entry.follow_up_outcome
        };
        if outcome_slot.is_some() {
            duplicate = row.unit_id;
        }
        *outcome_slot = Some(row.outcome);
        if row.time == earlier {
            entry.weight = row.weight;
            entry.covariates = row.covariates.as_slice();
        }
    }

    if let Some(unit_id) = duplicate {
        return Err(AttGtError::DuplicatePanelRow {
            unit_id,
            group,
            time,
        });
    }

    // Ascending unit position, which is ascending unit id: the design is
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
    Ok(())
}

/// The estimator's own word for why a cell's fit failed, one static label per
/// error kind, so a caller counting skipped cells can tell a non-finite
/// outcome from a singular design without parsing prose.
fn panel_fit_reason(error: &crate::types::DrDidError) -> &'static str {
    use crate::types::DrDidError as E;
    match error {
        E::EmptyInput => "panel_fit:empty",
        E::NoTreated => "panel_fit:no_treated",
        E::NoControl => "panel_fit:no_control",
        E::InvalidWeight { .. } => "panel_fit:invalid_weight",
        E::InvalidOutcome { .. } => "panel_fit:invalid_outcome",
        E::InvalidCovariate { .. } => "panel_fit:invalid_covariate",
        E::SingularSystem => "panel_fit:singular",
        _ => "panel_fit:other",
    }
}

/// The cell's estimate from a two-period fit and its influence vector over
/// all `unit_count` positions.
fn cell_estimate(
    fit: &DrDidEstimate,
    group: i32,
    time: i32,
    se: f64,
    ci: (f64, f64),
) -> AttGtEstimate {
    AttGtEstimate {
        group,
        time,
        event_time: time - group,
        att: fit.att,
        se,
        ci_low: ci.0,
        ci_high: ci.1,
        treated_n: fit.treated_n,
        control_n: fit.control_n,
        total_weight: fit.total_weight,
    }
}

/// Record the cell's overlap diagnostic and refuse the cell when `did` would
/// set it to NA: under [`AttGtConfig::refuse_poor_overlap`](crate::AttGtConfig),
/// for the doubly robust and weighting estimators, when the largest score is
/// at least 0.999 or the check's logistic fit fails.
pub(super) fn check_overlap(
    design: &[f64],
    feature_count: usize,
    treated: &[f64],
    (group, time, baseline_time): (i32, i32, i32),
    config: AttGtDrConfig,
    uses_propensity: impl Into<UsesPropensity>,
    overlap: &mut Vec<CellOverlap>,
) -> Result<(), &'static str> {
    let max_propensity = super::overlap::max_propensity(design, feature_count, treated);
    overlap.push(CellOverlap {
        group,
        time,
        baseline_time,
        max_propensity,
    });
    if !config.att_gt.refuse_poor_overlap || !uses_propensity.into().0 {
        return Ok(());
    }
    match max_propensity {
        None => Err("overlap_fit"),
        Some(score) if score >= super::overlap::OVERLAP_THRESHOLD => Err("overlap"),
        Some(_) => Ok(()),
    }
}

/// Whether an estimator fits a propensity score, which is when `did` checks
/// overlap.
pub(super) struct UsesPropensity(pub(super) bool);

impl From<Estimator> for UsesPropensity {
    fn from(estimator: Estimator) -> Self {
        Self(estimator != Estimator::OutcomeRegression)
    }
}

impl From<bool> for UsesPropensity {
    fn from(uses: bool) -> Self {
        Self(uses)
    }
}

/// Estimate one `(g, t)` cell and return the estimate plus a unit-aligned
/// influence vector of length `unit_count`.
fn estimate_panel_cell(
    scratch: &mut CellScratch<'_>,
    unit_count: usize,
    (group, time, baseline_time): (i32, i32, i32),
    config: AttGtDrConfig,
    estimator: Estimator,
    overlap: &mut Vec<CellOverlap>,
) -> CellFit {
    let treated_count = scratch.units.iter().filter(|unit| unit.treated).count();
    if treated_count == 0 {
        return Err("treated_panel");
    }
    if treated_count == scratch.units.len() {
        return Err("control_panel");
    }

    // The intercept is prepended here: `xformla = ~ x` means intercept plus x,
    // and the panel estimators take a finished design.
    let covariate_count = scratch
        .units
        .first()
        .map_or(0, |unit| unit.covariates.len());
    let feature_count = covariate_count + 1;
    scratch.treated.reserve(scratch.units.len());
    scratch.delta.reserve(scratch.units.len());
    scratch.weights.reserve(scratch.units.len());
    scratch.design.reserve(scratch.units.len() * feature_count);
    for unit in &scratch.units {
        if unit.covariates.len() != covariate_count {
            return Err("covariate_count");
        }
        scratch.treated.push(unit.treated);
        // Both are Some by construction: collect_cell_units filters on it.
        scratch.delta.push(
            unit.follow_up_outcome.unwrap_or_default() - unit.baseline_outcome.unwrap_or_default(),
        );
        scratch.weights.push(unit.weight);
        scratch.design.push(1.0);
        scratch.design.extend_from_slice(unit.covariates);
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
        (group, time, baseline_time),
        config,
        estimator,
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
    .map_err(|error| panel_fit_reason(&error))?;
    if fit.influence_function.len() != scratch.units.len() {
        return Err("influence_length");
    }
    // A solve that ran to a non-finite number is a failed cell, not a row.
    if !fit.att.is_finite() || !fit.se.is_finite() {
        return Err("non_finite");
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

    Ok((
        cell_estimate(&fit, group, time, fit.se, (fit.ci_low, fit.ci_high)),
        aligned,
        fit.design_columns_dropped,
    ))
}

/// Buffers of the unbalanced-panel route.
struct RepeatedScratch {
    rows: Vec<DrDidRepeatedObservation>,
    positions: Vec<usize>,
    /// Per unit: bit 0 set when the cell has its baseline row, bit 1 its
    /// follow-up row.
    seen: Vec<u8>,
    touched: Vec<usize>,
}

/// Estimate one cell of an unbalanced panel by the repeated cross-section
/// estimator on the rows of its two periods, with each unit's rows summed in
/// the influence function.
///
/// This is `did`'s `allow_unbalanced_panel = TRUE`: each row's influence is
/// scaled by `n / n_rows` and summed by unit, and the standard error is that
/// of the summed vector, which accounts for a unit observed in both periods.
fn estimate_repeated_cell(
    observations: &[AttGtDrObservation],
    prepared: &Prepared,
    (group, time, baseline_time): (i32, i32, i32),
    config: AttGtDrConfig,
    estimator: Estimator,
    scratch: &mut RepeatedScratch,
    overlap: &mut Vec<CellOverlap>,
) -> Result<CellFit, AttGtError> {
    scratch.rows.clear();
    scratch.positions.clear();
    for &unit in &scratch.touched {
        scratch.seen[unit] = 0;
    }
    scratch.touched.clear();
    for (k, &i) in prepared.rows.iter().enumerate() {
        let row = &observations[i];
        let Some(treated) = cell_role(row, prepared.cohort[k], group, time, baseline_time, config)
        else {
            continue;
        };
        let unit = prepared.position[k];
        let bit = if row.time == time { 2 } else { 1 };
        if scratch.seen[unit] & bit != 0 {
            return Err(AttGtError::DuplicatePanelRow {
                unit_id: row.unit_id.unwrap_or_default(),
                group,
                time,
            });
        }
        if scratch.seen[unit] == 0 {
            scratch.touched.push(unit);
        }
        scratch.seen[unit] |= bit;
        scratch.rows.push(DrDidRepeatedObservation {
            weight: row.weight,
            covariates: row.covariates.clone(),
            ..DrDidRepeatedObservation::new(
                DidCell::from_parts(
                    if treated {
                        TreatmentGroup::Treated
                    } else {
                        TreatmentGroup::Control
                    },
                    TimePeriod::from_bool(row.time == time),
                ),
                row.outcome,
            )
        });
        scratch.positions.push(unit);
    }
    if let Some(cell) = super::pair_estimators::missing_pair_cell(&scratch.rows) {
        return Ok(Err(cell));
    }
    let (design, feature_count, treated) = super::pair_estimators::overlap_design(&scratch.rows);
    if let Err(reason) = check_overlap(
        &design,
        feature_count,
        &treated,
        (group, time, baseline_time),
        config,
        estimator,
        overlap,
    ) {
        return Ok(Err(reason));
    }
    let fit = match estimator {
        Estimator::DoublyRobust => estimate_drdid_repeated_efficient(&scratch.rows, config.drdid),
        Estimator::InverseProbabilityWeighting => {
            estimate_ipw_did_repeated_cross_section(&scratch.rows, config.drdid)
        }
        Estimator::OutcomeRegression => {
            estimate_or_did_repeated_cross_section(&scratch.rows, config.drdid)
        }
    };
    let fit = match fit {
        Ok(fit) => fit,
        Err(error) => return Ok(Err(panel_fit_reason(&error))),
    };
    if !fit.att.is_finite() || !fit.se.is_finite() {
        return Ok(Err("non_finite"));
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
    Ok(Ok((
        cell_estimate(&fit, group, time, se, ci),
        folded,
        fit.design_columns_dropped,
    )))
}

fn estimate_att_gt_panel(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
    estimator: Estimator,
) -> Result<AttGtInfluenceOutput, AttGtError> {
    let prepared = preprocess(observations, config.att_gt, Route::Panel)?;
    let unit_count = prepared.position_count();

    let mut estimates = Vec::new();
    let mut influence_functions = Vec::new();
    let mut skipped = Vec::new();
    let mut pruned = Vec::new();
    let mut overlap = Vec::new();
    let mut scratch = CellScratch::new(unit_count);
    let mut repeated = RepeatedScratch {
        rows: Vec::new(),
        positions: Vec::new(),
        seen: vec![0; if prepared.as_repeated { unit_count } else { 0 }],
        touched: Vec::new(),
    };

    for &group in &prepared.sample.cohorts {
        for (time, baseline_time) in cell_grid(&prepared.sample.periods, group, config.att_gt) {
            let cell = if prepared.as_repeated {
                estimate_repeated_cell(
                    observations,
                    &prepared,
                    (group, time, baseline_time),
                    config,
                    estimator,
                    &mut repeated,
                    &mut overlap,
                )?
            } else {
                collect_cell_units(
                    observations,
                    &prepared,
                    group,
                    time,
                    baseline_time,
                    config,
                    &mut scratch,
                )?;
                estimate_panel_cell(
                    &mut scratch,
                    unit_count,
                    (group, time, baseline_time),
                    config,
                    estimator,
                    &mut overlap,
                )
            };
            match cell {
                Ok((estimate, influence, columns_dropped)) => {
                    if columns_dropped > 0 {
                        pruned.push(PrunedCell {
                            group,
                            time,
                            baseline_time,
                            columns_dropped,
                        });
                    }
                    estimates.push(estimate);
                    influence_functions.push(influence);
                }
                Err(cell) => {
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
            }
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

/// Panel `ATT(g,t)` by the doubly robust estimator, with influence functions
/// aligned to the unit index.
///
/// `did::att_gt(..., panel = TRUE, est_method = "dr")`. The input is first
/// pre-processed as `did` does it (see [`AttGtSample`](crate::AttGtSample)),
/// and each cell is fitted by the panel doubly robust estimator,
/// [`crate::estimate_drdid_panel`], on the units observed at both its periods,
/// with the covariates and the weight of the earlier period's row. Comparators with a fitted propensity score of 0.995
/// or higher are trimmed from the cell, as in `did` and `DRDID`. Under
/// [`UnbalancedPanel::RepeatedCrossSection`](crate::UnbalancedPanel) an
/// unbalanced panel is fitted cell by cell with
/// [`crate::estimate_drdid_repeated_efficient`], as `did` fits it with
/// `drdid_rc`.
///
/// Set [`BasePeriod::Universal`](crate::types::BasePeriod::Universal) to anchor
/// every cell at the last period before `g - anticipation`, which is what a
/// fixed-baseline event study means; R's default is `"varying"`.
///
/// The influence vectors are indexed by [`AttGtSample::ids`](crate::AttGtSample),
/// the sorted ids of the units kept, and
/// [`AttGtSample::unit_panel`](crate::AttGtSample::unit_panel) gives the panel
/// to aggregate them with.
///
/// # Errors
/// - [`AttGtError::MissingUnitId`] if any row lacks a unit id, and
///   [`AttGtError::InconsistentCohort`] if a unit's cohort differs between rows.
/// - [`AttGtError::DuplicatePanelRow`] if a unit has two rows in one period of
///   a cell.
/// - [`AttGtError::MissingCell`] if a cell cannot be fitted and
///   `skip_incomplete_pairs` is false.
/// - [`AttGtError::NoEstimablePairs`] if no cohort or no cell survives.
pub fn estimate_att_gt_dr_panel_with_influence(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<AttGtInfluenceOutput, AttGtError> {
    estimate_att_gt_panel(observations, config, Estimator::DoublyRobust)
}

/// Panel `ATT(g,t)` by inverse probability weighting,
/// `did::att_gt(..., panel = TRUE, est_method = "ipw")`: each cell is
/// [`crate::estimate_ipw_did_panel`], `DRDID::std_ipw_did_panel`, and an
/// unbalanced panel under
/// [`UnbalancedPanel::RepeatedCrossSection`](crate::UnbalancedPanel) is fitted
/// with [`crate::estimate_ipw_did_repeated_cross_section`]. Otherwise as
/// [`estimate_att_gt_dr_panel_with_influence`].
///
/// # Errors
/// As [`estimate_att_gt_dr_panel_with_influence`].
pub fn estimate_att_gt_ipw_panel_with_influence(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<AttGtInfluenceOutput, AttGtError> {
    estimate_att_gt_panel(observations, config, Estimator::InverseProbabilityWeighting)
}

/// Panel `ATT(g,t)` by outcome regression,
/// `did::att_gt(..., panel = TRUE, est_method = "reg")`: each cell is
/// [`crate::estimate_or_did_panel`], `DRDID::reg_did_panel`, and an
/// unbalanced panel under
/// [`UnbalancedPanel::RepeatedCrossSection`](crate::UnbalancedPanel) is fitted
/// with [`crate::estimate_or_did_repeated_cross_section`]. Otherwise as
/// [`estimate_att_gt_dr_panel_with_influence`].
///
/// # Errors
/// As [`estimate_att_gt_dr_panel_with_influence`].
pub fn estimate_att_gt_or_panel_with_influence(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<AttGtInfluenceOutput, AttGtError> {
    estimate_att_gt_panel(observations, config, Estimator::OutcomeRegression)
}

/// Each unit's cohort and sampling weight, indexed by the sorted distinct unit
/// ids of `observations`.
///
/// This is the index of the panel routes' influence vectors when their
/// pre-processing drops no unit and recodes no cohort, and every unit's
/// weight is the same on all its rows. Otherwise use the output's
/// [`AttGtSample::unit_panel`](crate::AttGtSample::unit_panel).
///
/// # Errors
/// [`AttGtError::MissingUnitId`] if any row lacks a unit id.
pub fn unit_panel(
    observations: &[AttGtDrObservation],
) -> Result<super::aggte::UnitPanel, AttGtError> {
    let mut ids = observations
        .iter()
        .map(|row| row.unit_id.ok_or(AttGtError::MissingUnitId))
        .collect::<Result<Vec<i64>, AttGtError>>()?;
    ids.sort_unstable();
    ids.dedup();
    let mut groups = vec![None; ids.len()];
    let mut weights = vec![1.0; ids.len()];
    for row in observations {
        let Some(id) = row.unit_id else { continue };
        let Ok(position) = ids.binary_search(&id) else {
            continue;
        };
        groups[position] = row.first_treated_time;
        weights[position] = row.weight;
    }
    Ok(super::aggte::UnitPanel {
        groups,
        weights,
        // Unclustered by default. A caller who knows units share a household,
        // a family or a matched set attaches the labels with
        // `UnitPanel::clustered_by`.
        clusters: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AttGtConfig, ComparisonGroup, DrDidConfig, PrunedCell, UnbalancedPanel};

    fn observation(unit: i64, group: Option<i32>, time: i32, outcome: f64) -> AttGtDrObservation {
        AttGtDrObservation {
            unit_id: Some(unit),
            first_treated_time: group,
            time,
            outcome,
            weight: 1.0,
            covariates: Vec::new(),
            comparison_cohort: None,
        }
    }

    /// Eight units treated at period 2 and eight never treated, over periods
    /// 1 to 3, with the treated units unobserved at period 3. The cell
    /// `(g = 2, t = 3)` then has no treated units and cannot be fitted. It is
    /// reported, not an error, and the one fittable cell is fitted. The first
    /// period is no cell's later period under a varying base, as in `did`.
    #[test]
    fn a_cell_with_no_treated_units_is_reported_rather_than_dropped() {
        let mut observations = Vec::new();
        for unit in 0..16_i64 {
            let group = (unit < 8).then_some(2);
            for time in 1..=3 {
                if group.is_some() && time == 3 {
                    continue;
                }
                let effect = if group.is_some() && time >= 2 {
                    5.0
                } else {
                    0.0
                };
                let outcome = 10.0 + f64::from(time) + (unit % 4) as f64 * 0.5 + effect;
                observations.push(observation(unit, group, time, outcome));
            }
        }
        let config = AttGtDrConfig {
            att_gt: AttGtConfig {
                unbalanced_panel: UnbalancedPanel::PerCell,
                ..AttGtConfig::default()
            },
            drdid: DrDidConfig::default(),
        };
        assert!(config.att_gt.skip_incomplete_pairs);

        let fitted = estimate_att_gt_dr_panel_with_influence(&observations, config).unwrap();
        let fitted_cells: Vec<(i32, i32)> =
            fitted.estimates.iter().map(|e| (e.group, e.time)).collect();
        assert_eq!(fitted_cells, vec![(2, 2)]);
        assert_eq!(
            fitted.skipped,
            vec![SkippedCell {
                group: 2,
                time: 3,
                baseline_time: 1,
                reason: "treated_panel",
            }]
        );
    }

    /// Twenty units treated at period 2 and twenty never treated over periods
    /// 1 to 4, one continuous covariate, and a rare indicator held by one
    /// treated unit and one comparator, the comparator observed at periods 1
    /// and 2 only. In cell (2, 2) the indicator varies in both arms and is
    /// kept; in cells (2, 3) and (2, 4) no comparator holds it, the outcome
    /// regression on the comparison arm cannot identify it, and the cell is
    /// fitted without it. Its estimate is then the one the continuous
    /// covariate alone gives, to the last bit, rather than a skipped cell.
    #[test]
    fn a_covariate_no_comparator_in_the_cell_holds_is_dropped_for_that_cell() {
        fn panel(with_indicator: bool) -> Vec<AttGtDrObservation> {
            let mut rows = Vec::new();
            for unit in 0..40_i64 {
                let group = (unit < 20).then_some(2);
                let holder = unit == 1 || unit == 21;
                let x = (unit % 7) as f64 * 0.3 - 1.0;
                for time in 1..=4 {
                    if unit == 21 && time > 2 {
                        continue;
                    }
                    let effect = if group.is_some() && time >= 2 {
                        3.0
                    } else {
                        0.0
                    };
                    let outcome =
                        5.0 + f64::from(time) * 0.5 + 1.2 * x + (unit % 3) as f64 * 0.25 + effect;
                    let mut covariates = vec![x];
                    if with_indicator {
                        covariates.push(f64::from(u8::from(holder)));
                    }
                    rows.push(AttGtDrObservation {
                        unit_id: Some(unit),
                        first_treated_time: group,
                        time,
                        outcome,
                        weight: 1.0,
                        covariates,
                        comparison_cohort: None,
                    });
                }
            }
            rows
        }
        let config = AttGtDrConfig {
            // The indicator separates the arms in the cells it is dropped
            // from, which `did`'s overlap check would refuse; the test is of
            // the pruning.
            att_gt: AttGtConfig {
                unbalanced_panel: UnbalancedPanel::PerCell,
                refuse_poor_overlap: false,
                ..AttGtConfig::default()
            },
            drdid: DrDidConfig::default(),
        };
        let full = estimate_att_gt_dr_panel_with_influence(&panel(true), config).unwrap();
        let reduced = estimate_att_gt_dr_panel_with_influence(&panel(false), config).unwrap();

        let cells: Vec<(i32, i32)> = full.estimates.iter().map(|e| (e.group, e.time)).collect();
        assert_eq!(cells, vec![(2, 2), (2, 3), (2, 4)]);
        assert!(full.skipped.iter().all(|cell| cell.time == 1));
        assert_eq!(
            full.pruned,
            vec![
                PrunedCell {
                    group: 2,
                    time: 3,
                    baseline_time: 1,
                    columns_dropped: 1,
                },
                PrunedCell {
                    group: 2,
                    time: 4,
                    baseline_time: 1,
                    columns_dropped: 1,
                },
            ]
        );
        for (with, without) in full.estimates.iter().zip(&reduced.estimates).skip(1) {
            assert_eq!(with.att.to_bits(), without.att.to_bits());
            assert_eq!(with.se.to_bits(), without.se.to_bits());
        }
        // The cell that kept the indicator is a different fit.
        assert_ne!(
            full.estimates[0].att.to_bits(),
            reduced.estimates[0].att.to_bits()
        );
        assert_eq!(reduced.pruned.len(), 0);
    }

    /// Three matched sets, indexed at periods 3, 4 and 5, each with treated
    /// units and their own never-treated comparators, observed from two
    /// periods before the index to three after. Every unit in a set follows
    /// the set's own trend, one unit of outcome per period times the index,
    /// and the treated units gain a constant 1.0 from their index on. Within a
    /// set the trends cancel exactly, so ATT(g,t) is 1.0 after the index and
    /// 0 before it. Pooling every comparator into each cell mixes three trends
    /// and reports something else, which is the 4 September 2026 Study I
    /// reading in miniature.
    #[test]
    fn matched_comparators_recover_the_effect_where_the_pool_reports_the_trends() {
        let mut rows = Vec::new();
        let mut unit = 0_i64;
        for index in 3..=5_i32 {
            let slope = f64::from(index);
            for member in 0..8 {
                for treated in [true, false] {
                    unit += 1;
                    let x = f64::from(member % 3);
                    for time in (index - 2)..=(index + 3) {
                        let effect = if treated && time >= index { 1.0 } else { 0.0 };
                        let outcome = 2.0 * x + slope * f64::from(time) + effect;
                        rows.push(AttGtDrObservation {
                            unit_id: Some(unit),
                            first_treated_time: treated.then_some(index),
                            time,
                            outcome,
                            weight: 1.0,
                            covariates: vec![x],
                            comparison_cohort: (!treated).then_some(index),
                        });
                    }
                }
            }
        }
        let config = |comparison_group| AttGtDrConfig {
            att_gt: AttGtConfig {
                comparison_group,
                base_period: crate::types::BasePeriod::Universal,
                unbalanced_panel: UnbalancedPanel::PerCell,
                ..AttGtConfig::default()
            },
            drdid: DrDidConfig::default(),
        };

        let matched = estimate_att_gt_dr_panel_with_influence(
            &rows,
            config(ComparisonGroup::MatchedNeverTreated),
        )
        .unwrap();
        // The only cells without a fit are the ones at periods a set is not
        // observed in, and the fit never fails.
        assert!(
            matched
                .skipped
                .iter()
                .all(|cell| cell.reason == "treated_panel"),
            "{:?}",
            matched.skipped
        );
        assert_eq!(matched.estimates.len(), 15);
        for cell in &matched.estimates {
            let expected = if cell.time >= cell.group { 1.0 } else { 0.0 };
            assert!(
                (cell.att - expected).abs() < 1e-9,
                "cell ({}, {}) gave {}",
                cell.group,
                cell.time,
                cell.att
            );
            // Eight comparators per cell: the set's own and nobody else's.
            assert_eq!(cell.control_n, 8);
        }

        let pooled =
            estimate_att_gt_dr_panel_with_influence(&rows, config(ComparisonGroup::NeverTreated))
                .unwrap();
        let off = pooled
            .estimates
            .iter()
            .filter(|cell| {
                let expected = if cell.time >= cell.group { 1.0 } else { 0.0 };
                (cell.att - expected).abs() > 0.1
            })
            .count();
        assert!(
            off > 0,
            "the pooled comparison should not recover the effect here"
        );
    }
}

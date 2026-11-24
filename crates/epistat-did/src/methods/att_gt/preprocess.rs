//! The pre-processing `did::att_gt` applies before it forms any cell.
//!
//! The rules are those of `did` 2.5.1 (`did_standardization`, the default
//! `faster_mode` route), in its order:
//!
//! 1. A cohort first treated after the last period plus the anticipation
//!    periods is recoded as never treated.
//! 2. If no unit is never treated, the periods from `c - anticipation` on are
//!    dropped, `c` the last cohort. Under the never-treated comparison group
//!    `c` is recoded as never treated; under the not-yet-treated group it
//!    stays a cohort, serves as a comparison in every remaining period and has
//!    no cells of its own. The matched comparison group has no counterpart in
//!    `did` and needs never-treated units.
//! 3. Units treated in the first remaining period, or within the anticipation
//!    periods of it, are dropped. The cohorts estimated are the remaining ones
//!    after the first period plus the anticipation periods.
//! 4. On a panel, units missing from some period are dropped, or kept, as
//!    [`UnbalancedPanel`] says.
//!
//! The base periods are then read from the sorted list of the remaining
//! periods ([`cell_grid`]), which is what lets the periods have gaps.

use std::collections::BTreeMap;

use crate::types::{
    AttGtConfig, AttGtDrObservation, AttGtError, AttGtSample, BasePeriod, ComparisonGroup,
    UnbalancedPanel,
};

/// Whether rows are units observed over time or independent observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Route {
    Panel,
    RepeatedCrossSection,
}

/// The rows an ATT(g,t) route estimates on.
pub(super) struct Prepared {
    /// The input rows kept, ascending.
    pub rows: Vec<usize>,
    /// The cohort of each kept row after recoding.
    pub cohort: Vec<Option<i32>>,
    /// The influence position of each kept row: its unit's on a panel route,
    /// its own on a repeated cross-section route.
    pub position: Vec<usize>,
    /// Whether a panel route estimates the unbalanced panel as repeated
    /// cross-sections.
    pub as_repeated: bool,
    pub sample: AttGtSample,
}

impl Prepared {
    pub(super) fn position_count(&self) -> usize {
        self.sample.ids.len()
    }
}

fn validate_rows(observations: &[AttGtDrObservation]) -> Result<(), AttGtError> {
    let covariate_count = observations.first().map_or(0, |row| row.covariates.len());
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
    }
    Ok(())
}

/// The unit id of every row, with each unit's cohort checked to be the same
/// on all its rows.
fn unit_ids(observations: &[AttGtDrObservation]) -> Result<Vec<i64>, AttGtError> {
    let ids = observations
        .iter()
        .map(|row| row.unit_id.ok_or(AttGtError::MissingUnitId))
        .collect::<Result<Vec<i64>, AttGtError>>()?;
    let mut cohorts: BTreeMap<i64, Option<i32>> = BTreeMap::new();
    for (row, &id) in observations.iter().zip(&ids) {
        match cohorts.get(&id) {
            Some(&cohort) if cohort != row.first_treated_time => {
                return Err(AttGtError::InconsistentCohort { unit_id: id });
            }
            Some(_) => {}
            None => {
                cohorts.insert(id, row.first_treated_time);
            }
        }
    }
    Ok(ids)
}

fn sorted_periods(observations: &[AttGtDrObservation], keep: &[bool]) -> Vec<i32> {
    let mut periods: Vec<i32> = observations
        .iter()
        .zip(keep)
        .filter(|(_, kept)| **kept)
        .map(|(row, _)| row.time)
        .collect();
    periods.sort_unstable();
    periods.dedup();
    periods
}

/// Units (when `ids` is given) or rows among `rows`.
fn count(rows: &[usize], ids: Option<&[i64]>) -> usize {
    ids.map_or(rows.len(), |ids| {
        let mut units: Vec<i64> = rows.iter().map(|&row| ids[row]).collect();
        units.sort_unstable();
        units.dedup();
        units.len()
    })
}

/// Apply `did`'s pre-processing.
///
/// # Errors
/// - [`AttGtError::EmptyInput`] for no rows, and the row and configuration
///   errors of the routes.
/// - [`AttGtError::MissingUnitId`] or [`AttGtError::InconsistentCohort`] on a
///   panel route.
/// - [`AttGtError::MissingNeverTreatedGroup`] under the matched comparison
///   group without never-treated units.
/// - [`AttGtError::NoEstimablePairs`] when no cohort or no row is left.
pub(super) fn preprocess(
    observations: &[AttGtDrObservation],
    config: AttGtConfig,
    route: Route,
) -> Result<Prepared, AttGtError> {
    super::validate_config(config)?;
    if observations.is_empty() {
        return Err(AttGtError::EmptyInput);
    }
    validate_rows(observations)?;
    let ids = match route {
        Route::Panel => Some(unit_ids(observations)?),
        Route::RepeatedCrossSection => None,
    };
    let ids = ids.as_deref();
    let anticipation = config.anticipation_periods;
    let n = observations.len();
    let mut keep = vec![true; n];
    let mut cohort: Vec<Option<i32>> = observations
        .iter()
        .map(|row| row.first_treated_time)
        .collect();
    let mut sample = AttGtSample::default();

    // 1. Cohorts after the last period.
    let last_period = observations.iter().map(|row| row.time).max().unwrap_or(0);
    let recoded: Vec<usize> = (0..n)
        .filter(|&i| cohort[i].is_some_and(|g| g > last_period + anticipation))
        .collect();
    for &i in &recoded {
        cohort[i] = None;
    }
    sample.recoded_never_treated = count(&recoded, ids);

    // 2. No never-treated units: the last cohort is the comparison group.
    let mut excluded_cohort = None;
    if cohort.iter().all(Option::is_some) {
        if config.comparison_group == ComparisonGroup::MatchedNeverTreated {
            return Err(AttGtError::MissingNeverTreatedGroup);
        }
        let latest = cohort.iter().flatten().copied().max().unwrap_or(0);
        let cutoff = latest - anticipation;
        for (i, row) in observations.iter().enumerate() {
            if row.time >= cutoff {
                keep[i] = false;
                sample.dropped_late_rows += 1;
            }
        }
        if config.comparison_group == ComparisonGroup::NeverTreated {
            for value in &mut cohort {
                if *value == Some(latest) {
                    *value = None;
                }
            }
        } else {
            excluded_cohort = Some(latest);
        }
        sample.last_cohort_as_comparison = Some(latest);
    }
    let periods = sorted_periods(observations, &keep);
    let Some(&first_period) = periods.first() else {
        return Err(AttGtError::NoEstimablePairs);
    };

    // 3. Units treated at the start.
    let started: Vec<usize> = (0..n)
        .filter(|&i| keep[i] && cohort[i].is_some_and(|g| g <= first_period + anticipation))
        .collect();
    for &i in &started {
        keep[i] = false;
    }
    sample.dropped_treated_at_start = count(&started, ids);
    let periods = sorted_periods(observations, &keep);
    let Some(&first_period) = periods.first() else {
        return Err(AttGtError::NoEstimablePairs);
    };
    let mut cohorts: Vec<i32> = (0..n)
        .filter(|&i| keep[i])
        .filter_map(|i| cohort[i])
        .filter(|&g| g > first_period + anticipation && Some(g) != excluded_cohort)
        .collect();
    cohorts.sort_unstable();
    cohorts.dedup();

    // 4. Balance.
    let mut as_repeated = false;
    if let Some(ids) = ids {
        let mut unit_periods: Vec<(i64, i32)> = (0..n)
            .filter(|&i| keep[i])
            .map(|i| (ids[i], observations[i].time))
            .collect();
        unit_periods.sort_unstable();
        unit_periods.dedup();
        let mut observed: BTreeMap<i64, usize> = BTreeMap::new();
        for (id, _) in &unit_periods {
            *observed.entry(*id).or_insert(0) += 1;
        }
        let balanced = observed.values().all(|&count| count == periods.len());
        match config.unbalanced_panel {
            UnbalancedPanel::Balance if !balanced => {
                let missing: Vec<i64> = observed
                    .iter()
                    .filter(|&(_, &count)| count < periods.len())
                    .map(|(&id, _)| id)
                    .collect();
                for (kept, id) in keep.iter_mut().zip(ids) {
                    if missing.binary_search(id).is_ok() {
                        *kept = false;
                    }
                }
                sample.dropped_unbalanced = missing.len();
            }
            UnbalancedPanel::RepeatedCrossSection => as_repeated = !balanced,
            _ => {}
        }
    }
    if cohorts.is_empty() || !keep.iter().any(|&kept| kept) {
        return Err(AttGtError::NoEstimablePairs);
    }

    let rows: Vec<usize> = (0..n).filter(|&i| keep[i]).collect();
    let kept_cohort: Vec<Option<i32>> = rows.iter().map(|&i| cohort[i]).collect();
    let (position, sample_ids, groups, weights) = match ids {
        Some(ids) => {
            let mut units: Vec<i64> = rows.iter().map(|&i| ids[i]).collect();
            units.sort_unstable();
            units.dedup();
            let position: Vec<usize> = rows
                .iter()
                .map(|&i| units.binary_search(&ids[i]).unwrap_or_default())
                .collect();
            let mut groups = vec![None; units.len()];
            let mut weights = vec![0.0; units.len()];
            let mut earliest = vec![i32::MAX; units.len()];
            let mut rows_of = vec![0_usize; units.len()];
            for (k, &i) in rows.iter().enumerate() {
                let unit = position[k];
                groups[unit] = kept_cohort[k];
                rows_of[unit] += 1;
                if as_repeated {
                    weights[unit] += observations[i].weight;
                } else if observations[i].time < earliest[unit] {
                    earliest[unit] = observations[i].time;
                    weights[unit] = observations[i].weight;
                }
            }
            if as_repeated {
                for (weight, &count) in weights.iter_mut().zip(&rows_of) {
                    *weight /= crate::util::usize_to_f64(count);
                }
            }
            (position, units, groups, weights)
        }
        None => (
            (0..rows.len()).collect(),
            rows.iter().map(|&i| crate::util::usize_to_i64(i)).collect(),
            kept_cohort.clone(),
            rows.iter().map(|&i| observations[i].weight).collect(),
        ),
    };
    sample.ids = sample_ids;
    sample.groups = groups;
    sample.weights = weights;
    sample.periods = periods;
    sample.cohorts = cohorts;
    sample.unbalanced_as_repeated_cross_section = as_repeated;

    Ok(Prepared {
        rows,
        cohort: kept_cohort,
        position,
        as_repeated,
        sample,
    })
}

/// The cells `(t, base)` of cohort `group` over the sorted `periods`, as
/// `did::att_gt` forms them.
///
/// The base of a post-treatment cell (`group <= t`), and of every cell under
/// a universal base, is the last period before `group - anticipation`. Under a
/// varying base a pre-treatment cell compares a period with the one before it,
/// so the first period is never a cell's later period. Under a universal base
/// the base period itself is no cell. `group` must exceed the first period
/// plus the anticipation periods, which [`preprocess`] ensures.
pub(super) fn cell_grid(periods: &[i32], group: i32, config: AttGtConfig) -> Vec<(i32, i32)> {
    let anticipation = config.anticipation_periods;
    let Some(base) = periods
        .iter()
        .rev()
        .copied()
        .find(|&period| period + anticipation < group)
    else {
        return Vec::new();
    };
    match config.base_period {
        BasePeriod::Varying => periods
            .windows(2)
            .map(|pair| {
                let (previous, t) = (pair[0], pair[1]);
                (t, if group <= t { base } else { previous })
            })
            .collect(),
        BasePeriod::Universal => periods
            .iter()
            .copied()
            .filter(|&t| t != base)
            .map(|t| (t, base))
            .collect(),
    }
}

/// The sample's unit panel for [`super::aggte::aggregate_att_gt`].
impl AttGtSample {
    /// The cohorts and weights of the influence positions, unclustered, for
    /// [`aggregate_att_gt`](crate::aggregate_att_gt) and the bands.
    #[must_use]
    pub fn unit_panel(&self) -> super::aggte::UnitPanel {
        super::aggte::UnitPanel {
            groups: self.groups.clone(),
            weights: self.weights.clone(),
            clusters: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_varying_grid_steps_over_gaps_between_periods() {
        let config = AttGtConfig::default();
        // Periods 1, 2, 4, 5, 7; cohort 5. The base of the post cells is 4.
        let grid = cell_grid(&[1, 2, 4, 5, 7], 5, config);
        assert_eq!(grid, vec![(2, 1), (4, 2), (5, 4), (7, 4)]);
        let universal = AttGtConfig {
            base_period: BasePeriod::Universal,
            ..config
        };
        assert_eq!(
            cell_grid(&[1, 2, 4, 5, 7], 5, universal),
            vec![(1, 4), (2, 4), (5, 4), (7, 4)]
        );
        let anticipating = AttGtConfig {
            anticipation_periods: 1,
            ..config
        };
        // With one period of anticipation the base is the last period before 4.
        assert_eq!(
            cell_grid(&[1, 2, 4, 5, 7], 5, anticipating),
            vec![(2, 1), (4, 2), (5, 2), (7, 2)]
        );
    }
}

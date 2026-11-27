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
//! 5. Under the never-treated comparison group, a never-treated group with
//!    fewer rows per period than the number of covariates plus 5 is refused
//!    ([`AttGtOptions::refuse_small_never_treated`](super::AttGtOptions)).
//!
//! The base periods are then read from the sorted list of the remaining
//! periods ([`cell_grid`]), which is what lets the periods have gaps.

use std::collections::BTreeMap;

use super::rows::Rows;
use super::{AttGtSample, BasePeriod, ComparisonGroup, Config, UnbalancedPanel};
use crate::error::Error;

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
    pub cohort: Vec<Option<i64>>,
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
        self.sample.units.len()
    }
}

fn sorted_periods(rows: &Rows<'_>, keep: &[bool]) -> Vec<i64> {
    let mut periods: Vec<i64> = rows
        .period
        .iter()
        .zip(keep)
        .filter(|(_, kept)| **kept)
        .map(|(period, _)| *period)
        .collect();
    periods.sort_unstable();
    periods.dedup();
    periods
}

/// Units (when `ids` is given) or rows among `rows`.
fn count(rows: &[usize], ids: Option<&[u32]>) -> usize {
    ids.map_or(rows.len(), |ids| {
        let mut units: Vec<u32> = rows.iter().map(|&row| ids[row]).collect();
        units.sort_unstable();
        units.dedup();
        units.len()
    })
}

/// Apply `did`'s pre-processing.
///
/// # Errors
/// - [`Error::NoNeverTreated`] under the matched comparison group without
///   never-treated units.
/// - [`Error::SmallNeverTreatedGroup`] when that check refuses the sample.
/// - [`Error::NoEstimableCells`] when no cohort or no row is left.
#[expect(
    clippy::too_many_lines,
    reason = "the five steps of `did`'s pre-processing in its order, which reads best as one"
)]
pub(super) fn preprocess(rows: &Rows<'_>, config: Config, route: Route) -> Result<Prepared, Error> {
    let ids = match route {
        Route::Panel => Some(rows.unit),
        Route::RepeatedCrossSection => None,
    };
    let anticipation = config.anticipation;
    let n = rows.len();
    let mut keep = vec![true; n];
    let mut cohort: Vec<Option<i64>> = rows.cohort.to_vec();
    let mut sample = AttGtSample::default();

    // 1. Cohorts after the last period.
    let last_period = rows.period.iter().copied().max().unwrap_or(0);
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
            return Err(Error::NoNeverTreated);
        }
        let latest = cohort.iter().flatten().copied().max().unwrap_or(0);
        let cutoff = latest - anticipation;
        for (i, &period) in rows.period.iter().enumerate() {
            if period >= cutoff {
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
    let periods = sorted_periods(rows, &keep);
    let Some(&first_period) = periods.first() else {
        return Err(Error::NoEstimableCells);
    };

    // 3. Units treated at the start.
    let started: Vec<usize> = (0..n)
        .filter(|&i| keep[i] && cohort[i].is_some_and(|g| g <= first_period + anticipation))
        .collect();
    for &i in &started {
        keep[i] = false;
    }
    sample.dropped_treated_at_start = count(&started, ids);
    let periods = sorted_periods(rows, &keep);
    let Some(&first_period) = periods.first() else {
        return Err(Error::NoEstimableCells);
    };
    let mut cohorts: Vec<i64> = (0..n)
        .filter(|&i| keep[i])
        .filter_map(|i| cohort[i])
        .filter(|&g| g > first_period + anticipation && Some(g) != excluded_cohort)
        .collect();
    cohorts.sort_unstable();
    cohorts.dedup();

    // 4. Balance.
    let mut as_repeated = false;
    if let Some(ids) = ids {
        let mut unit_periods: Vec<(u32, i64)> = (0..n)
            .filter(|&i| keep[i])
            .map(|i| (ids[i], rows.period[i]))
            .collect();
        unit_periods.sort_unstable();
        unit_periods.dedup();
        let mut observed: BTreeMap<u32, usize> = BTreeMap::new();
        for (id, _) in &unit_periods {
            *observed.entry(*id).or_insert(0) += 1;
        }
        let balanced = observed.values().all(|&count| count == periods.len());
        match config.unbalanced {
            UnbalancedPanel::Balance if !balanced => {
                let missing: Vec<u32> = observed
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
        return Err(Error::NoEstimableCells);
    }

    // 5. The size of the never-treated group: `did` stops when it has fewer
    // rows per period than the covariates plus 5.
    if config.refuse_small_never_treated && config.comparison_group == ComparisonGroup::NeverTreated
    {
        let never = (0..n).filter(|&i| keep[i] && cohort[i].is_none()).count();
        let per_period =
            crate::util::usize_to_f64(never) / crate::util::usize_to_f64(periods.len());
        let required = rows.n_covariates() + 5;
        if per_period < crate::util::usize_to_f64(required) {
            return Err(Error::SmallNeverTreatedGroup {
                per_period,
                required,
            });
        }
    }

    let kept: Vec<usize> = (0..n).filter(|&i| keep[i]).collect();
    let kept_cohort: Vec<Option<i64>> = kept.iter().map(|&i| cohort[i]).collect();
    let (position, sample_units, groups, weights) = match ids {
        Some(ids) => {
            let mut units: Vec<u32> = kept.iter().map(|&i| ids[i]).collect();
            units.sort_unstable();
            units.dedup();
            let position: Vec<usize> = kept
                .iter()
                .map(|&i| units.binary_search(&ids[i]).unwrap_or_default())
                .collect();
            let mut groups = vec![None; units.len()];
            let mut weights = vec![0.0; units.len()];
            let mut earliest = vec![i64::MAX; units.len()];
            let mut rows_of = vec![0_usize; units.len()];
            for (k, &i) in kept.iter().enumerate() {
                let unit = position[k];
                groups[unit] = kept_cohort[k];
                rows_of[unit] += 1;
                if as_repeated {
                    weights[unit] += rows.weight(i);
                } else if rows.period[i] < earliest[unit] {
                    earliest[unit] = rows.period[i];
                    weights[unit] = rows.weight(i);
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
            (0..kept.len()).collect(),
            kept.iter().map(|&i| row_code(i)).collect(),
            kept_cohort.clone(),
            kept.iter().map(|&i| rows.weight(i)).collect(),
        ),
    };
    sample.units = sample_units;
    sample.groups = groups;
    sample.weights = weights;
    sample.periods = periods;
    sample.cohorts = cohorts;
    sample.unbalanced_as_repeated_cross_section = as_repeated;

    Ok(Prepared {
        rows: kept,
        cohort: kept_cohort,
        position,
        as_repeated,
        sample,
    })
}

/// The number of row `i` as an influence unit on a repeated cross-section.
/// The estimator refuses more rows than `u32` can number, so it does not
/// saturate.
pub(super) fn row_code(i: usize) -> u32 {
    u32::try_from(i).unwrap_or(u32::MAX)
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
pub(super) fn cell_grid(periods: &[i64], group: i64, config: Config) -> Vec<(i64, i64)> {
    let anticipation = config.anticipation;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_varying_grid_steps_over_gaps_between_periods() {
        let config = Config::default();
        // Periods 1, 2, 4, 5, 7; cohort 5. The base of the post cells is 4.
        let grid = cell_grid(&[1, 2, 4, 5, 7], 5, config);
        assert_eq!(grid, vec![(2, 1), (4, 2), (5, 4), (7, 4)]);
        let universal = Config {
            base_period: BasePeriod::Universal,
            ..config
        };
        assert_eq!(
            cell_grid(&[1, 2, 4, 5, 7], 5, universal),
            vec![(1, 4), (2, 4), (5, 4), (7, 4)]
        );
        let anticipating = Config {
            anticipation: 1,
            ..config
        };
        // With one period of anticipation the base is the last period before 4.
        assert_eq!(
            cell_grid(&[1, 2, 4, 5, 7], 5, anticipating),
            vec![(2, 1), (4, 2), (5, 2), (7, 2)]
        );
    }
}

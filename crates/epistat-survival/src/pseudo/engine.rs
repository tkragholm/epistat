//! The infinitesimal jackknife of the Aalen-Johansen estimate, read from the
//! influence that [`aalen_johansen::variance`] returns.
//!
//! The incidence of a cause has no closed form that is cheaper than the
//! recursion of the estimate itself, so it is taken from
//! [`aalen_johansen::variance`], which keeps the influence of every cluster at
//! the times asked for. The area under a curve has the influence
//! `integral of U_g(s) ds`, the influence of the curve integrated, and the
//! integral needs the influence at every exit time up to the horizon. That
//! array is clusters by exit times by states values, and is why the restricted
//! time lost to a cause costs memory that the other targets do not. The caller can
//! bound it, see [`PseudoOptions::max_influence_values`](super::PseudoOptions).

use crate::aalen_johansen::{AalenJohansenOptions, VarianceOptions, variance};
use crate::error::{Error, Result};
use crate::surv::SurvData;
use crate::timefix::TimeFix;

use super::Target;
use super::table::{Group, Reading};

/// The influence of every cluster of every group on the target at the times
/// in `reading`, row-major by cluster, one vector per group in the order of
/// `groups`.
///
/// `data` already follows the tie rule.
pub(super) fn run(
    data: &SurvData<'_>,
    groups: &[Group],
    reading: &Reading<'_>,
    max_values: Option<usize>,
) -> Result<Vec<Vec<f64>>> {
    let at = reading.at;
    let k = at.len();
    let integrate = matches!(
        reading.target,
        Target::RestrictedMean | Target::RestrictedTimeLost { .. }
    );
    let state = reading.target.cause().map_or(0, |c| c as usize);
    let states = data.followup()?.n_codes() as usize + 1;
    let horizon = at[k - 1];
    let grid: Vec<f64> = if integrate {
        let mut all: Vec<f64> = groups
            .iter()
            .flat_map(|g| g.times.iter().copied())
            .filter(|&t| t <= horizon)
            .collect();
        all.sort_by(f64::total_cmp);
        all.dedup();
        all
    } else {
        at.to_vec()
    };
    let clusters: usize = groups.iter().map(Group::n_clusters).sum();
    let values = clusters.saturating_mul(grid.len()).saturating_mul(states);
    if let Some(limit) = max_values {
        if values > limit {
            return Err(Error::input(format!(
                "the influence of {clusters} clusters at {} times in {states} states is {values} \
                 values, more than the {limit} allowed",
                grid.len()
            )));
        }
    }
    let result = variance(
        data,
        &AalenJohansenOptions {
            timefix: TimeFix::Exact,
        },
        &VarianceOptions {
            horizons: Vec::new(),
            influence_times: grid.clone(),
        },
    )?;
    let mut out = Vec::with_capacity(groups.len());
    for (group, influence) in groups.iter().zip(&result.groups) {
        debug_assert_eq!(group.labels, influence.clusters);
        let n = group.n_clusters();
        let t = grid.len();
        let mut column = vec![0.0f64; n * k];
        for c in 0..n {
            let base = c * t;
            if integrate {
                // The influence is a step function, the value at each grid time
                // holding until the next, so the area is a sum of rectangles.
                let mut slot = 0;
                while slot < k && (t == 0 || at[slot] < grid[0]) {
                    column[c * k + slot] = 0.0;
                    slot += 1;
                }
                let mut accumulated = 0.0f64;
                for i in 0..t {
                    let u = influence.influence[(base + i) * states + state];
                    let next = if i + 1 < t {
                        grid[i + 1]
                    } else {
                        f64::INFINITY
                    };
                    while slot < k && at[slot] < next {
                        column[c * k + slot] = accumulated + u * (at[slot] - grid[i]);
                        slot += 1;
                    }
                    if i + 1 < t {
                        accumulated += u * (next - grid[i]);
                    }
                }
            } else {
                for slot in 0..k {
                    column[c * k + slot] = influence.influence[(base + slot) * states + state];
                }
            }
        }
        out.push(column);
    }
    Ok(out)
}

//! The infinitesimal jackknife of the Aalen-Johansen estimate, read from the
//! influence that [`aalen_johansen::variance`] returns.
//!
//! The incidence of a cause has no closed form that is cheaper than the
//! recursion of the estimate itself, so it is taken from
//! [`aalen_johansen::variance`], which returns the influence of every cluster
//! at the times asked for. The area under a curve has the influence
//! `integral of U_g(s) ds`, the influence of the curve integrated, and the
//! integral needs the influence at every exit time up to the horizon. Holding all
//! of it takes clusters by exit times by states values, so the influence is
//! asked for a chunk of exit times at a time and integrated as it arrives, see
//! [`steps`](super::steps). The caller bounds what is held at once with
//! [`PseudoOptions::max_influence_values`](super::PseudoOptions), and each pass
//! over a chunk costs a run of the recursion. [`aalen_johansen::variance`]
//! cannot give the integral itself, since it keeps each cluster's integral
//! inside the recursion and returns only the sum of squares; an output of the
//! integral at chosen horizons would make the area targets one pass.

use crate::aalen_johansen::{AalenJohansenOptions, AalenJohansenVarianceOptions, variance_dfbeta};
use crate::error::{Error, Result};
use crate::surv::SurvData;
use crate::timefix::TimeFix;

use super::Target;
use super::steps::{Steps, read};
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
    let states = data.followup().n_codes() as usize + 1;
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
    let blocks: Vec<usize> = groups.iter().map(Group::n_clusters).collect();
    let clusters: usize = blocks.iter().sum();
    let per_pass = match max_values {
        None => grid.len().max(1),
        Some(limit) => {
            let per_time = clusters.saturating_mul(states);
            let fit = limit.checked_div(per_time).unwrap_or(0);
            if fit == 0 {
                return Err(Error::options(format!(
                    "the influence of {clusters} clusters in {states} states at one time is \
                     {per_time} values, more than the {limit} allowed"
                )));
            }
            fit
        }
    };
    read(
        &Steps {
            knots: &grid,
            at,
            integrate,
        },
        &blocks,
        1,
        per_pass,
        |chunk| {
            let (result, dfbeta) = variance_dfbeta(
                data,
                &AalenJohansenOptions {
                    timefix: TimeFix::Exact,
                },
                &AalenJohansenVarianceOptions {
                    horizons: Vec::new(),
                    influence_times: chunk.to_vec(),
                },
            )?;
            let t = chunk.len();
            Ok(groups
                .iter()
                .zip(result.groups.iter().zip(&dfbeta))
                .map(|(group, (influence, values))| {
                    debug_assert_eq!(group.labels, influence.clusters);
                    let n = group.n_clusters();
                    let mut column = Vec::with_capacity(n * t);
                    for c in 0..n {
                        for i in 0..t {
                            column.push(values[(c * t + i) * states + state]);
                        }
                    }
                    column
                })
                .collect())
        },
    )
}

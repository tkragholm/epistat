//! The estimator by subgroup (`by`) and by treatment path (`by_path`).

#![expect(
    clippy::float_cmp,
    reason = "a baseline treatment is matched exactly, as R matches values"
)]

use std::collections::{BTreeSet, HashMap};

use super::design::ranked_paths;
use super::panel::prepare;
use super::{DcdhConfig, DcdhError, DcdhObservation, DcdhResult, counts, estimate_dcdh, validate};

/// The estimates for one level of [`estimate_dcdh_by`].
#[derive(Debug, Clone, PartialEq)]
pub struct DcdhSubgroup {
    /// The level.
    pub level: i64,
    /// The estimates on its groups.
    pub result: Result<DcdhResult, DcdhError>,
}

/// The estimates for one treatment path of [`estimate_dcdh_by_path`].
#[derive(Debug, Clone, PartialEq)]
pub struct DcdhPathResult {
    /// The treatments at `F_g - 1, ..., F_g - 1 + L`.
    pub path: Vec<f64>,
    /// The switchers on the path in the whole panel.
    pub n_groups: usize,
    /// The estimates from those switchers and the not-yet switchers with
    /// the same baseline treatment.
    pub result: Result<DcdhResult, DcdhError>,
}

/// The estimator on the groups of each level of a group-level variable
/// (`by`), in increasing order of the levels. `by[i]` is the level of row
/// `i`, constant within a group.
///
/// # Errors
/// [`DcdhError`] for invalid options, or when `by` is not one level per row
/// constant within groups; the estimates of each level carry their own.
pub fn estimate_dcdh_by(
    observations: &[DcdhObservation],
    config: &DcdhConfig,
    by: &[i64],
) -> Result<Vec<DcdhSubgroup>, DcdhError> {
    validate(config)?;
    if by.len() != observations.len() {
        return Err(DcdhError::InvalidConfig("by needs one level per row"));
    }
    let mut level_of: HashMap<i64, i64> = HashMap::new();
    for (o, &b) in observations.iter().zip(by) {
        if *level_of.entry(o.group).or_insert(b) != b {
            return Err(DcdhError::VaryingSubgroup { group: o.group });
        }
    }
    let levels: BTreeSet<i64> = by.iter().copied().collect();
    Ok(levels
        .into_iter()
        .map(|level| {
            let rows: Vec<DcdhObservation> = observations
                .iter()
                .zip(by)
                .filter(|(_, b)| **b == level)
                .map(|(o, _)| o.clone())
                .collect();
            DcdhSubgroup {
                level,
                result: estimate_dcdh(&rows, config),
            }
        })
        .collect())
}

/// The estimator separately for the most common treatment paths of the
/// switchers (`by_path`): `paths` of them, all with `None`, in the order of
/// [`super::dcdh_design`]. Each path is estimated from its switchers and,
/// as controls, the other groups with the same baseline treatment before
/// their first switch, with `same_switchers`.
///
/// # Errors
/// [`DcdhError`] as [`super::estimate_dcdh`] on the whole panel; the
/// estimates of each path carry their own.
///
/// # Panics
/// Only when a number of periods does not fit in `i64`.
pub fn estimate_dcdh_by_path(
    observations: &[DcdhObservation],
    config: &DcdhConfig,
    paths: Option<usize>,
) -> Result<Vec<DcdhPathResult>, DcdhError> {
    validate(config)?;
    if paths == Some(0) {
        return Err(DcdhError::InvalidConfig("by_path needs at least one path"));
    }
    if config.continuous.is_some() {
        return Err(DcdhError::InvalidConfig(
            "by_path cannot be combined with continuous",
        ));
    }
    let panel = prepare(observations, config)?;
    let l = counts(&panel, config)?.l;
    let (ranked, _) = ranked_paths(&panel, l);
    let k = paths.map_or(ranked.len(), |p| p.min(ranked.len()));
    let grid_of: HashMap<i64, usize> = panel
        .group_ids
        .iter()
        .enumerate()
        .map(|(g, &id)| (id, g))
        .collect();
    let rank_of = |time: i64| {
        let r = panel.times.partition_point(|&t| t < time);
        i64::try_from(r).expect("rank fits i64") + 1
    };
    let sub_config = DcdhConfig {
        same_switchers: true,
        ..*config
    };
    Ok(ranked
        .into_iter()
        .take(k)
        .map(|(path, members)| {
            let on_path: BTreeSet<usize> = members.into_iter().collect();
            let baseline = path.treatments[0];
            let rows: Vec<DcdhObservation> = observations
                .iter()
                .filter(|o| {
                    grid_of.get(&o.group).is_some_and(|&g| {
                        on_path.contains(&g)
                            || (panel.d_sq[g] == baseline && rank_of(o.time) < panel.f_g[g])
                    })
                })
                .cloned()
                .collect();
            DcdhPathResult {
                n_groups: path.n_groups,
                path: path.treatments,
                result: estimate_dcdh(&rows, &sub_config),
            }
        })
        .collect())
}

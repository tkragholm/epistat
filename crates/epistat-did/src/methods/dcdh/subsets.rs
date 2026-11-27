//! The estimator by subgroup (`by`) and by treatment path (`by_path`).

#![expect(
    clippy::float_cmp,
    reason = "a baseline treatment is matched exactly, as R matches values"
)]

use std::collections::{BTreeSet, HashMap};

use crate::error::Error;

use super::design::ranked_paths;
use super::panel::prepare;
use super::{Dcdh, DcdhOptions, SwitchingPanel, counts, domain, estimate, validate};

/// The estimates for one level of [`by`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct DcdhSubgroup {
    /// The subgroup code.
    pub level: u32,
    /// The estimates on its units.
    pub result: Result<Dcdh, Error>,
}

/// The estimates for one treatment path of [`by_path`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct DcdhPathResult {
    /// The treatments at `F_g - 1, ..., F_g - 1 + L`.
    pub path: Vec<f64>,
    /// The switchers on the path in the whole panel.
    pub n_units: usize,
    /// The estimates from those switchers and the not-yet switchers with
    /// the same baseline treatment.
    pub result: Result<Dcdh, Error>,
}

/// The estimator on the units of each subgroup of the panel
/// ([`SwitchingPanelBuilder::subgroups`](super::SwitchingPanelBuilder::subgroups)),
/// `by` of `did_multiplegt_dyn`, in increasing order of the subgroup codes.
///
/// # Errors
///
/// [`Error::Unsupported`] for a panel without subgroups, and the errors of
/// [`estimate`] for invalid options; the estimates of each subgroup carry
/// their own.
pub fn by(data: &SwitchingPanel<'_>, options: &DcdhOptions) -> Result<Vec<DcdhSubgroup>, Error> {
    validate(options)?;
    let Some(subgroups) = data.subgroups() else {
        return Err(Error::Unsupported {
            what: "by on a panel without subgroups",
        });
    };
    let levels: BTreeSet<u32> = subgroups.iter().copied().collect();
    Ok(levels
        .into_iter()
        .map(|level| {
            let rows: Vec<usize> = (0..data.len()).filter(|&i| subgroups[i] == level).collect();
            let columns = data.select(&rows, None);
            DcdhSubgroup {
                level,
                result: columns.panel().and_then(|panel| estimate(&panel, options)),
            }
        })
        .collect())
}

/// The estimator separately for the most common treatment paths of the
/// switchers (`by_path`): `paths` of them, all with `None`, in the order of
/// [`design`](super::design). Each path is estimated from its switchers
/// and, as controls, the other units with the same baseline treatment
/// before their first switch, with `same_switchers`.
///
/// # Errors
///
/// [`Error::Input`] for `paths` of zero, [`Error::Unsupported`] with
/// `continuous`, and the errors of [`estimate`] on the whole panel; the
/// estimates of each path carry their own.
///
/// # Panics
///
/// Only when a number of periods does not fit in `i64`.
pub fn by_path(
    data: &SwitchingPanel<'_>,
    options: &DcdhOptions,
    paths: Option<usize>,
) -> Result<Vec<DcdhPathResult>, Error> {
    validate(options)?;
    if paths == Some(0) {
        return Err(domain("paths", 0.0, "1 or more"));
    }
    if options.continuous.is_some() {
        return Err(Error::Unsupported {
            what: "by_path with continuous",
        });
    }
    let panel = prepare(data, options)?;
    let l = counts(&panel, options)?.l;
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
    let sub_options = DcdhOptions {
        same_switchers: true,
        ..*options
    };
    Ok(ranked
        .into_iter()
        .take(k)
        .map(|(path, members)| {
            let on_path: BTreeSet<usize> = members.into_iter().collect();
            let baseline = path.treatments[0];
            let rows: Vec<usize> = (0..data.len())
                .filter(|&i| {
                    grid_of.get(&i64::from(data.unit()[i])).is_some_and(|&g| {
                        on_path.contains(&g)
                            || (panel.d_sq[g] == baseline
                                && rank_of(data.period()[i]) < panel.f_g[g])
                    })
                })
                .collect();
            let columns = data.select(&rows, None);
            DcdhPathResult {
                n_units: path.n_groups,
                path: path.treatments,
                result: columns
                    .panel()
                    .and_then(|panel| estimate(&panel, &sub_options)),
            }
        })
        .collect())
}

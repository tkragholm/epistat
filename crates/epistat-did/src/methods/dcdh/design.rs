//! The descriptive outputs of `did_multiplegt_dyn`: the switchers' treatment
//! paths (`design`) and the dates of their first switch
//! (`date_first_switch`).

use std::collections::BTreeMap;

use super::panel::{Panel, ordered, prepare};
use crate::error::Error;

use super::{DcdhOptions, SwitchingPanel, counts, domain, validate};

/// One treatment path of [`design`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct DcdhPath {
    /// The treatments at `F_g - 1, F_g, ..., F_g - 1 + L`.
    pub treatments: Vec<f64>,
    /// The number of switchers on the path.
    pub n_groups: usize,
    /// Their share of the switchers, weighted by the sum of each switcher's
    /// weights over those periods.
    pub share: f64,
}

/// The result of [`design`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct DcdhDesign {
    /// `L`, the number of effects estimated.
    pub effects: usize,
    /// The most common paths, by decreasing number of switchers, then by
    /// the treatments in increasing lexicographic order.
    pub paths: Vec<DcdhPath>,
    /// The switchers observed over all of `F_g - 1..=F_g - 1 + L`.
    pub n_switchers: usize,
    /// The share of those switchers on the paths listed.
    pub coverage: f64,
}

/// One date of [`first_switch_dates`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct DcdhSwitchDate {
    /// The period, as in the input.
    pub period: i64,
    /// The number of switchers whose treatment first changes then.
    pub n_groups: usize,
    /// Their share of the switchers (of the baseline treatment).
    pub share: f64,
}

/// The first-switch dates of the switchers of one baseline treatment, or of
/// all switchers.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct DcdhSwitchDates {
    /// The baseline treatment, `None` for all switchers together.
    pub baseline: Option<f64>,
    /// The dates in increasing order.
    pub dates: Vec<DcdhSwitchDate>,
}

/// Each switcher's path over `F_g - 1..=F_g - 1 + l` with the sum of its
/// weights there, for the switchers observed in all of those periods.
pub(super) fn group_paths(panel: &Panel, l: usize) -> Vec<(usize, Vec<f64>, f64)> {
    let nt = panel.nt;
    let mut out = Vec::new();
    for g in 0..panel.n_grid_groups() {
        let Ok(first) = usize::try_from(panel.f_g[g] - 1 - panel.t_min) else {
            continue;
        };
        if first + l >= nt {
            continue;
        }
        let cells = g * nt + first..=g * nt + first + l;
        let path: Vec<f64> = panel.d[cells.clone()].to_vec();
        if path.iter().any(|v| v.is_nan()) {
            continue;
        }
        let weight: f64 = panel.weight[cells].iter().filter(|w| !w.is_nan()).sum();
        out.push((g, path, weight));
    }
    out
}

/// All paths with their switchers, in the order of [`DcdhDesign::paths`].
pub(super) fn ranked_paths(panel: &Panel, l: usize) -> (Vec<(DcdhPath, Vec<usize>)>, usize) {
    let groups = group_paths(panel, l);
    let total_weight: f64 = groups.iter().map(|(_, _, w)| w).sum();
    let mut by_path: BTreeMap<Vec<i64>, (Vec<f64>, Vec<usize>, f64)> = BTreeMap::new();
    for (g, path, w) in &groups {
        let key: Vec<i64> = path.iter().map(|&v| ordered(v)).collect();
        let e = by_path
            .entry(key)
            .or_insert_with(|| (path.clone(), Vec::new(), 0.0));
        e.1.push(*g);
        e.2 += w / total_weight;
    }
    // The map is in increasing lexicographic order of the treatments; a
    // stable sort by the number of switchers keeps it among ties.
    let mut paths: Vec<(DcdhPath, Vec<usize>)> = by_path
        .into_values()
        .map(|(treatments, members, share)| {
            (
                DcdhPath {
                    treatments,
                    n_groups: members.len(),
                    share,
                },
                members,
            )
        })
        .collect();
    paths.sort_by_key(|p| std::cmp::Reverse(p.0.n_groups));
    (paths, groups.len())
}

/// The treatment paths of the switchers from the period before their first
/// switch to the last effect estimated (`design`).
///
/// A switcher enters when its treatment is observed in all of
/// `F_g - 1..=F_g - 1 + L`, with the weight of the sum of its weights there.
/// The paths are listed by decreasing number of switchers until their
/// weighted share reaches `coverage`, and the first path past it is listed
/// too, as in `did_multiplegt_dyn_design`.
///
/// # Errors
///
/// The errors of [`estimate`](super::estimate), [`Error::Input`] for a
/// `coverage` outside `(0, 1]`, and [`Error::Unsupported`] with
/// `continuous`.
pub fn design(
    data: &SwitchingPanel<'_>,
    options: &DcdhOptions,
    coverage: f64,
) -> Result<DcdhDesign, Error> {
    validate(options)?;
    if !(coverage > 0.0 && coverage <= 1.0) {
        return Err(domain("coverage", coverage, "(0, 1]"));
    }
    if options.continuous.is_some() {
        return Err(Error::Unsupported {
            what: "the design with continuous",
        });
    }
    let panel = prepare(data, options)?;
    let l = counts(&panel, options)?.l;
    let (ranked, n_switchers) = ranked_paths(&panel, l);
    let mut paths = Vec::new();
    let mut cumulative = 0.0;
    let mut reached = 1.0;
    for (path, _) in ranked {
        cumulative += path.share;
        let within = cumulative <= coverage;
        paths.push(path);
        if !within {
            reached = cumulative;
            break;
        }
    }
    Ok(DcdhDesign {
        effects: l,
        paths,
        n_switchers,
        coverage: if coverage < 1.0 { reached } else { 1.0 },
    })
}

/// The periods in which the switchers' treatment first changes
/// (`date_first_switch`), for all switchers or by baseline treatment.
///
/// # Errors
///
/// The errors of [`estimate`](super::estimate).
///
/// # Panics
///
/// Only when a number of periods does not fit in `usize`.
pub fn first_switch_dates(
    data: &SwitchingPanel<'_>,
    options: &DcdhOptions,
    by_baseline: bool,
) -> Result<Vec<DcdhSwitchDates>, Error> {
    validate(options)?;
    let panel = prepare(data, options)?;
    let never = panel.t_max + 1;
    let mut tables: BTreeMap<Option<i64>, (f64, BTreeMap<i64, usize>)> = BTreeMap::new();
    for g in 0..panel.n_grid_groups() {
        let f = panel.f_g[g];
        if f >= never {
            continue;
        }
        let base = panel.base_value[g];
        let key = by_baseline.then(|| ordered(base));
        let e = tables.entry(key).or_insert_with(|| (base, BTreeMap::new()));
        *e.1.entry(f).or_insert(0) += 1;
    }
    Ok(tables
        .into_values()
        .map(|(base, dates)| {
            let total: usize = dates.values().sum();
            DcdhSwitchDates {
                baseline: by_baseline.then_some(base),
                dates: dates
                    .into_iter()
                    .map(|(f, n)| DcdhSwitchDate {
                        period: panel.times[usize::try_from(f - 1).expect("rank from 1")],
                        n_groups: n,
                        share: n as f64 / total as f64,
                    })
                    .collect(),
            }
        })
        .collect())
}

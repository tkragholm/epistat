//! The bootstrap of `did_multiplegt_dyn` (`bootstrap`): groups, or clusters,
//! drawn with replacement, each entering with its number of draws as a
//! weight.

use rayon::prelude::*;

use epistat_core::{Bootstrap, Estimate};
use epistat_rng::{replicate_seed, resample_counts};

use super::{Dcdh, DcdhOptions, SwitchingPanel, analytic, normal_estimate, z_of};

/// One estimate with its bootstrap standard error.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct DcdhBootstrapTerm {
    /// `l` for effect `l`, `-l` for placebo `l`, 0 for the average.
    pub relative_period: i64,
    /// The estimate on the data, with the standard deviation of the
    /// replicates' estimates (`B - 1` in the denominator, over the
    /// replicates that have one, `None` with fewer than two) as its standard
    /// error, the normal interval at the options' level and the normal
    /// p-value; `None` when the data give no estimate.
    pub estimate: Option<Estimate>,
    /// The replicates that have an estimate.
    pub replicates: usize,
}

/// The bootstrap inference of [`Dcdh::bootstrap`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct DcdhBootstrap {
    /// The effects with bootstrap inference.
    pub effects: Vec<DcdhBootstrapTerm>,
    /// The placebos.
    pub placebos: Vec<DcdhBootstrapTerm>,
    /// The average total effect.
    pub average_total_effect: DcdhBootstrapTerm,
    /// The estimates of each replicate: the effects, the placebos and the
    /// average, `None` where a replicate has none.
    pub draws: Vec<Vec<Option<f64>>>,
}

/// The bootstrap of `analytic`, the estimates on `data`.
///
/// Replicate `r` draws the units (with clusters, the clusters) of the panel
/// with replacement, as many as there are, from the stream seeded with
/// `epistat_rng::replicate_seed(draws.seed, r)`; a unit drawn `m` times
/// enters once with its weights times `m`, which is what the package does,
/// and one drawn no time leaves. The replicates run in parallel, and each
/// depends only on the seed and `r`, so the result does not depend on the
/// number of threads. A replicate in which an estimate cannot be computed
/// has none for it. Each thread holds one replicate's panel, so the memory
/// is that of one estimation times the threads of the rayon pool it runs
/// in; a smaller pool bounds it.
pub(super) fn run(
    data: &SwitchingPanel<'_>,
    options: &DcdhOptions,
    analytic_fit: &Dcdh,
    draws: Bootstrap,
) -> DcdhBootstrap {
    let n = data.len();
    let unit_of: Vec<u32> = (0..n)
        .map(|i| data.clusters().map_or(data.unit()[i], |c| c[i]))
        .collect();
    let mut units = unit_of.clone();
    units.sort_unstable();
    units.dedup();
    let index: Vec<usize> = unit_of
        .iter()
        .map(|u| units.partition_point(|v| v < u))
        .collect();
    let inner = DcdhOptions {
        bootstrap: None,
        ..*options
    };
    let (l, l_pl) = (analytic_fit.effects.len(), analytic_fit.placebos.len());
    let replicates: Vec<Vec<Option<f64>>> = (0..draws.replicates)
        .into_par_iter()
        .map(|r| {
            let counts = resample_counts(units.len(), replicate_seed(draws.seed, r as u64));
            let rows: Vec<usize> = (0..n).filter(|&i| counts[index[i]] > 0).collect();
            let scale: Vec<f64> = rows.iter().map(|&i| f64::from(counts[index[i]])).collect();
            let columns = data.select(&rows, Some(&scale));
            let fit = columns
                .panel()
                .ok()
                .and_then(|panel| analytic(&panel, &inner).ok());
            let pick = |v: Option<&Vec<super::DcdhTerm>>, i: usize| {
                v.and_then(|v| v.get(i))
                    .and_then(|e| e.estimate)
                    .map(|e| e.estimate)
            };
            (0..l)
                .map(|i| pick(fit.as_ref().map(|f| &f.effects), i))
                .chain((0..l_pl).map(|i| pick(fit.as_ref().map(|f| &f.placebos), i)))
                .chain(std::iter::once(fit.as_ref().and_then(|f| {
                    f.average_total_effect.estimate.map(|e| e.estimate)
                })))
                .collect()
        })
        .collect();
    let z = z_of(options.level);
    let summary = |column: usize, term: &super::DcdhTerm| {
        let values: Vec<f64> = replicates.iter().filter_map(|d| d[column]).collect();
        let n = values.len();
        let se = (n >= 2).then(|| {
            let mean = values.iter().sum::<f64>() / n as f64;
            let ss: f64 = values.iter().map(|v| (v - mean) * (v - mean)).sum();
            (ss / (n - 1) as f64).sqrt()
        });
        DcdhBootstrapTerm {
            relative_period: term.relative_period,
            estimate: term
                .estimate
                .map(|e| normal_estimate(e.estimate, se, z, options.level)),
            replicates: n,
        }
    };
    let effects = analytic_fit
        .effects
        .iter()
        .enumerate()
        .map(|(i, t)| summary(i, t))
        .collect();
    let placebos = analytic_fit
        .placebos
        .iter()
        .enumerate()
        .map(|(i, t)| summary(l + i, t))
        .collect();
    let average_total_effect = summary(l + l_pl, &analytic_fit.average_total_effect);
    DcdhBootstrap {
        effects,
        placebos,
        average_total_effect,
        draws: replicates,
    }
}

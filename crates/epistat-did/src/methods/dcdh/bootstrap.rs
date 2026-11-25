//! The bootstrap of `did_multiplegt_dyn` (`bootstrap`): groups, or clusters,
//! drawn with replacement, each entering with its number of draws as a
//! weight.

use rayon::prelude::*;

use epistat_rng::{replicate_seed, resample_counts};

use crate::inference::z_score_for_confidence;

use super::{DcdhConfig, DcdhError, DcdhObservation, DcdhResult, estimate_dcdh, validate};

/// One estimate with its bootstrap standard error.
#[derive(Debug, Clone, PartialEq)]
pub struct DcdhBootstrapEstimate {
    /// `l` for effect `l`, `-l` for placebo `l`, 0 for the average.
    pub relative_period: i64,
    /// The estimate on the data.
    pub estimate: Option<f64>,
    /// The standard deviation of the replicates' estimates, with `B - 1` in
    /// the denominator, over the replicates that have one; `None` with fewer
    /// than two.
    pub se: Option<f64>,
    /// `estimate +- z se`.
    pub ci: Option<(f64, f64)>,
    /// The replicates that have an estimate.
    pub replicates: usize,
}

/// The result of [`estimate_dcdh_bootstrap`].
#[derive(Debug, Clone, PartialEq)]
pub struct DcdhBootstrap {
    /// The estimates on the data, with the analytic inference.
    pub analytic: DcdhResult,
    /// The effects with bootstrap inference.
    pub effects: Vec<DcdhBootstrapEstimate>,
    /// The placebos.
    pub placebos: Vec<DcdhBootstrapEstimate>,
    /// The average total effect.
    pub average_total_effect: DcdhBootstrapEstimate,
    /// The estimates of each replicate: the effects, the placebos and the
    /// average, `None` where a replicate has none.
    pub draws: Vec<Vec<Option<f64>>>,
}

/// The estimates of `did_multiplegt_dyn` with bootstrap standard errors.
///
/// Replicate `r` draws the groups (with clusters, the clusters) of the input
/// with replacement, as many as there are, from the stream seeded with
/// `epistat_rng::replicate_seed(seed, r)`; a group drawn `m` times enters
/// once with its weights times `m`, which is what the package does, and
/// one drawn no time leaves. The replicates run in parallel, and each
/// depends only on `seed` and `r`, so the result does not depend on the
/// number of threads. A replicate in which an estimate cannot be computed
/// has none for it. Each thread holds one replicate's panel, so the memory
/// is that of one estimation times the threads of the rayon pool it runs
/// in; a smaller pool bounds it.
///
/// # Errors
/// [`DcdhError`] as [`estimate_dcdh`] on the data, or for fewer than two
/// replications.
///
/// # Panics
/// Only on more than `u32::MAX` resampling units.
pub fn estimate_dcdh_bootstrap(
    observations: &[DcdhObservation],
    config: &DcdhConfig,
    replications: usize,
    seed: u64,
) -> Result<DcdhBootstrap, DcdhError> {
    validate(config)?;
    if replications < 2 {
        return Err(DcdhError::InvalidConfig(
            "the bootstrap needs at least two replications",
        ));
    }
    let analytic = estimate_dcdh(observations, config)?;
    let unit_of: Vec<i64> = observations
        .iter()
        .map(|o| o.cluster.unwrap_or(o.group))
        .collect();
    let mut units = unit_of.clone();
    units.sort_unstable();
    units.dedup();
    let index: Vec<usize> = unit_of
        .iter()
        .map(|u| units.binary_search(u).expect("a unit"))
        .collect();
    let (l, l_pl) = (analytic.effects.len(), analytic.placebos.len());
    let draws: Vec<Vec<Option<f64>>> = (0..replications)
        .into_par_iter()
        .map(|r| {
            let counts = resample_counts(units.len(), replicate_seed(seed, r as u64));
            let rows: Vec<DcdhObservation> = observations
                .iter()
                .zip(&index)
                .filter(|(_, u)| counts[**u] > 0)
                .map(|(o, u)| DcdhObservation {
                    weight: o.weight * f64::from(counts[*u]),
                    ..o.clone()
                })
                .collect();
            let fit = estimate_dcdh(&rows, config).ok();
            let pick = |v: Option<&Vec<super::DcdhEstimate>>, i: usize| {
                v.and_then(|v| v.get(i)).and_then(|e| e.estimate)
            };
            (0..l)
                .map(|i| pick(fit.as_ref().map(|f| &f.effects), i))
                .chain((0..l_pl).map(|i| pick(fit.as_ref().map(|f| &f.placebos), i)))
                .chain(std::iter::once(
                    fit.as_ref().and_then(|f| f.average_total_effect.estimate),
                ))
                .collect()
        })
        .collect();
    let z = z_score_for_confidence(config.confidence_level);
    let summary = |column: usize, period: i64, estimate: Option<f64>| {
        let values: Vec<f64> = draws.iter().filter_map(|d| d[column]).collect();
        let n = values.len();
        let se = (n >= 2).then(|| {
            let mean = values.iter().sum::<f64>() / n as f64;
            let ss: f64 = values.iter().map(|v| (v - mean) * (v - mean)).sum();
            (ss / (n - 1) as f64).sqrt()
        });
        DcdhBootstrapEstimate {
            relative_period: period,
            estimate,
            se: estimate.and(se),
            ci: estimate.zip(se).map(|(e, s)| (e - z * s, e + z * s)),
            replicates: n,
        }
    };
    let effects = (0..l)
        .map(|i| {
            summary(
                i,
                analytic.effects[i].relative_period,
                analytic.effects[i].estimate,
            )
        })
        .collect();
    let placebos = (0..l_pl)
        .map(|i| {
            summary(
                l + i,
                analytic.placebos[i].relative_period,
                analytic.placebos[i].estimate,
            )
        })
        .collect();
    let average_total_effect = summary(l + l_pl, 0, analytic.average_total_effect.estimate);
    Ok(DcdhBootstrap {
        analytic,
        effects,
        placebos,
        average_total_effect,
        draws,
    })
}

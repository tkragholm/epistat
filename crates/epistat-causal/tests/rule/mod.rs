//! The ratio of the estimators' variance rule to the rule of the hand-written
//! R estimators, for the R comparisons.
//!
//! The R estimators report the variance `sum D^2` of the influences `D` on the
//! dfbeta scale, and with `G` clusters `G / (G - 1) sum S_g^2` of the cluster
//! sums. The estimators centre the influences and apply `G / (G - 1)` with
//! every person its own cluster when there are none. [`rule_ratio`] evaluates
//! both on the estimator's own influence function. Both rules are smooth in
//! the influences, which agree with R's to the tests' tolerances, so the ratio
//! adds an error of the second order of those tolerances.

use std::collections::BTreeMap;

use epistat_causal::{Clusters, Influence};

/// The estimator's variance of the combination `w` of the columns of
/// `influence` over the R rule's, with the cluster of each person (by row)
/// `cluster`.
pub fn rule_ratio(influence: &Influence, w: [f64; 2], cluster: Option<&[u32]>) -> f64 {
    let combined = influence.combine(&w).unwrap();
    let units = combined.units();
    let m = units.len() as f64;
    let d: Vec<f64> = combined.column(0).iter().map(|psi| psi / m).collect();
    let (old, clusters) = match cluster {
        None => (d.iter().map(|x| x * x).sum::<f64>(), None),
        Some(c) => {
            let mut sums = BTreeMap::<u32, f64>::new();
            for (&unit, x) in units.iter().zip(&d) {
                *sums.entry(c[unit as usize]).or_default() += x;
            }
            let g = sums.len() as f64;
            let all: Vec<u32> = (0..u32::try_from(c.len()).unwrap()).collect();
            (
                g / (g - 1.0) * sums.values().map(|s| s * s).sum::<f64>(),
                Some(Clusters::per_unit(&all, c).unwrap()),
            )
        }
    };
    let new = combined
        .variance(clusters.as_ref())
        .unwrap()
        .get(0, 0)
        .unwrap();
    new / old
}

/// The combinations of the two means whose standard errors the estimators
/// report: each mean, the difference and the ratio, by the delta method.
pub fn combinations(mean0: f64, ratio: f64) -> [[f64; 2]; 4] {
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, -1.0],
        [1.0 / mean0, -ratio / mean0],
    ]
}

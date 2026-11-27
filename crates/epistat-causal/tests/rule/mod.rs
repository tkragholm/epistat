//! The conversion of the variances of the estimators written out by hand in R
//! to the estimators' rule, for the R comparisons.
//!
//! The R estimators report the variance of the influences `D_i` on the dfbeta
//! scale over the `m` persons used without centring them,
//!
//! ```text
//! V_R = c_R sum_g S_g^2,       S_g = sum_{i in g} D_i,
//! ```
//!
//! every person its own cluster without clusters, with `c_R = G / (G - 1)` for
//! `G` clusters and `c_R = 1` without. The estimators centre the influences,
//!
//! ```text
//! V_E = c_E sum_g (S_g - n_g D_bar)^2,       D_bar = sum_i D_i / m,
//! ```
//!
//! with the factor `c_E` of the influence function's [`ClusterRule`]: `G / (G -
//! 1)` under `SmallSample` (`m / (m - 1)` without clusters) and 1 under
//! `NoSmallSample`. So `V_E = V_R (c_E / c_R) C` exactly, with the centring
//! ratio
//!
//! ```text
//! C = sum_g (S_g - n_g D_bar)^2 / sum_g S_g^2
//!   = 1 - (2 D_bar sum_g n_g S_g - D_bar^2 sum_g n_g^2) / sum_g S_g^2,
//! ```
//!
//! which is 1 up to the square of the rounding of `D_bar` when the influence
//! function has mean 0 and below 1 otherwise: on the one-step path of TMLE,
//! whose influence function has the mean `Tmle::score`, and in the survival
//! TMLE, where it has the mean of the targeting score. [`conversion`]
//! evaluates `(c_E / c_R) C` on the estimator's own influence function. `C`
//! is smooth in the influences, which agree with R's to the tests'
//! tolerances, so the conversion adds an error of the second order of those
//! tolerances.

use std::collections::BTreeMap;

use epistat_causal::{ClusterRule, Influence};

/// `V_E / V_R` for the combination `w` of the columns of `influence`, with
/// the cluster of each person (by row) `cluster`.
pub fn conversion(influence: &Influence, w: [f64; 2], cluster: Option<&[u32]>) -> f64 {
    let combined = influence.combine(&w).unwrap();
    let units = combined.units();
    let m = units.len() as f64;
    let d: Vec<f64> = combined.column(0).iter().map(|psi| psi / m).collect();
    let mean = d.iter().sum::<f64>() / m;
    // The sum `S_g` and the size `n_g` of each cluster of the persons used.
    let mut sums = BTreeMap::<u32, (f64, f64)>::new();
    for (k, (&unit, x)) in units.iter().zip(&d).enumerate() {
        let g = match cluster {
            Some(c) => c[unit as usize],
            None => u32::try_from(k).unwrap(),
        };
        let e = sums.entry(g).or_default();
        e.0 += x;
        e.1 += 1.0;
    }
    let groups = sums.len() as f64;
    let small_sample = groups / (groups - 1.0);
    let c_r = if cluster.is_some() { small_sample } else { 1.0 };
    let c_e = match influence.rule() {
        ClusterRule::SmallSample => small_sample,
        _ => 1.0,
    };
    let centred: f64 = sums.values().map(|(s, n)| (s - n * mean).powi(2)).sum();
    let plain: f64 = sums.values().map(|(s, _)| s * s).sum();
    c_e / c_r * centred / plain
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

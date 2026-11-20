//! The infinitesimal jackknife of the cumulative hazard, in closed form.
//!
//! # Algorithm
//!
//! With the exit times `t_1 < t_2 < ...` of a group, `Y_l` the weight at risk
//! and `n_l` the weight of the events counted at `t_l` (every event for the
//! hazard of any event, those of one cause for the hazard of a cause), the
//! Nelson-Aalen estimate is
//!
//! ```text
//! H(t) = sum over t_l <= t of n_l / Y_l.
//! ```
//!
//! Weight row `i`, at risk on `(s_i, u_i]`, by `1 + e` and differentiate at
//! `e = 0`. The numerator `n_l` moves with the weight of a row that has the event
//! at `t_l`, and `Y_l` with the weight of a row at risk at it, so
//!
//! ```text
//! dH(t)/dw_i = 1(i has the counted event, u_i <= t) / Y(u_i)
//!              - sum over t_l in (s_i, min(u_i, t)] of n_l / Y_l^2.
//! ```
//!
//! The influence of a row is its weight times this, and that of a cluster the
//! sum over its rows \[1\]\[2\]. The sum over exit times is a difference of prefix
//! sums, so a time costs a pass over the exit times and one over the rows.
//!
//! # References
//!
//! 1. Therneau T. M., Grambsch P. M. (2000) Modeling Survival Data, Springer,
//!    section 2.9.
//! 2. Therneau T. M. (2024) Methods of the survfit function, vignette `methods`
//!    of the survival package, on residuals of survival curves.

use rayon::prelude::*;

use super::table::{Group, Reading};

/// The influence of every cluster of the group on the cumulative hazard at the
/// times in `reading`, row-major by cluster.
pub(super) fn run(group: &Group, reading: &Reading<'_>) -> Vec<f64> {
    let m = group.times.len();
    let k = reading.at.len();
    let n = group.n_clusters();
    let cause = reading.target.cause();
    let counted = if cause.is_some() {
        &group.cause_events
    } else {
        &group.events
    };
    // `n_l / Y_l^2` and its prefix sums, 0 at a time with no counted event.
    let mut prefix = vec![0.0f64; m + 1];
    let mut inverse = vec![0.0f64; m];
    for l in 0..m {
        let (d, y) = (counted[l], group.risk[l]);
        let q = if d > 0.0 { d / (y * y) } else { 0.0 };
        prefix[l + 1] = prefix[l] + q;
        inverse[l] = if y > 0.0 { 1.0 / y } else { 0.0 };
    }
    let mut out = vec![0.0f64; n * k];
    for (slot, &through) in reading.reads.iter().enumerate() {
        let column: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|c| {
                let mut sum = 0.0f64;
                for row in group.cluster_rows(c) {
                    let (first, last) = (row.first as usize, row.last as usize);
                    let mut value = 0.0f64;
                    if first <= last {
                        value -= prefix[(last + 1).min(through)] - prefix[first.min(through)];
                        let own = row.status > 0 && cause.is_none_or(|c| c == row.status);
                        if own && last < through {
                            value += inverse[last];
                        }
                    }
                    sum += row.weight * value;
                }
                sum
            })
            .collect();
        for (c, v) in column.into_iter().enumerate() {
            out[c * k + slot] = v;
        }
    }
    out
}

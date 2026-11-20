//! The error bound of the infinitesimal jackknife tests.
//!
//! The influence of a cluster `g` obeys `U_g(t) = U_g(t-) (I + H(t)) + Q_g(t)`.
//! The matrix `I + H` has non-negative entries and rows that sum to 1, so
//! multiplying a row vector by it does not increase its absolute sum. The
//! increments `Q_g` are the terms
//!
//! ```text
//! w p_j(t-) / Y_j(t)        for each row of g that moves from j at t,
//! w_gj p_j(t-) n_jk / Y_j^2  for each transition j -> k at t, per weight w_gj of g at risk in j,
//! ```
//!
//! each entering two entries with opposite signs, and `p <= 1`. Let `A_g` be
//! twice the sum of those terms over all times with `p` replaced by 1, and
//! with `2 sum_i w_i / W` added when the initial distribution is estimated.
//! Then `A_g` bounds the absolute sum of every `U_g(t)` and of every `C_g(t)`.
//!
//! Two implementations of the recursion differ from each other by at most
//! `2 kappa eps A_g` in each entry, where `kappa = m (2 S + 5) + 2 n + 8` for
//! `m` times, `S` states and `n` rows: each step does at most `S + 3`
//! operations on an entry, each rounding by `eps` of the entry or of the
//! running absolute sum, which is below `A_g`, over `m` steps. The increments
//! themselves carry the relative error of `p`, which is `m (S + 2) eps` for a
//! product-sum of `m` steps, and of one or two weights at risk `Y`, each a sum
//! of `n` weights and so `n eps`, the quotients adding a few `eps`.
//!
//! A standard error is the Euclidean norm of the entries over the clusters,
//! which moves by at most the Euclidean norm of the differences,
//! `2 kappa eps sqrt(sum_g A_g^2)`. The integral over `[t0, tau]` of a step
//! function adds `T + 2` roundings and multiplies the bound by `tau - t0`.

#![expect(
    clippy::float_cmp,
    reason = "a row leaves at its own stop time, which is one of the times"
)]

/// One row of a process, in a curve.
pub struct Row {
    pub start: f64,
    pub stop: f64,
    pub from: usize,
    /// `None` when the row is censored.
    pub to: Option<usize>,
    pub weight: f64,
    pub cluster: usize,
}

/// `A_g` for every cluster `0..n_clusters`, from the rows, the event times and
/// `opening_weight_share`, `2 w / W` per cluster for an estimated initial
/// distribution or zeros.
pub fn influence_bound(
    rows: &[Row],
    times: &[f64],
    states: usize,
    n_clusters: usize,
    opening: &[f64],
) -> Vec<f64> {
    let mut a = opening.to_vec();
    a.resize(n_clusters, 0.0);
    for &t in times {
        // weight at risk in each state, and moves out of it
        let mut y = vec![0.0f64; states];
        let mut moved = vec![0.0f64; states];
        for r in rows {
            if r.start < t && t <= r.stop {
                y[r.from] += r.weight;
                if r.stop == t && r.to.is_some_and(|to| to != r.from) {
                    moved[r.from] += r.weight;
                }
            }
        }
        for r in rows {
            if r.start < t && t <= r.stop {
                let j = r.from;
                if moved[j] > 0.0 {
                    // the dN term of a row that moves, and the at-risk term
                    if r.stop == t && r.to.is_some_and(|to| to != j) {
                        a[r.cluster] += 2.0 * r.weight / y[j];
                    }
                    a[r.cluster] += 2.0 * r.weight * moved[j] / (y[j] * y[j]);
                }
            }
        }
    }
    a
}

/// `kappa` for `m` times, `s` states and `n` rows.
pub fn kappa(m: usize, s: usize, n: usize) -> f64 {
    (m * (2 * s + 5) + 2 * n + 8) as f64
}

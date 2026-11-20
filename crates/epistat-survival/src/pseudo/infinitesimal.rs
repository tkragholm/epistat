//! The infinitesimal jackknife of the Kaplan-Meier curve and its area, in
//! closed form.
//!
//! # Algorithm
//!
//! Weight row `i` by `1 + e` and differentiate the curve at `e = 0`. With
//! event times `t_1 < t_2 < ...`, `Y_l` the weight at risk and `d_l` the weight of
//! events at `t_l`, `h_l = d_l / Y_l`, the Kaplan-Meier curve `S` has the
//! derivative with respect to the weight of row `i`, which is at risk on
//! `(s_i, u_i]` and has the event `delta_i`,
//!
//! ```text
//! dS(t)/dw_i = S(t) [ A(min(u_i, t)) - A(min(s_i, t)) - delta_i 1(u_i <= t) b(u_i) ],
//! A(x) = sum over event times <= x of  h_l / (Y_l (1 - h_l)),
//! b(u) = 1 / (Y (1 - h)) at the event time u.
//! ```
//!
//! This follows from `log S = sum over l of log (Y_l - d_l) - log Y_l`, whose
//! derivative with respect to the weight of a row at risk at `t_l` is
//! `1 / (Y_l - d_l) - 1 / Y_l` when the row does not have the event there, and
//! `-1 / Y_l` when it does \[1\]. Where `h_l = 1` the curve is zero from `t_l`, and
//! `1 - h_l` is read as 1, as `survival` does. The influence of a row is its weight
//! times the derivative, and that of a cluster the sum over its rows.
//!
//! The restricted mean `R(tau)` is the integral of `S` to `tau`, so its
//! derivative is the integral of the above. Exchanging the sum and the integral,
//! with `Area_l` the area under `S` from `t_l` to `tau`,
//!
//! ```text
//! dR(tau)/dw_i = G(min(i_u, k)) - G(min(i_s, k)) - delta_i 1(i_u <= k) Area_{i_u} c_{i_u},
//! G(j) = sum over l <= j of Area_l h_l c_l,        c_l = 1 / (Y_l (1 - h_l)),
//! ```
//!
//! where `i_u` and `i_s` count the event times not after `u_i` and `s_i`, and
//! `k` those not after `tau`, and `c_l` is 0 where `h_l = 1`, where `Area_l` is
//! 0 as well. A time `tau` costs a pass over the event times and one over the
//! rows, and the rows' positions among the event times are found once.
//!
//! # References
//!
//! 1. Therneau T. M., Grambsch P. M. (2000) Modeling Survival Data, Springer,
//!    section 2.9.
//! 2. Therneau T. M. (2024) Methods of the survfit function, vignette
//!    `methods` of the survival package, on residuals of survival curves.

#![expect(
    clippy::float_cmp,
    reason = "a hazard of exactly 1 is where the curve reaches zero"
)]

use rayon::prelude::*;

use super::Target;
use super::table::{Group, Reading};
use crate::aalen_johansen::advance;

/// The event times of a group and what the derivatives read from them.
struct Events {
    times: Vec<f64>,
    survival: Vec<f64>,
    hazard: Vec<f64>,
    /// `h / (Y (1 - h))`, with `1 - h` read as 1 where `h` is 1.
    a: Vec<f64>,
    /// `1 / (Y (1 - h))`, with `1 - h` read as 1 where `h` is 1.
    b: Vec<f64>,
    /// `1 / (Y (1 - h))`, or 0 where that is infinite.
    c: Vec<f64>,
}

fn events(group: &Group) -> Events {
    let mut e = Events {
        times: Vec::new(),
        survival: Vec::new(),
        hazard: Vec::new(),
        a: Vec::new(),
        b: Vec::new(),
        c: Vec::new(),
    };
    let mut s = 1.0f64;
    for j in 0..group.times.len() {
        let (y, d) = (group.risk[j], group.events[j]);
        if d > 0.0 {
            let h = d / y;
            advance(&mut s, y, d, &[], &mut []);
            let temp = if h == 1.0 { 1.0 } else { 1.0 - h };
            e.times.push(group.times[j]);
            e.survival.push(s);
            e.hazard.push(h);
            e.a.push(h / (y * temp));
            e.b.push(1.0 / (temp * y));
            let c = 1.0 / (y * (1.0 - h));
            e.c.push(if c.is_finite() { c } else { 0.0 });
        }
    }
    e
}

/// The influence of every cluster of the group on the survival probability or
/// the restricted mean at the times in `reading`, row-major by cluster.
pub(super) fn run(group: &Group, reading: &Reading<'_>) -> Vec<f64> {
    let e = events(group);
    let k = reading.at.len();
    let n = group.n_clusters();
    let me = e.times.len();
    // The position of each row among the event times.
    let positions: Vec<(usize, usize)> = group
        .rows
        .iter()
        .map(|row| {
            (
                e.times.partition_point(|&t| t <= row.stop),
                e.times.partition_point(|&t| t <= row.start),
            )
        })
        .collect();
    let mut out = vec![0.0f64; n * k];
    let mut prefix = vec![0.0f64; me + 1];
    let mut area = vec![0.0f64; me];
    for (slot, &tau) in reading.at.iter().enumerate() {
        let through = e.times.partition_point(|&t| t <= tau);
        // The value that multiplies the prefix sums, and the term of a row's
        // own event, by event time.
        let (scale, own): (f64, Vec<f64>) = if reading.target == Target::Survival {
            prefix[0] = 0.0;
            for l in 0..me {
                prefix[l + 1] = prefix[l] + e.a[l];
            }
            let scale = if through == 0 {
                1.0
            } else {
                e.survival[through - 1]
            };
            (scale, e.b.clone())
        } else {
            if through > 0 {
                area[through - 1] = e.survival[through - 1] * (tau - e.times[through - 1]);
                for l in (0..through - 1).rev() {
                    area[l] = area[l + 1] + e.survival[l] * (e.times[l + 1] - e.times[l]);
                }
            }
            prefix[0] = 0.0;
            let mut weights = vec![0.0f64; me];
            for l in 0..me {
                let a = if l < through { area[l] } else { 0.0 };
                prefix[l + 1] = prefix[l] + a * e.hazard[l] * e.c[l];
                weights[l] = a * e.c[l];
            }
            (1.0, weights)
        };
        let column: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|c| {
                let mut sum = 0.0f64;
                let range = group.offsets[c]..group.offsets[c + 1];
                for (row, &(u, s)) in group.rows[range.clone()].iter().zip(&positions[range]) {
                    let mut value = prefix[u.min(through)] - prefix[s.min(through)];
                    if row.status > 0 && u <= through {
                        value -= own[u - 1];
                    }
                    sum += row.weight * (scale * value);
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

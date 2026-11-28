//! The log-rank type test of Grafféo et al. that the net survival is the same
//! in several groups.

#![expect(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "times are compared with the values they were copied from, counts of rows are far below 2^53, and the number of grid steps is a floor of a non-negative quotient"
)]

use epistat_core::{Covariance, check};
use epistat_linalg::cholesky::{Pivot, cholesky_lower};
use epistat_stats::dist::chi_square::sf as chi_square_sf;

use super::ExpectedHazard;
use super::cohort::{SWEEP_LIMIT, rows_by_stratum};
use crate::covariance;
use crate::error::{Error, Result};
use crate::par;
use crate::surv::SurvData;

const BLOCK: usize = 1024;
const WAVE: usize = 64;

/// The test of equal net survival in several groups.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct NetSurvivalTest {
    /// The chi-square statistic `z' V^-1 z` on the first `G - 1` groups.
    pub statistic: f64,
    /// The degrees of freedom, the number of groups less one.
    pub df: usize,
    /// The upper tail probability.
    pub p_value: f64,
    /// The weighted observed minus expected excess deaths of every group,
    /// summed over the strata; they sum to 0.
    pub z: Vec<f64>,
    /// The covariance of `z` over all `G` groups, `G` by `G`.
    pub covariance: Covariance,
}

/// The test of Grafféo, Castell, Belot and Giorgi that the net survival of
/// `groups` (codes `0..G`, one per row) is the same, stratified by the strata
/// of the data, as `relsurv::rs.diff`.
///
/// Every row is weighted by the inverse of its expected survival since the
/// origin, `w_i(t) = 1 / S*_i(t)`, as in the Pohar Perme estimator. On a grid
/// of points `t_j`, with `Y_g^w(t_j)` the weighted number at risk in group `g`
/// at `t_j`, `dN_g^w` the weighted deaths at `t_j`, `dE_g^w` the weighted
/// expected deaths over `(t_{j-1}, t_j]`, `sum_i w_i(t_j) (H*_i(t_j) -
/// H*_i(t_{j-1}))` over those at risk, and `p_g = Y_g^w / Y^w`,
///
/// ```text
/// z_g     = sum_s sum_j [ (dN_g^w - dE_g^w) - p_g (dN^w - dE^w) ]
/// V_gh    = sum_s sum_j sum_k (delta_gk - p_g) (delta_hk - p_h) dN_k^{w^2}
/// ```
///
/// with `dN_k^{w^2}` the deaths of group `k` weighted by `w^2`, over the points
/// where someone is at risk. The statistic is `z' V^-1 z` over the first
/// `G - 1` groups, chi-square on `G - 1` degrees of freedom. The integrals of
/// the expected deaths and of `p` are sums on the grid with the weights at
/// the right end of each step, the rule of `rs.diff`, so they approximate the
/// integrals of the paper with an error of the order of the square of the
/// expected hazard over a step. The grid is every distinct exit time of the
/// data, every entry time, and, when `step` is given, its multiples up to the
/// last exit: `rs.diff`'s default `precision = 1` is `grid_step = 1` with the time
/// in days. A row is at risk at `t_j` when it entered before `t_j` and exits
/// at or after it.
///
/// # Scale
///
/// Every row visits the points of its follow-up, in parallel blocks summed in
/// order, so the result does not depend on the number of threads. The memory
/// is the number of points times the number of groups, per block of rows.
///
/// # Errors
///
/// An input error for groups of another length than the data or with fewer
/// than two levels with rows, a status with more than one event code,
/// weights other than 0 and 1, a grid step that is not positive and finite, an
/// entry before time 0, follow-up beyond what `expected` covers, more than
/// 3e8 pairs of a person and a point, and a covariance that is not positive
/// definite, as when a group has no deaths.
pub fn net_survival_test(
    data: &SurvData<'_>,
    groups: &[u32],
    expected: &ExpectedHazard,
    grid_step: Option<f64>,
) -> Result<NetSurvivalTest> {
    check::same_length("groups", data.n_rows(), groups.len())?;
    if let Some(w) = grid_step
        && !(w.is_finite() && w > 0.0)
    {
        return Err(Error::options("the step must be finite and positive"));
    }
    if data.followup()?.n_codes() != 1 {
        return Err(Error::not_applicable(
            "the test needs a binary status of death from any cause",
        ));
    }
    let strata = rows_by_stratum(data, expected)?;
    let surv = data.followup()?;
    let (start, stop, status) = (surv.start(), surv.stop(), surv.status());
    let begin = |i: usize| start.map_or(0.0, |s| s[i]);
    let used: Vec<usize> = strata.iter().flatten().copied().collect();
    let n_groups = used
        .iter()
        .map(|&i| groups[i] as usize + 1)
        .max()
        .unwrap_or(0);
    let mut present = vec![false; n_groups];
    for &i in &used {
        present[groups[i] as usize] = true;
    }
    if present.iter().filter(|&&p| p).count() < 2 || present.iter().any(|&p| !p) {
        return Err(Error::no_information(
            "the groups must be coded 0 to G - 1, each with rows, and G at least 2",
        ));
    }
    let mut points: Vec<f64> = used.iter().map(|&i| stop[i]).collect();
    points.extend(used.iter().map(|&i| begin(i)).filter(|&s| s > 0.0));
    let t_max = points.iter().copied().fold(0.0_f64, f64::max);
    if let Some(w) = grid_step {
        let k_max = (t_max / w).floor() as usize;
        points.extend((1..=k_max).map(|k| k as f64 * w));
    }
    points.sort_by(f64::total_cmp);
    points.dedup();
    let m = points.len();
    let work: f64 = used
        .iter()
        .map(|&i| {
            let lo = points.partition_point(|&p| p <= begin(i));
            let hi = points.partition_point(|&p| p <= stop[i]);
            (hi - lo) as f64
        })
        .sum();
    if work > SWEEP_LIMIT {
        return Err(Error::options(format!(
            "the test would visit {work:.3e} pairs of a person and a point, above the limit of \
             {SWEEP_LIMIT:.0e}: use a coarser step"
        )));
    }
    let g = n_groups;
    let mut z = vec![0.0_f64; g];
    let mut covariance = vec![0.0_f64; g * g];
    for rows in &strata {
        // Per point and group: the weighted number at risk, the weighted
        // deaths less the weighted expected deaths, and the deaths weighted by
        // the square of the weight.
        let sums = par::try_ordered_reduce(
            rows.len(),
            BLOCK,
            WAVE,
            |range| {
                let mut s = vec![[0.0_f64; 3]; m * g];
                let mut reader = expected.reader();
                for &i in &rows[range] {
                    let pieces = reader.row_until(i, expected.horizon(i)).ok_or_else(|| {
                        Error::out_of_range(format!("row {i} has no expected hazard"))
                    })?;
                    let first = points.partition_point(|&p| p <= begin(i));
                    let last = points.partition_point(|&p| p <= stop[i]);
                    let mut h_prev = if first == 0 {
                        0.0
                    } else {
                        pieces.cumulative_within(points[first - 1])
                    };
                    let k = groups[i] as usize;
                    for (j, &t) in points.iter().enumerate().take(last).skip(first) {
                        let h = pieces.cumulative_within(t);
                        let w = h.exp();
                        let cell = &mut s[j * g + k];
                        cell[0] += w;
                        cell[1] -= w * (h - h_prev);
                        if stop[i] == t && status[i] != 0 {
                            cell[1] += w;
                            cell[2] += w * w;
                        }
                        h_prev = h;
                    }
                }
                Ok::<_, Error>(s)
            },
            |total: &mut Vec<[f64; 3]>, part| {
                for (t, p) in total.iter_mut().zip(part) {
                    for c in 0..3 {
                        t[c] += p[c];
                    }
                }
            },
        )?;
        if sums.is_empty() {
            continue;
        }
        for j in 0..m {
            let cells = &sums[j * g..(j + 1) * g];
            let risk: f64 = cells.iter().map(|c| c[0]).sum();
            if risk <= 0.0 {
                continue;
            }
            let event: f64 = cells.iter().map(|c| c[1]).sum();
            for a in 0..g {
                let pa = cells[a][0] / risk;
                z[a] += cells[a][1] - pa * event;
                for b in 0..g {
                    let pb = cells[b][0] / risk;
                    let mut v = 0.0;
                    for (k, cell) in cells.iter().enumerate() {
                        let da = f64::from(u8::from(a == k)) - pa;
                        let db = f64::from(u8::from(b == k)) - pb;
                        v += da * db * cell[2];
                    }
                    covariance[a * g + b] += v;
                }
            }
        }
    }
    let r = g - 1;
    let reduced: Vec<f64> = (0..r * r)
        .map(|e| covariance[(e / r) * g + e % r])
        .collect();
    let lower = cholesky_lower(&reduced, r, Pivot::Strict).ok_or_else(|| {
        Error::not_positive_definite(
            "covariance",
            "the covariance of the test is not positive definite, as when a group has no deaths",
        )
    })?;
    // z' V^-1 z = |L^-1 z|^2 by forward substitution.
    let mut y = vec![0.0_f64; r];
    for a in 0..r {
        let mut v = z[a];
        for b in 0..a {
            v -= lower[a * r + b] * y[b];
        }
        y[a] = v / lower[a * r + a];
    }
    let statistic: f64 = y.iter().map(|v| v * v).sum();
    Ok(NetSurvivalTest {
        statistic,
        df: r,
        p_value: chi_square_sf(statistic, r as f64),
        z,
        covariance: covariance::from_values(g, &covariance)?,
    })
}

//! Two summaries of the switchers' treatment changes: the weights of the
//! normalised effects on the lags of the treatment (`normalized_weights`),
//! and the average number of periods over which a dose of treatment
//! accumulates (`avg_time_periods`).

#![expect(
    clippy::float_cmp,
    reason = "the indicators of the package are exactly 0, 1 or NaN"
)]

use super::core::DirectionTerms;
use super::panel::Panel;
use super::{DcdhAvgTimePeriods, DcdhOptions, DcdhSwitchers, DcdhTerm};

/// `normalized_weights`: the weight of lag `k` in normalised effect `l` is
///
/// ```text
/// sum_g N_{g,F_g-1+l} |D_{g,F_g-1+l-k} - D_g1| / (delta_l N_l),
/// ```
///
/// over the switchers of effect `l`, with `delta_l` its normalisation and
/// `N_l` its weighted number of switchers.
pub(super) fn lag_weights(
    panel: &Panel,
    inn: Option<&DirectionTerms>,
    out: Option<&DirectionTerms>,
    normalization: &[Option<f64>],
    effects: &[DcdhTerm],
) -> Vec<Option<Vec<f64>>> {
    let nt = panel.nt;
    let cells = panel.n_grid_groups() * nt;
    let dist = |d: Option<&DirectionTerms>, i: usize, c: usize| {
        d.and_then(|d| d.effects.get(i))
            .is_some_and(|t| t.dist[c] == 1.0)
    };
    (0..effects.len())
        .map(|i| {
            let delta = normalization
                .get(i)
                .copied()
                .flatten()
                .filter(|d| *d != 0.0 && d.is_finite())?;
            let n = effects[i].n_switchers_weighted;
            let mut sums = vec![0.0; i + 1];
            for c in 0..cells {
                if !(dist(inn, i, c) || dist(out, i, c)) {
                    continue;
                }
                let (g, tp) = (c / nt, c % nt);
                for (k, sum) in sums.iter_mut().enumerate() {
                    let Some(tk) = tp.checked_sub(k) else {
                        continue;
                    };
                    let v = panel.n_gt[c] * (panel.d[g * nt + tk] - panel.d_sq[g]).abs();
                    if !v.is_nan() {
                        *sum += v;
                    }
                }
            }
            Some(sums.iter().map(|s| s / delta / n).collect())
        })
        .collect()
}

/// `avg_time_periods`, as `avg_cumul_cpp` of `DIDmultiplegtDYN` 2.4.0.
///
/// A switcher `g` contributes when its outcome is observed at `F_g - 1`.
/// Its horizon `T'_g` is the last period `P` in `F_g..=T_g` at which its
/// outcome is observed and some group with the same baseline that has not
/// switched by `P` has its outcome observed at `F_g - 1` and at `P`; then
/// `M_g = min(L, T'_g - F_g + 1)`. Each `k < M_g` at which the switcher's
/// outcome and treatment are observed and such a control exists adds
/// `|D_{g,F_g+k} - D_g1| (M_g - k)` to the numerator and
/// `|D_{g,F_g+k} - D_g1|` to the denominator. With `same_switchers` only
/// switchers with `M_g = L` contribute.
pub(super) fn time_periods(panel: &Panel, l: usize, options: &DcdhOptions) -> DcdhAvgTimePeriods {
    let nt = panel.nt;
    let gn = panel.n_grid_groups();
    let observed = |c: usize| !panel.y[c].is_nan();
    // controls[(level * nt + a) * nt + b]: the groups of the level with the
    // outcome observed at periods a < b that have not switched by b.
    let mut controls = vec![0_u32; panel.n_level * nt * nt];
    for g in 0..gn {
        let base = panel.level[g] * nt;
        for a in 0..nt {
            if !observed(g * nt + a) {
                continue;
            }
            for b in a + 1..nt {
                if panel.time_of(b) >= panel.f_g[g] {
                    break;
                }
                if observed(g * nt + b) {
                    controls[(base + a) * nt + b] += 1;
                }
            }
        }
    }
    let never_f = panel.t_max + 1;
    let tp_of = |t: i64| usize::try_from(t - panel.t_min).ok().filter(|&tp| tp < nt);
    let (mut numerator, mut denominator) = (0.0, 0.0);
    let mut switchers = vec![0.0; l];
    for g in 0..gn {
        let f = panel.f_g[g];
        if f >= never_f {
            continue;
        }
        let Some(r) = tp_of(f - 1).filter(|&r| observed(g * nt + r)) else {
            continue;
        };
        let has_control = |p: usize| controls[(panel.level[g] * nt + r) * nt + p] > 0;
        let mut last = f - 1;
        for t in f..=panel.t_g[g] {
            if let Some(p) = tp_of(t)
                && observed(g * nt + p)
                && has_control(p)
            {
                last = t;
            }
        }
        if f > last {
            continue;
        }
        let li = i64::try_from(l).expect("effects fit i64");
        let m = li.min(last - f + 1);
        if options.same_switchers && m < li {
            continue;
        }
        let direction_ok = match options.switchers {
            DcdhSwitchers::Both => true,
            DcdhSwitchers::In => panel.s_g[g] == 1.0,
            DcdhSwitchers::Out => panel.s_g[g] == 0.0,
        };
        if !direction_ok {
            continue;
        }
        for (k, slot) in switchers
            .iter_mut()
            .enumerate()
            .take(usize::try_from(m).expect("m >= 1"))
        {
            let ki = i64::try_from(k).expect("horizon fits i64");
            let Some(p) = tp_of(f + ki) else {
                continue;
            };
            let c = g * nt + p;
            if !observed(c) || panel.d[c].is_nan() || !has_control(p) {
                continue;
            }
            let change = (panel.d[c] - panel.d_sq[g]).abs();
            numerator += change * (m - ki) as f64;
            denominator += change;
            *slot += panel.n_gt[c];
        }
    }
    DcdhAvgTimePeriods {
        average: (denominator != 0.0).then(|| numerator / denominator),
        numerator,
        denominator,
        switchers,
    }
}

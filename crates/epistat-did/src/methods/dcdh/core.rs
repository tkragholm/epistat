//! `did_multiplegt_dyn_core`: the event-study effects and placebos of one
//! direction of switching, with the terms of their variance.

#![expect(
    clippy::float_cmp,
    reason = "the indicators of the package are exactly 0, 1 or NaN, and R compares them and the treatment values exactly"
)]

use super::controls::Controls;
use super::panel::{NA, Panel};

/// The terms of one effect or placebo in one direction.
pub(super) struct Terms {
    /// `N^1_l` or `N^0_l`: the weighted number of switchers.
    pub n: f64,
    /// The same, unweighted.
    pub n_dw: f64,
    /// `U_{G,g,l}` of each grid group, empty when `n` is 0.
    pub u: Vec<f64>,
    /// `U^var_{G,g,l}` of each grid group, empty when `n` is 0.
    pub u_var: Vec<f64>,
    /// The weight of each cell in the count of observations, empty when `n`
    /// is 0.
    pub count: Vec<f64>,
    /// Weighted mean of the switchers' cumulated change in treatment, for the
    /// normalised estimator.
    pub delta_norm: f64,
    /// Whether the cell is a switcher of this effect (1, 0 or NaN).
    pub dist: Vec<f64>,
}

/// Everything one direction contributes.
pub(super) struct DirectionTerms {
    pub effects: Vec<Terms>,
    pub placebos: Vec<Terms>,
    /// `sum_l N_l`.
    pub sum_n: f64,
    /// The average total effect's numerator and variance terms per grid
    /// group, divided by `ate_den`.
    pub ate_u: Vec<f64>,
    pub ate_u_var: Vec<f64>,
    pub ate_den: f64,
    /// Per effect and cell: the switcher's change in treatment at `F_g - 1 +
    /// l` (0 or NaN in other cells), as `delta_D_g_l_XX`.
    pub delta_d_g: Vec<Vec<f64>>,
}

pub(super) struct Options {
    pub only_never_switchers: bool,
    pub normalized: bool,
}

/// Sums of a value over the cells that pass a filter, by a key of the
/// cell, with the count of rows or of distinct clusters.
struct Cohort {
    count: Vec<f64>,
    total: Vec<f64>,
    dof: Vec<f64>,
}

fn cohort_sums(
    panel: &Panel,
    key: &dyn Fn(usize) -> usize,
    n_keys: usize,
    member: &[bool],
    diff: &[f64],
) -> Cohort {
    let nt = panel.nt;
    let mut count = vec![0.0; n_keys];
    let mut total = vec![0.0; n_keys];
    let mut rows = vec![0.0; n_keys];
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for (c, &m) in member.iter().enumerate() {
        if !m {
            continue;
        }
        let k = key(c);
        count[k] += panel.n_gt[c];
        total[k] += panel.n_gt[c] * diff[c];
        rows[k] += 1.0;
        if let Some(cl) = &panel.cluster {
            pairs.push((k, cl[c / nt]));
        }
    }
    let dof = if panel.cluster.is_some() {
        pairs.sort_unstable();
        pairs.dedup();
        let mut distinct = vec![0.0; n_keys];
        for (k, _) in pairs {
            distinct[k] += 1.0;
        }
        distinct
    } else {
        rows
    };
    Cohort { count, total, dof }
}

/// `E_hat_gt` and `DOF_gt` of one cell, the demeaning and the small-sample
/// factor of its variance term, by the rules of `compute_E_hat_gt_polars`
/// and `compute_DOF_gt_polars` of 2.4.0.
fn demeaning(
    t: i64,
    f: i64,
    i: i64,
    ns: Option<(f64, f64)>,
    s: Option<(f64, f64)>,
    nss: Option<(f64, f64)>,
) -> (f64, f64) {
    let switcher = f - 1 + i == t;
    let before = t < f;
    let mut e = NA;
    let mut dof = NA;
    if before || switcher {
        e = 0.0;
        dof = 1.0;
    }
    if before && let Some((m, n)) = ns {
        if n >= 2.0 {
            e = m;
        }
        if n > 1.0 {
            dof = (n / (n - 1.0)).sqrt();
        }
    }
    if switcher && let Some((m, n)) = s {
        if n >= 2.0 {
            e = m;
        }
        if n > 1.0 {
            dof = (n / (n - 1.0)).sqrt();
        }
    }
    if let Some((m, n)) = nss
        && n >= 2.0
        && ((switcher && s.is_some_and(|(_, k)| k == 1.0))
            || (before && ns.is_some_and(|(_, k)| k == 1.0)))
    {
        e = m;
        dof = (n / (n - 1.0)).sqrt();
    }
    (e, dof)
}

/// Inputs shared by an effect and a placebo once the switchers and controls
/// are known.
struct Sample<'a> {
    i: usize,
    /// The long difference, after the controls.
    diff: &'a [f64],
    never: &'a [f64],
    dist: &'a [f64],
    /// Switchers by cell, by period, and controls by cell.
    n_t_g: &'a [f64],
    n_t: &'a [f64],
    n_ctrl: &'a [f64],
    n: f64,
    part2: &'a [f64],
}

/// `U`, `U^var` and the counts of one effect or placebo.
fn u_terms(panel: &Panel, s: &Sample<'_>) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let nt = panel.nt;
    let gn = panel.n_grid_groups();
    let cells = gn * nt;
    let i = i64::try_from(s.i).expect("effect index");
    let cell_of = |c: usize| panel.base_id[c / nt] * nt + c % nt;

    // The cells of the not-yet switchers and of the switchers that enter the
    // demeaning, and their means.
    let not_yet: Vec<bool> = (0..cells)
        .map(|c| {
            panel.n_gt[c] != 0.0 && !s.diff[c].is_nan() && s.never[c] == 1.0 && s.n_t[c % nt] > 0.0
        })
        .collect();
    let switching: Vec<bool> = (0..cells)
        .map(|c| panel.n_gt[c] != 0.0 && s.dist[c] == 1.0)
        .collect();
    let union: Vec<bool> = (0..cells).map(|c| not_yet[c] || switching[c]).collect();
    let n_cells = panel.n_base * nt;
    let ns = cohort_sums(panel, &cell_of, n_cells, &not_yet, s.diff);
    let sw = cohort_sums(
        panel,
        &|c| panel.cohort_id[c / nt],
        panel.n_cohort,
        &switching,
        s.diff,
    );
    let nss = cohort_sums(panel, &cell_of, n_cells, &union, s.diff);
    let pick = |co: &Cohort, k: usize, member: bool| {
        member.then(|| (co.total[k] / co.count[k], co.dof[k]))
    };

    let g_count = panel.n_groups;
    let mut u = vec![0.0; gn];
    let mut u_var = vec![0.0; gn];
    let mut count = vec![0.0; cells];
    for g in 0..gn {
        let (f, t_g) = (panel.f_g[g], panel.t_g[g]);
        let use_l = f64::from(u8::from(i < t_g));
        let (mut su, mut sv) = (0.0, 0.0);
        for tp in 0..nt {
            let c = g * nt + tp;
            let t = panel.time_of(tp);
            let window = f64::from(u8::from(t > i && t <= t_g));
            let cell = cell_of(c);
            let weight = s.dist[c] - (s.n_t_g[cell] / s.n_ctrl[cell]) * s.never[c];
            let coef = use_l * (g_count / s.n) * window * panel.n_gt[c] * weight;
            let term = coef * s.diff[c];
            if !term.is_nan() {
                su += term;
            }
            let counted = (term != 0.0 && !term.is_nan())
                || (term == 0.0
                    && s.diff[c] == 0.0
                    && ((!s.dist[c].is_nan() && s.dist[c] != 0.0)
                        || (s.n_t_g[cell] != 0.0 && !s.never[c].is_nan() && s.never[c] != 0.0)));
            if counted {
                count[c] = panel.n_gt[c];
            }
            let (e, dof) = demeaning(
                t,
                f,
                i,
                pick(&ns, cell, not_yet[c]),
                pick(&sw, panel.cohort_id[g], switching[c]),
                pick(&nss, cell, union[c]),
            );
            let vterm = coef * dof * (s.diff[c] - e);
            if !vterm.is_nan() {
                sv += vterm;
            }
        }
        u[g] = su;
        u_var[g] = sv - s.part2[g];
    }
    (u, u_var, count)
}

/// Switchers per cell, per period and in all, of a switcher indicator.
fn switcher_counts(panel: &Panel, dist: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64, f64) {
    let nt = panel.nt;
    let mut n_t = vec![0.0; nt];
    let mut n_dw_t = vec![0.0; nt];
    let mut n_t_g = vec![0.0; panel.n_base * nt];
    for (c, &v) in dist.iter().enumerate() {
        let w = v * panel.n_gt[c];
        if !w.is_nan() {
            n_t[c % nt] += w;
            n_t_g[panel.base_id[c / nt] * nt + c % nt] += w;
        }
        if !v.is_nan() {
            n_dw_t[c % nt] += v;
        }
    }
    let n: f64 = n_t.iter().sum();
    let n_dw: f64 = n_dw_t.iter().sum();
    (n_t, n_dw_t, n_t_g, n, n_dw)
}

fn controls_sum(panel: &Panel, never: &[f64]) -> Vec<f64> {
    let nt = panel.nt;
    let mut n_ctrl = vec![0.0; panel.n_base * nt];
    for (c, &v) in never.iter().enumerate() {
        let w = v * panel.n_gt[c];
        if !w.is_nan() {
            n_ctrl[panel.base_id[c / nt] * nt + c % nt] += w;
        }
    }
    n_ctrl
}

/// The weighted mean change of treatment from `F_g` to `F_g - 1 + l` of the
/// switchers, for the normalised estimator.
fn delta_norm(panel: &Panel, i: usize, inc: f64, dist: &[f64], n: f64) -> f64 {
    let nt = panel.nt;
    let i = i64::try_from(i).expect("effect index");
    let mut total = 0.0;
    for g in 0..panel.n_grid_groups() {
        if panel.s_g[g] != inc {
            continue;
        }
        let f = panel.f_g[g];
        let mut sum = 0.0;
        for tp in 0..nt {
            let t = panel.time_of(tp);
            let v = panel.d[g * nt + tp] - panel.d_sq[g];
            if t >= f && t <= f - 1 + i && !v.is_nan() {
                sum += v;
            }
        }
        let s = panel.s_g[g];
        let signed = s * sum + (1.0 - s) * (-sum);
        for tp in 0..nt {
            let c = g * nt + tp;
            if dist[c] == 1.0 {
                let v = (panel.n_gt[c] / n) * signed;
                if !v.is_nan() {
                    total += v;
                }
            }
        }
    }
    total
}

/// The effects and placebos of one direction: switchers in (`inc` 1) or out
/// (`inc` 0), `effects` effects and `placebos` placebos.
#[expect(
    clippy::too_many_lines,
    reason = "one pass of did_multiplegt_dyn_core, in its order"
)]
pub(super) fn direction(
    panel: &Panel,
    controls: Option<&Controls>,
    inc: f64,
    effects: usize,
    placebos: usize,
    options: &Options,
) -> DirectionTerms {
    let nt = panel.nt;
    let gn = panel.n_grid_groups();
    let cells = gn * nt;
    let never_f = panel.t_max + 1;
    let zero_part2 = vec![0.0; gn];
    let mut out_effects = Vec::with_capacity(effects);
    let mut nevers: Vec<Vec<f64>> = Vec::with_capacity(effects);

    for i in 1..=effects {
        let ii = i64::try_from(i).expect("effect index");
        let mut diff = vec![NA; cells];
        let mut never = vec![NA; cells];
        for g in 0..gn {
            let f = panel.f_g[g];
            for tp in 0..nt {
                let c = g * nt + tp;
                let t = panel.time_of(tp);
                if tp >= i {
                    diff[c] = panel.y[c] - panel.y[c - i];
                }
                if !diff[c].is_nan() {
                    never[c] = f64::from(u8::from(f > t));
                }
                if options.only_never_switchers && f > t && f < never_f {
                    never[c] = 0.0;
                }
            }
        }
        let n_ctrl = controls_sum(panel, &never);
        let mut dist = vec![NA; cells];
        for g in 0..gn {
            let f = panel.f_g[g];
            for tp in 0..nt {
                let c = g * nt + tp;
                if diff[c].is_nan() {
                    continue;
                }
                let t = panel.time_of(tp);
                let has_ctrl = n_ctrl[panel.base_id[g] * nt + tp] > 0.0;
                dist[c] = if t == f - 1 + ii && ii <= panel.l_g[g] {
                    if panel.s_g[g].is_nan() {
                        if has_ctrl { NA } else { 0.0 }
                    } else {
                        f64::from(u8::from(panel.s_g[g] == inc && has_ctrl))
                    }
                } else {
                    0.0
                };
            }
        }
        let (n_t, _, n_t_g, n, n_dw) = switcher_counts(panel, &dist);

        let mut part2 = zero_part2.clone();
        if let Some(ctl) = controls {
            part2 = ctl.adjust(panel, i, 0, &mut diff, &dist, &never, &n_t_g, &n_ctrl, n);
        }

        let (u, u_var, count) = if n == 0.0 {
            (Vec::new(), Vec::new(), Vec::new())
        } else {
            u_terms(
                panel,
                &Sample {
                    i,
                    diff: &diff,
                    never: &never,
                    dist: &dist,
                    n_t_g: &n_t_g,
                    n_t: &n_t,
                    n_ctrl: &n_ctrl,
                    n,
                    part2: &part2,
                },
            )
        };
        let delta = if options.normalized {
            delta_norm(panel, i, inc, &dist, n)
        } else {
            0.0
        };
        nevers.push(never);
        out_effects.push(Terms {
            n,
            n_dw,
            u,
            u_var,
            count,
            delta_norm: delta,
            dist,
        });
    }

    let mut out_placebos = Vec::with_capacity(placebos);
    for i in 1..=placebos {
        let never_i = &nevers[i - 1];
        let dist_i = &out_effects[i - 1].dist;
        let mut diff = vec![NA; cells];
        for g in 0..gn {
            for tp in 0..nt {
                let c = g * nt + tp;
                if tp >= 2 * i {
                    diff[c] = panel.y[c - 2 * i] - panel.y[c - i];
                }
            }
        }
        let never: Vec<f64> = (0..cells)
            .map(|c| never_i[c] * f64::from(u8::from(!diff[c].is_nan())))
            .collect();
        let n_ctrl = controls_sum(panel, &never);
        let dist: Vec<f64> = (0..cells)
            .map(|c| {
                if dist_i[c].is_nan() {
                    NA
                } else {
                    let ok = !diff[c].is_nan() && n_ctrl[panel.base_id[c / nt] * nt + c % nt] > 0.0;
                    dist_i[c] * f64::from(u8::from(ok))
                }
            })
            .collect();
        let (n_t, _, n_t_g, n, n_dw) = switcher_counts(panel, &dist);
        let mut part2 = zero_part2.clone();
        if let Some(ctl) = controls {
            part2 = ctl.adjust(panel, i, i, &mut diff, &dist, &never, &n_t_g, &n_ctrl, n);
        }
        let (u, u_var, count) = if n == 0.0 {
            (Vec::new(), Vec::new(), Vec::new())
        } else {
            u_terms(
                panel,
                &Sample {
                    i,
                    diff: &diff,
                    never: &never,
                    dist: &dist,
                    n_t_g: &n_t_g,
                    n_t: &n_t,
                    n_ctrl: &n_ctrl,
                    n,
                    part2: &part2,
                },
            )
        };
        let delta = if options.normalized {
            delta_norm(panel, i, inc, &dist, n)
        } else {
            0.0
        };
        out_placebos.push(Terms {
            n,
            n_dw,
            u,
            u_var,
            count,
            delta_norm: delta,
            dist,
        });
    }

    // The average total effect: the effects weighted by their numbers of
    // switchers, over the switchers' mean change in treatment.
    let sum_n: f64 = out_effects.iter().map(|e| e.n).sum();
    let mut num = vec![0.0; gn];
    let mut num_var = vec![0.0; gn];
    let mut den = 0.0;
    let mut delta_d_g = Vec::with_capacity(effects);
    for e in &out_effects {
        let mut dd = vec![0.0; cells];
        if e.n != 0.0 {
            let w = e.n / sum_n;
            let mut delta = 0.0;
            let mut temp = vec![0.0; cells];
            for g in 0..gn {
                let s = panel.s_g[g];
                for tp in 0..nt {
                    let c = g * nt + tp;
                    if e.dist[c] == 1.0 {
                        let change = panel.d[c] - panel.d_sq[g];
                        temp[c] = (panel.n_gt[c] / e.n) * (change * s + (1.0 - s) * (-change));
                    }
                    if !temp[c].is_nan() {
                        delta += temp[c];
                    }
                }
            }
            for c in 0..cells {
                dd[c] = temp[c] * (e.n / panel.n_gt[c]);
            }
            for g in 0..gn {
                num[g] += w * e.u[g];
                num_var[g] += w * e.u_var[g];
            }
            den += w * delta;
        }
        delta_d_g.push(dd);
    }
    let ate_u = num.iter().map(|v| v / den).collect();
    let ate_u_var = num_var.iter().map(|v| v / den).collect();
    DirectionTerms {
        effects: out_effects,
        placebos: out_placebos,
        sum_n,
        ate_u,
        ate_u_var,
        ate_den: den,
        delta_d_g,
    }
}

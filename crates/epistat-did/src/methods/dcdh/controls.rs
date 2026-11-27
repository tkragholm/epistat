//! The `controls` option: the outcome's long differences are residualised on
//! those of the controls, with coefficients fitted on the not-yet switchers
//! of each baseline treatment, and the variance gains the term of that fit.

use epistat_linalg::{Attempt, Mat, Method, inverse, solve};

use super::panel::{NA, Panel};
use crate::error::Error;

pub(super) struct Controls {
    k: usize,
    /// `theta_d` of each baseline level.
    theta: Vec<Vec<f64>>,
    /// `in_brackets_{d,j}` of each level, control and grid group.
    in_brackets: Vec<Vec<Vec<f64>>>,
}

fn level_value(panel: &Panel, l: usize) -> f64 {
    (0..panel.n_grid_groups())
        .find(|&g| panel.level[g] == l)
        .map_or(NA, |g| panel.d_sq[g])
}

/// Whether `x` takes two values in one cell among `rows`.
fn varies(rows: impl Iterator<Item = usize>, cell: &dyn Fn(usize) -> usize, x: &[f64]) -> bool {
    let mut first: std::collections::HashMap<usize, u64> = std::collections::HashMap::new();
    for c in rows {
        let v = (x[c] + 0.0).to_bits();
        if *first.entry(cell(c)).or_insert(v) != v {
            return true;
        }
    }
    false
}

fn solve_spd(a: &[f64], b: &[f64], k: usize) -> Option<Vec<f64>> {
    let am = Mat::from_fn(k, k, |i, j| a[i * k + j]);
    let bm = Mat::from_fn(k, 1, |i, _| b[i]);
    let s = solve(
        am.as_ref(),
        bm.as_ref(),
        &[Attempt::plain(Method::Cholesky)],
    )
    .ok()?;
    Some((0..k).map(|i| s.x[(i, 0)]).collect())
}

impl Controls {
    #[expect(
        clippy::too_many_lines,
        reason = "the controls block of did_multiplegt_main, in its order"
    )]
    pub fn new(panel: &Panel) -> Result<Self, Error> {
        let k = panel.controls.len();
        let user_k = k - panel.generated.len();
        let nt = panel.nt;
        let gn = panel.n_grid_groups();
        let cells = gn * nt;
        let g_count = panel.n_groups;
        let first_diff = |v: &[f64]| -> Vec<f64> {
            (0..cells)
                .map(|c| if c % nt >= 1 { v[c] - v[c - 1] } else { NA })
                .collect()
        };
        let diff_y = first_diff(&panel.y);
        let diff_x: Vec<Vec<f64>> = panel.controls.iter().map(|x| first_diff(x)).collect();
        let fd_all: Vec<bool> = (0..cells)
            .map(|c| diff_x.iter().all(|x| !x[c].is_nan()))
            .collect();
        let mask: Vec<bool> = (0..cells)
            .map(|c| panel.ever_change[c] == 0.0 && !diff_y[c].is_nan() && fd_all[c])
            .collect();

        // The controls' first differences, demeaned within the cell of
        // period, baseline and stratum over the not-yet switchers.
        let n_cells = panel.n_base * nt;
        let cell_of = |c: usize| panel.base_id[c / nt] * nt + c % nt;
        let mut sum_w = vec![0.0; n_cells];
        let mut sum_x = vec![vec![0.0; n_cells]; k];
        for c in 0..cells {
            if mask[c] {
                sum_w[cell_of(c)] += panel.n_gt[c];
                for j in 0..k {
                    let v = panel.n_gt[c] * diff_x[j][c];
                    if !v.is_nan() {
                        sum_x[j][cell_of(c)] += v;
                    }
                }
            }
        }
        let mut resid = vec![vec![0.0; cells]; k];
        let mut prod = vec![vec![0.0; cells]; k];
        for j in 0..k {
            for c in 0..cells {
                if mask[c] {
                    let avg = sum_x[j][cell_of(c)] / sum_w[cell_of(c)];
                    let r = panel.n_gt[c].sqrt() * (diff_x[j][c] - avg);
                    resid[j][c] = if r.is_nan() { 0.0 } else { r };
                    let p = panel.n_gt[c].sqrt() * resid[j][c];
                    prod[j][c] = if p.is_nan() { 0.0 } else { p };
                }
            }
        }

        let mut theta = Vec::with_capacity(panel.n_level);
        let mut inv_denom = Vec::with_capacity(panel.n_level);
        let mut prediction = vec![NA; cells];
        for l in 0..panel.n_level {
            let value = level_value(panel, l);
            let mut dates: Vec<i64> = (0..gn)
                .filter(|&g| panel.level[g] == l)
                .map(|g| panel.f_g[g])
                .collect();
            dates.sort_unstable();
            dates.dedup();
            if dates.len() <= 1 {
                return Err(Error::Controls {
                    baseline: value,
                    reason: "its groups share one first-switch date",
                });
            }
            let rows: Vec<usize> = (0..cells)
                .filter(|&c| mask[c] && panel.level[c / nt] == l)
                .collect();
            if rows.is_empty() {
                return Err(Error::Controls {
                    baseline: value,
                    reason: "no not-yet switcher has the controls' first differences",
                });
            }
            let mut m = vec![0.0; k * k];
            let mut v = vec![0.0; k];
            for &c in &rows {
                let yw = panel.n_gt[c].sqrt() * diff_y[c];
                for a in 0..k {
                    v[a] += resid[a][c] * yw;
                    for b in 0..k {
                        m[a * k + b] += resid[a][c] * resid[b][c];
                    }
                }
            }
            // The period controls of `continuous` that are constant within
            // every cell of the regression are zero once demeaned; they are
            // left out, with coefficient 0.
            let act: Vec<usize> = (0..k)
                .filter(|&j| {
                    j < user_k
                        || varies(
                            rows.iter().copied().filter(|&c| panel.n_gt[c] > 0.0),
                            &cell_of,
                            &diff_x[j],
                        )
                })
                .collect();
            let ka = act.len();
            let m_act: Vec<f64> = (0..ka * ka)
                .map(|i| m[act[i / ka] * k + act[i % ka]])
                .collect();
            let v_act: Vec<f64> = act.iter().map(|&j| v[j]).collect();
            let collinear = || Error::Controls {
                baseline: value,
                reason: "the controls are collinear among its not-yet switchers",
            };
            let mut th = vec![0.0; k];
            let mut inv = vec![0.0; k * k];
            if ka > 0 {
                let solved = solve_spd(&m_act, &v_act, ka).ok_or_else(collinear)?;
                let mm = Mat::from_fn(ka, ka, |a, b| m_act[a * ka + b]);
                let inverted = inverse(mm.as_ref(), &[Attempt::plain(Method::Cholesky)])
                    .map_err(|_| collinear())?;
                for a in 0..ka {
                    th[act[a]] = solved[a];
                    for b in 0..ka {
                        inv[act[a] * k + act[b]] = inverted.x[(a, b)];
                    }
                }
            }
            // R 2.4.0: the sum of N_gt over the level's control sample.
            let rmax = rows.iter().map(|&c| panel.f_g[c / nt]).max().unwrap_or(0);
            let mut rsum = 0.0;
            for &c in &rows {
                let t = panel.time_of(c % nt);
                if t >= 2 && t < rmax && t < panel.f_g[c / nt] {
                    rsum += panel.n_gt[c];
                }
            }
            inv_denom.push(
                (0..k)
                    .map(|a| {
                        (0..k)
                            .map(|b| inv[a * k + b] * rsum * g_count)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>(),
            );
            theta.push(th);

            // The weighted regression of the outcome's first difference on the
            // controls' with period effects, among the level's not-yet
            // switchers, and its fitted values.
            let in_reg =
                |c: usize| panel.level[c / nt] == l && panel.f_g[c / nt] > panel.time_of(c % nt);
            let est: Vec<usize> = (0..cells)
                .filter(|&c| {
                    in_reg(c) && !diff_y[c].is_nan() && fd_all[c] && !panel.weight[c].is_nan()
                })
                .collect();
            let mut wt = vec![0.0; nt];
            let mut ybar = vec![0.0; nt];
            let mut xbar = vec![vec![0.0; nt]; k];
            for &c in &est {
                let (tp, w) = (c % nt, panel.weight[c]);
                wt[tp] += w;
                ybar[tp] += w * diff_y[c];
                for j in 0..k {
                    xbar[j][tp] += w * diff_x[j][c];
                }
            }
            for tp in 0..nt {
                ybar[tp] /= wt[tp];
                for x in &mut xbar {
                    x[tp] /= wt[tp];
                }
            }
            let mut a = vec![0.0; k * k];
            let mut b = vec![0.0; k];
            for &c in &est {
                let (tp, w) = (c % nt, panel.weight[c]);
                let dy = diff_y[c] - ybar[tp];
                for p in 0..k {
                    let dp = diff_x[p][c] - xbar[p][tp];
                    b[p] += w * dp * dy;
                    for q in 0..k {
                        a[p * k + q] += w * dp * (diff_x[q][c] - xbar[q][tp]);
                    }
                }
            }
            let act_b: Vec<usize> = (0..k)
                .filter(|&j| {
                    j < user_k
                        || varies(
                            est.iter().copied().filter(|&c| panel.weight[c] > 0.0),
                            &|c| c % nt,
                            &diff_x[j],
                        )
                })
                .collect();
            let kb = act_b.len();
            let a_act: Vec<f64> = (0..kb * kb)
                .map(|i| a[act_b[i / kb] * k + act_b[i % kb]])
                .collect();
            let b_act: Vec<f64> = act_b.iter().map(|&j| b[j]).collect();
            let mut beta = vec![0.0; k];
            if kb > 0 {
                let solved = solve_spd(&a_act, &b_act, kb).ok_or(Error::Controls {
                    baseline: value,
                    reason: "the controls are collinear in the regression with period effects",
                })?;
                for (i, &j) in act_b.iter().enumerate() {
                    beta[j] = solved[i];
                }
            }
            for c in 0..cells {
                let tp = c % nt;
                if in_reg(c) && fd_all[c] && wt[tp] > 0.0 {
                    let mut fit = ybar[tp];
                    for j in 0..k {
                        fit += beta[j] * (diff_x[j][c] - xbar[j][tp]);
                    }
                    prediction[c] = fit;
                }
            }
        }

        // The influence of theta_d: sums over each group's periods before its
        // switch.
        let mut denom: Vec<f64> = vec![0.0; panel.n_level * nt];
        let mut t_d = vec![i64::MIN; panel.n_level];
        for g in 0..gn {
            t_d[panel.level[g]] = t_d[panel.level[g]].max(panel.f_g[g]);
            for tp in 0..nt {
                let c = g * nt + tp;
                if panel.f_g[g] > panel.time_of(tp) && !diff_y[c].is_nan() {
                    denom[panel.level[g] * nt + tp] += 1.0;
                }
            }
        }
        let mut n_c = vec![0.0; panel.n_level];
        for g in 0..gn {
            let l = panel.level[g];
            for tp in 0..nt {
                let c = g * nt + tp;
                let t = panel.time_of(tp);
                if t >= 2 && t < t_d[l] && t < panel.f_g[g] && !diff_y[c].is_nan() {
                    n_c[l] += panel.n_gt[c];
                }
            }
        }
        let mut in_sum = vec![vec![0.0; gn]; k];
        for g in 0..gn {
            let l = panel.level[g];
            for tp in 0..nt {
                let c = g * nt + tp;
                let t = panel.time_of(tp);
                if !(t >= 2 && t < panel.f_g[g]) {
                    continue;
                }
                let e = denom[l * nt + tp];
                let (fit, factor) = if e >= 2.0 {
                    (prediction[c], (e / (e - 1.0)).sqrt())
                } else {
                    (prediction[c] * 0.0, 1.0)
                };
                for (j, s) in in_sum.iter_mut().enumerate() {
                    let v = prod[j][c] * factor * (diff_y[c] - fit) / n_c[l];
                    if !v.is_nan() {
                        s[g] += v;
                    }
                }
            }
        }
        let in_brackets: Vec<Vec<Vec<f64>>> = (0..panel.n_level)
            .map(|l| {
                (0..k)
                    .map(|j| {
                        (0..gn)
                            .map(|g| {
                                let mut v = 0.0;
                                if panel.level[g] == l && panel.f_g[g] >= 3 {
                                    for (q, s) in in_sum.iter().enumerate() {
                                        v += inv_denom[l][j][q] * s[g];
                                    }
                                }
                                v - theta[l][j]
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            k,
            theta,
            in_brackets,
        })
    }

    /// Residualises the long difference `diff` of effect `i` (or of placebo
    /// `i` when `placebo` is `i`) and returns the adjustment of each grid
    /// group's variance term.
    pub fn adjust(
        &self,
        panel: &Panel,
        i: usize,
        placebo: usize,
        diff: &mut [f64],
        dist: &[f64],
        never: &[f64],
        n_t_g: &[f64],
        n_ctrl: &[f64],
        n: f64,
    ) -> Vec<f64> {
        let nt = panel.nt;
        let gn = panel.n_grid_groups();
        let ii = i64::try_from(i).expect("effect index");
        let g_count = panel.n_groups;
        let mut big_m = vec![vec![0.0; self.k]; panel.n_level];
        for j in 0..self.k {
            let x = &panel.controls[j];
            for g in 0..gn {
                let l = panel.level[g];
                let t_g = panel.t_g[g];
                let mut m = 0.0;
                for tp in 0..nt {
                    let c = g * nt + tp;
                    let dx = if placebo == 0 {
                        if tp >= i { x[c] - x[c - i] } else { NA }
                    } else if tp >= 2 * i {
                        x[c - 2 * i] - x[c - i]
                    } else {
                        NA
                    };
                    let t = panel.time_of(tp);
                    let cell = panel.base_id[g] * nt + tp;
                    let use_l = f64::from(u8::from(ii <= t_g - 2));
                    let window = f64::from(u8::from(t > ii && t <= t_g));
                    let v = use_l
                        * (g_count / n)
                        * ((dist[c] - (n_t_g[cell] / n_ctrl[cell]) * never[c])
                            * window
                            * (panel.n_gt[c] * dx));
                    if !v.is_nan() {
                        m += v;
                    }
                    diff[c] -= self.theta[l][j] * dx;
                }
                big_m[l][j] += m;
            }
        }
        for row in &mut big_m {
            for v in row.iter_mut() {
                *v /= g_count;
            }
        }
        (0..gn)
            .map(|g| {
                let mut part2 = 0.0;
                for (l, row) in big_m.iter().enumerate() {
                    let mut combined = 0.0;
                    for (j, mj) in row.iter().enumerate() {
                        combined += mj * self.in_brackets[l][j][g];
                    }
                    part2 += combined;
                }
                part2
            })
            .collect()
    }
}

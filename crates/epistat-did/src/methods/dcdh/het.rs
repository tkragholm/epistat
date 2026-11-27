//! `predict_het`: regressions of the switchers' outcome changes on
//! group-level covariates.

use std::collections::HashMap;

use epistat_core::{Estimate, InputError, Interval};
use epistat_linalg::{Attempt, Mat, Method, inverse};
use epistat_stats::dist::f::sf as f_sf;
use epistat_stats::dist::t::{quantile as t_quantile, sf as t_sf};

use crate::error::Error;

use super::panel::{index_of, ordered, prepare};
use super::{DcdhOptions, SwitchingPanel, counts, domain, validate};

/// One coefficient of [`heterogeneity`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct DcdhHeterogeneity {
    /// `l` for effect `l`, `-l` for placebo `l`.
    pub relative_period: i64,
    /// The column of the covariate in the panel's heterogeneity covariates.
    pub covariate: usize,
    /// The coefficient with its HC2 standard error, the t interval at the
    /// options' level on the residual degrees of freedom and the t p-value;
    /// the interval and the p-value `None` without residual degrees of
    /// freedom.
    pub estimate: Estimate,
    /// `estimate / se`, `None` without a positive standard error.
    pub t: Option<f64>,
    /// The switchers in the regression.
    pub n: usize,
    /// The residual degrees of freedom.
    pub df: usize,
    /// The F test that all covariates' coefficients are zero, `None` when
    /// the covariance does not invert.
    pub p_joint: Option<f64>,
}

/// The regressions of `predict_het`: for effect `l`, the switchers'
/// `S_g (Y_{g,F_g-1+l} - Y_{g,F_g-1})` (`S_g` 1 for switchers in and -1 for
/// switchers out, the outcome in levels) on the covariates and fixed effects
/// of the cells of `F_g`, the baseline treatment, `S_g` and the stratum,
/// among the switchers with `F_g - 1 + l <= T_g`, weighted by the weight of
/// each group's first period, with HC2 standard errors and the F test that
/// the covariates' coefficients are zero. With `trends_lin` the change is
/// less `l (Y_{g,F_g-1} - Y_{g,F_g-2})`. The placebos use
/// `Y_{g,F_g-1-l} - Y_{g,F_g-1}` in the same sample. A dimension of the
/// cells enters only when it takes two values among those switchers.
///
/// The covariates are the panel's
/// [`het_covariates`](super::SwitchingPanelBuilder::het_covariates),
/// constant within a unit. `effects` selects the effects (and placebos)
/// from 1; `None` takes all that are estimated.
///
/// # Errors
///
/// The errors of [`estimate`](super::estimate); [`Error::Unsupported`] for
/// a panel without heterogeneity covariates or with controls, and with
/// `normalized`, which the package refuses too; [`Error::Input`] for an
/// empty set of covariates or a selected effect that is not estimated; and
/// [`Error::Collinear`] for covariates that are collinear with the cells.
///
/// # Panics
///
/// Only when a number of effects does not fit in `i64`.
#[expect(
    clippy::too_many_lines,
    reason = "the predict_het block of did_multiplegt_main, in its order"
)]
pub fn heterogeneity(
    data: &SwitchingPanel<'_>,
    options: &DcdhOptions,
    effects: Option<&[usize]>,
) -> Result<Vec<DcdhHeterogeneity>, Error> {
    validate(options)?;
    if options.normalized {
        return Err(Error::Unsupported {
            what: "predict_het with normalized",
        });
    }
    if data.controls().is_some_and(|x| x.ncols() > 0) {
        return Err(Error::Unsupported {
            what: "predict_het with controls",
        });
    }
    let Some(covariates) = data.het_covariates() else {
        return Err(Error::Unsupported {
            what: "predict_het on a panel without heterogeneity covariates",
        });
    };
    let k = covariates.ncols();
    if k == 0 {
        return Err(InputError::Empty {
            what: "het_covariates",
        }
        .into());
    }
    let mut of_group: HashMap<i64, Vec<f64>> = HashMap::new();
    for i in 0..data.len() {
        of_group
            .entry(i64::from(data.unit()[i]))
            .or_insert_with(|| (0..k).map(|j| covariates[(i, j)]).collect());
    }
    let panel = prepare(data, options)?;
    let c = counts(&panel, options)?;
    let not_estimated = |i: usize| domain("effects", i as f64, "the effects estimated");
    let selected: Vec<usize> = match effects {
        None => (1..=c.l).collect(),
        Some(e) => {
            if let Some(&i) = e.iter().find(|&&i| i == 0 || i > c.l) {
                return Err(not_estimated(i));
            }
            e.to_vec()
        }
    };
    let placebos: Vec<usize> = if c.l_pl == 0 {
        Vec::new()
    } else if effects.is_none() {
        (1..=c.l_pl).collect()
    } else if let Some(&i) = selected.iter().find(|&&i| i > c.l_pl) {
        return Err(domain("effects", i as f64, "the placebos estimated"));
    } else {
        selected.clone()
    };

    let x: Vec<&[f64]> = panel
        .group_ids
        .iter()
        .map(|id| of_group[id].as_slice())
        .collect();
    let nt = panel.nt;
    let level_at = |g: usize, t: i64| -> f64 {
        usize::try_from(t - panel.t_min)
            .ok()
            .filter(|&tp| tp < nt)
            .map_or(f64::NAN, |tp| panel.y_level[g * nt + tp])
    };
    let mut out = Vec::new();
    for (i, placebo) in selected
        .iter()
        .map(|&i| (i, false))
        .chain(placebos.iter().map(|&i| (i, true)))
    {
        let ii = i64::try_from(i).expect("effect index");
        let sample: Vec<usize> = (0..panel.n_grid_groups())
            .filter(|&g| panel.f_g[g] - 1 + ii <= panel.t_g[g])
            .collect();
        let feasible = |g: usize| {
            !level_at(g, panel.f_g[g] - 1).is_nan()
                && (!options.trends_lin || !level_at(g, panel.f_g[g] - 2).is_nan())
        };
        // The dimensions of the cells that vary among the feasible switchers.
        let dims: [Box<dyn Fn(usize) -> i64>; 4] = [
            Box::new(|g| panel.f_g[g]),
            Box::new(|g| ordered(panel.base_value[g])),
            Box::new(|g| ordered(panel.s_g[g])),
            // The baseline-and-stratum index: with the baseline it gives the
            // same cells as the stratum.
            Box::new(|g| i64::try_from(panel.base_id[g]).expect("index")),
        ];
        let varies: Vec<bool> = dims
            .iter()
            .map(|f| {
                let mut v: Vec<i64> = sample
                    .iter()
                    .filter(|&&g| feasible(g))
                    .map(|&g| f(g))
                    .collect();
                v.sort_unstable();
                v.dedup();
                v.len() > 1
            })
            .collect();
        let rows: Vec<(usize, f64, f64)> = sample
            .iter()
            .filter_map(|&g| {
                let base = level_at(g, panel.f_g[g] - 1);
                let target = if placebo {
                    level_at(g, panel.f_g[g] - 1 - ii)
                } else {
                    level_at(g, panel.f_g[g] - 1 + ii)
                };
                let mut change = target - base;
                if options.trends_lin {
                    change -= ii as f64 * (base - level_at(g, panel.f_g[g] - 2));
                }
                let direction = if panel.s_g[g] == 0.0 {
                    -1.0
                } else {
                    panel.s_g[g]
                };
                let y = direction * change;
                let w = panel.weight[g * nt];
                (!y.is_nan() && !w.is_nan() && w > 0.0).then_some((g, y, w))
            })
            .collect();
        let cell_key = |g: usize| -> [i64; 4] {
            let mut key = [0_i64; 4];
            for (j, f) in dims.iter().enumerate() {
                if varies[j] {
                    key[j] = f(g);
                }
            }
            key
        };
        let (cell, n_cells) = index_of(
            &rows
                .iter()
                .map(|&(g, _, _)| cell_key(g))
                .collect::<Vec<_>>(),
        );
        let fit = within_hc2(&rows, &cell, n_cells, &x, k).ok_or(Error::Collinear {
            what: "the predict_het covariates with the cells",
        })?;
        let df = rows.len().saturating_sub(k + n_cells);
        let level = options.level;
        let t_crit = (df > 0)
            .then(|| t_quantile(f64::midpoint(1.0, level.confidence_value()), df as f64))
            .filter(|q| q.is_finite());
        let p_joint = Some(joint_f(&fit.beta, &fit.v, k, df)).filter(|p| !p.is_nan());
        let period = if placebo { -ii } else { ii };
        for j in 0..k {
            let se = Some(fit.v[j * k + j].sqrt()).filter(|s| s.is_finite());
            let b = fit.beta[j];
            let t = se.filter(|&s| s > 0.0).map(|s| b / s);
            let interval = se
                .zip(t_crit)
                .map(|(s, q)| Interval::new(Some(b - q * s), Some(b + q * s), level));
            let p_value = t.filter(|_| df > 0).map(|t| 2.0 * t_sf(t.abs(), df as f64));
            out.push(DcdhHeterogeneity {
                relative_period: period,
                covariate: j,
                estimate: Estimate::point(b)
                    .with_se(se)
                    .with_interval(interval)
                    .with_p_value(p_value),
                t,
                n: rows.len(),
                df,
                p_joint,
            });
        }
    }
    out.sort_by_key(|r| (r.covariate, r.relative_period));
    Ok(out)
}

struct Fit {
    beta: Vec<f64>,
    /// The HC2 covariance, row-major.
    v: Vec<f64>,
}

/// The weighted regression of `y` on `x` with cell fixed effects, by
/// demeaning within cells, and its HC2 covariance. The hat value of a row is
/// `w / W_cell + w x~' A^-1 x~` with `x~` the demeaned covariates and
/// `A = sum w x~ x~'`; a row alone in its cell has `x~ = 0`, leverage 1 and
/// no part in the covariance.
fn within_hc2(
    rows: &[(usize, f64, f64)],
    cell: &[usize],
    n_cells: usize,
    x: &[&[f64]],
    k: usize,
) -> Option<Fit> {
    let mut size = vec![0_usize; n_cells];
    for &c in cell {
        size[c] += 1;
    }
    let mut w_cell = vec![0.0; n_cells];
    let mut y_cell = vec![0.0; n_cells];
    let mut x_cell = vec![0.0; n_cells * k];
    for (r, &(g, y, w)) in rows.iter().enumerate() {
        let c = cell[r];
        w_cell[c] += w;
        y_cell[c] += w * y;
        for j in 0..k {
            x_cell[c * k + j] += w * x[g][j];
        }
    }
    let xt: Vec<Vec<f64>> = rows
        .iter()
        .enumerate()
        .map(|(r, &(g, _, _))| {
            let c = cell[r];
            (0..k)
                .map(|j| x[g][j] - x_cell[c * k + j] / w_cell[c])
                .collect()
        })
        .collect();
    let yt: Vec<f64> = rows
        .iter()
        .enumerate()
        .map(|(r, &(_, y, _))| y - y_cell[cell[r]] / w_cell[cell[r]])
        .collect();
    let mut a = vec![0.0; k * k];
    let mut b = vec![0.0; k];
    for (r, &(_, _, w)) in rows.iter().enumerate() {
        for p in 0..k {
            b[p] += w * xt[r][p] * yt[r];
            for q in 0..k {
                a[p * k + q] += w * xt[r][p] * xt[r][q];
            }
        }
    }
    let am = Mat::from_fn(k, k, |i, j| a[i * k + j]);
    let inv = inverse(am.as_ref(), &[Attempt::plain(Method::Cholesky)]).ok()?;
    let ai = |i: usize, j: usize| inv.x[(i, j)];
    let beta: Vec<f64> = (0..k)
        .map(|p| (0..k).map(|q| ai(p, q) * b[q]).sum())
        .collect();
    let mut meat = vec![0.0; k * k];
    for (r, &(_, _, w)) in rows.iter().enumerate() {
        if size[cell[r]] == 1 {
            continue;
        }
        let e = yt[r] - (0..k).map(|j| xt[r][j] * beta[j]).sum::<f64>();
        let mut quad = 0.0;
        for p in 0..k {
            for q in 0..k {
                quad += xt[r][p] * ai(p, q) * xt[r][q];
            }
        }
        let h = w / w_cell[cell[r]] + w * quad;
        let omega = (w * e).powi(2) / (1.0 - h);
        for p in 0..k {
            for q in 0..k {
                meat[p * k + q] += omega * xt[r][p] * xt[r][q];
            }
        }
    }
    let mut v = vec![0.0; k * k];
    for p in 0..k {
        for q in 0..k {
            let mut acc = 0.0;
            for s in 0..k {
                for u in 0..k {
                    acc += ai(p, s) * meat[s * k + u] * ai(u, q);
                }
            }
            v[p * k + q] = acc;
        }
    }
    Some(Fit { beta, v })
}

/// `P(F_{k, df} > b' V^-1 b / k)`.
fn joint_f(beta: &[f64], v: &[f64], k: usize, df: usize) -> f64 {
    let vm = Mat::from_fn(k, k, |i, j| v[i * k + j]);
    let Ok(inv) = inverse(vm.as_ref(), &[Attempt::plain(Method::FullPivLu)]) else {
        return f64::NAN;
    };
    let mut s = 0.0;
    for p in 0..k {
        for q in 0..k {
            s += beta[p] * inv.x[(p, q)] * beta[q];
        }
    }
    f_sf(s / k as f64, k as f64, df as f64)
}

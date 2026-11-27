//! The Poisson and logit models of `etwfe(family = "poisson" / "logit")`
//! and their average effects. See the [module](crate::etwfe) for the model, the
//! effects and the covariances.

use epistat_core::Covariance;
use epistat_glm::rows::{Block, Rows};
use epistat_glm::{Error as GlmError, Family, Glm, Link};
use epistat_linalg::ROW_BLOCK;
use rayon::prelude::*;

use super::{
    EtwfeCoefficient, EtwfeEffect, EtwfeFamily, EtwfeFit, EtwfeOptions, EtwfeTerm, EtwfeVcov,
    Layout, effect_influence, unit_codes,
};
use crate::error::Error;
use crate::methods::fe_output::{self, Reference};
use crate::methods::staggered::StaggeredPanel;

/// The family and link of a Poisson or logit model, `None` for the linear
/// one.
pub(super) const fn model(family: EtwfeFamily) -> Option<(Family, Link)> {
    match family {
        EtwfeFamily::Poisson => Some((Family::Poisson, Link::Log)),
        EtwfeFamily::Logit => Some((Family::Binomial, Link::Logit)),
        EtwfeFamily::Gaussian => None,
    }
}

/// The positions of the column groups of the nonlinear design.
struct Columns {
    p: usize,
    gc: usize,
    tc: usize,
    n_cells: usize,
    /// The first column of each group: covariates, cohort slopes, period
    /// slopes, cohort dummies, period dummies, cells.
    covariate: usize,
    cohort_slope: usize,
    period_slope: usize,
    cohort: usize,
    period: usize,
    cell: usize,
    total: usize,
}

impl Columns {
    fn new(layout: &Layout<'_>) -> Self {
        let p = layout.p;
        let gc = layout.slope_cohorts.len();
        let tc = layout.slope_periods.len();
        let n_cells = layout.cells.len();
        let covariate = 1;
        let cohort_slope = covariate + p;
        let period_slope = cohort_slope + p * gc;
        let cohort = period_slope + p * tc;
        let period = cohort + gc;
        let cell = period + tc;
        Self {
            p,
            gc,
            tc,
            n_cells,
            covariate,
            cohort_slope,
            period_slope,
            cohort,
            period,
            cell,
            total: cell + n_cells * (1 + p),
        }
    }
}

/// The rows of the nonlinear design, formed on request.
struct Design<'l> {
    layout: &'l Layout<'l>,
    columns: Columns,
    /// The input rows of the fit, in order.
    rows: Vec<usize>,
}

impl Design<'_> {
    /// Calls `f(j, x_j, false)` for the nonzero columns of row `i` other than
    /// its cell, in column order, then `f(j, x_j, true)` for those of `cell`:
    /// the indicator and the demeaned covariates.
    fn entries(&self, i: usize, cell: Option<usize>, mut f: impl FnMut(usize, f64, bool)) {
        let mut base = |j: usize, v: f64| f(j, v, false);
        let layout = self.layout;
        let panel = layout.panel;
        let c = &self.columns;
        let cohort_slot =
            panel.first_treated()[i].and_then(|g| layout.slope_cohorts.binary_search(&g).ok());
        let period_slot = layout.slope_periods.binary_search(&panel.period()[i]).ok();
        base(0, 1.0);
        if let Some(x) = panel.covariates() {
            for k in 0..c.p {
                let v = x[(i, k)];
                base(c.covariate + k, v);
                if let Some(a) = cohort_slot {
                    base(c.cohort_slope + k * c.gc + a, v);
                }
                if let Some(b) = period_slot {
                    base(c.period_slope + k * c.tc + b, v);
                }
            }
        }
        if let Some(a) = cohort_slot {
            base(c.cohort + a, 1.0);
        }
        if let Some(b) = period_slot {
            base(c.period + b, 1.0);
        }
        if let Some(cell) = cell {
            f(c.cell + cell, 1.0, true);
            for k in 0..c.p {
                f(
                    c.cell + c.n_cells * (k + 1) + cell,
                    layout.demeaned(i, k),
                    true,
                );
            }
        }
    }

    fn weight(&self, i: usize) -> f64 {
        self.layout.panel.weights().map_or(1.0, |w| w[i])
    }
}

impl Rows for Design<'_> {
    fn n_rows(&self) -> usize {
        self.rows.len()
    }

    fn n_columns(&self) -> usize {
        self.columns.total
    }

    fn fill(&self, start: usize, mut block: Block<'_>) {
        let panel = self.layout.panel;
        for r in 0..block.y.len() {
            let i = self.rows[start + r];
            let x = &mut block.x;
            self.entries(i, self.layout.row_cell[i], |j, v, _| x[(r, j)] = v);
            block.y[r] = panel.outcome()[i];
            block.weight[r] = self.weight(i);
            block.cluster[r] = 0;
        }
    }
}

/// `x' b` over the entries `(j, x_j)` added in order from `-0`.
fn dot(entries: &[(usize, f64)], b: &[f64]) -> f64 {
    let mut total = -0.0_f64;
    for &(j, v) in entries {
        total += v * b[j];
    }
    total
}

/// Sums over the blocks of `ROW_BLOCK` rows of `0..n` of the terms `term`
/// gives, added in block order, so that the result does not depend on the
/// number of threads.
fn blocked_sum(n: usize, len: usize, term: impl Fn(usize, usize, &mut [f64]) + Sync) -> Vec<f64> {
    let blocks = n.div_ceil(ROW_BLOCK);
    let parts: Vec<Vec<f64>> = (0..blocks)
        .into_par_iter()
        .map(|b| {
            let mut out = vec![0.0; len];
            term(b * ROW_BLOCK, (b * ROW_BLOCK + ROW_BLOCK).min(n), &mut out);
            out
        })
        .collect();
    let mut total = vec![0.0; len];
    for part in &parts {
        for (t, v) in total.iter_mut().zip(part) {
            *t += v;
        }
    }
    total
}

const fn glm_error(e: GlmError) -> Error {
    Error::Glm(e)
}

/// `sum_g S_g S_g'` for the sums `sums`, one row of `q` values per group.
fn outer_sums(sums: &[f64], q: usize) -> Vec<f64> {
    let groups = sums.len().checked_div(q).unwrap_or(0);
    let mut meat = vec![0.0; q * q];
    for g in 0..groups {
        let s = &sums[g * q..(g + 1) * q];
        for a in 0..q {
            for c in 0..q {
                meat[a * q + c] += s[a] * s[c];
            }
        }
    }
    meat
}

/// `B M B` for the inverse information `b` and the meat `meat`, both
/// row-major `q x q`.
fn sandwich(b: &[f64], meat: &[f64], q: usize) -> Vec<f64> {
    let product = |x: &[f64], y: &[f64]| -> Vec<f64> {
        (0..q * q)
            .map(|ac| {
                let (a, c) = (ac / q, ac % q);
                let mut total = 0.0;
                for k in 0..q {
                    total += x[a * q + k] * y[k * q + c];
                }
                total
            })
            .collect()
    };
    product(&product(b, meat), b)
}

/// A count of rows summed as a float, which is exact below `2^53`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a count of rows, a whole number below 2^53"
)]
const fn count(v: f64) -> usize {
    v as usize
}

/// The Poisson or logit model of [`fit`](super::fit), with cohort and
/// period dummies, and its average effects on the scale of the mean.
#[expect(
    clippy::too_many_lines,
    reason = "one pass from the layout to the fit, the covariance and the averages"
)]
pub(super) fn fit(
    panel: &StaggeredPanel<'_>,
    options: &EtwfeOptions,
    family: Family,
    link: Link,
) -> Result<EtwfeFit, Error> {
    let layout = Layout::new(panel, options.control)?;
    let n = layout.n();
    let way = panel.clusters().unwrap_or(panel.unit());
    let columns = Columns::new(&layout);
    let k_all = columns.total;
    let rows: Vec<usize> = (0..n)
        .filter(|&i| !layout.excluded[i] && panel.weights().is_none_or(|w| w[i] > 0.0))
        .collect();
    let excluded_rows = n - rows.len();
    let design = Design {
        layout: &layout,
        columns,
        rows,
    };
    let model = Glm {
        tolerance: options.tolerance,
        max_iterations: options.max_iterations,
        ..Glm::new(family, link)
    };
    let fit = model.fit_rows(&design).map_err(|e| match e {
        GlmError::Response { row, value } => glm_error(GlmError::Response {
            row: design.rows.get(row).copied().unwrap_or(row),
            value,
        }),
        other => glm_error(other),
    })?;
    let kept: Vec<usize> = (0..k_all)
        .filter(|&j| fit.coefficients[j].is_some())
        .collect();
    let q = kept.len();
    let beta: Vec<f64> = fit.coefficients.iter().map(|b| b.unwrap_or(0.0)).collect();
    let position: Vec<Option<usize>> = {
        let mut out = vec![None; k_all];
        for (a, &j) in kept.iter().enumerate() {
            out[j] = Some(a);
        }
        out
    };
    // (X'WX)^-1 over the kept columns: the covariance at dispersion one.
    let bread: Vec<f64> = (0..q * q)
        .map(|ac| {
            fit.covariance.as_slice()[kept[ac / q] * k_all + kept[ac % q]].unwrap_or(f64::NAN)
        })
        .collect();

    // The score factor w (y - mu) of each row of the fit at the estimates.
    let n_fit = design.rows.len();
    let score: Vec<f64> = design
        .rows
        .par_iter()
        .map_init(Vec::new, |entries: &mut Vec<(usize, f64)>, &i| {
            entries.clear();
            let cell = layout.row_cell[i];
            design.entries(i, cell, |j, v, _| entries.push((j, v)));
            let mu = link.inverse(dot(entries, &beta));
            design.weight(i) * (panel.outcome()[i] - mu)
        })
        .collect();
    // Sums of the scores within groups numbered from 0, over the kept columns.
    let group_sums = |codes: &[u32], count: usize| -> Vec<f64> {
        let mut sums = vec![0.0; count * q];
        let mut entries: Vec<(usize, f64)> = Vec::new();
        for (r, &i) in design.rows.iter().enumerate() {
            entries.clear();
            design.entries(i, layout.row_cell[i], |j, v, _| entries.push((j, v)));
            let g = codes[r] as usize;
            for &(j, v) in &entries {
                if let Some(a) = position[j] {
                    sums[g * q + a] += score[r] * v;
                }
            }
        }
        sums
    };

    let k = q;
    let factor_k = |numerator: usize| -> Option<f64> {
        if !options.ssc.k_adj {
            return Some(1.0);
        }
        (n_fit > k).then(|| numerator as f64 / (n_fit - k) as f64)
    };
    let covariance_kept: Option<Vec<f64>> = match options.vcov {
        EtwfeVcov::Iid => {
            factor_k(n_fit.saturating_sub(1)).map(|f| bread.iter().map(|v| f * v).collect())
        }
        EtwfeVcov::Hetero => factor_k(n_fit).map(|f| {
            let meat = blocked_sum(n_fit, q * q, |start, end, out| {
                let mut entries: Vec<(usize, f64)> = Vec::new();
                for r in start..end {
                    let i = design.rows[r];
                    entries.clear();
                    design.entries(i, layout.row_cell[i], |j, v, _| {
                        if let Some(a) = position[j] {
                            entries.push((a, score[r] * v));
                        }
                    });
                    for &(a, u) in &entries {
                        for &(c, v) in &entries {
                            out[a * q + c] += u * v;
                        }
                    }
                }
            });
            sandwich(&bread, &meat, q).iter().map(|v| f * v).collect()
        }),
        EtwfeVcov::Clustered => {
            // Clusters numbered in order of first appearance among the rows.
            let mut number: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
            let codes: Vec<u32> = design
                .rows
                .iter()
                .map(|&i| {
                    let next = u32::try_from(number.len()).unwrap_or(u32::MAX);
                    *number.entry(way[i]).or_insert(next)
                })
                .collect();
            let g = number.len();
            let meat = outer_sums(&group_sums(&codes, g), q);
            let g_factor = if options.ssc.g_adj {
                (g >= 2).then(|| g as f64 / (g - 1) as f64)
            } else {
                Some(1.0)
            };
            match (g_factor, factor_k(n_fit.saturating_sub(1))) {
                (Some(a), Some(b)) => Some(
                    sandwich(&bread, &meat, q)
                        .iter()
                        .map(|v| a * b * v)
                        .collect(),
                ),
                _ => None,
            }
        }
    };
    let mut covariance = vec![None; k_all * k_all];
    if let Some(v) = &covariance_kept {
        for (a, &i) in kept.iter().enumerate() {
            for (c, &j) in kept.iter().enumerate() {
                covariance[i * k_all + j] = Some(v[a * q + c]);
            }
        }
    }

    // Terms.
    let mut terms: Vec<EtwfeTerm> = vec![EtwfeTerm::Intercept];
    let linear = &layout.terms;
    terms.extend_from_slice(&linear[..layout.first_cell]);
    terms.extend(layout.slope_cohorts.iter().map(|&g| EtwfeTerm::Cohort(g)));
    terms.extend(layout.slope_periods.iter().map(|&t| EtwfeTerm::Period(t)));
    terms.extend_from_slice(&linear[layout.first_cell..]);
    let coefficients: Vec<EtwfeCoefficient> = terms
        .iter()
        .enumerate()
        .map(|(j, &term)| EtwfeCoefficient {
            term,
            estimate: fit.coefficients[j].map(|b| {
                let se = covariance[j * k_all + j].map(|v| v.max(0.0).sqrt());
                fe_output::estimate(b, se, options.level, Reference::Normal)
            }),
        })
        .collect();

    // Influence by unit.
    let (units, codes) = unit_codes(panel.unit(), &design.rows);
    let n_units = units.len();
    let unit_sums = group_sums(&codes, n_units);
    let mut influence = vec![None; k_all];
    for (a, &j) in kept.iter().enumerate() {
        influence[j] = Some(
            (0..n_units)
                .map(|u| {
                    let mut total = 0.0;
                    for c in 0..q {
                        total += bread[a * q + c] * unit_sums[u * q + c];
                    }
                    n_units as f64 * total
                })
                .collect::<Vec<f64>>(),
        );
    }

    // The averages: per set of rows, the count, the sum of the contrasts and
    // the sums of their derivatives in the coefficients.
    let averages = layout.averages();
    let m = averages.keys.len();
    let width = 2 + k_all;
    let sums = blocked_sum(n, m * width, |start, end, out| {
        let mut base: Vec<(usize, f64)> = Vec::new();
        let mut cell: Vec<(usize, f64)> = Vec::new();
        for i in start..end {
            let Some(targets) = averages.targets(&layout, i) else {
                continue;
            };
            base.clear();
            cell.clear();
            design.entries(i, layout.cell_of(i), |j, v, in_cell| {
                if in_cell {
                    cell.push((j, v));
                } else {
                    base.push((j, v));
                }
            });
            let eta0 = dot(&base, &beta);
            let eta1 = eta0 + dot(&cell, &beta);
            let (mu0, d0) = link.inverse_and_derivative(eta0);
            let (mu1, d1) = link.inverse_and_derivative(eta1);
            for &a in targets.iter().filter(|&&a| a != usize::MAX) {
                let row = &mut out[a * width..(a + 1) * width];
                row[0] += 1.0;
                row[1] += mu1 - mu0;
                for &(j, v) in &base {
                    row[2 + j] += (d1 - d0) * v;
                }
                for &(j, v) in &cell {
                    row[2 + j] += d1 * v;
                }
            }
        }
    });
    let rows_in: Vec<f64> = (0..m).map(|a| sums[a * width]).collect();
    let jacobian: Vec<Vec<f64>> = (0..m)
        .map(|a| {
            (0..k_all)
                .map(|j| {
                    if position[j].is_some() {
                        sums[a * width + 2 + j] / rows_in[a]
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    let mut effect_covariance = vec![None; m * m];
    for a in 0..m {
        for b in 0..m {
            let mut total = 0.0;
            let mut defined = true;
            for (i, &ja) in jacobian[a].iter().enumerate() {
                if ja == 0.0 {
                    continue;
                }
                for (j, &jb) in jacobian[b].iter().enumerate() {
                    if jb == 0.0 {
                        continue;
                    }
                    match covariance[i * k_all + j] {
                        Some(v) => total += ja * jb * v,
                        None => defined = false,
                    }
                }
            }
            effect_covariance[a * m + b] = defined.then_some(total);
        }
    }
    let effects: Vec<EtwfeEffect> = averages
        .keys
        .iter()
        .enumerate()
        .map(|(a, &term)| {
            let zero = jacobian[a].iter().all(|&v| v == 0.0);
            let se = if zero {
                None
            } else {
                effect_covariance[a * m + a].map(|v| v.max(0.0).sqrt())
            };
            EtwfeEffect {
                term,
                estimate: fe_output::estimate(
                    sums[a * width + 1] / rows_in[a],
                    se,
                    options.level,
                    Reference::Normal,
                ),
                n_rows: count(rows_in[a]),
            }
        })
        .collect();
    let effect_influence = effect_influence(&jacobian, &influence, n_units);
    let rule = fe_output::fixest_rule(options.ssc.g_adj);

    Ok(EtwfeFit {
        coefficients,
        covariance: Covariance::new(k_all, covariance)?,
        effects,
        effect_covariance: Covariance::new(m, effect_covariance)?,
        effect_influence: fe_output::influence(effect_influence, units.clone(), rule)?,
        influence: fe_output::influence(influence, units, rule)?,
        df_k: k,
        n_units,
        n_rows: n_fit,
        removed_singletons: 0,
        excluded_rows,
        deviance: Some(fit.deviance),
        iterations: Some(fit.iterations),
        converged: fit.converged,
    })
}

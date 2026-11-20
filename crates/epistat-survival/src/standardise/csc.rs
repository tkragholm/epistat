//! The standardised absolute risk of a cause from cause-specific Cox models
//! and its influence function. See the module documentation of
//! [`standardise`](super) for the formulas.
//!
//! The influence function of the absolute risk of one covariate row is linear
//! in a set of arrays indexed by the grid position: the running sums behind
//! [`absrisk`](crate::absrisk)'s variance. Their coefficients depend on the
//! row only through its risk path and its covariates. The average over the
//! rows of the population is therefore taken once, on those arrays, and the
//! influence function of the training rows is formed from the averages in one
//! further pass. The memory is `O(J p)` and the cost `O(n J p)` for the paths
//! plus `O(n p)` per distinct time for the influence functions.

use super::model::{Baselines, Influence, ModelInput};
use super::{LevelRows, ordered_reduce};
use crate::error::{Error, Result};

/// What the evaluation reads.
pub(super) struct Csc<'a> {
    pub(super) models: &'a [ModelInput<'a>],
    pub(super) base: &'a Baselines,
    pub(super) cause_model: usize,
    pub(super) product_limit: bool,
    /// The rows of positive weight, their strata and weights divided by the
    /// total weight.
    pub(super) rows: &'a [usize],
    pub(super) stratum: &'a [usize],
    pub(super) weight: &'a [f64],
}

/// The event-free survival and the absolute risk along the grid for one row.
struct Path {
    surv: Vec<f64>,
    cif: Vec<f64>,
    /// The first grid position at which the survival is not a probability.
    valid: usize,
}

/// Running sums of one stratum, averaged over the rows of the stratum.
struct StratumAcc {
    hk: Vec<f64>,
    w: Vec<f64>,
    cum_c: Vec<Vec<f64>>,
    prefix_a: Vec<Vec<f64>>,
}

/// The averages of one level.
struct LevelAcc {
    /// The standardised risk at each needed count.
    fbar: Vec<f64>,
    min_valid: usize,
    strata: Vec<StratumAcc>,
    /// The coefficient vector of the influence function of every model's
    /// coefficients, `(top + 1) * p` per model.
    g: Vec<Vec<f64>>,
}

impl LevelAcc {
    fn new(n_needed: usize, n_strata: usize, top: usize, ps: &[usize], arrays: bool) -> Self {
        let m = ps.len();
        let sized = |len: usize| if arrays { len } else { 0 };
        Self {
            fbar: vec![0.0; n_needed],
            min_valid: usize::MAX,
            strata: (0..n_strata)
                .map(|_| StratumAcc {
                    hk: vec![0.0; sized(top + 1)],
                    w: vec![0.0; sized(top)],
                    cum_c: vec![vec![0.0; sized(top + 1)]; m],
                    prefix_a: vec![vec![0.0; sized(top + 1)]; m],
                })
                .collect(),
            g: ps
                .iter()
                .map(|&p| vec![0.0; sized((top + 1) * p)])
                .collect(),
        }
    }

    fn merge(&mut self, other: &Self) {
        add(&mut self.fbar, &other.fbar);
        self.min_valid = self.min_valid.min(other.min_valid);
        for (a, b) in self.strata.iter_mut().zip(&other.strata) {
            add(&mut a.hk, &b.hk);
            add(&mut a.w, &b.w);
            for (x, y) in a.cum_c.iter_mut().zip(&b.cum_c) {
                add(x, y);
            }
            for (x, y) in a.prefix_a.iter_mut().zip(&b.prefix_a) {
                add(x, y);
            }
        }
        for (a, b) in self.g.iter_mut().zip(&other.g) {
            add(a, b);
        }
    }
}

pub(super) fn add(a: &mut [f64], b: &[f64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += *y;
    }
}

/// The per-row work arrays of one chunk.
struct Scratch {
    hk: Vec<f64>,
    wl: Vec<f64>,
    we: Vec<f64>,
    w: Vec<f64>,
    c: Vec<Vec<f64>>,
    cum_c: Vec<Vec<f64>>,
    prefix_a: Vec<Vec<f64>>,
    prefix_vec: Vec<Vec<f64>>,
    xs: Vec<Vec<f64>>,
    theta: Vec<f64>,
}

impl Scratch {
    fn new(top: usize, ps: &[usize]) -> Self {
        let m = ps.len();
        let pk = ps.iter().copied().max().unwrap_or(0);
        Self {
            hk: vec![0.0; top + 1],
            wl: vec![0.0; top + 1],
            we: vec![0.0; (top + 1) * pk],
            w: vec![0.0; top],
            c: vec![vec![0.0; top]; m],
            cum_c: vec![vec![0.0; top + 1]; m],
            prefix_a: vec![vec![0.0; top + 1]; m],
            prefix_vec: ps.iter().map(|&p| vec![0.0; (top + 1) * p]).collect(),
            xs: ps.iter().map(|&p| vec![0.0; p]).collect(),
            theta: vec![0.0; m],
        }
    }
}

/// The result of the evaluation: the risk of every level at every needed
/// count and, with the influence function, the Gram matrix of the levels at
/// every needed count.
pub(super) struct Evaluated {
    /// `levels * needed`, `None` where the risk is not defined.
    pub(super) risk: Vec<Option<f64>>,
    /// `needed * levels * levels`, zero where a risk is not defined.
    pub(super) gram: Option<Vec<f64>>,
}

impl Csc<'_> {
    fn fill_row(&self, level: &LevelRows<'_>, row: usize, sc: &mut Scratch) -> Result<()> {
        for (m, model) in self.models.iter().enumerate() {
            level.fill(m, model.x, row, &mut sc.xs[m]);
            let mut eta = 0.0;
            for (a, b) in self.base.models[m].beta.iter().enumerate() {
                eta += sc.xs[m][a] * b;
            }
            let value = eta.exp();
            if !value.is_finite() {
                return Err(Error::input(format!(
                    "levels: the risk score of row {row} in model {m} is not finite"
                )));
            }
            sc.theta[m] = value;
        }
        Ok(())
    }

    fn path(&self, theta: &[f64], s: usize, top: usize) -> Path {
        let k = self.cause_model;
        let hazard_k = &self.base.models[k].strata[s].lam;
        let mut surv = Vec::with_capacity(top);
        let mut cif = Vec::with_capacity(top);
        let mut valid = top;
        let mut s_j = 1.0;
        let mut total = 0.0;
        for (j, &increment) in hazard_k.iter().enumerate().take(top) {
            if j > 0 {
                s_j = if self.product_limit {
                    let mut hazard = 0.0;
                    for (m, th) in theta.iter().enumerate() {
                        hazard += th * self.base.models[m].strata[s].lam[j - 1];
                    }
                    s_j * (1.0 - hazard)
                } else {
                    let mut cumulative = 0.0;
                    for (m, th) in theta.iter().enumerate() {
                        cumulative += self.base.models[m].strata[s].cum[j - 1] * th;
                    }
                    (-cumulative).exp()
                };
            }
            if valid == top && (s_j.is_nan() || s_j < 0.0) {
                valid = j;
            }
            total += s_j * (increment * theta[k]);
            surv.push(s_j);
            cif.push(total);
        }
        Path { surv, cif, valid }
    }

    /// Add the row `i` of the population to the averages of a level.
    #[expect(
        clippy::too_many_arguments,
        reason = "the arrays of one row of one pass"
    )]
    fn add_row(
        &self,
        acc: &mut LevelAcc,
        infl: Option<&Influence>,
        i: usize,
        path: &Path,
        needed: &[usize],
        top: usize,
        sc: &mut Scratch,
    ) {
        let wt = self.weight[i];
        let s = self.stratum[i];
        acc.min_valid = acc.min_valid.min(path.valid);
        for (u, &c) in needed.iter().enumerate() {
            if c <= path.valid {
                acc.fbar[u] += wt * path.cif[c - 1];
            }
        }
        let Some(infl) = infl else {
            return;
        };
        let k = self.cause_model;
        let n_models = self.models.len();
        let pk = infl.models[k].p;
        let theta = &sc.theta;
        let lam_k = &self.base.models[k].strata[s].lam;
        let sums_k = &infl.models[k].strata[s];
        sc.w.fill(0.0);
        for c in &mut sc.c {
            c.fill(0.0);
        }
        for (j, &increment) in lam_k.iter().enumerate().take(top) {
            let hazard = theta[k] * increment;
            if hazard > 0.0 {
                sc.w[j] = path.surv[j] * theta[k];
                if j >= 1 {
                    for (c, th) in sc.c.iter_mut().zip(theta) {
                        c[j] = path.surv[j] * hazard * th;
                    }
                }
            }
        }
        sc.hk[0] = 0.0;
        sc.wl[0] = 0.0;
        sc.we[..pk].fill(0.0);
        for (j, &increment) in lam_k.iter().enumerate().take(top) {
            sc.hk[j + 1] = sc.hk[j] + sc.w[j] * sums_k.lam_over_s0[j];
            sc.wl[j + 1] = sc.wl[j] + sc.w[j] * increment;
            for a in 0..pk {
                sc.we[(j + 1) * pk + a] = sc.we[j * pk + a] + sc.w[j] * sums_k.e_lam[j * pk + a];
            }
        }
        for m in 0..n_models {
            let p = infl.models[m].p;
            let sums = &infl.models[m].strata[s];
            let cum = &self.base.models[m].strata[s].cum;
            sc.cum_c[m][0] = 0.0;
            sc.prefix_a[m][0] = 0.0;
            sc.prefix_vec[m][..p].fill(0.0);
            for j in 0..top {
                sc.cum_c[m][j + 1] = sc.cum_c[m][j] + sc.c[m][j];
                let (a_prev, cum_prev) = if j >= 1 {
                    (sums.cum_lam_over_s0[j - 1], cum[j - 1])
                } else {
                    (0.0, 0.0)
                };
                sc.prefix_a[m][j + 1] = sc.prefix_a[m][j] + sc.c[m][j] * a_prev;
                for a in 0..p {
                    let e_prev = if j >= 1 {
                        sums.cum_e_lam[(j - 1) * p + a]
                    } else {
                        0.0
                    };
                    sc.prefix_vec[m][(j + 1) * p + a] = sc.prefix_vec[m][j * p + a]
                        + sc.c[m][j] * (e_prev - sc.xs[m][a] * cum_prev);
                }
            }
        }
        let sa = &mut acc.strata[s];
        for jt in 0..=top {
            sa.hk[jt] += wt * sc.hk[jt];
            for m in 0..n_models {
                sa.cum_c[m][jt] += wt * sc.cum_c[m][jt];
                sa.prefix_a[m][jt] += wt * sc.prefix_a[m][jt];
            }
        }
        for j in 0..top {
            sa.w[j] += wt * sc.w[j];
        }
        for m in 0..n_models {
            let p = infl.models[m].p;
            for jt in 0..=top {
                for a in 0..p {
                    let mut value = sc.prefix_vec[m][jt * p + a];
                    if m == k {
                        value += sc.xs[k][a] * sc.wl[jt] - sc.we[jt * pk + a];
                    }
                    acc.g[m][jt * p + a] += wt * value;
                }
            }
        }
    }

    /// The influence of the models on the standardised risk of a level at
    /// the grid count `jt`, for the training row `i`.
    fn model_part(&self, infl: &Influence, acc: &LevelAcc, i: usize, jt: usize) -> f64 {
        let k = self.cause_model;
        let q = infl.count_le[i];
        let s = infl.stratum[i];
        let mut value = 0.0;
        for (train, g) in infl.models.iter().zip(&acc.g) {
            let p = train.p;
            let row = &train.if_beta[i * p..(i + 1) * p];
            let coefficient = &g[jt * p..(jt + 1) * p];
            for (r, c) in row.iter().zip(coefficient) {
                value += r * c;
            }
        }
        let sa = &acc.strata[s];
        value -= infl.models[k].risk[i] * sa.hk[jt.min(q)];
        if infl.models[k].event[i] && q <= jt {
            value += sa.w[q - 1] * infl.models[k].event_over_s0[i];
        }
        for (m, train) in infl.models.iter().enumerate() {
            let sums = &train.strata[s];
            let upto = q.min(jt - 1);
            let mut baseline = sa.prefix_a[m][upto + 1];
            if q >= 1 && jt > q + 1 {
                baseline += sums.cum_lam_over_s0[q - 1] * (sa.cum_c[m][jt] - sa.cum_c[m][q + 1]);
            }
            value += train.risk[i] * baseline;
            if train.event[i] {
                let from = q.max(1);
                if jt > from {
                    value -= train.event_over_s0[i] * (sa.cum_c[m][jt] - sa.cum_c[m][from]);
                }
            }
        }
        value
    }

    /// Evaluate the levels at the sorted distinct counts `needed`, all positive.
    pub(super) fn evaluate(
        &self,
        levels: &[LevelRows<'_>],
        infl: Option<&Influence>,
        needed: &[usize],
    ) -> Result<Evaluated> {
        let n_levels = levels.len();
        let top = needed.last().copied().unwrap_or(0);
        let n_needed = needed.len();
        let ps: Vec<usize> = self.models.iter().map(|m| m.x.ncols()).collect();
        let n_strata = self.base.n_strata;
        let n = self.rows.len();
        let mut accs = Vec::with_capacity(n_levels);
        if n_needed > 0 {
            for level in levels {
                let acc = ordered_reduce(
                    n,
                    |range| {
                        let mut acc = LevelAcc::new(n_needed, n_strata, top, &ps, infl.is_some());
                        let mut sc = Scratch::new(top, &ps);
                        for i in range {
                            self.fill_row(level, self.rows[i], &mut sc)?;
                            let path = self.path(&sc.theta, self.stratum[i], top);
                            self.add_row(&mut acc, infl, i, &path, needed, top, &mut sc);
                        }
                        Ok(acc)
                    },
                    |a, b| a.merge(&b),
                )?;
                accs.push(acc);
            }
        }
        let mut risk = vec![None; n_levels * n_needed];
        for (l, acc) in accs.iter().enumerate() {
            for (u, &c) in needed.iter().enumerate() {
                if c <= acc.min_valid {
                    risk[l * n_needed + u] = Some(acc.fbar[u]);
                }
            }
        }
        let Some(infl) = infl else {
            return Ok(Evaluated { risk, gram: None });
        };
        if n_needed == 0 {
            return Ok(Evaluated {
                risk,
                gram: Some(Vec::new()),
            });
        }
        let size = n_needed * n_levels * n_levels;
        let gram = ordered_reduce(
            n,
            |range| {
                let mut gram = vec![0.0; size];
                let mut sc = Scratch::new(top, &ps);
                let mut own = vec![0.0; n_levels];
                for i in range {
                    // The row's own risk at each needed count, per level.
                    let mut own_all = vec![0.0; n_levels * n_needed];
                    for (l, level) in levels.iter().enumerate() {
                        self.fill_row(level, self.rows[i], &mut sc)?;
                        let path = self.path(&sc.theta, self.stratum[i], top);
                        for (u, &c) in needed.iter().enumerate() {
                            own_all[l * n_needed + u] = if c <= path.valid {
                                path.cif[c - 1]
                            } else {
                                f64::NAN
                            };
                        }
                    }
                    let wt = self.weight[i];
                    for (u, &c) in needed.iter().enumerate() {
                        for l in 0..n_levels {
                            own[l] = match risk[l * n_needed + u] {
                                Some(fbar) => {
                                    self.model_part(infl, &accs[l], i, c)
                                        + wt * (own_all[l * n_needed + u] - fbar)
                                }
                                None => 0.0,
                            };
                        }
                        let base = u * n_levels * n_levels;
                        for a in 0..n_levels {
                            for b in 0..n_levels {
                                gram[base + a * n_levels + b] += own[a] * own[b];
                            }
                        }
                    }
                }
                Ok(gram)
            },
            |a, b| add(a, &b),
        )?;
        Ok(Evaluated {
            risk,
            gram: Some(gram),
        })
    }
}

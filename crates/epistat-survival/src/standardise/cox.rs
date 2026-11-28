//! The standardised risk of one Cox model, `1 - exp(-theta L(t))` averaged
//! over the population, and its influence function. See the module
//! documentation of [`standardise`](super) for the formulas.

use super::csc::{Evaluated, add};
use super::{LevelRows, ordered_reduce};
use crate::absrisk::influence::{Baselines, Influence, ModelInput};
use crate::error::{Error, Result};

pub(super) struct CoxEval<'a> {
    pub(super) model: &'a ModelInput<'a>,
    pub(super) base: &'a Baselines,
    pub(super) rows: &'a [usize],
    pub(super) stratum: &'a [usize],
    pub(super) weight: &'a [f64],
}

/// The averages of one level at one count.
struct Acc {
    fbar: f64,
    /// `sum S theta` per stratum.
    a: Vec<f64>,
    /// `sum S theta (L x - int E dL)`, `p` values.
    g: Vec<f64>,
}

impl CoxEval<'_> {
    fn theta(&self, level: &LevelRows<'_>, row: usize, x: &mut [f64]) -> Result<f64> {
        level.fill(0, self.model.x, row, x);
        let mut eta = 0.0;
        for (a, b) in self.base.models[0].beta.iter().enumerate() {
            eta += x[a] * b;
        }
        // The person's own offset, measured from the mean offset the baseline
        // is at.
        if let Some(offset) = self.model.data.offset() {
            eta += offset[row] - self.base.models[0].offset_centre;
        }
        let value = f64::exp(eta);
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Error::undefined(
                "risk score",
                format!("levels: the risk score of row {row} is not finite"),
            ))
        }
    }

    pub(super) fn evaluate(
        &self,
        levels: &[LevelRows<'_>],
        infl: Option<&Influence>,
        needed: &[usize],
    ) -> Result<Evaluated> {
        let n_levels = levels.len();
        let n_needed = needed.len();
        let n = self.rows.len();
        let p = self.model.x.ncols();
        let n_strata = self.base.n_strata;
        let mut risk = vec![None; n_levels * n_needed];
        let mut accs: Vec<Vec<Acc>> = Vec::with_capacity(n_levels);
        for (l, level) in levels.iter().enumerate() {
            let mut per_count = Vec::with_capacity(n_needed);
            for (u, &c) in needed.iter().enumerate() {
                let acc = ordered_reduce(
                    n,
                    |range| {
                        let mut acc = Acc {
                            fbar: 0.0,
                            a: vec![0.0; n_strata],
                            g: vec![0.0; if infl.is_some() { p } else { 0 }],
                        };
                        let mut x = vec![0.0; p];
                        for i in range {
                            let s = self.stratum[i];
                            let theta = self.theta(level, self.rows[i], &mut x)?;
                            let cum = self.base.models[0].strata[s].cum[c - 1];
                            let surv = (-theta * cum).exp();
                            let wt = self.weight[i];
                            acc.fbar += wt * (1.0 - surv);
                            if let Some(infl) = infl {
                                let st = wt * surv * theta;
                                acc.a[s] += st;
                                let sums = &infl.models[0].strata[s];
                                for (a, g) in acc.g.iter_mut().enumerate() {
                                    *g += st * (cum * x[a] - sums.cum_e_lam[(c - 1) * p + a]);
                                }
                            }
                        }
                        Ok(acc)
                    },
                    |a, b| {
                        a.fbar += b.fbar;
                        add(&mut a.a, &b.a);
                        add(&mut a.g, &b.g);
                    },
                )?;
                risk[l * n_needed + u] = Some(acc.fbar);
                per_count.push(acc);
            }
            accs.push(per_count);
        }
        let Some(infl) = infl else {
            let std_ok = vec![true; risk.len()];
            return Ok(Evaluated {
                risk,
                std_ok,
                gram: None,
            });
        };
        let size = n_needed * n_levels * n_levels;
        let gram = ordered_reduce(
            n,
            |range| {
                let mut gram = vec![0.0; size];
                let mut x = vec![0.0; p];
                let mut own = vec![0.0; n_levels];
                for i in range {
                    let s = infl.stratum[i];
                    let q = infl.count_le[i];
                    let q0 = infl.count_le0[i];
                    let w_train = infl.weight[i];
                    let train = &infl.models[0];
                    let sums = &train.strata[s];
                    let wt = self.weight[i];
                    for (u, &c) in needed.iter().enumerate() {
                        for (l, level) in levels.iter().enumerate() {
                            let acc = &accs[l][u];
                            let theta = self.theta(level, self.rows[i], &mut x)?;
                            let cum = self.base.models[0].strata[s].cum[c - 1];
                            let f_i = 1.0 - (-theta * cum).exp();
                            // The coefficients' part has the case weight in the
                            // score residual already.
                            let mut beta_part = 0.0;
                            for (r, g) in train.if_beta[i * p..(i + 1) * p].iter().zip(&acc.g) {
                                beta_part += r * g;
                            }
                            let cum_at = |m: usize| {
                                if m >= 1 {
                                    sums.cum_lam_over_s0[m - 1]
                                } else {
                                    0.0
                                }
                            };
                            let mut value =
                                -train.risk[i] * acc.a[s] * (cum_at(q.min(c)) - cum_at(q0.min(c)));
                            if train.event[i] && q <= c {
                                value += acc.a[s] * train.event_over_s0[i];
                            }
                            own[l] = beta_part + w_train * value + wt * (f_i - acc.fbar);
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
        let std_ok = vec![true; risk.len()];
        Ok(Evaluated {
            risk,
            std_ok,
            gram: Some(gram),
        })
    }
}

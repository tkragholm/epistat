//! The influence function of the absolute risk, by prefix sums over the event
//! times. See the module documentation of [`absrisk`](super) for the formulas.
//!
//! Index conventions: the grid positions are `0..J`, the position `j` is the
//! event time `s_j`. For a training row `i` with exit time `T_i`, `q_i` is the
//! number of grid times at or before `T_i`, so that `s_j <= T_i` is `j < q_i`
//! and the position of `T_i` itself, when it is a grid time, is `q_i - 1`.
//! A prefix array `P` holds `P[a] = sum over j < a`.

use super::{CauseSpecificCox, Context, Path};
use crate::cox::{CoxOptions, Ties, score_residuals};
use crate::error::{Error, Result};

/// What one model contributes about the training rows.
struct TrainModel {
    p: usize,
    /// `exp(x_i' beta)` of every training row.
    risk: Vec<f64>,
    /// The influence function of the coefficients, `n` rows of `p`.
    if_beta: Vec<f64>,
    /// `1 / S0(T_i)` for a row with the event of the model at `T_i`, else 0.
    event_over_s0: Vec<f64>,
    /// Whether the row has the event of the model.
    event: Vec<bool>,
    strata: Vec<StratumSums>,
}

/// The risk-set sums of one model in one stratum on the grid.
struct StratumSums {
    /// `dL(s_j) / S0(s_j)`, 0 at a position without an event of the model.
    lam_over_s0: Vec<f64>,
    /// The running sum of `lam_over_s0`.
    cum_lam_over_s0: Vec<f64>,
    /// `E(s_j) dL(s_j)`, `p` values per position.
    e_lam: Vec<f64>,
    /// The running sum of `e_lam`, `p` values per position.
    cum_e_lam: Vec<f64>,
}

/// The influence functions of the models at the training rows, for the rows
/// of positive weight.
pub(super) struct Influence {
    n: usize,
    /// `q_i` of every training row.
    count_le: Vec<usize>,
    stratum: Vec<usize>,
    models: Vec<TrainModel>,
}

impl Influence {
    pub(super) fn new(model: &CauseSpecificCox<'_>, ctx: &Context) -> Result<Self> {
        let data = &model.data;
        if data.followup()?.start().is_some() {
            return Err(Error::input(
                "std_err: the influence function is for data without entry times",
            ));
        }
        if data
            .weight()
            .is_some_and(|w| w.iter().any(|&w| w > 0.0 && w - 1.0 != 0.0))
        {
            return Err(Error::input(
                "std_err: the influence function is for case weights of 1",
            ));
        }
        if model.options.iter().any(|o| o.offset.is_some()) {
            return Err(Error::input(
                "std_err: the influence function is for models without an offset",
            ));
        }
        let rows: Vec<usize> = data.used_rows().map(|i| i as usize).collect();
        let n = rows.len();
        let stop = data.followup()?.stop();
        let stratum: Vec<usize> = rows
            .iter()
            .map(|&i| data.stratum().map_or(0, |c| c[i] as usize))
            .collect();
        let count_le: Vec<usize> = rows
            .iter()
            .map(|&i| ctx.grid.partition_point(|&g| g <= stop[i]))
            .collect();
        let n_strata = data.n_strata().max(1);
        let mut models = Vec::with_capacity(ctx.models.len());
        for m in 0..ctx.models.len() {
            let x = model.designs[m];
            let p = x.ncols();
            let beta = &ctx.models[m].beta;
            let model_data = model.model_data(m);
            let event: Vec<bool> = {
                let surv = model_data.followup()?;
                let code = model.options[m].cause.unwrap_or(1);
                let status = surv.status();
                rows.iter()
                    .map(|&i| i64::from(status[i]) == i64::from(code))
                    .collect()
            };
            let risk: Vec<f64> = rows
                .iter()
                .map(|&i| {
                    let mut eta = 0.0;
                    for (a, b) in beta.iter().enumerate() {
                        eta += x[(i, a)] * b;
                    }
                    eta.exp()
                })
                .collect();
            if risk.iter().any(|v| !v.is_finite() || *v <= 0.0) {
                return Err(Error::input(format!(
                    "model {m}: a risk score of the fit is not finite"
                )));
            }
            let v = covariance(model, m)?;
            let options = CoxOptions {
                ties: Ties::Breslow,
                ..model.options[m]
            };
            let coefficients: Vec<Option<f64>> = beta.iter().copied().map(Some).collect();
            let scores = score_residuals(&model_data, x, &coefficients, &options)?;
            let mut if_beta = vec![0.0; n * p];
            for (i, &row) in rows.iter().enumerate() {
                for a in 0..p {
                    let mut total = 0.0;
                    for (b, column) in scores.iter().enumerate() {
                        let u = column.as_ref().map_or(0.0, |c| c[row]);
                        total += v[a * p + b] * u;
                    }
                    if_beta[i * p + a] = total;
                }
            }
            let mut sums = Vec::with_capacity(n_strata);
            let mut event_over_s0 = vec![0.0; n];
            for s in 0..n_strata {
                let jumps = &ctx.models[m].strata[s].jumps;
                let lam = &ctx.models[m].strata[s].lam;
                let (s0, e) = risk_sets(
                    &rows,
                    &stratum,
                    s,
                    stop,
                    &risk,
                    x,
                    p,
                    jumps.iter().map(|&j| ctx.grid[j]),
                );
                let mut lam_over_s0 = vec![0.0; ctx.grid.len()];
                let mut e_lam = vec![0.0; ctx.grid.len() * p];
                for (l, &j) in jumps.iter().enumerate() {
                    lam_over_s0[j] = lam[j] / s0[l];
                    for a in 0..p {
                        e_lam[j * p + a] = e[l * p + a] * lam[j];
                    }
                }
                let mut cum_lam_over_s0 = Vec::with_capacity(lam_over_s0.len());
                let mut running = 0.0;
                for &v in &lam_over_s0 {
                    running += v;
                    cum_lam_over_s0.push(running);
                }
                let mut cum_e_lam = vec![0.0; e_lam.len()];
                let mut running = vec![0.0; p];
                for j in 0..ctx.grid.len() {
                    for a in 0..p {
                        running[a] += e_lam[j * p + a];
                        cum_e_lam[j * p + a] = running[a];
                    }
                }
                // 1 / S0 at the event time of the rows of this stratum.
                for (i, &row) in rows.iter().enumerate() {
                    if stratum[i] == s && event[i] {
                        let t = stop[row];
                        let l = jumps
                            .binary_search_by(|&j| ctx.grid[j].total_cmp(&t))
                            .map_err(|_| {
                                Error::input("internal: an event time is not a jump of its model")
                            })?;
                        event_over_s0[i] = 1.0 / s0[l];
                    }
                }
                sums.push(StratumSums {
                    lam_over_s0,
                    cum_lam_over_s0,
                    e_lam,
                    cum_e_lam,
                });
            }
            models.push(TrainModel {
                p,
                risk,
                if_beta,
                event_over_s0,
                event,
                strata: sums,
            });
        }
        Ok(Self {
            n,
            count_le,
            stratum,
            models,
        })
    }

    /// The standard error of the risk of one new row at each of the positive
    /// counts `needed` of grid times, sorted. `x` is the covariate row of each
    /// model.
    pub(super) fn row_std_err(
        &self,
        ctx: &Context,
        s: usize,
        theta: &[f64],
        x: &[Vec<f64>],
        path: &Path,
        needed: &[usize],
    ) -> Vec<f64> {
        let Some(&top) = needed.last() else {
            return Vec::new();
        };
        let k = ctx.cause_model;
        let n_models = ctx.models.len();
        let in_all = |m: usize| ctx.survival_models.contains(&m);
        let sums_k = &self.models[k].strata[s];
        let lam_k = &ctx.models[k].strata[s].lam;
        let pk = self.models[k].p;

        // Prefix arrays over j < a, for a up to top.
        let mut hk = vec![0.0; top + 1];
        let mut wl = vec![0.0; top + 1];
        let mut we = vec![0.0; (top + 1) * pk];
        let mut w = vec![0.0; top];
        let mut c: Vec<Vec<f64>> = vec![vec![0.0; top]; n_models];
        for j in 0..top {
            let hazard = theta[k] * lam_k[j];
            if hazard > 0.0 {
                w[j] = path.surv[j] * theta[k];
                if j >= 1 {
                    for m in 0..n_models {
                        if in_all(m) {
                            c[m][j] = path.surv[j] * hazard * theta[m];
                        }
                    }
                }
            }
        }
        for j in 0..top {
            hk[j + 1] = hk[j] + w[j] * sums_k.lam_over_s0[j];
            wl[j + 1] = wl[j] + w[j] * lam_k[j];
            for a in 0..pk {
                we[(j + 1) * pk + a] = we[j * pk + a] + w[j] * sums_k.e_lam[j * pk + a];
            }
        }
        let mut cum_c: Vec<Vec<f64>> = vec![vec![0.0; top + 1]; n_models];
        let mut prefix_a: Vec<Vec<f64>> = vec![vec![0.0; top + 1]; n_models];
        let mut prefix_vec: Vec<Vec<f64>> = Vec::with_capacity(n_models);
        for m in 0..n_models {
            let p = self.models[m].p;
            let sums = &self.models[m].strata[s];
            let cum = &ctx.models[m].strata[s].cum;
            let mut ce = vec![0.0; (top + 1) * p];
            for j in 0..top {
                cum_c[m][j + 1] = cum_c[m][j] + c[m][j];
                let (a_prev, cum_prev) = if j >= 1 {
                    (sums.cum_lam_over_s0[j - 1], cum[j - 1])
                } else {
                    (0.0, 0.0)
                };
                prefix_a[m][j + 1] = prefix_a[m][j] + c[m][j] * a_prev;
                for a in 0..p {
                    let e_prev = if j >= 1 {
                        sums.cum_e_lam[(j - 1) * p + a]
                    } else {
                        0.0
                    };
                    ce[(j + 1) * p + a] = ce[j * p + a] + c[m][j] * (e_prev - x[m][a] * cum_prev);
                }
            }
            prefix_vec.push(ce);
        }

        needed
            .iter()
            .map(|&jt| {
                // The coefficient vector of the influence function of each model.
                let g: Vec<Vec<f64>> = (0..n_models)
                    .map(|m| {
                        let p = self.models[m].p;
                        (0..p)
                            .map(|a| {
                                let mut value = 0.0;
                                if m == k {
                                    value += x[k][a] * wl[jt] - we[jt * pk + a];
                                }
                                if in_all(m) {
                                    value += prefix_vec[m][jt * p + a];
                                }
                                value
                            })
                            .collect()
                    })
                    .collect();
                let mut total = 0.0;
                for i in 0..self.n {
                    let q = self.count_le[i];
                    let mut value = 0.0;
                    for (train, coefficient) in self.models.iter().zip(&g) {
                        let p = train.p;
                        let row = &train.if_beta[i * p..(i + 1) * p];
                        for (r, c) in row.iter().zip(coefficient) {
                            value += r * c;
                        }
                    }
                    if self.stratum[i] == s {
                        value -= self.models[k].risk[i] * hk[jt.min(q)];
                        if self.models[k].event[i] && q <= jt {
                            value += w[q - 1] * self.models[k].event_over_s0[i];
                        }
                        for m in 0..n_models {
                            if !in_all(m) {
                                continue;
                            }
                            let train = &self.models[m];
                            let sums = &train.strata[s];
                            let upto = q.min(jt - 1);
                            let mut baseline = prefix_a[m][upto + 1];
                            if q >= 1 && jt > q + 1 {
                                baseline +=
                                    sums.cum_lam_over_s0[q - 1] * (cum_c[m][jt] - cum_c[m][q + 1]);
                            }
                            value += train.risk[i] * baseline;
                            if train.event[i] {
                                let from = q.max(1);
                                if jt > from {
                                    value -=
                                        train.event_over_s0[i] * (cum_c[m][jt] - cum_c[m][from]);
                                }
                            }
                        }
                    }
                    total += value * value;
                }
                total.sqrt()
            })
            .collect()
    }
}

/// The covariance of the coefficients of a model: the robust one when the fit
/// has it, as `var` of a `coxph` fit.
fn covariance(model: &CauseSpecificCox<'_>, m: usize) -> Result<Vec<f64>> {
    let fit = &model.fits[m];
    let chosen = fit.robust_covariance.as_ref().unwrap_or(&fit.covariance);
    let p = model.designs[m].ncols();
    if chosen.len() != p * p {
        return Err(Error::input(format!(
            "model {m}: the covariance is not p by p"
        )));
    }
    chosen
        .iter()
        .map(|v| v.ok_or_else(|| Error::input(format!("model {m}: the covariance is not defined"))))
        .collect()
}

/// `S0` and `E` of the rows of stratum `s` at the event times, which are
/// ascending: the sum of the risk scores of the rows with exit time at or after
/// the time, and the risk-score weighted mean of their covariates. The rows are
/// added from the latest exit time down.
#[expect(
    clippy::too_many_arguments,
    reason = "the columns of one risk-set pass"
)]
fn risk_sets(
    rows: &[usize],
    stratum: &[usize],
    s: usize,
    stop: &[f64],
    risk: &[f64],
    x: faer::MatRef<'_, f64>,
    p: usize,
    times: impl DoubleEndedIterator<Item = f64> + ExactSizeIterator,
) -> (Vec<f64>, Vec<f64>) {
    let mut order: Vec<usize> = (0..rows.len()).filter(|&i| stratum[i] == s).collect();
    order.sort_by(|&a, &b| stop[rows[a]].total_cmp(&stop[rows[b]]));
    let n_times = times.len();
    let mut s0 = vec![0.0; n_times];
    let mut e = vec![0.0; n_times * p];
    let mut s1 = vec![0.0; p];
    let mut acc = 0.0;
    let mut ptr = order.len();
    for (l, t) in times.enumerate().rev() {
        while ptr > 0 && stop[rows[order[ptr - 1]]] >= t {
            ptr -= 1;
            let i = order[ptr];
            acc += risk[i];
            for a in 0..p {
                s1[a] += risk[i] * x[(rows[i], a)];
            }
        }
        s0[l] = acc;
        for a in 0..p {
            e[l * p + a] = s1[a] / acc;
        }
    }
    (s0, e)
}

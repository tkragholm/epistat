//! A Cox model for the censoring times: the weights it gives and the influence
//! of its estimation on a metric.
//!
//! The censoring distribution of row `i` is `G_i(s) = exp(-q_i L0(s))`, with
//! `q_i = exp(x_i' b)` and `L0` the Breslow estimate of the baseline hazard of
//! the censoring times, whose jumps are `d_l / S0(u_l)` at the distinct
//! censoring times `u_l`. A row that had the event of interest or a competing
//! event is at risk of being censored until its own time, so a death at a
//! censoring time stays in the risk set `S0(u) = sum of q_j over T_j >= u`.
//!
//! # Influence
//!
//! A metric that depends on the weights `1 / G_i(s_i)` has, to first order, the
//! change `(1 / N) sum_i g_i dL_i(s_i)` when the cumulative hazards `L_i = q_i
//! L0` move by `dL_i`, where `g_i` is the derivative of `N` times the metric with
//! respect to the relative size of the weight of row `i`. If one row `k` is
//! given a little more weight, the cumulative hazard of row `i` moves by
//!
//! ```text
//! q_i { c_k(s) + (L0(s) x_i - H(s))' f_k },
//! c_k(s) = d_k 1(T_k <= s) / S0(T_k) - q_k C(min(T_k, s)),
//! C(v) = sum over u_l <= v of d_l / S0(u_l)^2,
//! H(s) = sum over u_l <= s of d_l E(u_l) / S0(u_l),
//! ```
//!
//! with `d_k` 1 for a row that was censored, `E` the mean of `x` in the risk set
//! weighted by `q`, and `f_k` the influence of row `k` on the coefficients,
//! `V u_k` for the covariance `V` and the score residual `u_k`. The first part
//! of `c_k` is row `k`'s own jump and the second its presence in the risk sets,
//! and the `H` term is the effect of estimating `b` on the baseline. The sum
//! over `i` is done by cumulative sums over the distinct censoring times, so it
//! costs `O(n + m p)` per evaluation time.
//!
//! The score residuals and `V` are those of the Breslow likelihood at the
//! fitted coefficients, so with tied censoring times and the Efron fit the
//! covariance and the residuals are not exactly each other's inverse and
//! derivative. This is the form of `riskRegression::iidCox`.

use epistat_linalg::{Mat, MatRef};

use super::prepare::{Kind, Sorted};
use crate::cox::{CoxModelFit, CoxOptions, Ties, fit, score_residuals};
use crate::error::{Error, Result};
use crate::surv::SurvData;
use crate::timefix::TimeFix;

/// A Cox model for the censoring times, fitted by [`fit`](crate::cox::fit).
#[derive(Debug, Clone)]
pub struct CoxCensoring<'a> {
    /// The covariates of the censoring model, one row per row of the data in
    /// the data's order. No intercept column.
    pub x: MatRef<'a, f64>,
    /// How the model is fitted. `Score` fits the censoring model with
    /// `rms::cph`, which has `Efron` ties and stops at a relative change of
    /// the log partial likelihood below `1e-6`: that is `eps` here to reproduce
    /// its coefficients. The default is that of [`CoxOptions`]. The rule for
    /// near-ties, `cause` and `start` are not used: the times are those
    /// of the metric, already ordered by its own rule, and the status is
    /// censored or not.
    pub options: CoxOptions,
}

/// The fitted censoring model at the used rows, in sorted order.
#[derive(Debug, Clone)]
pub(super) struct CoxCens {
    p: usize,
    /// `exp(x_i' b)`.
    q: Vec<f64>,
    /// The covariates, `p` per sorted position.
    x: Vec<f64>,
    /// The distinct censoring times.
    grid: Vec<f64>,
    /// `L0` after the first `m` jumps, `m = 0..=grid.len()`.
    lam: Vec<f64>,
    /// `C` after the first `m` jumps.
    c_cum: Vec<f64>,
    /// `S0` at each jump.
    s0: Vec<f64>,
    /// `H` after the first `m` jumps, `p` per `m`.
    h_cum: Vec<f64>,
    /// `f_k`, `p` per sorted position.
    dfbeta: Vec<f64>,
}

impl CoxCens {
    pub(super) fn new(sorted: &Sorted, spec: &CoxCensoring<'_>) -> Result<Self> {
        let n = sorted.n();
        let p = spec.x.ncols();
        if spec.x.nrows() != sorted.n_rows {
            return Err(Error::invalid(format!(
                "the censoring model's x has {} rows for {} rows of data",
                spec.x.nrows(),
                sorted.n_rows
            )));
        }
        if spec.options.ties == Ties::Exact {
            return Err(Error::unsupported(
                "the censoring model cannot use the exact ties method: its score residuals \
                 are not available",
            ));
        }
        let status: Vec<i32> = sorted
            .kind
            .iter()
            .map(|&k| i32::from(k == Kind::Censored))
            .collect();
        let data = SurvData::builder(&sorted.time, &status).build()?;
        let x = Mat::from_fn(n, p, |i, j| spec.x[(sorted.row[i] as usize, j)]);
        let options = CoxOptions {
            timefix: TimeFix::Exact,
            cause: None,
            start: None,
            ..spec.options.clone()
        };
        let model = fit(&data, x.as_ref(), &options)?;
        if !model.converged {
            return Err(Error::NotConverged {
                iterations: model.iterations,
            });
        }
        let beta: Vec<f64> = model
            .coefficients
            .iter()
            .map(|c| {
                c.ok_or_else(|| {
                    Error::undefined("the censoring model has an aliased or diverging coefficient")
                })
            })
            .collect::<Result<_>>()?;
        let q: Vec<f64> = (0..n)
            .map(|i| {
                let mut eta = 0.0;
                for (j, b) in beta.iter().enumerate() {
                    eta += x[(i, j)] * b;
                }
                eta.exp()
            })
            .collect();
        if q.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(Error::invalid(
                "a risk score of the censoring model is not finite",
            ));
        }
        let xs: Vec<f64> = (0..n)
            .flat_map(|i| (0..p).map(move |j| (i, j)))
            .map(|(i, j)| x[(i, j)])
            .collect();

        // Risk-set sums at each distinct censoring time, from the end.
        let mut grid: Vec<f64> = Vec::new();
        let mut d: Vec<f64> = Vec::new();
        for pos in 0..n {
            if sorted.kind[pos] == Kind::Censored {
                if grid.last() == Some(&sorted.time[pos]) {
                    *d.last_mut().expect("d and grid have the same length") += 1.0;
                } else {
                    grid.push(sorted.time[pos]);
                    d.push(1.0);
                }
            }
        }
        let m = grid.len();
        let mut s0 = vec![0.0; m];
        let mut e = vec![0.0; m * p];
        let mut acc = 0.0;
        let mut s1 = vec![0.0; p];
        let mut pos = n;
        for l in (0..m).rev() {
            while pos > 0 && sorted.time[pos - 1] >= grid[l] {
                pos -= 1;
                acc += q[pos];
                for (j, s) in s1.iter_mut().enumerate() {
                    *s += q[pos] * xs[pos * p + j];
                }
            }
            s0[l] = acc;
            for j in 0..p {
                e[l * p + j] = s1[j] / acc;
            }
        }
        let mut lam = vec![0.0; m + 1];
        let mut c_cum = vec![0.0; m + 1];
        let mut h_cum = vec![0.0; (m + 1) * p];
        for l in 0..m {
            lam[l + 1] = lam[l] + d[l] / s0[l];
            c_cum[l + 1] = c_cum[l] + d[l] / (s0[l] * s0[l]);
            for j in 0..p {
                h_cum[(l + 1) * p + j] = h_cum[l * p + j] + d[l] * e[l * p + j] / s0[l];
            }
        }
        let dfbeta = influence_of_coefficients(&data, x.as_ref(), &model, &options, p, n)?;
        Ok(Self {
            p,
            q,
            x: xs,
            grid,
            lam,
            c_cum,
            s0,
            h_cum,
            dfbeta,
        })
    }

    /// `L0(s-)`: the baseline hazard of the jumps before `time`.
    pub(super) fn lambda_before(&self, time: f64) -> f64 {
        self.lam[self.grid.partition_point(|&u| u < time)]
    }

    /// `L0(s)`: the baseline hazard of the jumps at or before `time`.
    pub(super) fn lambda_through(&self, time: f64) -> f64 {
        self.lam[self.grid.partition_point(|&u| u <= time)]
    }

    /// `G_i(s) = exp(-q_i L0)` for the baseline hazard `lambda` at sorted
    /// position `pos`.
    pub(super) fn survival(&self, pos: usize, lambda: f64) -> f64 {
        (-self.q[pos] * lambda).exp()
    }

    /// The influence of estimating the censoring model on a metric with the
    /// derivatives `g` (see the module documentation) at evaluation time `t`,
    /// for each sorted position.
    pub(super) fn term(&self, sorted: &Sorted, t: f64, g: &[f64]) -> Vec<f64> {
        let n = sorted.n();
        let p = self.p;
        let m = self.grid.len();
        // The number of jumps that enter the cumulative hazard of each row:
        // those before its own time if it had an event by t, else those up to t.
        let mut by_count = vec![0.0; m + 1];
        let mut w = vec![0.0; p];
        for (pos, &g_pos) in g.iter().enumerate().take(n) {
            let (count, used) = if sorted.time[pos] <= t {
                (
                    self.grid.partition_point(|&u| u < sorted.time[pos]),
                    sorted.kind[pos] != Kind::Censored,
                )
            } else {
                (self.grid.partition_point(|&u| u <= t), true)
            };
            if !used || g_pos == 0.0 {
                continue;
            }
            let a = g_pos * self.q[pos];
            by_count[count] += a;
            for (j, wj) in w.iter_mut().enumerate() {
                *wj += a * (self.lam[count] * self.x[pos * p + j] - self.h_cum[count * p + j]);
            }
        }
        // suffix[c] = sum of a over rows with count >= c.
        let mut suffix = vec![0.0; m + 2];
        for c in (0..=m).rev() {
            suffix[c] = suffix[c + 1] + by_count[c];
        }
        // below[c] = sum of a C(count) over rows with count < c.
        let mut below = vec![0.0; m + 2];
        for c in 0..=m {
            below[c + 1] = below[c] + by_count[c] * self.c_cum[c];
        }
        (0..n)
            .map(|k| {
                // Jumps at or before the row's own time.
                let reach = self.grid.partition_point(|&u| u <= sorted.time[k]);
                let own = if sorted.kind[k] == Kind::Censored {
                    // Its own jump is the `reach`-th, which counts for the rows
                    // whose count is above `reach - 1`.
                    suffix[reach] / self.s0[reach - 1]
                } else {
                    0.0
                };
                let risk_sets = self.q[k] * (below[reach] + self.c_cum[reach] * suffix[reach]);
                let mut beta_part = 0.0;
                for (j, wj) in w.iter().enumerate() {
                    beta_part += self.dfbeta[k * p + j] * wj;
                }
                own - risk_sets + beta_part
            })
            .collect()
    }
}

/// `V u_k` for every row: the model covariance times the score residual of the
/// Breslow likelihood at the fitted coefficients.
fn influence_of_coefficients(
    data: &SurvData<'_>,
    x: MatRef<'_, f64>,
    model: &CoxModelFit,
    options: &CoxOptions,
    p: usize,
    n: usize,
) -> Result<Vec<f64>> {
    if p == 0 {
        return Ok(Vec::new());
    }
    let breslow = CoxOptions {
        ties: Ties::Breslow,
        ..options.clone()
    };
    let scores = score_residuals(data, x, &model.coefficients, &breslow)?;
    let v: Vec<f64> = model
        .covariance
        .as_slice()
        .iter()
        .map(|c| c.ok_or_else(|| Error::invalid("the censoring model's covariance is not defined")))
        .collect::<Result<_>>()?;
    let mut out = vec![0.0; n * p];
    for i in 0..n {
        for a in 0..p {
            let mut total = 0.0;
            for (b, column) in scores.iter().enumerate() {
                let u = column.as_ref().map_or(0.0, |c| c[i]);
                total += v[a * p + b] * u;
            }
            out[i * p + a] = total;
        }
    }
    Ok(out)
}

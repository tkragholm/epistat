//! The influence of the Kaplan-Meier or Cox model of censoring on the weights of
//! [`ipcw`](super::ipcw).
//!
//! # Algorithm
//!
//! The weights are `w_k = exp(l_k)` with, for a row `k` and the censoring event
//! times `t_l` of its stratum,
//!
//! ```text
//! l_k = Lambda^cox_k - s Lambda^km_k              (Cox model, s = 1 if stabilised)
//! l_k = (1 - s) Lambda^km_k                       (Kaplan-Meier model)
//! Lambda^km_k  = sum over t_l in (entry_k, stop_k) of delta_l        (weight at the stop)
//!              = sum over t_l in (entry_k, start_k] of delta_l       (weight at the start)
//! Lambda^cox_k = sum over the rows q of the subject up to k of rho_q
//!                (sum over t_l in the interval of q of h_l)
//! ```
//!
//! with `delta_l = log Y_l - log (Y_l - d_l)` the jump of the Kaplan-Meier
//! censoring hazard, `Y_l` the number at risk and `d_l` the number censored at
//! `t_l`, `rho = exp(x' b)` the risk score, and `h_l` the increment of the
//! baseline cumulative hazard of the Cox model. The interval of the row `q` is
//! `(start_q, stop_q]` for the rows before `k` and `(start_k, stop_k)` for `k`
//! itself with the weight at the stop, which has no part for `k` with the weight at
//! the start.
//!
//! The increment is `h = W / S_0` for Breslow ties and `h = (W / d) sum over j < d of
//! 1 / (S_0 - (j / d) E_0)` for Efron ties, with `S_0` the sum of `rho` over the
//! risk set, `E_0` its sum over the censored rows at `t_l`, and `W` the weight of
//! those rows, as in [`crate::cox::basehaz`].
//!
//! Perturbing the case weight `omega_u` of a unit `u`, who owns some of the rows
//! of the censoring process, changes the quantities of the process as follows.
//! `Y_l` changes by the number of its rows at risk at `t_l` and `d_l` by the
//! number of its rows censored at `t_l`; `S_0` by the sum of `rho` over those at
//! risk, `E_0` by that over those censored, and `W` by the number censored; and the
//! coefficients move by `V s_u`, with `V` the covariance of the Cox fit and `s_u`
//! the sum of its score residuals over the rows of `u` (the derivative of the
//! partial log likelihood with respect to the case weight). With
//!
//! ```text
//! IF_u(delta_l) = at_u(l) (1 / Y_l - 1 / (Y_l - d_l)) + e_u(l) / (Y_l - d_l)
//! IF_u(h_l)     = h_W e_u(l) + h_S R_u(l) + h_E E_u(l) + (h_S S_1 + h_E E_1)' V s_u
//! ```
//!
//! and `at_u`, `e_u`, `R_u`, `E_u` the counts and sums of `u`, and with `h_W`, `h_S`,
//! `h_E` the derivatives of `h` with respect to `W`, `S_0`, `E_0` and `S_1`, `E_1`
//! the sums of `rho x` that `S_0` and `E_0` have, the influence of the weights
//! needed by [`Nuisance::add`] is a sum over the censoring event times, of the
//! coefficients of the log weights, called `A` below, times these. The coefficient
//! of time `t_l` is the sum of `a_k` over the rows whose log weight has the time in
//! its interval, found for all times with differences of a cumulative sum, so that
//! the cost is `O((n + L) log n)` per column of `a`.
//!
//! The part that goes through the coefficients is `kappa' V s_u` with `kappa` the
//! sum over rows of `a_k` times the derivative of `l_k` with respect to the
//! coefficients.

use epistat_linalg::{Mat, MatRef};
use epistat_stats::sum::Compensated;

use super::adjust::{Nuisance, WeightAt};
use super::ipcw::{CensoringModel, Ipcw, IpcwOptions, Parts, fit_parts};
use crate::cox::{CoxOptions, Ties, score_residuals};
use crate::error::{Error, Result};
use crate::surv::{SurvBuf, SurvData};
use crate::timefix::TimeFix;

/// The censoring event times of a stratum and what the risk sets there give.
struct Group {
    /// The distinct stops of the censoring events.
    times: Vec<f64>,
    /// `1 / Y - 1 / (Y - d)` and `1 / (Y - d)`: the derivatives of `delta`.
    km_at_risk: Vec<f64>,
    km_event: Vec<f64>,
    /// The Cox model: the increment of the baseline hazard, its derivatives with
    /// respect to `W`, `S_0` and `E_0`, `S_1` and `E_1` with `p` values per time,
    /// and the cumulative hazard after each time with 0 first.
    h_w: Vec<f64>,
    h_s: Vec<f64>,
    h_e: Vec<f64>,
    s1: Vec<f64>,
    e1: Vec<f64>,
    cumulative: Vec<f64>,
}

struct CoxPart {
    p: usize,
    risk: Vec<f64>,
    /// `V` row-major.
    covariance: Vec<f64>,
    /// `s_u`, one row per unit.
    score: Mat<f64>,
}

pub(super) struct ProcessNuisance<'a> {
    parts: Parts<'a>,
    at: WeightAt,
    unit_of_row: Vec<usize>,
    groups: Vec<Group>,
    cox: Option<CoxPart>,
    /// The factor of `Lambda^km` in the log weight.
    km_sign: f64,
}

/// For the rows sorted by stop and by start: the sums of `value` over the rows
/// with `start < t <= stop` at each time of `times`, ascending.
fn at_risk_sums(
    by_stop: &[usize],
    by_start: &[usize],
    stop: &[f64],
    start: &[f64],
    times: &[f64],
    value: &dyn Fn(usize) -> f64,
) -> Vec<f64> {
    let mut out = vec![0.0; times.len()];
    let sweep = |order: &[usize], key: &[f64]| -> Vec<f64> {
        let mut sums = vec![0.0; times.len()];
        let mut acc = Compensated::new();
        let mut j = order.len();
        for l in (0..times.len()).rev() {
            while j > 0 && key[order[j - 1]] >= times[l] {
                acc.add(value(order[j - 1]));
                j -= 1;
            }
            sums[l] = acc.value();
        }
        sums
    };
    let (a, b) = (sweep(by_stop, stop), sweep(by_start, start));
    for l in 0..times.len() {
        out[l] = a[l] - b[l];
    }
    out
}

/// The sums of `value` over the censoring rows at each time, `events` sorted by
/// stop.
fn event_sums(
    events: &[usize],
    stop: &[f64],
    times: &[f64],
    value: &dyn Fn(usize) -> f64,
) -> Vec<f64> {
    let mut out = vec![0.0; times.len()];
    let mut l = 0;
    for &r in events {
        while times[l] < stop[r] {
            l += 1;
        }
        out[l] += value(r);
    }
    out
}

/// The number of censoring rows at each time, `events` sorted by stop.
fn event_counts(events: &[usize], stop: &[f64], times: &[f64]) -> Vec<usize> {
    let mut out = vec![0; times.len()];
    let mut l = 0;
    for &r in events {
        while times[l] < stop[r] {
            l += 1;
        }
        out[l] += 1;
    }
    out
}

impl<'a> ProcessNuisance<'a> {
    /// Fit the model of censoring, and give the weights of the rows that the
    /// outcome uses with the term of the variance.
    pub fn new(
        data: &SurvData<'_>,
        model: CensoringModel<'a>,
        opts: &IpcwOptions,
        at: WeightAt,
        unit_of_row: Vec<usize>,
        n_units: usize,
    ) -> Result<(Vec<f64>, Self)> {
        if data.weight().is_some() {
            return Err(Error::invalid(
                "data: carries case weights, but the weights come from the weighting",
            ));
        }
        if opts.ties == Ties::Exact && matches!(model, CensoringModel::Cox { .. }) {
            return Err(Error::invalid(
                "ties: the variance of weights from a Cox model of censoring is not \
                 available for the exact method",
            ));
        }
        let (
            Ipcw {
                weight_at_stop,
                weight_at_start,
                ..
            },
            parts,
        ) = fit_parts(data, model, opts)?;
        let weights = match at {
            WeightAt::Stop => weight_at_stop,
            WeightAt::Start => weight_at_start,
        };
        let n = parts.stop.len();
        let zero_start = vec![0.0; n];
        let start: &[f64] = parts.start.as_deref().unwrap_or(&zero_start);
        let stop = &parts.stop;

        // The Cox part: risk scores, covariance and the unit sums of the score
        // residuals.
        let cox = match (&parts.x, &parts.coefficients, &parts.covariance) {
            (Some(x), Some(coefficients), Some(covariance)) => {
                let p = x.ncols();
                let mut beta = Vec::with_capacity(p);
                for (j, c) in coefficients.iter().enumerate() {
                    beta.push(c.ok_or_else(|| {
                        Error::invalid(format!("the censoring model: column {j} is aliased"))
                    })?);
                }
                let v: Vec<f64> = covariance
                    .as_slice()
                    .iter()
                    .map(|c| {
                        c.ok_or_else(|| {
                            Error::undefined("the covariance of the censoring model is undefined")
                        })
                    })
                    .collect::<Result<_>>()?;
                let risk: Vec<f64> = (0..n)
                    .map(|r| {
                        let eta: f64 = (0..p).map(|j| x[(r, j)] * beta[j]).sum();
                        eta.exp()
                    })
                    .collect();
                let flipped: Vec<i32> = parts.censored.iter().map(|&c| i32::from(c)).collect();
                let codes: Vec<u32> = parts.group.iter().map(|&g| g as u32).collect();
                let mut builder = SurvBuf::builder(stop.clone(), flipped);
                if let Some(s) = &parts.start {
                    builder = builder.start(s.clone());
                }
                let buf = builder.stratum(codes).build()?;
                let process = buf.as_data();
                let cox_opts = CoxOptions {
                    ties: parts.ties,
                    timefix: TimeFix::Exact,
                    ..CoxOptions::default()
                };
                let residuals = score_residuals(&process, *x, coefficients, &cox_opts)?;
                let mut score = Mat::<f64>::zeros(n_units, p);
                for (j, column) in residuals.iter().enumerate() {
                    let column = column.as_ref().ok_or_else(|| {
                        Error::undefined("a score residual of the censoring model is undefined")
                    })?;
                    for r in 0..n {
                        score[(unit_of_row[r], j)] += column[r];
                    }
                }
                Some(CoxPart {
                    p,
                    risk,
                    covariance: v,
                    score,
                })
            }
            _ => None,
        };
        let km_sign = if cox.is_some() {
            -f64::from(u8::from(parts.stabilise))
        } else {
            1.0 - f64::from(u8::from(parts.stabilise))
        };

        let mut groups = Vec::with_capacity(parts.n_groups);
        for g in 0..parts.n_groups {
            let rows: Vec<usize> = (0..n).filter(|&r| parts.group[r] == g).collect();
            let mut events: Vec<usize> = rows
                .iter()
                .copied()
                .filter(|&r| parts.censored[r])
                .collect();
            events.sort_unstable_by(|&a, &b| stop[a].total_cmp(&stop[b]));
            let mut times: Vec<f64> = events.iter().map(|&r| stop[r]).collect();
            times.dedup();
            let mut by_stop = rows.clone();
            by_stop.sort_unstable_by(|&a, &b| stop[a].total_cmp(&stop[b]));
            let mut by_start = rows.clone();
            by_start.sort_unstable_by(|&a, &b| start[a].total_cmp(&start[b]));
            let l = times.len();
            let y = at_risk_sums(&by_stop, &by_start, stop, start, &times, &|_| 1.0);
            let counts = event_counts(&events, stop, &times);
            let d: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
            // A time at which everyone at risk is censored has an infinite jump
            // `delta`, which only rows with an infinite weight pass, and those are an
            // error of the fit of the weights. No row has it in its interval, so its
            // derivatives are set to 0 and the rounding of the cumulative sums of
            // coefficients cannot multiply an infinity.
            let live = |i: usize| y[i] - d[i] > 0.0;
            let km_at_risk: Vec<f64> = (0..l)
                .map(|i| {
                    if live(i) {
                        1.0 / y[i] - 1.0 / (y[i] - d[i])
                    } else {
                        0.0
                    }
                })
                .collect();
            let km_event: Vec<f64> = (0..l)
                .map(|i| if live(i) { 1.0 / (y[i] - d[i]) } else { 0.0 })
                .collect();
            let (mut h_w, mut h_s, mut h_e) = (vec![0.0; l], vec![0.0; l], vec![0.0; l]);
            let (mut s1, mut e1) = (Vec::new(), Vec::new());
            let mut cumulative = vec![0.0; l + 1];
            if let (Some(c), Some(x)) = (&cox, &parts.x) {
                let p = c.p;
                let risk = &c.risk;
                let s0 = at_risk_sums(&by_stop, &by_start, stop, start, &times, &|r| risk[r]);
                let e0 = event_sums(&events, stop, &times, &|r| risk[r]);
                s1 = vec![0.0; l * p];
                e1 = vec![0.0; l * p];
                for j in 0..p {
                    let sj = at_risk_sums(&by_stop, &by_start, stop, start, &times, &|r| {
                        risk[r] * x[(r, j)]
                    });
                    let ej = event_sums(&events, stop, &times, &|r| risk[r] * x[(r, j)]);
                    for i in 0..l {
                        s1[i * p + j] = sj[i];
                        e1[i * p + j] = ej[i];
                    }
                }
                let bad =
                    || Error::undefined("the sum of the risk scores of a risk set is not positive");
                for i in 0..l {
                    let count = counts[i];
                    let w = count as f64;
                    let h = if parts.ties == Ties::Breslow {
                        if !(s0[i] > 0.0 && s0[i].is_finite()) {
                            return Err(bad());
                        }
                        h_w[i] = 1.0 / s0[i];
                        h_s[i] = -w / (s0[i] * s0[i]);
                        w / s0[i]
                    } else {
                        let (mut sum1, mut sum2, mut sum3) = (0.0, 0.0, 0.0);
                        for j in 0..count {
                            let f = j as f64 / w;
                            let den = s0[i] - e0[i] * f;
                            if !(den > 0.0 && den.is_finite()) {
                                return Err(bad());
                            }
                            let t = 1.0 / den;
                            sum1 += t / w;
                            sum2 += t * t / w;
                            sum3 += f * t * t / w;
                        }
                        h_w[i] = sum1;
                        h_s[i] = -w * sum2;
                        h_e[i] = w * sum3;
                        w * sum1
                    };
                    cumulative[i + 1] = cumulative[i] + h;
                }
            }
            groups.push(Group {
                times,
                km_at_risk,
                km_event,
                h_w,
                h_s,
                h_e,
                s1,
                e1,
                cumulative,
            });
        }
        Ok((
            weights,
            Self {
                parts,
                at,
                unit_of_row,
                groups,
                cox,
                km_sign,
            },
        ))
    }
}

impl Nuisance for ProcessNuisance<'_> {
    fn add(&self, a: MatRef<'_, f64>, total: &mut Mat<f64>) -> Result<()> {
        let parts = &self.parts;
        let n = parts.stop.len();
        let zero_start = vec![0.0; n];
        let start: &[f64] = parts.start.as_deref().unwrap_or(&zero_start);
        let stop = &parts.stop;
        let stop_at = self.at == WeightAt::Stop;
        let mut s_row = vec![0.0; n];
        let mut p_row = vec![0.0; n];
        for c in 0..a.ncols() {
            // The sum of a over the rows of the subject from each row on, and
            // after it.
            let mut end = n;
            while end > 0 {
                let subject = parts.subject[parts.order[end - 1]];
                let mut begin = end - 1;
                while begin > 0 && parts.subject[parts.order[begin - 1]] == subject {
                    begin -= 1;
                }
                let mut later = 0.0;
                for pos in (begin..end).rev() {
                    let r = parts.order[pos];
                    p_row[r] = later;
                    later += a[(r, c)];
                    s_row[r] = if stop_at { later } else { p_row[r] };
                }
                end = begin;
            }
            // The coefficients of the censoring times, for the Kaplan-Meier part
            // and for the Cox part.
            let mut km_a: Vec<Vec<f64>> = self
                .groups
                .iter()
                .map(|g| vec![0.0; g.times.len() + 1])
                .collect();
            let mut cox_a: Vec<Vec<f64>> = km_a.clone();
            let mut cox_point: Vec<Vec<f64>> = km_a.clone();
            let mut kappa = vec![0.0; self.cox.as_ref().map_or(0, |c| c.p)];
            for r in 0..n {
                let g = parts.group[r];
                let grp = &self.groups[g];
                if self.km_sign != 0.0 {
                    let lo = grp.times.partition_point(|&t| t <= parts.entry[r]);
                    let hi = if stop_at {
                        grp.times.partition_point(|&t| t < stop[r])
                    } else {
                        grp.times.partition_point(|&t| t <= start[r])
                    };
                    if hi > lo {
                        let v = self.km_sign * a[(r, c)];
                        km_a[g][lo] += v;
                        km_a[g][hi] -= v;
                    }
                }
                if let Some(cox) = &self.cox {
                    let lo = grp.times.partition_point(|&t| t <= start[r]);
                    let hi = grp.times.partition_point(|&t| t < stop[r]);
                    let (open, point) = (cox.risk[r] * s_row[r], cox.risk[r] * p_row[r]);
                    if hi > lo {
                        cox_a[g][lo] += open;
                        cox_a[g][hi] -= open;
                    }
                    if hi < grp.times.len() && grp.times[hi].total_cmp(&stop[r]).is_eq() {
                        cox_point[g][hi] += point;
                    }
                    let x = parts
                        .x
                        .as_ref()
                        .map_or_else(|| unreachable!("a Cox part has a design"), |x| *x);
                    let h_open = grp.cumulative[hi]
                        - grp.cumulative[grp.times.partition_point(|&t| t <= start[r])];
                    let h_point = grp.cumulative[grp.times.partition_point(|&t| t <= stop[r])]
                        - grp.cumulative[hi];
                    let w = open * h_open + point * h_point;
                    for (j, k) in kappa.iter_mut().enumerate() {
                        *k += x[(r, j)] * w;
                    }
                }
            }
            // From differences to the coefficient of each time.
            for g in 0..self.groups.len() {
                let l = self.groups[g].times.len();
                let (mut ak, mut ac) = (0.0, 0.0);
                for i in 0..l {
                    ak += km_a[g][i];
                    ac += cox_a[g][i];
                    km_a[g][i] = ak;
                    cox_a[g][i] = ac + cox_point[g][i];
                }
                km_a[g].truncate(l);
                cox_a[g].truncate(l);
            }
            // The prefix sums of the coefficients times the at-risk derivatives.
            let mut pk: Vec<Vec<f64>> = Vec::with_capacity(self.groups.len());
            let mut pc: Vec<Vec<f64>> = Vec::with_capacity(self.groups.len());
            for (g, grp) in self.groups.iter().enumerate() {
                let l = grp.times.len();
                let (mut sk, mut sc) = (vec![0.0; l + 1], vec![0.0; l + 1]);
                for i in 0..l {
                    sk[i + 1] = sk[i]
                        + if km_a[g][i] == 0.0 {
                            0.0
                        } else {
                            km_a[g][i] * grp.km_at_risk[i]
                        };
                    sc[i + 1] = sc[i]
                        + if cox_a[g][i] == 0.0 {
                            0.0
                        } else {
                            cox_a[g][i] * grp.h_s[i]
                        };
                    if let Some(cox) = &self.cox {
                        let p = cox.p;
                        for (j, k) in kappa.iter_mut().enumerate() {
                            *k += cox_a[g][i]
                                * (grp.h_s[i] * grp.s1[i * p + j] + grp.h_e[i] * grp.e1[i * p + j]);
                        }
                    }
                }
                pk.push(sk);
                pc.push(sc);
            }
            // The rows' own terms.
            for r in 0..n {
                let g = parts.group[r];
                let grp = &self.groups[g];
                let lo = grp.times.partition_point(|&t| t <= start[r]);
                let hi = grp.times.partition_point(|&t| t <= stop[r]);
                let rho = self.cox.as_ref().map_or(0.0, |c| c.risk[r]);
                let mut v = pk[g][hi] - pk[g][lo] + rho * (pc[g][hi] - pc[g][lo]);
                if parts.censored[r] {
                    let i = hi - 1;
                    if km_a[g][i] != 0.0 {
                        v += km_a[g][i] * grp.km_event[i];
                    }
                    if self.cox.is_some() {
                        v += cox_a[g][i] * (grp.h_w[i] + rho * grp.h_e[i]);
                    }
                }
                total[(self.unit_of_row[r], c)] += v;
            }
            // The part through the coefficients.
            if let Some(cox) = &self.cox {
                let p = cox.p;
                let m: Vec<f64> = (0..p)
                    .map(|i| (0..p).map(|j| cox.covariance[i * p + j] * kappa[j]).sum())
                    .collect();
                for u in 0..cox.score.nrows() {
                    let mut s = 0.0;
                    for (j, &mj) in m.iter().enumerate() {
                        s += cox.score[(u, j)] * mj;
                    }
                    total[(u, c)] += s;
                }
            }
        }
        Ok(())
    }
}

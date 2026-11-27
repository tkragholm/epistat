//! A direct evaluation of the influence function of the absolute risk, one
//! training row and one event time at a time, written from the description of
//! the influence function in `epistat_survival::absrisk`. It is slow, has no
//! prefix sums, and also returns the sum of the absolute values of the terms,
//! which is the size the rounding error of the standard error scales with.

#![allow(dead_code, reason = "each test file uses part of this module")]
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    reason = "a plain transcription of the formulas; event times are compared exactly"
)]

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::predict::{BaselineOptions, Centring, FittedModel, basehaz};
use epistat_survival::cox::{CoxModelFit, CoxOptions, Ties, score_residuals};

/// One cause-specific model as the influence function needs it.
pub struct NModel {
    pub p: usize,
    /// `n` rows of `p`.
    pub x: Vec<Vec<f64>>,
    pub event: Vec<bool>,
    pub beta: Vec<f64>,
    /// Row-major `p` by `p`.
    pub v: Vec<f64>,
    /// `lam[stratum][grid position]`.
    pub lam: Vec<Vec<f64>>,
    /// The influence function of the coefficients of every training row.
    pub if_beta: Vec<Vec<f64>>,
}

pub struct NCase {
    pub stop: Vec<f64>,
    pub stratum: Vec<usize>,
    pub grid: Vec<f64>,
    pub models: Vec<NModel>,
    pub cause_model: usize,
    pub survival_models: Vec<usize>,
    pub product_limit: bool,
    /// Whether the derivative of the product limit is the exact one, else the
    /// first-order one of the exponential form.
    pub exact: bool,
}

/// The baselines of one model at zero covariates on a grid, per stratum, and
/// the influence function of its coefficients.
pub fn model_pieces(
    data: &SurvData<'_>,
    x: &Mat<f64>,
    fit: &CoxModelFit,
    options: &CoxOptions,
    grid: &[f64],
    n_strata: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
    let p = x.ncols();
    let zero = vec![0.0; p];
    let fitted = FittedModel {
        data,
        x: x.as_ref(),
        fit,
        options,
    };
    let base = basehaz(
        &fitted,
        &BaselineOptions {
            centring: Centring::Given(&zero),
            ..BaselineOptions::default()
        },
    )
    .unwrap();
    let mut lam = vec![vec![0.0; grid.len()]; n_strata];
    for (s, b) in base.strata.iter().enumerate() {
        for (t, h) in b.time.iter().zip(&b.hazard) {
            let j = grid.iter().position(|g| g == t).unwrap();
            lam[s][j] = *h;
        }
    }
    let coefficients: Vec<Option<f64>> = fit.coefficients.clone();
    let breslow = CoxOptions {
        ties: Ties::Breslow,
        ..options.clone()
    };
    let scores = score_residuals(data, x.as_ref(), &coefficients, &breslow).unwrap();
    let v: Vec<f64> = fit
        .covariance
        .as_slice()
        .iter()
        .map(|v| v.unwrap())
        .collect();
    let v = fit
        .robust_covariance
        .as_ref()
        .map_or(v, |r| r.as_slice().iter().map(|v| v.unwrap()).collect());
    let n = x.nrows();
    let mut if_beta = vec![vec![0.0; p]; n];
    for i in 0..n {
        for a in 0..p {
            for b in 0..p {
                if_beta[i][a] += v[a * p + b] * scores[b].as_ref().unwrap()[i];
            }
        }
    }
    (lam, if_beta, v)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The standard error and the size of its terms at the counts `counts` of
/// grid times, for one new row: `theta` and `xr` per model, in stratum `s`.
pub fn naive_se(
    case: &NCase,
    s: usize,
    theta: &[f64],
    xr: &[Vec<f64>],
    counts: &[usize],
) -> Vec<(f64, f64)> {
    let n = case.stop.len();
    let big_j = case.grid.len();
    let k = case.cause_model;
    let m_all = case.models.len();
    let in_all: Vec<bool> = (0..m_all)
        .map(|m| case.survival_models.contains(&m))
        .collect();
    // S0 and E of every model in the stratum at every grid position with an
    // event of the model.
    let mut s0 = vec![vec![0.0; big_j]; m_all];
    let mut e = vec![vec![vec![0.0; 0]; big_j]; m_all];
    let mut risk = vec![vec![0.0; n]; m_all];
    for m in 0..m_all {
        let md = &case.models[m];
        for i in 0..n {
            risk[m][i] = dot(&md.x[i], &md.beta).exp();
        }
        for j in 0..big_j {
            if md.lam[s][j] > 0.0 {
                let mut sum = 0.0;
                let mut s1 = vec![0.0; md.p];
                for i in 0..n {
                    if case.stratum[i] == s && case.stop[i] >= case.grid[j] {
                        sum += risk[m][i];
                        for a in 0..md.p {
                            s1[a] += risk[m][i] * md.x[i][a];
                        }
                    }
                }
                s0[m][j] = sum;
                e[m][j] = s1.iter().map(|v| v / sum).collect();
            }
        }
    }
    // The survival along the grid.
    let mut surv = vec![1.0; big_j];
    for j in 1..big_j {
        surv[j] = if case.product_limit {
            let hz: f64 = case
                .survival_models
                .iter()
                .map(|&m| theta[m] * case.models[m].lam[s][j - 1])
                .sum();
            surv[j - 1] * (1.0 - hz)
        } else {
            let cum: f64 = case
                .survival_models
                .iter()
                .map(|&m| theta[m] * case.models[m].lam[s][..j].iter().sum::<f64>())
                .sum();
            (-cum).exp()
        };
    }
    // The weight of the hazard increment at each position in the derivative of
    // the survival: 1 for the exponential form, 1 / (1 - h) for the exact
    // derivative of the product limit.
    let rw: Vec<f64> = (0..big_j)
        .map(|l| {
            if case.product_limit && case.exact {
                let hz: f64 = case
                    .survival_models
                    .iter()
                    .map(|&m| theta[m] * case.models[m].lam[s][l])
                    .sum();
                1.0 / (1.0 - hz)
            } else {
                1.0
            }
        })
        .collect();
    let cum_lam =
        |m: usize, u: usize| -> f64 { (0..=u).map(|l| rw[l] * case.models[m].lam[s][l]).sum() };
    let cum_e_lam = |m: usize, u: usize| -> Vec<f64> {
        let md = &case.models[m];
        let mut out = vec![0.0; md.p];
        for j in 0..=u {
            for a in 0..md.p {
                if md.lam[s][j] > 0.0 {
                    out[a] += rw[j] * e[m][j][a] * md.lam[s][j];
                }
            }
        }
        out
    };
    let mut result = Vec::new();
    for &jt in counts {
        let mut total = 0.0;
        let mut total_abs = 0.0;
        for i in 0..n {
            let same = case.stratum[i] == s;
            let ti = case.stop[i];
            let mut value = 0.0;
            let mut size = 0.0;
            for j in 0..jt {
                let hk = theta[k] * case.models[k].lam[s][j];
                if hk <= 0.0 {
                    continue;
                }
                let mk = &case.models[k];
                // the influence function of the hazard of the cause at j
                let mut ihaz = 0.0;
                let mut isize_ = 0.0;
                {
                    let mut put = |t: f64| {
                        ihaz += t;
                        isize_ += t.abs();
                    };
                    let lk = mk.lam[s][j];
                    put(-dot(&mk.if_beta[i], &e[k][j]) * lk);
                    if same && case.grid[j] <= ti {
                        put(-risk[k][i] * lk / s0[k][j]);
                    }
                    if same && ti == case.grid[j] && mk.event[i] {
                        put(1.0 / s0[k][j]);
                    }
                    put(dot(&xr[k], &mk.if_beta[i]) * lk);
                }
                value += surv[j] * theta[k] * ihaz;
                size += surv[j] * theta[k] * isize_;
                if j >= 1 {
                    let mut icum = 0.0;
                    let mut isz = 0.0;
                    for m in 0..m_all {
                        if !in_all[m] {
                            continue;
                        }
                        let md = &case.models[m];
                        let u = j - 1;
                        let mut part = 0.0;
                        let mut part_abs = 0.0;
                        let mut put = |t: f64| {
                            part += t;
                            part_abs += t.abs();
                        };
                        put(-dot(&md.if_beta[i], &cum_e_lam(m, u)));
                        if same {
                            let mut a_sum = 0.0;
                            for l in 0..=u {
                                if md.lam[s][l] > 0.0 && case.grid[l] <= ti {
                                    a_sum += rw[l] * md.lam[s][l] / s0[m][l];
                                }
                            }
                            put(-risk[m][i] * a_sum);
                            if md.event[i] && ti <= case.grid[u] {
                                let l = (0..big_j)
                                    .find(|&l| case.grid[l] == ti)
                                    .expect("an event time is on the grid");
                                put(rw[l] / s0[m][l]);
                            }
                        }
                        put(dot(&xr[m], &md.if_beta[i]) * cum_lam(m, u));
                        icum += theta[m] * part;
                        isz += theta[m] * part_abs;
                    }
                    value -= surv[j] * hk * icum;
                    size += surv[j] * hk * isz;
                }
            }
            total += value * value;
            total_abs += size * size;
        }
        result.push((total.sqrt(), total_abs.sqrt()));
    }
    result
}

//! The influence-function standard error of the absolute risk and of the
//! standardised risk with case weights, entry times and an offset, against the
//! derivative of the estimate with respect to the case weight of each person.
//!
//! The influence function of a person is `w_i dF / dw_i`, the weight times the
//! derivative, which is `dF / d log w_i`. It is formed here as a central
//! difference of the estimate refitted with the weight of the person moved by
//! the relative step `h`, with Richardson extrapolation from `h` and `h / 2`.
//! The outcome models are fitted to convergence at machine precision with
//! Breslow ties, the likelihood the influence function describes.
//!
//! Integer weights give the estimates of repeated rows. Their standard errors
//! are those of weights, not of copies: a person of weight `w` is one unit of
//! the influence function, `w` copies are `w` units of `1 / w` of the weight
//! each, so `se^2` of the weighted data is `sum w_i^2 d_i^2` and of the
//! repeated rows `sum w_i d_i^2`. Both are checked against the numerical
//! derivatives `d_i`.
//!
//! # Tolerances
//!
//! The extrapolated derivative has the truncation error of the difference of
//! the two central differences, `|D(h) - D(h / 2)|`, and the rounding error
//! `c eps |F| / h` of a difference of two estimates, `c = 4 (n + p + J)` for
//! the sums of the fit, the baseline and the average, and a factor 2 for the
//! compensated risk sets of data with entry times. The tolerance of an
//! influence function is twice the first plus the second. That of a squared
//! standard error or a covariance is `sum (|a_i| tb_i + |b_i| ta_i + ta_i
//! tb_i)` over the persons. They are not changed after a run.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "the inputs are a few hundred rows"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::absrisk::{CauseSpecificCox, CscOptions, NewRows, RiskOptions, SurvType};
use epistat_survival::cox::{self, CoxModelFit, CoxOptions, Ties};
use epistat_survival::standardise::{
    Counterfactual, ExposureLevel, InfluenceForm, StandardiseOptions, Standardised,
    standardise_cause_specific, standardise_cox,
};

struct Sim {
    stop: Vec<f64>,
    start: Vec<f64>,
    status: Vec<i32>,
    stratum: Vec<u32>,
    offset: Vec<f64>,
    design: Mat<f64>,
}

fn sim(n: usize, seed: u64, causes: i32) -> Sim {
    let mut rng = Stream::seeded(seed);
    let (mut stop, mut start, mut status) = (Vec::new(), Vec::new(), Vec::new());
    let (mut stratum, mut offset, mut cols) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let x1 = rng.normal();
        let x2 = f64::from(u8::from(rng.below(2) == 0));
        let a = f64::from(u8::from(rng.uniform() < 1.0 / (1.0 + (-0.5 * x1).exp())));
        let o = 0.4 * rng.normal();
        let t1 = rng.exp1() / (0.10 * (0.5 * x1 - 0.4 * a + 0.3 * x2 + o).exp());
        let t2 = rng.exp1() / (0.06 * (-0.4 * a).exp());
        let c = 2.0 + 14.0 * rng.uniform();
        let first = if causes == 1 { t1 } else { t1.min(t2) };
        let t = ((first.min(c)) * 4.0).ceil() / 4.0;
        stop.push(t);
        status.push(if c < first {
            0
        } else if causes == 1 || t1 < t2 {
            1
        } else {
            2
        });
        let entered = rng.uniform() < 0.5;
        let u = rng.uniform();
        start.push(if entered { 0.35 * t * u } else { 0.0 });
        stratum.push(rng.below(2) as u32);
        offset.push(o);
        cols.push([a, x1, x2]);
    }
    Sim {
        stop,
        start,
        status,
        stratum,
        offset,
        design: Mat::from_fn(n, 3, |i, j| cols[i][j]),
    }
}

/// Weights in `[0.5, 3]`.
fn weights(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Stream::seeded(seed);
    (0..n).map(|_| 0.5 + 2.5 * rng.uniform()).collect()
}

#[derive(Clone, Copy)]
struct Cfg {
    weight: bool,
    entry: bool,
    offset: bool,
}

const CONFIGS: [(&str, Cfg); 5] = [
    (
        "weights",
        Cfg {
            weight: true,
            entry: false,
            offset: false,
        },
    ),
    (
        "entry",
        Cfg {
            weight: false,
            entry: true,
            offset: false,
        },
    ),
    (
        "offset",
        Cfg {
            weight: false,
            entry: false,
            offset: true,
        },
    ),
    (
        "weights and entry",
        Cfg {
            weight: true,
            entry: true,
            offset: false,
        },
    ),
    (
        "weights, entry and offset",
        Cfg {
            weight: true,
            entry: true,
            offset: true,
        },
    ),
];

fn data_of<'a>(s: &'a Sim, w: &'a [f64], cfg: Cfg, n_codes: u32) -> SurvData<'a> {
    let mut builder = SurvData::builder(&s.stop, &s.status)
        .n_codes(n_codes)
        .stratum(&s.stratum)
        .weight(w);
    if cfg.entry {
        builder = builder.start(&s.start);
    }
    if cfg.offset {
        builder = builder.offset(&s.offset);
    }
    builder.build().unwrap()
}

fn precise(cause: Option<u32>) -> CoxOptions {
    CoxOptions {
        ties: Ties::Breslow,
        cause,
        tolerance: 1e-14,
        max_iterations: 100,
        ..CoxOptions::default()
    }
}

/// The numerical influence functions of a table of estimates, one list per
/// entry, and their tolerances.
struct Numeric {
    influence: Vec<Vec<f64>>,
    tol: Vec<Vec<f64>>,
}

fn numeric(
    n: usize,
    w0: &[f64],
    estimate: &dyn Fn(&[f64]) -> Vec<f64>,
    values: &[f64],
    n_terms: usize,
    entry: bool,
) -> Numeric {
    let h = 1e-4;
    let c = if entry { 8.0 } else { 4.0 } * n_terms as f64 * f64::EPSILON;
    let mut influence = vec![vec![0.0; n]; values.len()];
    let mut tol = vec![vec![0.0; n]; values.len()];
    for i in 0..n {
        let mut d = [vec![], vec![]];
        for (k, step) in [h, h / 2.0].into_iter().enumerate() {
            let mut w = w0.to_vec();
            w[i] = w0[i] * (1.0 + step);
            let up = estimate(&w);
            w[i] = w0[i] * (1.0 - step);
            let down = estimate(&w);
            d[k] = up
                .iter()
                .zip(&down)
                .map(|(a, b)| (a - b) / (2.0 * step))
                .collect();
        }
        for q in 0..values.len() {
            influence[q][i] = (4.0 * d[1][q] - d[0][q]) / 3.0;
            tol[q][i] = 2.0 * (d[0][q] - d[1][q]).abs() + c * values[q].abs() / (h / 2.0);
        }
    }
    Numeric { influence, tol }
}

/// `sum_i a_i b_i` with its tolerance.
fn product(num: &Numeric, a: usize, b: usize, scale: &[f64]) -> (f64, f64) {
    let (ia, ib) = (&num.influence[a], &num.influence[b]);
    let (ta, tb) = (&num.tol[a], &num.tol[b]);
    let mut direct = 0.0;
    let mut bound = 0.0;
    for i in 0..ia.len() {
        direct += scale[i] * ia[i] * ib[i];
        bound += scale[i] * (ia[i].abs() * tb[i] + ib[i].abs() * ta[i] + ta[i] * tb[i]);
    }
    (direct, bound)
}

fn check_covariance(name: &str, unit: &Standardised, num: &Numeric, scale: &[f64]) {
    let l = unit.n_levels;
    let t = unit.n_times;
    let cov = unit.covariance.as_ref().unwrap();
    let mut worst = 0.0f64;
    for (k, block) in cov.iter().enumerate().take(t) {
        for a in 0..l {
            for b in 0..l {
                let (direct, bound) = product(num, a * t + k, b * t + k, scale);
                let got = block.get(a, b).unwrap();
                worst = worst.max((got - direct).abs() / bound);
                assert!(
                    (got - direct).abs() <= bound,
                    "{name}: covariance {a},{b} at time {k}: {got} against {direct}, bound {bound}"
                );
            }
        }
    }
    eprintln!("{name}: worst covariance deviation {worst:.3} of its bound");
}

const SET0: [(usize, f64); 1] = [(0, 0.0)];
const SET1: [(usize, f64); 1] = [(0, 1.0)];

fn levels() -> [ExposureLevel<'static>; 2] {
    [
        ExposureLevel::single(Counterfactual::Columns(&SET0)),
        ExposureLevel::single(Counterfactual::Columns(&SET1)),
    ]
}

fn cox_run(s: &Sim, w: &[f64], cfg: Cfg, times: &[f64], std_err: bool) -> Standardised {
    let data = data_of(s, w, cfg, 1);
    let options = precise(None);
    let fit = cox::fit(&data, s.design.as_ref(), &options).unwrap();
    standardise_cox(
        &data,
        s.design.as_ref(),
        &fit,
        &options,
        &levels(),
        times,
        &StandardiseOptions {
            se: std_err,
            ..StandardiseOptions::default()
        },
    )
    .unwrap()
}

#[test]
fn cox_standard_error_is_the_derivative() {
    let n = 70;
    let s = sim(n, 21, 1);
    let times = [1.0, 3.0, 6.5, 11.0];
    for (name, cfg) in CONFIGS {
        let w0 = if cfg.weight {
            weights(n, 5)
        } else {
            vec![1.0; n]
        };
        let unit = cox_run(&s, &w0, cfg, &times, true);
        let estimate = |w: &[f64]| -> Vec<f64> {
            cox_run(&s, w, cfg, &times, false)
                .risk
                .iter()
                .map(|r| r.unwrap())
                .collect()
        };
        let values: Vec<f64> = unit.risk.iter().map(|r| r.unwrap()).collect();
        let num = numeric(n, &w0, &estimate, &values, 3 * n + 40, cfg.entry);
        check_covariance(&format!("cox, {name}"), &unit, &num, &vec![1.0; n]);
    }
}

fn csc_run(
    s: &Sim,
    w: &[f64],
    cfg: Cfg,
    product_limit: bool,
    form: InfluenceForm,
    times: &[f64],
    std_err: bool,
    cause: u32,
) -> Standardised {
    let data = data_of(s, w, cfg, 2);
    let x2 = Mat::from_fn(s.design.nrows(), 2, |i, j| s.design[(i, j)]);
    let designs = [s.design.as_ref(), x2.as_ref()];
    let fits: Vec<CoxModelFit> = (0..2)
        .map(|m| cox::fit(&data, designs[m], &precise(Some(m as u32 + 1))).unwrap())
        .collect();
    let level = |set: &'static [(usize, f64)]| ExposureLevel {
        designs: vec![Counterfactual::Columns(set), Counterfactual::Columns(set)],
    };
    standardise_cause_specific(
        &data,
        &designs,
        &fits,
        cause,
        &CscOptions {
            ties: Ties::Breslow,
            ..CscOptions::default()
        },
        &[level(&SET0), level(&SET1)],
        times,
        &StandardiseOptions {
            se: std_err,
            product_limit,
            influence: form,
            ..StandardiseOptions::default()
        },
    )
    .unwrap()
}

#[test]
fn cause_specific_standard_error_is_the_derivative() {
    let n = 80;
    let s = sim(n, 9, 2);
    let times = [1.0, 2.5, 4.0, 5.0];
    for (name, cfg) in &CONFIGS {
        let w0 = if cfg.weight {
            weights(n, 6)
        } else {
            vec![1.0; n]
        };
        for product_limit in [false, true] {
            for cause in [1, 2] {
                let unit = csc_run(
                    &s,
                    &w0,
                    *cfg,
                    product_limit,
                    InfluenceForm::Exact,
                    &times,
                    true,
                    cause,
                );
                let estimate = |w: &[f64]| -> Vec<f64> {
                    csc_run(
                        &s,
                        w,
                        *cfg,
                        product_limit,
                        InfluenceForm::Exact,
                        &times,
                        false,
                        cause,
                    )
                    .risk
                    .iter()
                    .map(|r| r.unwrap())
                    .collect()
                };
                let values: Vec<f64> = unit.risk.iter().map(|r| r.unwrap()).collect();
                let num = numeric(n, &w0, &estimate, &values, 6 * n + 80, cfg.entry);
                check_covariance(
                    &format!("csc cause {cause}, product limit {product_limit}, {name}"),
                    &unit,
                    &num,
                    &vec![1.0; n],
                );
            }
        }
    }
}

/// Without the product limit the two forms are one function, with weights and
/// entry times as without.
#[test]
fn forms_agree_without_the_product_limit() {
    let n = 80;
    let s = sim(n, 9, 2);
    let times = [1.0, 2.5, 4.0, 5.0];
    let w0 = weights(n, 6);
    let cfg = CONFIGS[3].1;
    let a = csc_run(&s, &w0, cfg, false, InfluenceForm::Exact, &times, true, 1);
    let b = csc_run(
        &s,
        &w0,
        cfg,
        false,
        InfluenceForm::Exponential,
        &times,
        true,
        1,
    );
    for (x, y) in a.se.unwrap().iter().zip(b.se.unwrap().iter()) {
        let (x, y) = (x.unwrap(), y.unwrap());
        assert!(
            (x - y).abs() <= 8.0 * n as f64 * f64::EPSILON * x.abs(),
            "{x} {y}"
        );
    }
}

fn predict_run(
    s: &Sim,
    w: &[f64],
    cfg: Cfg,
    surv_type: SurvType,
    product_limit: bool,
    times: &[f64],
    std_err: bool,
    cause: u32,
) -> (Vec<f64>, Vec<f64>) {
    let n = s.design.nrows();
    let data = data_of(s, w, cfg, 2);
    let x2 = Mat::from_fn(n, 2, |i, j| s.design[(i, j)]);
    let designs = [s.design.as_ref(), x2.as_ref()];
    let opts = CscOptions {
        ties: Ties::Breslow,
        surv_type,
        ..CscOptions::default()
    };
    let fits: Vec<CoxModelFit> = match surv_type {
        SurvType::Hazard => (0..2)
            .map(|m| cox::fit(&data, designs[m], &precise(Some(m as u32 + 1))).unwrap())
            .collect(),
        SurvType::Survival { cause } => {
            let any = {
                let status: Vec<i32> = s.status.iter().map(|&v| i32::from(v != 0)).collect();
                let mut builder = SurvData::builder(&s.stop, &status)
                    .stratum(&s.stratum)
                    .weight(w);
                if cfg.entry {
                    builder = builder.start(&s.start);
                }
                if cfg.offset {
                    builder = builder.offset(&s.offset);
                }
                let d = builder.build().unwrap();
                cox::fit(&d, designs[1], &precise(None)).unwrap()
            };
            vec![
                cox::fit(&data, designs[0], &precise(Some(cause))).unwrap(),
                any,
            ]
        }
    };
    let model = CauseSpecificCox::from_fits(&data, &designs, fits, &opts).unwrap();
    let new1 = Mat::from_fn(2, 3, |i, j| [[1.0, 0.4, 1.0], [0.0, -0.8, 0.0]][i][j]);
    let new2 = Mat::from_fn(2, 2, |i, j| [[1.0, 0.4], [0.0, -0.8]][i][j]);
    let new_designs = [new1.as_ref(), new2.as_ref()];
    let rows = [1u32, 0];
    let new = NewRows {
        designs: &new_designs,
        stratum: Some(&rows),
        offset: cfg.offset.then_some(&[0.3, -0.2]),
    };
    let out = model
        .predict(
            &new,
            times,
            cause,
            &RiskOptions {
                product_limit,
                se: std_err,
                influence: InfluenceForm::Exact,
                ..RiskOptions::default()
            },
        )
        .unwrap();
    (
        out.risk.iter().map(|r| r.unwrap()).collect(),
        out.se
            .map(|v| v.iter().map(|r| r.unwrap()).collect())
            .unwrap_or_default(),
    )
}

#[test]
fn absolute_risk_standard_error_is_the_derivative() {
    let n = 80;
    let s = sim(n, 13, 2);
    let times = [1.0, 2.5, 4.0, 5.0];
    for (name, cfg) in &CONFIGS {
        let w0 = if cfg.weight {
            weights(n, 7)
        } else {
            vec![1.0; n]
        };
        for (label, surv_type, cause) in [
            ("hazard", SurvType::Hazard, 1),
            ("hazard", SurvType::Hazard, 2),
            ("survival", SurvType::Survival { cause: 1 }, 1),
        ] {
            for product_limit in [false, true] {
                let (risk, se) =
                    predict_run(&s, &w0, *cfg, surv_type, product_limit, &times, true, cause);
                let estimate = |w: &[f64]| -> Vec<f64> {
                    predict_run(&s, w, *cfg, surv_type, product_limit, &times, false, cause).0
                };
                let num = numeric(n, &w0, &estimate, &risk, 6 * n + 80, cfg.entry);
                let mut worst = 0.0f64;
                for (q, se_q) in se.iter().enumerate() {
                    let (direct, bound) = product(&num, q, q, &vec![1.0; n]);
                    let got = se_q * se_q;
                    worst = worst.max((got - direct).abs() / bound);
                    assert!(
                        (got - direct).abs() <= bound,
                        "{label} cause {cause}, product limit {product_limit}, {name}: \
                         se^2 {got} against {direct}, bound {bound}, entry {q}"
                    );
                }
                eprintln!(
                    "absrisk {label} cause {cause}, product limit {product_limit}, {name}: worst {worst:.3}"
                );
            }
        }
    }
}

/// Integer weights and repeated rows: the same estimate, and the standard error
/// of the repeated rows is `sum w_i d_i^2` where that of the weights is `sum
/// w_i^2 d_i^2`.
#[test]
fn integer_weights_are_repeated_rows() {
    let n = 60;
    let s = sim(n, 31, 1);
    let mut rng = Stream::seeded(77);
    let w0: Vec<f64> = (0..n).map(|_| f64::from(1 + rng.below(3) as u8)).collect();
    let cfg = Cfg {
        weight: true,
        entry: true,
        offset: true,
    };
    let times = [1.0, 3.0, 6.5, 11.0];
    // The rows repeated.
    let index: Vec<usize> = (0..n)
        .flat_map(|i| std::iter::repeat_n(i, w0[i] as usize))
        .collect();
    let rep = Sim {
        stop: index.iter().map(|&i| s.stop[i]).collect(),
        start: index.iter().map(|&i| s.start[i]).collect(),
        status: index.iter().map(|&i| s.status[i]).collect(),
        stratum: index.iter().map(|&i| s.stratum[i]).collect(),
        offset: index.iter().map(|&i| s.offset[i]).collect(),
        design: Mat::from_fn(index.len(), 3, |r, j| s.design[(index[r], j)]),
    };
    let ones = vec![1.0; index.len()];
    let weighted = cox_run(&s, &w0, cfg, &times, true);
    let repeated = cox_run(
        &rep,
        &ones,
        Cfg {
            weight: false,
            ..cfg
        },
        &times,
        true,
    );
    let estimate = |w: &[f64]| -> Vec<f64> {
        cox_run(&s, w, cfg, &times, false)
            .risk
            .iter()
            .map(|r| r.unwrap())
            .collect()
    };
    let values: Vec<f64> = weighted.risk.iter().map(|r| r.unwrap()).collect();
    let tol_est = 16.0 * (3 * index.len() + 40) as f64 * f64::EPSILON;
    for (a, b) in weighted.risk.iter().zip(&repeated.risk) {
        let (a, b) = (a.unwrap(), b.unwrap());
        assert!((a - b).abs() <= tol_est * a.abs().max(1e-3), "{a} {b}");
    }
    let num = numeric(n, &w0, &estimate, &values, 3 * n + 40, true);
    let inverse: Vec<f64> = w0.iter().map(|w| 1.0 / w).collect();
    check_covariance("cox weights", &weighted, &num, &vec![1.0; n]);
    check_covariance("cox repeated rows", &repeated, &num, &inverse);
}

/// `surv.type = "survival"`: the cause model and the model of the time to any
/// event, standardised.
fn survival_type_run(
    s: &Sim,
    w: &[f64],
    cfg: Cfg,
    product_limit: bool,
    times: &[f64],
    std_err: bool,
) -> Standardised {
    let n = s.design.nrows();
    let data = data_of(s, w, cfg, 2);
    let any_status: Vec<i32> = s.status.iter().map(|&v| i32::from(v != 0)).collect();
    let mut builder = SurvData::builder(&s.stop, &any_status)
        .stratum(&s.stratum)
        .weight(w);
    if cfg.entry {
        builder = builder.start(&s.start);
    }
    if cfg.offset {
        builder = builder.offset(&s.offset);
    }
    let any = builder.build().unwrap();
    let x2 = Mat::from_fn(n, 2, |i, j| s.design[(i, j)]);
    let designs = [s.design.as_ref(), x2.as_ref()];
    let fits = vec![
        cox::fit(&data, designs[0], &precise(Some(1))).unwrap(),
        cox::fit(&any, designs[1], &precise(None)).unwrap(),
    ];
    let level = |set: &'static [(usize, f64)]| ExposureLevel {
        designs: vec![Counterfactual::Columns(set), Counterfactual::Columns(set)],
    };
    standardise_cause_specific(
        &data,
        &designs,
        &fits,
        1,
        &CscOptions {
            ties: Ties::Breslow,
            surv_type: SurvType::Survival { cause: 1 },
            ..CscOptions::default()
        },
        &[level(&SET0), level(&SET1)],
        times,
        &StandardiseOptions {
            se: std_err,
            product_limit,
            ..StandardiseOptions::default()
        },
    )
    .unwrap()
}

#[test]
fn survival_type_standard_error_is_the_derivative() {
    let n = 80;
    let s = sim(n, 9, 2);
    let times = [1.0, 2.5, 4.0, 5.0];
    for (name, cfg) in &CONFIGS {
        let w0 = if cfg.weight {
            weights(n, 6)
        } else {
            vec![1.0; n]
        };
        for product_limit in [false, true] {
            let unit = survival_type_run(&s, &w0, *cfg, product_limit, &times, true);
            let estimate = |w: &[f64]| -> Vec<f64> {
                survival_type_run(&s, w, *cfg, product_limit, &times, false)
                    .risk
                    .iter()
                    .map(|r| r.unwrap())
                    .collect()
            };
            let values: Vec<f64> = unit.risk.iter().map(|r| r.unwrap()).collect();
            let num = numeric(n, &w0, &estimate, &values, 6 * n + 80, cfg.entry);
            check_covariance(
                &format!("survival type, product limit {product_limit}, {name}"),
                &unit,
                &num,
                &vec![1.0; n],
            );
        }
    }
}

/// Cause-specific models with an offset: integer weights and repeated rows give
/// the same estimates.
#[test]
fn integer_weights_are_repeated_rows_cause_specific() {
    let n = 60;
    let s = sim(n, 32, 2);
    let mut rng = Stream::seeded(78);
    let w0: Vec<f64> = (0..n).map(|_| f64::from(1 + rng.below(3) as u8)).collect();
    let cfg = Cfg {
        weight: true,
        entry: true,
        offset: true,
    };
    let times = [1.0, 2.5, 4.0];
    let index: Vec<usize> = (0..n)
        .flat_map(|i| std::iter::repeat_n(i, w0[i] as usize))
        .collect();
    let rep = Sim {
        stop: index.iter().map(|&i| s.stop[i]).collect(),
        start: index.iter().map(|&i| s.start[i]).collect(),
        status: index.iter().map(|&i| s.status[i]).collect(),
        stratum: index.iter().map(|&i| s.stratum[i]).collect(),
        offset: index.iter().map(|&i| s.offset[i]).collect(),
        design: Mat::from_fn(index.len(), 3, |r, j| s.design[(index[r], j)]),
    };
    let ones = vec![1.0; index.len()];
    let tol = 16.0 * (6 * index.len() + 80) as f64 * f64::EPSILON;
    for cause in [1, 2] {
        let a = csc_run(
            &s,
            &w0,
            cfg,
            true,
            InfluenceForm::Exact,
            &times,
            false,
            cause,
        );
        let b = csc_run(
            &rep,
            &ones,
            cfg,
            true,
            InfluenceForm::Exact,
            &times,
            false,
            cause,
        );
        for (x, y) in a.risk.iter().zip(&b.risk) {
            let (x, y) = (x.unwrap(), y.unwrap());
            assert!((x - y).abs() <= tol * x.abs().max(1e-3), "{x} {y}");
        }
    }
}

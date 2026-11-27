//! The influence function of the standardised risk against the derivative of
//! the estimate with respect to the case weight of each person, which is what
//! an influence function is, and the standard error of a population with one
//! covariate profile against [`absrisk`](epistat_survival::absrisk).
//!
//! The derivative is a central difference of the estimate refitted with the
//! weight of one person moved by `h`, with Richardson extrapolation from the
//! steps `h` and `h / 2`. The outcome model is fitted to convergence at
//! machine precision, with Breslow ties, which is the likelihood the influence
//! function describes, so that data with tied event times and strata can be
//! used.
//!
//! # Tolerances
//!
//! The extrapolated derivative has the truncation error of the difference of
//! the two central differences, `|D(h) - D(h / 2)|`, which bounds that of
//! the better of them by a factor of 3 and that of the extrapolation by far
//! more, and the rounding error `c eps |F| / h` of a difference of two
//! estimates, with `c = 4 (n + p + J)` for the sums of the fit, the baseline
//! and the average. The tolerance of an influence function is twice the first
//! plus the second, that of a standard error or a covariance is the sum of
//! the tolerances of the terms of the sums of products. They are not changed
//! after a run.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "the inputs are a few hundred rows"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::absrisk::{CauseSpecificCox, CscOptions, NewRows, RiskOptions};
use epistat_survival::cox::{self, CoxModelFit, CoxOptions, Ties};
use epistat_survival::standardise::{
    Counterfactual, InfluenceForm, Level, StandardiseOptions, standardise_cause_specific,
    standardise_cox,
};

struct Sim {
    stop: Vec<f64>,
    status: Vec<i32>,
    stratum: Vec<u32>,
    design: Mat<f64>,
}

fn sim(n: usize, seed: u64, tied: bool, causes: i32) -> Sim {
    let mut rng = Stream::seeded(seed);
    let mut stop = Vec::new();
    let mut status = Vec::new();
    let mut stratum = Vec::new();
    let mut cols = Vec::new();
    for _ in 0..n {
        let x1 = rng.normal();
        let x2 = f64::from(u8::from(rng.below(2) == 0));
        let a = f64::from(u8::from(rng.uniform() < 1.0 / (1.0 + (-0.5 * x1).exp())));
        let t1 = rng.exp1() / (0.10 * (0.5 * x1 - 0.4 * a + 0.3 * x2).exp());
        let t2 = rng.exp1() / (0.06 * (-0.4 * a).exp());
        let c = 2.0 + 14.0 * rng.uniform();
        let t = if causes == 1 { t1 } else { t1.min(t2) }.min(c);
        stop.push(if tied { (t * 4.0).ceil() / 4.0 } else { t });
        status.push(if c < if causes == 1 { t1 } else { t1.min(t2) } {
            0
        } else if causes == 1 || t1 < t2 {
            1
        } else {
            2
        });
        stratum.push(rng.below(3) as u32);
        cols.push([a, x1, x2]);
    }
    let design = Mat::from_fn(n, 3, |i, j| cols[i][j]);
    Sim {
        stop,
        status,
        stratum,
        design,
    }
}

const SET0: [(usize, f64); 1] = [(0, 0.0)];
const SET1: [(usize, f64); 1] = [(0, 1.0)];

fn precise(cause: Option<u32>) -> CoxOptions<'static> {
    CoxOptions {
        ties: Ties::Breslow,
        cause,
        eps: 1e-14,
        iter_max: 100,
        ..CoxOptions::default()
    }
}

/// The estimate for case weights `w`: a table of levels by times.
type Estimator<'a> = dyn Fn(&[f64]) -> Vec<f64> + 'a;

fn check(
    name: &str,
    n: usize,
    n_terms: usize,
    estimator: &Estimator<'_>,
    unit: &epistat_survival::standardise::Standardised,
) {
    let eps = f64::EPSILON;
    let l = unit.n_levels;
    let t = unit.n_times;
    let h = 1e-4;
    let mut influence = vec![vec![0.0; n]; l * t];
    let mut tol = vec![vec![0.0; n]; l * t];
    for i in 0..n {
        let mut d = [vec![], vec![]];
        for (k, step) in [h, h / 2.0].into_iter().enumerate() {
            let mut w = vec![1.0; n];
            w[i] = 1.0 + step;
            let up = estimator(&w);
            w[i] = 1.0 - step;
            let down = estimator(&w);
            d[k] = up
                .iter()
                .zip(&down)
                .map(|(a, b)| (a - b) / (2.0 * step))
                .collect();
        }
        for q in 0..l * t {
            let rich = (4.0 * d[1][q] - d[0][q]) / 3.0;
            influence[q][i] = rich;
            let f = unit.risk[q].unwrap();
            tol[q][i] =
                2.0 * (d[0][q] - d[1][q]).abs() + 4.0 * n_terms as f64 * eps * f.abs() / (h / 2.0);
        }
    }
    let mut worst_cov = 0.0f64;
    let cov = unit.covariance.as_ref().unwrap();
    for k in 0..t {
        for a in 0..l {
            for b in 0..l {
                let ia = &influence[a * t + k];
                let ib = &influence[b * t + k];
                let ta = &tol[a * t + k];
                let tb = &tol[b * t + k];
                let mut direct = 0.0;
                let mut bound = 0.0;
                for i in 0..n {
                    direct += ia[i] * ib[i];
                    bound += ia[i].abs() * tb[i] + ib[i].abs() * ta[i] + ta[i] * tb[i];
                }
                let got = cov[k * l * l + a * l + b].unwrap();
                worst_cov = worst_cov.max((got - direct).abs() / bound);
                assert!(
                    (got - direct).abs() <= bound,
                    "{name}: covariance {a},{b} at time {k}: {got} against {direct}, bound {bound}"
                );
            }
        }
    }
    eprintln!("{name}: worst covariance deviation {worst_cov:.3} of its bound");
}

#[test]
fn influence_cox_tied_strata() {
    let n = 60;
    let s = sim(n, 5, true, 1);
    let times = [1.0, 3.0, 6.5, 11.0];
    let levels = [
        Level::single(Counterfactual::Columns(&SET0)),
        Level::single(Counterfactual::Columns(&SET1)),
    ];
    let status = &s.status;
    let estimator = |w: &[f64]| -> Vec<f64> {
        let data = SurvData::builder(&s.stop, status)
            .stratum(&s.stratum)
            .weight(w)
            .build()
            .unwrap();
        let options = precise(None);
        let fit = cox::fit(&data, s.design.as_ref(), &options).unwrap();
        let out = standardise_cox(
            &data,
            s.design.as_ref(),
            &fit,
            &options,
            &levels,
            &times,
            &StandardiseOptions::default(),
        )
        .unwrap();
        out.risk.iter().map(|r| r.unwrap()).collect()
    };
    let data = SurvData::builder(&s.stop, status)
        .stratum(&s.stratum)
        .build()
        .unwrap();
    let options = precise(None);
    let fit = cox::fit(&data, s.design.as_ref(), &options).unwrap();
    let unit = standardise_cox(
        &data,
        s.design.as_ref(),
        &fit,
        &options,
        &levels,
        &times,
        &StandardiseOptions {
            se: true,
            ..StandardiseOptions::default()
        },
    )
    .unwrap();
    check("cox tied strata", n, 3 * n + 40, &estimator, &unit);
}

fn csc_run(
    s: &Sim,
    w: Option<&[f64]>,
    product_limit: bool,
    influence: InfluenceForm,
    times: &[f64],
    std_err: bool,
    cause: u32,
) -> epistat_survival::standardise::Standardised {
    let mut builder = SurvData::builder(&s.stop, &s.status)
        .n_codes(2)
        .stratum(&s.stratum);
    if let Some(w) = w {
        builder = builder.weight(w);
    }
    let data = builder.build().unwrap();
    let x2 = Mat::from_fn(s.design.nrows(), 2, |i, j| s.design[(i, j)]);
    let designs = [s.design.as_ref(), x2.as_ref()];
    let fits: Vec<CoxModelFit> = (0..2)
        .map(|m| cox::fit(&data, designs[m], &precise(Some(m as u32 + 1))).unwrap())
        .collect();
    let levels = [
        Level {
            designs: vec![
                Counterfactual::Columns(&SET0),
                Counterfactual::Columns(&SET0),
            ],
        },
        Level {
            designs: vec![
                Counterfactual::Columns(&SET1),
                Counterfactual::Columns(&SET1),
            ],
        },
    ];
    standardise_cause_specific(
        &data,
        &designs,
        &fits,
        cause,
        &CscOptions {
            ties: Ties::Breslow,
            ..CscOptions::default()
        },
        &levels,
        times,
        &StandardiseOptions {
            se: std_err,
            product_limit,
            influence,
            ..StandardiseOptions::default()
        },
    )
    .unwrap()
}

#[test]
fn influence_cause_specific_tied_strata() {
    let n = 90;
    let s = sim(n, 9, true, 2);
    let times = [1.0, 2.5, 4.0, 5.0];
    // The exact form is the derivative of the estimate, with the product limit
    // and without it.
    for product_limit in [false, true] {
        for cause in [1, 2] {
            let estimator = |w: &[f64]| -> Vec<f64> {
                csc_run(
                    &s,
                    Some(w),
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
            let unit = csc_run(
                &s,
                None,
                product_limit,
                InfluenceForm::Exact,
                &times,
                true,
                cause,
            );
            check(
                &format!("csc cause {cause}, product limit {product_limit}"),
                n,
                6 * n + 80,
                &estimator,
                &unit,
            );
        }
    }
}

/// The first-order form is not the derivative of the product limit: its
/// standard errors differ from the exact ones, by the order of the total
/// hazard, and agree without the product limit, where the two forms are the
/// same function.
#[test]
fn first_order_form_differs_from_exact_under_the_product_limit() {
    let n = 90;
    let s = sim(n, 9, true, 2);
    let times = [1.0, 2.5, 4.0, 5.0];
    for product_limit in [true, false] {
        let mut gap = 0.0f64;
        for cause in [1, 2] {
            let exact = csc_run(
                &s,
                None,
                product_limit,
                InfluenceForm::Exact,
                &times,
                true,
                cause,
            );
            let first = csc_run(
                &s,
                None,
                product_limit,
                InfluenceForm::Exponential,
                &times,
                true,
                cause,
            );
            for (a, b) in exact
                .se
                .as_ref()
                .unwrap()
                .iter()
                .zip(first.se.as_ref().unwrap())
            {
                let (a, b) = (a.unwrap(), b.unwrap());
                gap = gap.max((a - b).abs() / a);
            }
        }
        eprintln!(
            "first-order against exact standard error, product limit {product_limit}: up to {gap:.3e} relative"
        );
        if product_limit {
            assert!(gap > 1e-3, "the two forms should differ, by {gap}");
        } else {
            assert!(gap <= 8.0 * (n as f64) * f64::EPSILON, "{gap}");
        }
    }
}

fn build<'a>(s: &'a Sim, w: Option<&'a [f64]>) -> SurvData<'a> {
    let mut builder = SurvData::builder(&s.stop, &s.status)
        .n_codes(2)
        .stratum(&s.stratum);
    if let Some(w) = w {
        builder = builder.weight(w);
    }
    builder.build().unwrap()
}

/// The standard error of one prediction of
/// [`CauseSpecificCox::predict`](epistat_survival::absrisk::CauseSpecificCox::predict)
/// against the square root of the sum of squares of the numerical derivatives
/// of the risk with respect to the case weight of each person. The tolerance of
/// the derivative of a person is that of [`check`], and that of the square root
/// of a sum of squares is `sum (2 |d_i| tol_i + tol_i^2)` divided by the larger
/// of the two standard errors.
#[test]
fn absrisk_standard_error_is_the_derivative() {
    let n = 90;
    let s = sim(n, 13, true, 2);
    let times = [1.0, 2.5, 4.0, 5.0];
    let eps = f64::EPSILON;
    let profile1 = [1.0, 0.4, 1.0];
    let profile2 = [1.0, 0.4];
    let new1 = Mat::from_fn(1, 3, |_, j| profile1[j]);
    let new2 = Mat::from_fn(1, 2, |_, j| profile2[j]);
    let new_designs = [new1.as_ref(), new2.as_ref()];
    let stratum_row = [1u32];
    let new = NewRows {
        designs: &new_designs,
        stratum: Some(&stratum_row),
        offset: None,
    };
    let x2 = Mat::from_fn(n, 2, |i, j| s.design[(i, j)]);
    let designs = [s.design.as_ref(), x2.as_ref()];
    let predict = |w: Option<&[f64]>, product_limit: bool, form: InfluenceForm, se: bool| {
        let data = build(&s, w);
        let fits: Vec<CoxModelFit> = (0..2)
            .map(|m| cox::fit(&data, designs[m], &precise(Some(m as u32 + 1))).unwrap())
            .collect();
        let model = CauseSpecificCox::from_fits(
            &data,
            &designs,
            fits,
            &CscOptions {
                ties: Ties::Breslow,
                ..CscOptions::default()
            },
        )
        .unwrap();
        let mut out = Vec::new();
        for cause in [1, 2] {
            out.push(
                model
                    .predict(
                        &new,
                        &times,
                        cause,
                        &RiskOptions {
                            product_limit,
                            se,
                            influence: form,
                            ..RiskOptions::default()
                        },
                    )
                    .unwrap(),
            );
        }
        out
    };
    let h = 1e-4;
    for product_limit in [true, false] {
        let unit = predict(None, product_limit, InfluenceForm::Exact, true);
        let first = predict(None, product_limit, InfluenceForm::Exponential, true);
        // The numerical influence function of each cause at each time.
        let mut influence = vec![vec![vec![0.0; n]; times.len()]; 2];
        let mut tol = vec![vec![vec![0.0; n]; times.len()]; 2];
        for i in 0..n {
            let mut d = [vec![], vec![]];
            for (kk, step) in [h, h / 2.0].into_iter().enumerate() {
                let mut w = vec![1.0; n];
                w[i] = 1.0 + step;
                let up = predict(Some(&w), product_limit, InfluenceForm::Exact, false);
                w[i] = 1.0 - step;
                let down = predict(Some(&w), product_limit, InfluenceForm::Exact, false);
                d[kk] = (0..2)
                    .map(|c| {
                        (0..times.len())
                            .map(|t| {
                                (up[c].risk[t].unwrap() - down[c].risk[t].unwrap()) / (2.0 * step)
                            })
                            .collect::<Vec<f64>>()
                    })
                    .collect::<Vec<_>>();
            }
            for c in 0..2 {
                for t in 0..times.len() {
                    influence[c][t][i] = (4.0 * d[1][c][t] - d[0][c][t]) / 3.0;
                    let f = unit[c].risk[t].unwrap();
                    tol[c][t][i] = 2.0 * (d[0][c][t] - d[1][c][t]).abs()
                        + 4.0 * (6 * n + 80) as f64 * eps * f.abs() / (h / 2.0);
                }
            }
        }
        let mut worst = 0.0f64;
        let mut gap_first = 0.0f64;
        for c in 0..2 {
            for t in 0..times.len() {
                let direct = influence[c][t].iter().map(|v| v * v).sum::<f64>().sqrt();
                let bound_sq: f64 = (0..n)
                    .map(|i| {
                        2.0 * influence[c][t][i].abs() * tol[c][t][i] + tol[c][t][i] * tol[c][t][i]
                    })
                    .sum();
                let got = unit[c].se.as_ref().unwrap()[t].unwrap();
                let bound = bound_sq / got.max(direct);
                worst = worst.max((got - direct).abs() / bound);
                assert!(
                    (got - direct).abs() <= bound,
                    "product limit {product_limit}, cause {}, time {t}: {got} against {direct}, bound {bound}",
                    c + 1
                );
                let f = first[c].se.as_ref().unwrap()[t].unwrap();
                gap_first = gap_first.max((f - direct).abs() / direct);
            }
        }
        eprintln!(
            "absrisk, product limit {product_limit}: worst deviation {worst:.3} of its bound; the first-order standard error is off by up to {gap_first:.3e} relative"
        );
    }
}

/// When every person is given the same covariates the average is one
/// prediction, with no spread of the covariates, so that the estimate and the
/// standard error are those of `CauseSpecificCox::predict` for that row. The
/// two are the same sums formed in a different order, so they agree to a few
/// `n J eps`, relative to the value.
#[test]
fn one_profile_is_one_prediction() {
    let n = 300;
    let s = sim(n, 21, true, 2);
    let data = SurvData::builder(&s.stop, &s.status)
        .n_codes(2)
        .build()
        .unwrap();
    let x2 = Mat::from_fn(n, 2, |i, j| s.design[(i, j)]);
    let designs = [s.design.as_ref(), x2.as_ref()];
    let opts = CscOptions::default();
    let model = CauseSpecificCox::fit(&data, &designs, &opts).unwrap();
    let times = [0.5, 2.0, 4.0, 6.0, 9.0];
    let profile = [(0, 1.0), (1, 0.3), (2, 1.0)];
    let profile2 = [(0, 1.0), (1, 0.3)];
    let new1 = Mat::from_fn(1, 3, |_, j| profile[j].1);
    let new2 = Mat::from_fn(1, 2, |_, j| profile2[j].1);
    let new_designs = [new1.as_ref(), new2.as_ref()];
    let new = NewRows {
        designs: &new_designs,
        stratum: None,
        offset: None,
    };
    let levels = [Level {
        designs: vec![
            Counterfactual::Columns(&profile),
            Counterfactual::Columns(&profile2),
        ],
    }];
    let n_events = s.status.iter().filter(|&&c| c != 0).count() as f64;
    let mut worst = 0.0f64;
    for cause in [1, 2] {
        for product_limit in [true, false] {
            let want = model
                .predict(
                    &new,
                    &times,
                    cause,
                    &RiskOptions {
                        product_limit,
                        se: true,
                        ..RiskOptions::default()
                    },
                )
                .unwrap();
            let got = standardise_cause_specific(
                &data,
                &designs,
                model.fits(),
                cause,
                &opts,
                &levels,
                &times,
                &StandardiseOptions {
                    se: true,
                    product_limit,
                    ..StandardiseOptions::default()
                },
            )
            .unwrap();
            let bound = 8.0 * (n as f64 + n_events) * f64::EPSILON;
            for k in 0..times.len() {
                let (a, b) = (want.risk[k], got.risk[k]);
                let (sa, sb) = (want.se.as_ref().unwrap()[k], got.se.as_ref().unwrap()[k]);
                assert_eq!(a.is_some(), b.is_some());
                if let (Some(a), Some(b)) = (a, b) {
                    let e = (a - b).abs() / a.abs().max(f64::MIN_POSITIVE);
                    worst = worst.max(e / bound);
                    assert!(e <= bound, "risk {a} against {b} at {k}");
                    let (sa, sb) = (sa.unwrap(), sb.unwrap());
                    let e = (sa - sb).abs() / sa;
                    worst = worst.max(e / bound);
                    assert!(e <= bound, "se {sa} against {sb} at {k}");
                }
            }
        }
    }
    eprintln!("one profile: worst deviation {worst:.3} of its bound");
}

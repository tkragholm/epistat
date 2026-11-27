//! The pooled logistic censoring weights and the variances with estimated
//! treatment and censoring weights give identical bits in rayon pools of 1, 2, 3,
//! 4 and 8 threads.

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::cox::CoxOptions;
use epistat_survival::weighting::cox as wcox;
use epistat_survival::weighting::ipcw::{CensoringModel, IpcwOptions};
use epistat_survival::weighting::iptw::{self, IptwOptions};
use epistat_survival::weighting::km::{self, KmOptions};
use epistat_survival::weighting::pooled;
use epistat_survival::weighting::pooled::PooledEvent;
use epistat_survival::weighting::pooled::PooledOptions;
use epistat_survival::weighting::{OutcomeCensoring, WeightAt, Weights};
use epistat_survival::{Covariance, SurvData};
use epistat_testkit::{POOLS, assert_same_on_pools};

struct Simulated {
    id: Vec<u32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    arm: Vec<u32>,
    cluster: Vec<u32>,
    /// Per row: intercept, interval, covariate at the start of the interval.
    z: Mat<f64>,
    treat: Vec<bool>,
    z_subject: Mat<f64>,
}

fn simulate() -> Simulated {
    let n = 600;
    let mut rng = Stream::seeded(21);
    let (mut id, mut start, mut stop, mut status, mut arm, mut cluster) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let (mut cols, mut treat, mut zs) = ([Vec::new(), Vec::new()], Vec::new(), Vec::new());
    for i in 0..n {
        let (x1, x2) = (rng.normal(), f64::from(u8::from(rng.uniform() < 0.4)));
        let p = 1.0 / (1.0 + (-(0.3 * x1 - 0.2 * x2)).exp());
        let a = rng.uniform() < p;
        treat.push(a);
        zs.push([1.0, x1, x2]);
        let mut l = rng.normal();
        for k in 1..=40u32 {
            l = 0.6 * l + 0.7 * rng.normal() + 0.2 * f64::from(u8::from(a));
            let hazard = 1.0 / (1.0 + (3.0 - 0.4 * l + 0.5 * f64::from(u8::from(a))).exp());
            let event = rng.uniform() < hazard;
            let drop = 1.0 / (1.0 + (2.2 - 0.6 * l - 0.4 * f64::from(u8::from(a))).exp());
            let censor = !event && rng.uniform() < drop;
            id.push(i);
            start.push(f64::from(k - 1));
            stop.push(f64::from(k));
            status.push(i32::from(event));
            arm.push(u32::from(a));
            cluster.push(i % 60);
            cols[0].push(f64::from(k));
            cols[1].push(l);
            if event || censor {
                break;
            }
        }
    }
    let rows = id.len();
    let z = Mat::from_fn(rows, 4, |i, j| match j {
        0 => 1.0,
        1 => cols[0][i],
        2 => cols[1][i],
        _ => f64::from(arm[i]),
    });
    let z_subject = Mat::from_fn(n as usize, 3, |i, j| zs[i][j]);
    Simulated {
        id,
        start,
        stop,
        status,
        arm,
        cluster,
        z,
        treat,
        z_subject,
    }
}

#[test]
fn estimated_weight_variances_are_the_same_on_every_pool() {
    let s = simulate();
    let rows = s.stop.len();
    // The interval is constant in every risk set of the outcome, and the arm in
    // the strata of the data, so the models have the covariate and its square.
    let x = Mat::from_fn(rows, 2, |i, j| s.z[(i, 2)].powi(if j == 0 { 1 } else { 2 }));
    let x_out = Mat::from_fn(rows, 2, |i, j| s.z[(i, 3 - j)]);
    let numerator = Mat::from_fn(rows, 2, |i, j| s.z[(i, 3 * j)]);
    let plain = SurvData::builder(&s.stop, &s.status)
        .start(&s.start)
        .id(&s.id)
        .build()
        .unwrap();
    let arms = SurvData::builder(&s.stop, &s.status)
        .start(&s.start)
        .id(&s.id)
        .stratum(&s.arm)
        .cluster(&s.cluster)
        .build()
        .unwrap();
    let times = [3.0, 6.0, 10.0];
    let cox_opts = CoxOptions::default();
    for stabilise in [false, true] {
        assert_same_on_pools(&POOLS, || {
            let num = stabilise.then(|| numerator.as_ref());
            let pooled_fit = pooled::fit(
                &plain,
                PooledEvent::Censoring { censored: None },
                s.z.as_ref(),
                num,
                &PooledOptions::default(),
            )
            .unwrap();
            let propensity =
                iptw::fit(&s.treat, s.z_subject.as_ref(), &IptwOptions::default()).unwrap();
            let ipcw_opts = IpcwOptions {
                stabilise,
                ..IpcwOptions::default()
            };
            let mut out = pooled_fit.weight.clone();
            out.extend(&pooled_fit.weight_before);
            out.extend(&pooled_fit.coefficients);
            let cox_cases = [
                Weights {
                    iptw: None,
                    subject: None,
                    censoring: Some(OutcomeCensoring::Pooled {
                        fit: &pooled_fit,
                        at: WeightAt::Stop,
                    }),
                },
                Weights {
                    iptw: Some(&propensity),
                    subject: Some(&s.id),
                    censoring: Some(OutcomeCensoring::Hazard {
                        model: CensoringModel::Cox { x: x.as_ref() },
                        options: &ipcw_opts,
                        at: WeightAt::Start,
                    }),
                },
                Weights {
                    iptw: None,
                    subject: None,
                    censoring: Some(OutcomeCensoring::Hazard {
                        model: CensoringModel::KaplanMeier,
                        options: &ipcw_opts,
                        at: WeightAt::Stop,
                    }),
                },
            ];
            for est in &cox_cases {
                let fit = wcox::fit(&plain, x_out.as_ref(), &cox_opts, est).unwrap();
                out.extend(&fit.coefficients);
                out.extend(fit.covariance.as_slice().iter().map(|v| v.unwrap()));
                out.extend(fit.robust_covariance.as_slice().iter().map(|v| v.unwrap()));
            }
            let km_cases = [
                Weights {
                    iptw: Some(&propensity),
                    subject: Some(&s.id),
                    censoring: Some(OutcomeCensoring::Pooled {
                        fit: &pooled_fit,
                        at: WeightAt::Start,
                    }),
                },
                Weights {
                    iptw: None,
                    subject: None,
                    censoring: Some(OutcomeCensoring::Hazard {
                        model: CensoringModel::Cox { x: x.as_ref() },
                        options: &ipcw_opts,
                        at: WeightAt::Stop,
                    }),
                },
            ];
            for est in &km_cases {
                let curves = km::estimate(&arms, &times, &KmOptions::default(), est).unwrap();
                out.extend(&curves.surv);
                out.extend(
                    curves
                        .covariance
                        .iter()
                        .flat_map(Covariance::as_slice)
                        .map(|v| v.unwrap_or(f64::NAN)),
                );
                out.extend(
                    curves
                        .covariance_known
                        .iter()
                        .flat_map(Covariance::as_slice)
                        .map(|v| v.unwrap_or(f64::NAN)),
                );
            }
            out
        });
    }
}

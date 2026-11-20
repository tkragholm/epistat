//! The weighting entry points give identical bits in rayon pools of 1, 2, 3, 4
//! and 8 threads: the propensity weights and their summary, the weighted Cox
//! fit with both variances, the weighted Kaplan-Meier curves with both
//! covariances, and the censoring weights from both models.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::cox::CoxOptions;
use epistat_survival::weighting::cox as wcox;
use epistat_survival::weighting::ipcw::{self, CensoringModel, IpcwOptions};
use epistat_survival::weighting::iptw::{self, Estimand, IptwOptions};
use epistat_survival::weighting::km::{self, KmOptions};
use epistat_testkit::{POOLS, assert_same_on_pools};
use faer::Mat;

struct Simulated {
    z: Mat<f64>,
    treat: Vec<bool>,
    stop: Vec<f64>,
    start: Vec<f64>,
    status: Vec<i32>,
    subject: Vec<u32>,
}

fn simulate() -> Simulated {
    let n = 1500;
    let mut rng = Stream::seeded(8);
    let mut cols = [Vec::new(), Vec::new()];
    let mut treat = Vec::new();
    let (mut stop, mut start, mut status, mut subject) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let (a, b) = (rng.normal(), f64::from(u8::from(rng.uniform() < 0.4)));
        let p = 1.0 / (1.0 + (-(0.3 * a + 0.5 * b - 0.2)).exp());
        let t = rng.uniform() < p;
        let time =
            (rng.exp1() * 8.0 * (-0.4 * a - 0.5 * f64::from(u8::from(t))).exp()).ceil() + 1.0;
        let cens = (rng.exp1() * 12.0).ceil() + 1.0;
        let (ts, ev) = if time <= cens { (time, 1) } else { (cens, 0) };
        cols[0].push(a);
        cols[1].push(b);
        treat.push(t);
        // Two rows for a subject followed beyond the cut.
        let cut = 2.0 + f64::from((rng.below(3)) as u32);
        if ts > cut {
            for (s, e, st) in [(0.0, cut, 0), (cut, ts, ev)] {
                start.push(s);
                stop.push(e);
                status.push(st);
                subject.push(i as u32);
            }
        } else {
            start.push(0.0);
            stop.push(ts);
            status.push(ev);
            subject.push(i as u32);
        }
    }
    let z = Mat::from_fn(n, 3, |i, j| if j == 0 { 1.0 } else { cols[j - 1][i] });
    Simulated {
        z,
        treat,
        stop,
        start,
        status,
        subject,
    }
}

#[test]
fn weighting_is_the_same_on_every_pool() {
    let s = simulate();
    let rows = s.stop.len();
    let x = Mat::from_fn(rows, 2, |i, j| {
        if j == 0 {
            f64::from(u8::from(s.treat[s.subject[i] as usize]))
        } else {
            s.z[(s.subject[i] as usize, 1)]
        }
    });
    let stratum: Vec<u32> = (0..rows)
        .map(|i| u32::from(s.treat[s.subject[i] as usize]))
        .collect();
    let data = SurvData::builder(&s.stop, &s.status)
        .start(&s.start)
        .build()
        .unwrap();
    let arms = SurvData::builder(&s.stop, &s.status)
        .start(&s.start)
        .stratum(&stratum)
        .build()
        .unwrap();
    let times = [3.0, 6.0, 10.0, 15.0];
    for (estimand, stabilise) in [
        (Estimand::Ate, false),
        (Estimand::Ate, true),
        (Estimand::Att, false),
    ] {
        assert_same_on_pools(&POOLS, || {
            let w = iptw::fit(
                &s.treat,
                s.z.as_ref(),
                &IptwOptions {
                    estimand,
                    stabilise,
                },
            )
            .unwrap();
            let mut out = w.weights.clone();
            out.extend(&w.propensity);
            out.extend(&w.coefficients);
            for g in [w.summary.treated.unwrap(), w.summary.control.unwrap()] {
                out.extend([
                    g.sum,
                    g.min,
                    g.max,
                    g.mean,
                    g.coefficient_of_variation.unwrap(),
                    g.effective_sample_size,
                ]);
            }
            let cox = wcox::fit(
                &data,
                x.as_ref(),
                &CoxOptions::default(),
                &w,
                Some(&s.subject),
            )
            .unwrap();
            out.extend(&cox.coefficients);
            out.extend(&cox.covariance);
            out.extend(&cox.robust_covariance);
            out.extend(&cox.weight_covariance);
            let curves =
                km::estimate(&arms, &times, &KmOptions::default(), &w, Some(&s.subject)).unwrap();
            out.extend(&curves.surv);
            out.extend(curves.covariance.iter().map(|v| v.unwrap_or(f64::NAN)));
            out.extend(
                curves
                    .covariance_known
                    .iter()
                    .map(|v| v.unwrap_or(f64::NAN)),
            );
            out
        });
    }
}

#[test]
fn censoring_weights_are_the_same_on_every_pool() {
    let s = simulate();
    let rows = s.stop.len();
    let x = Mat::from_fn(rows, 2, |i, j| s.z[(s.subject[i] as usize, j + 1)]);
    let data = SurvData::builder(&s.stop, &s.status)
        .start(&s.start)
        .id(&s.subject)
        .build()
        .unwrap();
    for stabilise in [false, true] {
        let opts = IpcwOptions {
            stabilise,
            ..IpcwOptions::default()
        };
        assert_same_on_pools(&POOLS, || {
            let mut out = Vec::new();
            for model in [
                CensoringModel::KaplanMeier,
                CensoringModel::Cox { x: x.as_ref() },
            ] {
                let w = ipcw::fit(&data, model, &opts).unwrap();
                out.extend(&w.weight_at_stop);
                out.extend(&w.weight_at_start);
            }
            out
        });
    }
}

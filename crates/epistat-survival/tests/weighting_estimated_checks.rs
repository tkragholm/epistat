//! The refusals of the pooled censoring model and of the variances with estimated
//! weights, and the identity of the generic route with the propensity-only one.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few dozen rows, far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::{CoxOptions, Ties};
use epistat_survival::weighting::cox as wcox;
use epistat_survival::weighting::ipcw::{CensoringModel, IpcwOptions};
use epistat_survival::weighting::iptw::{self, IptwOptions};
use epistat_survival::weighting::km::{self, KmOptions};
use epistat_survival::weighting::pooled;
use epistat_survival::weighting::{Censoring, Estimated, WeightAt};

/// Four subjects with two or three rows each.
struct Tiny {
    id: Vec<u32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    design: Mat<f64>,
}

fn tiny() -> Tiny {
    let id = vec![0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7];
    let stop = vec![
        1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0,
        2.0,
    ];
    let status = vec![0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0];
    let start: Vec<f64> = stop.iter().map(|s| s - 1.0).collect();
    let n = id.len();
    let design = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            1.0
        } else {
            f64::from((i * 7 % 5) as u32) / 3.0
        }
    });
    Tiny {
        id,
        start,
        stop,
        status,
        design,
    }
}

fn data(t: &Tiny) -> SurvData<'_> {
    SurvData::builder(&t.stop, &t.status)
        .start(&t.start)
        .id(&t.id)
        .build()
        .unwrap()
}

#[test]
fn pooled_fit_refuses_invalid_input() {
    let t = tiny();
    let d = data(&t);
    let short = Mat::from_fn(3, 2, |_, _| 1.0);
    let err = |r: epistat_survival::Result<pooled::PooledIpcw<'_>>| r.err().unwrap().to_string();
    assert!(err(pooled::fit(&d, None, short.as_ref(), None)).contains("design: 3 rows"));
    assert!(
        err(pooled::fit(&d, Some(&[true]), t.design.as_ref(), None)).contains("censored: 1 values")
    );
    // No censoring event among the rows of the fit.
    let none = vec![false; t.id.len()];
    assert!(
        err(pooled::fit(&d, Some(&none), t.design.as_ref(), None)).contains("censoring events")
    );
    // A column that repeats another.
    let aliased = Mat::from_fn(t.id.len(), 3, |i, j| t.design[(i, j.min(1))]);
    assert!(err(pooled::fit(&d, None, aliased.as_ref(), None)).contains("aliased"));
    // Two rows of a subject with the same stop.
    let same = vec![1.0; t.id.len()];
    let status = vec![0; t.id.len()];
    let bad = SurvData::builder(&same, &status).id(&t.id).build().unwrap();
    assert!(err(pooled::fit(&bad, None, t.design.as_ref(), None)).contains("same stop"));
}

#[test]
fn estimated_variances_refuse_invalid_input() {
    let t = tiny();
    let d = data(&t);
    let x = Mat::from_fn(t.id.len(), 1, |i, _| t.design[(i, 1)]);
    let opts = CoxOptions::default();
    let nothing = Estimated::default();
    let e = wcox::fit_estimated(&d, x.as_ref(), &opts, &nothing).unwrap_err();
    assert!(e.to_string().contains("neither treatment nor censoring"));
    let e = km::estimate_estimated(&d, &[1.0], &KmOptions::default(), &nothing).unwrap_err();
    assert!(e.to_string().contains("neither treatment nor censoring"));
    // A subject map without treatment weights.
    let ids = t.id.clone();
    let km_model = IpcwOptions::default();
    let est = Estimated {
        iptw: None,
        subject: Some(&ids),
        censoring: Some(Censoring::Hazard {
            model: CensoringModel::KaplanMeier,
            options: &km_model,
            at: WeightAt::Stop,
        }),
    };
    let e = wcox::fit_estimated(&d, x.as_ref(), &opts, &est).unwrap_err();
    assert!(e.to_string().contains("subject: it indexes"));
    // Case weights in the data.
    let w = vec![1.0; t.id.len()];
    let weighted = SurvData::builder(&t.stop, &t.status)
        .start(&t.start)
        .id(&t.id)
        .weight(&w)
        .build()
        .unwrap();
    let est = Estimated {
        iptw: None,
        subject: None,
        censoring: Some(Censoring::Hazard {
            model: CensoringModel::KaplanMeier,
            options: &km_model,
            at: WeightAt::Stop,
        }),
    };
    let e = wcox::fit_estimated(&weighted, x.as_ref(), &opts, &est).unwrap_err();
    assert!(e.to_string().contains("case weights"));
    // Exact ties with a Cox model of censoring.
    let exact = IpcwOptions {
        ties: Ties::Exact,
        ..IpcwOptions::default()
    };
    let est = Estimated {
        iptw: None,
        subject: None,
        censoring: Some(Censoring::Hazard {
            model: CensoringModel::Cox { x: x.as_ref() },
            options: &exact,
            at: WeightAt::Stop,
        }),
    };
    let e = wcox::fit_estimated(&d, x.as_ref(), &opts, &est).unwrap_err();
    assert!(e.to_string().contains("exact method"));
    // A pooled fit of other rows than the data.
    let other = SurvData::builder(&t.stop[..10], &t.status[..10])
        .start(&t.start[..10])
        .id(&t.id[..10])
        .build()
        .unwrap();
    let fit = pooled::fit(&d, None, t.design.as_ref(), None);
    if let Ok(fit) = fit {
        let est = Estimated {
            iptw: None,
            subject: None,
            censoring: Some(Censoring::Pooled {
                fit: &fit,
                at: WeightAt::Stop,
            }),
        };
        let x10 = Mat::from_fn(10, 1, |i, _| t.design[(i, 1)]);
        let e = wcox::fit_estimated(&other, x10.as_ref(), &opts, &est).unwrap_err();
        assert!(e.to_string().contains("pooled fit has"));
    }
}

#[test]
fn propensity_weights_alone_give_the_propensity_only_variance() {
    // 40 subjects with one or two rows and a treatment that depends on a covariate.
    let n = 40usize;
    let mut stop = Vec::new();
    let mut start = Vec::new();
    let mut status = Vec::new();
    let mut subject = Vec::new();
    let mut treat = Vec::new();
    let z = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            1.0
        } else {
            ((i * 13 % 11) as f64 - 5.0) / 5.0
        }
    });
    for i in 0..n {
        treat.push((i * 7 + i / 3) % 3 != 0);
        let end = 3.0 + (i * 5 % 9) as f64;
        let event = i32::from(i % 4 != 0);
        if i % 2 == 0 {
            start.extend([0.0, 2.0]);
            stop.extend([2.0, end]);
            status.extend([0, event]);
            subject.extend([i as u32, i as u32]);
        } else {
            start.push(0.0);
            stop.push(end);
            status.push(event);
            subject.push(i as u32);
        }
    }
    let rows = stop.len();
    let x = Mat::from_fn(rows, 2, |r, j| {
        if j == 0 {
            f64::from(u8::from(treat[subject[r] as usize]))
        } else {
            z[(subject[r] as usize, 1)]
        }
    });
    let arm: Vec<u32> = (0..rows)
        .map(|r| u32::from(treat[subject[r] as usize]))
        .collect();
    let cluster: Vec<u32> = (0..rows).map(|r| subject[r] / 4).collect();
    let d = SurvData::builder(&stop, &status)
        .start(&start)
        .cluster(&cluster)
        .build()
        .unwrap();
    let arms = SurvData::builder(&stop, &status)
        .start(&start)
        .stratum(&arm)
        .cluster(&cluster)
        .build()
        .unwrap();
    let times = [2.0, 4.0, 6.0];
    let mut worst = 0.0f64;
    for stabilise in [false, true] {
        let w = iptw::fit(
            &treat,
            z.as_ref(),
            &IptwOptions {
                stabilise,
                ..IptwOptions::default()
            },
        )
        .unwrap();
        let opts = CoxOptions::default();
        let old = wcox::fit(&d, x.as_ref(), &opts, &w, Some(&subject)).unwrap();
        let est = Estimated {
            iptw: Some(&w),
            subject: Some(&subject),
            censoring: None,
        };
        let new = wcox::fit_estimated(&d, x.as_ref(), &opts, &est).unwrap();
        assert_eq!(old.n_units, new.n_units);
        assert_eq!(old.coefficients, new.coefficients);
        for (a, b) in [
            (&old.covariance, &new.covariance),
            (&old.robust_covariance, &new.robust_covariance),
        ] {
            for (u, v) in a.iter().zip(b) {
                let scale = old.covariance[0].abs().max(old.covariance[3].abs());
                worst = worst.max((u - v).abs() / scale);
            }
        }
        let old = km::estimate(&arms, &times, &KmOptions::default(), &w, Some(&subject)).unwrap();
        let new = km::estimate_estimated(&arms, &times, &KmOptions::default(), &est).unwrap();
        assert_eq!(old.surv, new.surv);
        for (a, b) in old.covariance.iter().zip(&new.covariance) {
            let (a, b) = (a.unwrap(), b.unwrap());
            worst = worst.max((a - b).abs() / a.abs().max(1e-12));
        }
    }
    // The routes sum the same terms in another order, so they agree to rounding.
    println!("propensity only against the generic route: {worst:.2e}");
    assert!(worst < 1e-10, "{worst}");
}

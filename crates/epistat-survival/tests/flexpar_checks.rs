//! Properties and edge cases of `flexpar`: refusals, aliasing, weights, offsets,
//! delayed entry, the hazard ratio of a proportional hazards model and the
//! flags of a fit.

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::flexpar::{
    Contrast, FlexParOptions, Predict, PredictOptions, Scale, TimeVarying, contrast, fit, predict,
};

struct Sim {
    time: Vec<f64>,
    start: Vec<f64>,
    status: Vec<i32>,
    x: Mat<f64>,
}

/// Weibull times with shape 1.4 and covariate effects 0.5 and -0.4, censoring,
/// and delayed entry for about a third of the rows. Times are rounded to two
/// decimals so that some tie.
fn simulate(n: usize, seed: u64) -> Sim {
    let mut rng = Stream::seeded(seed);
    let (mut time, mut start, mut status) = (Vec::new(), Vec::new(), Vec::new());
    let mut x = Mat::<f64>::zeros(n, 2);
    for i in 0..n {
        let a = rng.normal();
        let b = f64::from(u8::from(rng.below(5) < 2));
        let t = ((rng.exp1().powf(1.0 / 1.4)) * (-(0.5 * a - 0.4 * b) / 1.4).exp() * 2.0 * 100.0)
            .round()
            / 100.0
            + 0.01;
        let c = (rng.exp1() * 5.0 * 100.0).round() / 100.0 + 0.01;
        let entry = if rng.below(3) == 0 {
            (0.4 * t * 100.0).round() / 100.0
        } else {
            0.0
        };
        time.push(t.min(c));
        status.push(i32::from(t <= c));
        start.push(if entry < t.min(c) { entry } else { 0.0 });
        x[(i, 0)] = a;
        x[(i, 1)] = b;
    }
    Sim {
        time,
        start,
        status,
        x,
    }
}

fn data(s: &Sim, entry: bool) -> SurvData<'_> {
    let mut b = SurvData::builder(&s.time, &s.status);
    if entry {
        b = b.start(&s.start);
    }
    b.build().unwrap()
}

#[test]
fn refusals() {
    let s = simulate(200, 1);
    let opts = FlexParOptions::new(Scale::Hazard, 3);
    let d = data(&s, false);
    // a time of 0
    let mut zero = s.time.clone();
    zero[0] = 0.0;
    let dz = SurvData::builder(&zero, &s.status).build().unwrap();
    assert!(fit(&dz, s.x.as_ref(), &opts).is_err());
    // all censored
    let none = vec![0; 200];
    let dn = SurvData::builder(&s.time, &none).build().unwrap();
    assert!(fit(&dn, s.x.as_ref(), &opts).is_err());
    // events all at one time: the knots coincide
    let tied = vec![2.0; 200];
    let dt = SurvData::builder(&tied, &s.status).build().unwrap();
    assert!(fit(&dt, s.x.as_ref(), &opts).is_err());
    // more parameters than events
    let few: Vec<i32> = (0..200).map(|i| i32::from(i < 4)).collect();
    let df = SurvData::builder(&s.time, &few).build().unwrap();
    assert!(fit(&df, s.x.as_ref(), &opts).is_err());
    // a time-varying column out of range, a repeated one, df 0
    for tvc in [
        vec![TimeVarying { column: 2, df: 1 }],
        vec![
            TimeVarying { column: 0, df: 1 },
            TimeVarying { column: 0, df: 2 },
        ],
        vec![TimeVarying { column: 0, df: 0 }],
    ] {
        let o = FlexParOptions { tvc: &tvc, ..opts };
        assert!(fit(&d, s.x.as_ref(), &o).is_err());
    }
    assert!(fit(&d, s.x.as_ref(), &FlexParOptions::new(Scale::Hazard, 0)).is_err());
    // knots of the wrong number
    let o = FlexParOptions {
        knots: Some(&[0.0, 1.0]),
        ..opts
    };
    assert!(fit(&d, s.x.as_ref(), &o).is_err());
    // covariates of another length
    let short = Mat::<f64>::zeros(10, 2);
    assert!(fit(&d, short.as_ref(), &opts).is_err());
    // competing causes
    let codes: Vec<i32> = s
        .status
        .iter()
        .enumerate()
        .map(|(i, &e)| e * (1 + i32::from(i % 3 == 0)))
        .collect();
    let dc = SurvData::builder(&s.time, &codes)
        .n_codes(2)
        .build()
        .unwrap();
    assert!(fit(&dc, s.x.as_ref(), &opts).is_err());
    // split follow-up of a subject is accepted on the hazard scale only
    let ids: Vec<u32> = (0..200).map(|i| i / 2).collect();
    let di = SurvData::builder(&s.time, &s.status)
        .start(&s.start)
        .id(&ids)
        .build()
        .unwrap();
    assert!(fit(&di, s.x.as_ref(), &FlexParOptions::new(Scale::Odds, 2)).is_err());
    assert!(fit(&di, s.x.as_ref(), &FlexParOptions::new(Scale::Hazard, 2)).is_ok());
    // a stratified follow-up
    let strata: Vec<u32> = (0..200).map(|i| i % 2).collect();
    let ds = SurvData::builder(&s.time, &s.status)
        .stratum(&strata)
        .build()
        .unwrap();
    assert!(fit(&ds, s.x.as_ref(), &opts).is_err());
}

#[test]
fn prediction_refusals() {
    let s = simulate(300, 2);
    let f = fit(
        &data(&s, false),
        s.x.as_ref(),
        &FlexParOptions::new(Scale::Hazard, 3),
    )
    .unwrap();
    let x = Mat::from_fn(1, 2, |_, j| if j == 0 { 0.3 } else { 1.0 });
    let ok = PredictOptions::default();
    assert!(predict(&f, x.as_ref(), &[1.0], Predict::Survival, &ok).is_ok());
    assert!(predict(&f, x.as_ref(), &[0.0], Predict::Survival, &ok).is_err());
    assert!(predict(&f, x.as_ref(), &[f64::INFINITY], Predict::Survival, &ok).is_err());
    let bad_alpha = PredictOptions { alpha: 1.5, ..ok };
    assert!(predict(&f, x.as_ref(), &[1.0], Predict::Survival, &bad_alpha).is_err());
    let wide = Mat::<f64>::zeros(1, 3);
    assert!(predict(&f, wide.as_ref(), &[1.0], Predict::Survival, &ok).is_err());
    // no clusters, so no robust covariance
    let robust = PredictOptions { robust: true, ..ok };
    assert!(f.robust_covariance.is_none());
    assert!(predict(&f, x.as_ref(), &[1.0], Predict::Survival, &robust).is_err());
    let two = Mat::<f64>::zeros(2, 2);
    assert!(
        contrast(
            &f,
            x.as_ref(),
            two.as_ref(),
            &[1.0],
            Contrast::HazardRatio,
            &ok
        )
        .is_err()
    );
}

#[test]
fn an_aliased_column_has_no_coefficient_and_changes_nothing_else() {
    let s = simulate(300, 3);
    let n = 300;
    // columns: a, b, 2a - 1 (aliased with a and the intercept), a constant
    let wide = Mat::from_fn(n, 4, |i, j| match j {
        0 => s.x[(i, 0)],
        1 => s.x[(i, 1)],
        2 => 2.0 * s.x[(i, 0)] - 1.0,
        _ => 3.0,
    });
    let opts = FlexParOptions::new(Scale::Probit, 3);
    let d = data(&s, true);
    let full = fit(&d, wide.as_ref(), &opts).unwrap();
    let base = fit(&d, s.x.as_ref(), &opts).unwrap();
    assert_eq!(full.beta[2], None);
    assert_eq!(full.beta[3], None);
    assert!((full.log_likelihood - base.log_likelihood).abs() < 1e-9);
    for j in 0..2 {
        assert!((full.beta[j].unwrap() - base.beta[j].unwrap()).abs() < 1e-6);
    }
    let cov = full.covariance.as_ref().unwrap();
    let q = full.dimension();
    let (i2, i3) = (full.gamma.len() + 2, full.gamma.len() + 3);
    assert!(cov[i2 * q + i2].is_none() && cov[i3 * q + i3].is_none());
    // predictions ignore the aliased columns
    let x = Mat::from_fn(1, 4, |_, j| [0.2, 1.0, 5.0, -7.0][j]);
    let xb = Mat::from_fn(1, 2, |_, j| [0.2, 1.0][j]);
    let o = PredictOptions::default();
    let a = predict(&full, x.as_ref(), &[1.0, 2.0], Predict::Survival, &o).unwrap();
    let b = predict(&base, xb.as_ref(), &[1.0, 2.0], Predict::Survival, &o).unwrap();
    assert!((a[0][1].estimate - b[0][1].estimate).abs() < 1e-7);
    // a time-varying term on an aliased column is refused
    let tvc = [TimeVarying { column: 3, df: 1 }];
    assert!(fit(&d, wide.as_ref(), &FlexParOptions { tvc: &tvc, ..opts }).is_err());
}

#[test]
fn integer_weights_are_repeated_rows_and_zero_weights_are_absent_rows() {
    let s = simulate(120, 4);
    let weights: Vec<f64> = (0..120).map(|i| [0.0, 1.0, 2.0, 3.0][i % 4]).collect();
    let opts = FlexParOptions::new(Scale::Odds, 2);
    let weighted = SurvData::builder(&s.time, &s.status)
        .weight(&weights)
        .build()
        .unwrap();
    let a = fit(&weighted, s.x.as_ref(), &opts).unwrap();
    // the same data written out
    let (mut time, mut status, mut rows) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..120 {
        for _ in 0..usize::from([0_u8, 1, 2, 3][i % 4]) {
            time.push(s.time[i]);
            status.push(s.status[i]);
            rows.push(i);
        }
    }
    let x = Mat::from_fn(rows.len(), 2, |r, j| s.x[(rows[r], j)]);
    let repeated = SurvData::builder(&time, &status).build().unwrap();
    // the knots are placed on the rows of positive weight, once each, so give them
    let b = fit(
        &repeated,
        x.as_ref(),
        &FlexParOptions {
            knots: Some(&a.knots),
            ..opts
        },
    )
    .unwrap();
    let a = fit(
        &weighted,
        s.x.as_ref(),
        &FlexParOptions {
            knots: Some(&a.knots),
            ..opts
        },
    )
    .unwrap();
    assert!((a.log_likelihood - b.log_likelihood).abs() < 1e-8 * b.log_likelihood.abs());
    for j in 0..2 {
        assert!((a.beta[j].unwrap() - b.beta[j].unwrap()).abs() < 1e-7);
    }
    assert_eq!(a.n_rows, 90);
    assert!((a.n_events - b.n_events).abs() < 1e-9);
}

#[test]
fn an_offset_moves_the_intercept_and_leaves_no_null_likelihood() {
    let s = simulate(250, 5);
    let d = data(&s, false);
    let opts = FlexParOptions::new(Scale::Hazard, 3);
    let base = fit(&d, s.x.as_ref(), &opts).unwrap();
    assert!(base.null_log_likelihood.unwrap() < base.log_likelihood);
    // an offset of 0.7 on every row shifts the intercept by -0.7
    let off = vec![0.7; 250];
    let shifted = fit(
        &d,
        s.x.as_ref(),
        &FlexParOptions {
            offset: Some(&off),
            ..opts
        },
    )
    .unwrap();
    assert!(shifted.null_log_likelihood.is_none());
    assert!((shifted.log_likelihood - base.log_likelihood).abs() < 1e-8);
    assert!((shifted.gamma[0] - (base.gamma[0] - 0.7)).abs() < 1e-6);
    let short = vec![0.0; 3];
    assert!(
        fit(
            &d,
            s.x.as_ref(),
            &FlexParOptions {
                offset: Some(&short),
                ..opts
            }
        )
        .is_err()
    );
}

#[test]
fn the_hazard_ratio_is_constant_without_time_varying_effects_and_is_not_with_them() {
    let s = simulate(500, 6);
    let d = data(&s, true);
    let x1 = Mat::from_fn(1, 2, |_, j| [0.4, 1.0][j]);
    let x0 = Mat::from_fn(1, 2, |_, j| [0.4, 0.0][j]);
    let times = [0.2, 0.5, 1.0, 2.0, 4.0];
    let o = PredictOptions::default();
    let f = fit(&d, s.x.as_ref(), &FlexParOptions::new(Scale::Hazard, 3)).unwrap();
    let hr = contrast(
        &f,
        x1.as_ref(),
        x0.as_ref(),
        &times,
        Contrast::HazardRatio,
        &o,
    )
    .unwrap();
    let expected = f.beta[1].unwrap().exp();
    for p in &hr[0] {
        assert!((p.estimate - expected).abs() < 1e-9 * expected);
        assert!(p.lower.unwrap() < p.estimate && p.estimate < p.upper.unwrap());
    }
    let tvc = [TimeVarying { column: 1, df: 2 }];
    let g = fit(
        &d,
        s.x.as_ref(),
        &FlexParOptions {
            tvc: &tvc,
            ..FlexParOptions::new(Scale::Hazard, 3)
        },
    )
    .unwrap();
    let hr = contrast(
        &g,
        x1.as_ref(),
        x0.as_ref(),
        &times,
        Contrast::HazardRatio,
        &o,
    )
    .unwrap();
    assert!((hr[0][0].estimate - hr[0][4].estimate).abs() > 1e-6);
    // a model with time-varying effects fits at least as well
    assert!(g.log_likelihood >= f.log_likelihood - 1e-9);
    // the survival difference and the hazard difference are finite and ordered
    let sd = contrast(
        &g,
        x1.as_ref(),
        x0.as_ref(),
        &times,
        Contrast::SurvivalDifference,
        &o,
    )
    .unwrap();
    assert!(
        sd[0]
            .iter()
            .all(|p| p.lower.unwrap() <= p.estimate && p.estimate <= p.upper.unwrap())
    );
    let hd = contrast(
        &g,
        x1.as_ref(),
        x0.as_ref(),
        &times,
        Contrast::HazardDifference,
        &o,
    )
    .unwrap();
    assert!(hd[0].iter().all(|p| p.standard_error.is_finite()));
}

#[test]
fn survival_is_a_decreasing_function_within_its_limits() {
    let s = simulate(400, 7);
    let d = data(&s, true);
    for scale in [Scale::Hazard, Scale::Odds, Scale::Probit] {
        let f = fit(&d, s.x.as_ref(), &FlexParOptions::new(scale, 4)).unwrap();
        assert!(f.converged && f.monotone_on_data && !f.diverging);
        let x = Mat::from_fn(1, 2, |_, j| [0.0, 1.0][j]);
        let times: Vec<f64> = (1..40).map(|k| 0.1 * f64::from(k)).collect();
        let curve = &predict(
            &f,
            x.as_ref(),
            &times,
            Predict::Survival,
            &PredictOptions::default(),
        )
        .unwrap()[0];
        for w in curve.windows(2) {
            assert!(w[1].estimate <= w[0].estimate);
        }
        for p in curve {
            assert!(0.0 <= p.lower.unwrap() && p.lower.unwrap() <= p.estimate);
            assert!(p.estimate <= p.upper.unwrap() && p.upper.unwrap() <= 1.0);
        }
        // the cumulative hazard is minus the log of the survivor function
        let h = &predict(
            &f,
            x.as_ref(),
            &times,
            Predict::CumulativeHazard,
            &PredictOptions::default(),
        )
        .unwrap()[0];
        for (a, b) in h.iter().zip(curve) {
            assert!((a.estimate + b.estimate.ln()).abs() < 1e-10 * (1.0 + a.estimate));
        }
    }
}

/// With more spline degrees of freedom than the data support, the fitted curve
/// can decrease between event times. The flag says so and the predictions refuse.
#[test]
fn a_fit_that_is_not_monotone_is_flagged_and_refused() {
    for seed in 0..200 {
        let s = simulate(45, 100 + seed);
        let d = data(&s, false);
        let Ok(f) = fit(&d, s.x.as_ref(), &FlexParOptions::new(Scale::Hazard, 6)) else {
            continue;
        };
        if f.monotone_on_data {
            continue;
        }
        let x = Mat::from_fn(1, 2, |_, j| [0.0, 1.0][j]);
        assert!(
            predict(
                &f,
                x.as_ref(),
                &[1.0],
                Predict::Survival,
                &PredictOptions::default()
            )
            .is_err()
        );
        let allowed = PredictOptions {
            allow_non_monotone: true,
            ..PredictOptions::default()
        };
        assert!(predict(&f, x.as_ref(), &[1.0], Predict::Survival, &allowed).is_ok());
        return;
    }
    panic!("no simulated fit was non-monotone");
}

/// One Newton iteration is not enough, so the fit goes to BFGS and back, and
/// reaches the same maximum.
#[test]
fn the_bfgs_fallback_reaches_the_same_maximum() {
    let s = simulate(300, 8);
    let d = data(&s, true);
    for scale in [Scale::Hazard, Scale::Odds, Scale::Probit] {
        let opts = FlexParOptions {
            max_iterations: 1,
            ..FlexParOptions::new(scale, 4)
        };
        let f = fit(&d, s.x.as_ref(), &opts).unwrap();
        let g = fit(&d, s.x.as_ref(), &FlexParOptions::new(scale, 4)).unwrap();
        assert!(g.converged && f.iterations > 1, "{scale:?}");
        assert!((f.log_likelihood - g.log_likelihood).abs() < 1e-8 * g.log_likelihood.abs());
        assert!(!f.diverging);
    }
}

//! The edges of the standardised risk: refused input, times outside the
//! follow-up, a zero time, and case weights against repeated rows.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the inputs are a few hundred rows and the weights are 1 or 2"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::cox::{self, CoxOptions, Ties};
use epistat_survival::standardise::{Counterfactual, Level, StandardiseOptions, standardise_cox};
use faer::Mat;

fn sim(n: usize) -> (Vec<f64>, Vec<i32>, Mat<f64>) {
    let mut rng = Stream::seeded(3);
    let mut stop = Vec::new();
    let mut status = Vec::new();
    let mut cols = Vec::new();
    for _ in 0..n {
        let x = rng.normal();
        let a = f64::from(u8::from(rng.below(2) == 0));
        let t = rng.exp1() / (0.1 * (0.5 * x - 0.3 * a).exp());
        let c = 2.0 + 10.0 * rng.uniform();
        stop.push(t.min(c));
        status.push(i32::from(t <= c));
        cols.push([a, x]);
    }
    (stop, status, Mat::from_fn(n, 2, |i, j| cols[i][j]))
}

const SET: [(usize, f64); 1] = [(0, 1.0)];

#[test]
fn refused_input_and_undefined_times() {
    let (stop, status, x) = sim(80);
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let options = CoxOptions::default();
    let fit = cox::fit(&data, x.as_ref(), &options).unwrap();
    let level = Level::single(Counterfactual::Columns(&SET));
    let run = |levels: &[Level<'_>], times: &[f64], opts: &StandardiseOptions| {
        standardise_cox(&data, x.as_ref(), &fit, &options, levels, times, opts)
    };
    let plain = StandardiseOptions::default();
    assert!(run(&[], &[1.0], &plain).is_err());
    assert!(run(std::slice::from_ref(&level), &[f64::NAN], &plain).is_err());
    let bad_column = Level::single(Counterfactual::Columns(&[(5, 1.0)]));
    assert!(run(&[bad_column], &[1.0], &plain).is_err());
    let bad_value = Level::single(Counterfactual::Columns(&[(0, f64::INFINITY)]));
    assert!(run(&[bad_value], &[1.0], &plain).is_err());
    let wrong_models = Level {
        designs: vec![Counterfactual::Columns(&SET), Counterfactual::Columns(&SET)],
    };
    assert!(run(&[wrong_models], &[1.0], &plain).is_err());
    let narrow = Mat::<f64>::zeros(80, 1);
    let wrong_shape = Level::single(Counterfactual::Design(narrow.as_ref()));
    assert!(run(&[wrong_shape], &[1.0], &plain).is_err());
    let bad_level = StandardiseOptions {
        conf_level: 1.0,
        ..plain
    };
    assert!(run(std::slice::from_ref(&level), &[1.0], &bad_level).is_err());

    // A time before the first event has risk 0 and standard error 0, a time
    // after the last follow-up has none.
    let last = stop.iter().copied().fold(0.0, f64::max);
    let out = run(
        std::slice::from_ref(&level),
        &[0.0, 1.0, last, last + 1.0],
        &StandardiseOptions {
            std_err: true,
            ..plain
        },
    )
    .unwrap();
    assert_eq!(out.risk[0], Some(0.0));
    assert_eq!(out.std_err.as_ref().unwrap()[0], Some(0.0));
    assert!(out.risk[1].is_some() && out.risk[2].is_some());
    assert_eq!(out.risk[3], None);
    assert_eq!(out.std_err.as_ref().unwrap()[3], None);
    assert_eq!(out.lower.as_ref().unwrap()[3], None);
    let c = out.contrast(0, 0).unwrap();
    assert_eq!(c.difference[1], Some(0.0));
    assert_eq!(c.ratio[1], Some(1.0));
    assert_eq!(c.ratio[0], None, "a ratio of two zeros is not defined");
    assert!(out.contrast(0, 1).is_err());
}

#[test]
fn case_weights_act_as_repeated_rows() {
    let (stop, status, x) = sim(60);
    let weight: Vec<f64> = (0..60)
        .map(|i| if i % 3 == 0 { 2.0 } else { 1.0 })
        .collect();
    let options = CoxOptions {
        ties: Ties::Breslow,
        eps: 1e-14,
        iter_max: 100,
        ..CoxOptions::default()
    };
    let weighted = SurvData::builder(&stop, &status)
        .weight(&weight)
        .build()
        .unwrap();
    let mut rows = Vec::new();
    for (i, w) in weight.iter().enumerate() {
        for _ in 0..*w as usize {
            rows.push(i);
        }
    }
    let stop2: Vec<f64> = rows.iter().map(|&i| stop[i]).collect();
    let status2: Vec<i32> = rows.iter().map(|&i| status[i]).collect();
    let x2 = Mat::from_fn(rows.len(), 2, |i, j| x[(rows[i], j)]);
    let repeated = SurvData::builder(&stop2, &status2).build().unwrap();
    let level = [Level::single(Counterfactual::Columns(&SET))];
    let times = [1.0, 3.0, 6.0];
    let est = |data: &SurvData<'_>, design: &Mat<f64>| {
        let fit = cox::fit(data, design.as_ref(), &options).unwrap();
        standardise_cox(
            data,
            design.as_ref(),
            &fit,
            &options,
            &level,
            &times,
            &StandardiseOptions::default(),
        )
        .unwrap()
    };
    let a = est(&weighted, &x);
    let b = est(&repeated, &x2);
    // Both are sums of at most n = 90 terms through a fit converged to
    // rounding with p = 2 columns, J <= 60 event times, |x beta| below 4 and a
    // cumulative hazard below 2, so the bound of `r_standardise.rs`,
    // 4 eps (n + J + p + 4 + xb) (1 + H), applies.
    let bound = 4.0 * f64::EPSILON * (90.0 + 60.0 + 2.0 + 4.0 + 8.0) * 3.0;
    for k in 0..times.len() {
        let (u, v) = (a.risk[k].unwrap(), b.risk[k].unwrap());
        assert!((u - v).abs() <= bound * u, "{u} against {v}");
    }
    // The standard error describes unit weights only.
    let refused = standardise_cox(
        &weighted,
        x.as_ref(),
        &cox::fit(&weighted, x.as_ref(), &options).unwrap(),
        &options,
        &level,
        &times,
        &StandardiseOptions {
            std_err: true,
            ..StandardiseOptions::default()
        },
    );
    assert!(refused.is_err());
}

//! What the Cox censoring model, the contrasts, the recalibration and the
//! counting-process forms of `metrics` refuse.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::{CoxOptions, Ties};
use epistat_survival::metrics::contrast::{ContrastOptions, auc_difference, brier_difference};
use epistat_survival::metrics::recalibration::{
    Bandwidth, RecalibrationOptions, RecalibrationScale, SmoothOptions, slope_intercept,
    smoothed_curve,
};
use epistat_survival::metrics::rttright::{self, RttrightOptions};
use epistat_survival::metrics::score::{CoxCensoring, ScoreCensoring, ScoreOptions, auc, brier};

fn make(x: &Mat<f64>, options: CoxOptions) -> ScoreCensoring<'_> {
    ScoreCensoring::Cox(CoxCensoring {
        x: x.as_ref(),
        options,
    })
}

fn build<'a>(id: &'a [u32], start: &'a [f64], stop: &'a [f64], status: &'a [i32]) -> SurvData<'a> {
    SurvData::builder(stop, status)
        .start(start)
        .id(id)
        .build()
        .unwrap()
}

fn data() -> (Vec<f64>, Vec<i32>, Vec<f64>) {
    let time = vec![1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0];
    let status = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0];
    let x = vec![
        0.1, -0.3, 0.5, 0.2, -0.1, 0.4, 0.0, -0.5, 0.3, 0.1, -0.2, 0.6,
    ];
    (time, status, x)
}

#[test]
fn the_cox_censoring_model_refuses_what_it_cannot_use() {
    let (time, status, x) = data();
    let n = time.len();
    let data = SurvData::builder(&time, &status).build().unwrap();
    let risk = Mat::from_fn(n, 1, |i, _| 0.2 + 0.03 * i as f64);
    let col = Mat::from_fn(n, 1, |i, _| x[i]);
    let ok = brier(
        &data,
        risk.as_ref(),
        &[4.0],
        &make(&col, CoxOptions::default()),
        &ScoreOptions::default(),
    )
    .unwrap();
    assert!(ok.se[0].is_some());
    let short = Mat::from_fn(n - 1, 1, |i, _| x[i]);
    let e = brier(
        &data,
        risk.as_ref(),
        &[4.0],
        &make(&short, CoxOptions::default()),
        &ScoreOptions::default(),
    )
    .unwrap_err();
    assert!(e.to_string().contains("rows of data"), "{e}");
    let exact = CoxOptions {
        ties: Ties::Exact,
        ..CoxOptions::default()
    };
    let e = brier(
        &data,
        risk.as_ref(),
        &[4.0],
        &make(&col, exact),
        &ScoreOptions::default(),
    )
    .unwrap_err();
    assert!(e.to_string().contains("exact"), "{e}");
    // No covariates is the Breslow estimate of the censoring hazard with no
    // coefficients: it works.
    let none = Mat::<f64>::zeros(n, 0);
    assert!(
        auc(
            &data,
            risk.as_ref(),
            &[4.0],
            &make(&none, CoxOptions::default()),
            &ScoreOptions::default()
        )
        .is_ok()
    );
}

#[test]
fn contrasts_need_matching_scores_with_influence() {
    let (time, status, _) = data();
    let n = time.len();
    let data = SurvData::builder(&time, &status).build().unwrap();
    let a = Mat::from_fn(n, 2, |i, _| 0.2 + 0.03 * i as f64);
    let b = Mat::from_fn(n, 2, |i, _| 0.6 - 0.02 * i as f64);
    let kept = ScoreOptions {
        keep_influence: true,
        ..ScoreOptions::default()
    };
    let sa = brier(
        &data,
        a.as_ref(),
        &[3.0, 6.0],
        &ScoreCensoring::KaplanMeier,
        &kept,
    )
    .unwrap();
    let sb = brier(
        &data,
        b.as_ref(),
        &[3.0, 6.0],
        &ScoreCensoring::KaplanMeier,
        &kept,
    )
    .unwrap();
    let c = brier_difference(&sa, &sb, &ContrastOptions::default()).unwrap();
    assert!((c.difference[0].unwrap() - (sa.brier[0] - sb.brier[0])).abs() < 1e-15);
    assert!(c.lower[0].unwrap() < c.upper[0].unwrap());
    let p = c.p_value[0].unwrap();
    assert!((0.0..=1.0).contains(&p));
    // A model against itself has no difference and no variance.
    let same = brier_difference(&sa, &sa, &ContrastOptions::default()).unwrap();
    assert_eq!(same.difference[0], Some(0.0));
    assert!(same.se[0].is_none() && same.p_value[0].is_none());
    let plain = brier(
        &data,
        a.as_ref(),
        &[3.0, 6.0],
        &ScoreCensoring::KaplanMeier,
        &ScoreOptions::default(),
    )
    .unwrap();
    let e = brier_difference(&plain, &sb, &ContrastOptions::default()).unwrap_err();
    assert!(e.to_string().contains("influence"), "{e}");
    let other = brier(
        &data,
        a.as_ref().get(.., ..1),
        &[3.0],
        &ScoreCensoring::KaplanMeier,
        &kept,
    )
    .unwrap();
    let e = brier_difference(&other, &sb, &ContrastOptions::default()).unwrap_err();
    assert!(e.to_string().contains("same data and times"), "{e}");
    let aa = auc(
        &data,
        a.as_ref(),
        &[3.0, 6.0],
        &ScoreCensoring::KaplanMeier,
        &kept,
    )
    .unwrap();
    let ab = auc(
        &data,
        b.as_ref(),
        &[3.0, 6.0],
        &ScoreCensoring::KaplanMeier,
        &kept,
    )
    .unwrap();
    assert!(auc_difference(&aa, &ab, &ContrastOptions::default()).is_ok());
}

#[test]
fn recalibration_refuses_what_it_cannot_use() {
    let n = 200;
    let mut rng = epistat_rng::Stream::seeded(5);
    let time: Vec<f64> = (0..n).map(|_| 0.1 + rng.exp1()).collect();
    let status: Vec<i32> = (0..n).map(|_| i32::from(rng.below(3) > 0)).collect();
    let data = SurvData::builder(&time, &status).build().unwrap();
    let t = time.iter().copied().fold(0.0f64, f64::max);
    let risk: Vec<f64> = (0..n)
        .map(|_| 0.05 + 0.9 * rng.below(1000) as f64 / 1000.0)
        .collect();
    let curve =
        |r: &[f64], t: f64, b: Bandwidth| smoothed_curve(&data, r, t, &SmoothOptions::new(b));
    assert!(curve(&risk, 1.0, Bandwidth::Fixed(0.3)).is_ok());
    let e = curve(&risk, 1.0, Bandwidth::Fixed(0.0)).unwrap_err();
    assert!(e.to_string().contains("bandwidth"), "{e}");
    let e = curve(&risk, t + 1.0, Bandwidth::Smooth).unwrap_err();
    assert!(
        e.to_string().contains("after the last observed time"),
        "{e}"
    );
    let e = curve(&risk[..n - 1], 1.0, Bandwidth::Smooth).unwrap_err();
    assert!(e.to_string().contains("has length"), "{e}");
    let mut bad = risk.clone();
    bad[2] = 1.5;
    assert!(curve(&bad, 1.0, Bandwidth::Smooth).is_err());
    let opts = RecalibrationOptions::new(RecalibrationScale::Logit);
    let fit = slope_intercept(&data, &risk, 1.0, &opts).unwrap();
    assert!(fit.converged && fit.se_slope().unwrap() > 0.0 && fit.se_intercept().unwrap() > 0.0);
    // The link is infinite at 0 and 1.
    let mut edge = risk.clone();
    edge[0] = 0.0;
    let e = slope_intercept(&data, &edge, 1.0, &opts).unwrap_err();
    assert!(e.to_string().contains("strictly between"), "{e}");
    // The slope is not estimable when every prediction is the same.
    let flat = vec![0.3; n];
    assert!(slope_intercept(&data, &flat, 1.0, &opts).is_err());
    // Strata, clusters and ids change what a pseudo-value is.
    let stratum: Vec<u32> = (0..n).map(|i| u32::from(i % 2 == 1)).collect();
    let stratified = SurvData::builder(&time, &status)
        .stratum(&stratum)
        .build()
        .unwrap();
    let e = slope_intercept(&stratified, &risk, 1.0, &opts).unwrap_err();
    assert!(e.to_string().contains("whole sample"), "{e}");
}

fn counting() -> (Vec<u32>, Vec<f64>, Vec<f64>, Vec<i32>) {
    // Subject 1: (0, 2] then (2, 5], event. Subject 2: (0, 4] censored.
    // Subject 3: (0, 1], (1, 3], (3, 6], censored.
    (
        vec![1, 1, 2, 3, 3, 3],
        vec![0.0, 2.0, 0.0, 0.0, 1.0, 3.0],
        vec![2.0, 5.0, 4.0, 1.0, 3.0, 6.0],
        vec![0, 1, 0, 0, 0, 0],
    )
}

#[test]
fn rttright_with_a_start_refuses_inconsistent_follow_up() {
    let (id, start, stop, status) = counting();
    let opts = RttrightOptions::default();
    let good = build(&id, &start, &stop, &status);
    let w = rttright::estimate(&good, None, &opts).unwrap();
    // Only the last row of the subject with the event has a weight.
    assert!(w.weights[(1, 0)] > 0.0);
    assert!((0..6).filter(|&i| i != 1).all(|i| w.weights[(i, 0)] == 0.0));
    // At 3 the first subject is still at risk on its second row, subject 3 on its
    // third row, and the weights sum to one.
    let w = rttright::estimate(&good, Some(&[3.0]), &opts).unwrap();
    let total: f64 = (0..6).map(|i| w.weights[(i, 0)]).sum();
    assert!((total - 1.0).abs() < 1e-12, "{total}");
    assert!(w.weights[(1, 0)] > 0.0 && w.weights[(2, 0)] > 0.0 && w.weights[(4, 0)] > 0.0);
    assert!(w.weights[(0, 0)] == 0.0 && w.weights[(3, 0)] == 0.0 && w.weights[(5, 0)] == 0.0);

    let e = rttright::estimate(&good, Some(&[0.0]), &opts).unwrap_err();
    assert!(e.to_string().contains("not after the common start"), "{e}");
    let no_id = SurvData::builder(&stop, &status)
        .start(&start)
        .build()
        .unwrap();
    let e = rttright::estimate(&no_id, None, &opts).unwrap_err();
    assert!(e.to_string().contains("delayed entry"), "{e}");
    let mut gap = start.clone();
    gap[1] = 2.5;
    let e = rttright::estimate(&build(&id, &gap, &stop, &status), None, &opts).unwrap_err();
    assert!(e.to_string().contains("gap"), "{e}");
    let mut overlap = start.clone();
    overlap[1] = 1.5;
    let e = rttright::estimate(&build(&id, &overlap, &stop, &status), None, &opts).unwrap_err();
    assert!(e.to_string().contains("overlap"), "{e}");
    let mut early = status.clone();
    early[0] = 1;
    let e = rttright::estimate(&build(&id, &start, &stop, &early), None, &opts).unwrap_err();
    assert!(e.to_string().contains("not the last of its subject"), "{e}");
    let mut late = start.clone();
    late[2] = 0.5;
    let e = rttright::estimate(&build(&id, &late, &stop, &status), None, &opts).unwrap_err();
    assert!(e.to_string().contains("different times"), "{e}");
    let weight = [1.0, 2.0, 1.0, 1.0, 1.0, 1.0];
    let weighted = SurvData::builder(&stop, &status)
        .start(&start)
        .id(&id)
        .weight(&weight)
        .build()
        .unwrap();
    let e = rttright::estimate(&weighted, None, &opts).unwrap_err();
    assert!(e.to_string().contains("several case weights"), "{e}");
}

//! Properties of the Kalbfleisch-Prentice curve, `start_time`, the counts at
//! censoring times and the curves of covariate paths that do not need R: the
//! Kaplan-Meier estimate when the coefficients are zero, the estimating
//! equation at tied times, the agreement of a path of constant covariates with
//! the ordinary curve, and the refusals.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few hundred rows, far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::predict::{
    CurveType, FittedModel, HazardType, NewData, PathData, SurvfitOptions, survfit, survfit_paths,
};
use epistat_survival::cox::{CoxModelFit, CoxOptions, Ties, fit};

fn with_coefficients(mut model: CoxModelFit, beta: &[f64], variance: &[f64]) -> CoxModelFit {
    model.coefficients = beta.iter().copied().map(Some).collect();
    model.covariance = variance.iter().copied().map(Some).collect();
    model
}

fn kp() -> SurvfitOptions<'static> {
    SurvfitOptions {
        curve_type: CurveType::KalbfleischPrentice,
        ..SurvfitOptions::default()
    }
}

/// Ten rows with ties: the events at times 2 (three), 3 (one), 5 (two), the
/// others censored at 3, 4, 6, 7. With zero coefficients the curve is
/// Kaplan-Meier: `(1 - 3/10)(1 - 1/6)(1 - 2/4)` at time 5, and the variance
/// part is Greenwood's.
#[test]
fn zero_coefficients_give_kaplan_meier() {
    let stop = [2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0];
    let status = [1, 1, 1, 1, 0, 0, 1, 1, 0, 0];
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let x = Mat::from_fn(10, 1, |i, _| (i as f64 * 0.37).sin());
    let opts = CoxOptions::default();
    let model = with_coefficients(fit(&data, x.as_ref(), &opts).unwrap(), &[0.0], &[0.0]);
    let fitted = FittedModel {
        data: &data,
        x: x.as_ref(),
        fit: &model,
        options: &opts,
    };
    let new_x = Mat::from_fn(1, 1, |_, _| 0.7);
    let new = NewData::new(new_x.as_ref());
    let c = survfit(
        &fitted,
        &new,
        &SurvfitOptions {
            censor_times: true,
            ..kp()
        },
    )
    .unwrap();
    let times = &c.strata[0];
    assert_eq!(times.time, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    assert_eq!(times.n_event, [3.0, 1.0, 0.0, 2.0, 0.0, 0.0]);
    assert_eq!(times.n_censor, [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    assert_eq!(times.n_risk, [10.0, 7.0, 5.0, 4.0, 2.0, 1.0]);
    let expected = [
        0.7,
        0.7 * (1.0 - 1.0 / 7.0),
        0.7 * (1.0 - 1.0 / 7.0),
        0.7 * (1.0 - 1.0 / 7.0) * 0.5,
        0.7 * (1.0 - 1.0 / 7.0) * 0.5,
        0.7 * (1.0 - 1.0 / 7.0) * 0.5,
    ];
    for (got, want) in c.curves[0].survival.iter().zip(expected) {
        assert!((got - want).abs() <= 1e-15, "{got} against {want}");
    }
    // The cumulative hazard is the Breslow one, not minus the log of the curve.
    let h = c.curves[0].cumulative_hazard[3];
    assert!((h - (0.3 + 1.0 / 7.0 + 0.5)).abs() < 1e-15, "{h}");
    // Greenwood's sum of d / (n (n - d)), the coefficient being known.
    let se = c.curves[0].se.as_ref().unwrap();
    let first = 3.0 / (10.0 * 7.0);
    let second = first + 1.0 / (7.0 * 6.0);
    let third = second + 2.0 / (4.0 * 2.0);
    for (k, v) in [(0, first), (1, second), (3, third)] {
        assert!((se[k] - f64::sqrt(v)).abs() < 1e-15, "{k}");
    }
}

/// The product-limit increment at tied times solves its estimating equation.
#[test]
fn tied_increment_solves_the_estimating_equation() {
    // Two rows with the events at time 1, risks 0.5 and 2, a third row alive.
    let stop = [1.0, 1.0, 2.0];
    let status = [1, 1, 0];
    let x = Mat::from_fn(3, 1, |i, _| [(0.5f64).ln(), (2.0f64).ln(), 0.0][i]);
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let opts = CoxOptions::default();
    let model = with_coefficients(fit(&data, x.as_ref(), &opts).unwrap(), &[1.0], &[1.0]);
    let fitted = FittedModel {
        data: &data,
        x: x.as_ref(),
        fit: &model,
        options: &opts,
    };
    // The curve of a row with risk 1 relative to the centre.
    let centre = epistat_survival::cox::predict::reference_means(
        &fitted,
        epistat_survival::cox::predict::Centring::Given(&[0.0]),
    )
    .unwrap();
    assert_eq!(centre, [0.0]);
    let new_x = Mat::from_fn(1, 1, |_, _| 0.0);
    let new = NewData::new(new_x.as_ref());
    let c = survfit(
        &fitted,
        &new,
        &SurvfitOptions {
            centring: epistat_survival::cox::predict::Centring::Given(&[0.0]),
            ..kp()
        },
    )
    .unwrap();
    let a = c.curves[0].survival[0];
    // The risk set at time 1 is 0.5 + 2 + 1 = 3.5.
    let f = 0.5 / (1.0 - a.powf(0.5)) + 2.0 / (1.0 - a.powf(2.0));
    assert!((f - 3.5).abs() < 1e-12 * 3.5, "{f}");
}

/// Rows of an ordinary curve and of a path of constant covariates agree, with
/// the path split anywhere.
#[test]
fn constant_path_equals_the_curve() {
    let n = 60usize;
    let stop: Vec<f64> = (0..n).map(|i| 1.0 + ((i * 7) % 23) as f64).collect();
    let status: Vec<i32> = (0..n).map(|i| i32::from(i % 3 != 0)).collect();
    let stratum: Vec<u32> = (0..n).map(|i| (i % 2) as u32).collect();
    let weight: Vec<f64> = (0..n).map(|i| 1.0 + (i % 4) as f64 * 0.5).collect();
    let data = SurvData::builder(&stop, &status)
        .stratum(&stratum)
        .weight(&weight)
        .build()
        .unwrap();
    let x = Mat::from_fn(n, 2, |i, j| ((i * (j + 3)) % 11) as f64 / 5.0);
    let opts = CoxOptions {
        ties: Ties::Efron,
        ..CoxOptions::default()
    };
    let model = with_coefficients(
        fit(&data, x.as_ref(), &opts).unwrap(),
        &[0.4, -0.3],
        &[0.04, 0.01, 0.01, 0.05],
    );
    let fitted = FittedModel {
        data: &data,
        x: x.as_ref(),
        fit: &model,
        options: &opts,
    };
    let new_x = Mat::from_fn(1, 2, |_, j| [0.6, 1.1][j]);
    let mut new = NewData::new(new_x.as_ref());
    new.stratum = Some(&[1]);
    for curve_type in [
        CurveType::ExpCumulativeHazard,
        CurveType::KalbfleischPrentice,
    ] {
        let o = SurvfitOptions {
            curve_type,
            censor_times: true,
            ..SurvfitOptions::default()
        };
        let c = survfit(&fitted, &new, &o).unwrap();
        let curve = &c.curves[0];
        // One interval, and the same split in three.
        let px = Mat::from_fn(3, 2, |_, j| [0.6, 1.1][j]);
        let split = PathData {
            x: px.as_ref(),
            id: &[7, 7, 7],
            start: &[0.0, 5.0, 11.0],
            stop: &[5.0, 11.0, 100.0],
            stratum: Some(&[1, 1, 1]),
            offset: None,
        };
        let one = PathData {
            x: px.as_ref().subrows(0, 1),
            id: &[7],
            start: &[0.0],
            stop: &[100.0],
            stratum: Some(&[1]),
            offset: None,
        };
        for paths in [&one, &split] {
            let q = survfit_paths(&fitted, paths, &o).unwrap();
            let got = &q.curves[0];
            assert_eq!(got.time, c.strata[curve.stratum].time);
            assert_eq!(got.n_censor, c.strata[curve.stratum].n_censor);
            for k in 0..got.time.len() {
                let close = |a: f64, b: f64| (a - b).abs() <= 1e-13 * b.abs().max(1e-3);
                assert!(close(got.cumulative_hazard[k], curve.cumulative_hazard[k]));
                assert!(close(got.survival[k], curve.survival[k]));
                assert!(close(
                    got.se.as_ref().unwrap()[k],
                    curve.se.as_ref().unwrap()[k]
                ));
            }
        }
    }
}

/// With `start_time` the curve starts at 1 there and equals the full curve
/// divided by its value before.
#[test]
fn start_time_restarts_the_curve() {
    let n = 50usize;
    let stop: Vec<f64> = (0..n).map(|i| 1.0 + ((i * 5) % 19) as f64).collect();
    let status: Vec<i32> = (0..n).map(|i| i32::from(i % 4 != 0)).collect();
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let x = Mat::from_fn(n, 1, |i, _| ((i * 3) % 7) as f64 / 3.0);
    let opts = CoxOptions::default();
    let model = with_coefficients(fit(&data, x.as_ref(), &opts).unwrap(), &[0.5], &[0.02]);
    let fitted = FittedModel {
        data: &data,
        x: x.as_ref(),
        fit: &model,
        options: &opts,
    };
    let new_x = Mat::from_fn(1, 1, |_, _| 0.8);
    let new = NewData::new(new_x.as_ref());
    let full = survfit(&fitted, &new, &SurvfitOptions::default()).unwrap();
    let from = survfit(
        &fitted,
        &new,
        &SurvfitOptions {
            start_time: Some(6.0),
            ..SurvfitOptions::default()
        },
    )
    .unwrap();
    let (tf, tp) = (&full.strata[0], &from.strata[0]);
    assert!(tp.time.iter().all(|&t| t >= 6.0));
    let skip = tf.time.iter().filter(|&&t| t < 6.0).count();
    assert!(skip > 0);
    assert_eq!(tf.time[skip..], tp.time);
    assert_eq!(tf.n_risk[skip..], tp.n_risk);
    let before = full.curves[0].survival[skip - 1];
    for (k, s) in from.curves[0].survival.iter().enumerate() {
        let want = full.curves[0].survival[skip + k] / before;
        assert!((s - want).abs() < 1e-14, "{s} against {want}");
    }
}

#[test]
fn refusals() {
    let stop = [1.0, 2.0, 3.0, 4.0];
    let status = [1, 1, 0, 1];
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let x = Mat::from_fn(4, 1, |i, _| i as f64);
    let opts = CoxOptions::default();
    let model = with_coefficients(fit(&data, x.as_ref(), &opts).unwrap(), &[0.1], &[0.2]);
    let fitted = FittedModel {
        data: &data,
        x: x.as_ref(),
        fit: &model,
        options: &opts,
    };
    let new = NewData::new(x.as_ref().subrows(0, 1));
    let bad = |o: SurvfitOptions<'_>| survfit(&fitted, &new, &o).map(|_| ());
    // The hazard form has no meaning for the product-limit curve.
    assert!(
        bad(SurvfitOptions {
            hazard: Some(HazardType::Efron),
            ..kp()
        })
        .is_err()
    );
    // No event at or after the start time.
    assert!(
        bad(SurvfitOptions {
            start_time: Some(4.5),
            ..SurvfitOptions::default()
        })
        .is_err()
    );
    assert!(
        bad(SurvfitOptions {
            start_time: Some(f64::NAN),
            ..SurvfitOptions::default()
        })
        .is_err()
    );
    // A row that leaves at the start time is kept.
    assert!(
        bad(SurvfitOptions {
            start_time: Some(4.0),
            ..SurvfitOptions::default()
        })
        .is_ok()
    );

    let px = Mat::from_fn(2, 1, |i, _| i as f64);
    let ok = PathData {
        x: px.as_ref(),
        id: &[1, 1],
        start: &[0.0, 2.0],
        stop: &[2.0, 4.0],
        stratum: None,
        offset: None,
    };
    assert!(survfit_paths(&fitted, &ok, &SurvfitOptions::default()).is_ok());
    let path_bad = |p: PathData<'_>| survfit_paths(&fitted, &p, &SurvfitOptions::default());
    assert!(
        path_bad(PathData {
            start: &[0.0, 1.5],
            ..ok
        })
        .is_err()
    );
    assert!(
        path_bad(PathData {
            start: &[2.0, 3.0],
            ..ok
        })
        .is_err()
    );
    assert!(path_bad(PathData { id: &[1], ..ok }).is_err());
    assert!(
        path_bad(PathData {
            stop: &[2.0, f64::NAN],
            ..ok
        })
        .is_err()
    );
    // A gap is allowed and the hazard of the gap is not counted.
    let gap = path_bad(PathData {
        start: &[0.0, 3.0],
        stop: &[1.5, 4.0],
        ..ok
    })
    .unwrap();
    assert_eq!(gap.curves[0].time, [1.0, 4.0]);
}

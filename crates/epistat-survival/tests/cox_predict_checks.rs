//! Properties of the baseline, the predictions and the curves of a Cox fit
//! that do not need R: a case worked by hand, rows of weight zero, the
//! independence of the curves from the means, and the refusals.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few hundred rows, far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::predict::{
    BaselineOptions, Centring, ConfType, FittedModel, HazardType, NewData, PredictOptions,
    PredictType, Reference, SurvfitOptions, Variance, basehaz, predict, predict_terms, survfit,
};
use epistat_survival::cox::{CoxModelFit, CoxOptions, Ties, fit};

fn with_coefficients(mut model: CoxModelFit, beta: &[f64], variance: &[f64]) -> CoxModelFit {
    model.coefficients = beta.iter().copied().map(Some).collect();
    model.covariance = variance.iter().copied().map(Some).collect();
    model
}

/// Three rows at times 1, 2, 3 with the events at 1 and 2, and beta 0: the
/// risk sets are 3 and 2 rows, so the Breslow increments are 1/3 and 1/2 and
/// the variance parts 1/9 and 1/4.
#[test]
fn worked_case() {
    let stop = [1.0, 2.0, 3.0];
    let status = [1, 1, 0];
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let x = Mat::from_fn(3, 1, |i, _| i as f64);
    let opts = CoxOptions {
        ties: Ties::Breslow,
        ..CoxOptions::default()
    };
    let model = with_coefficients(fit(&data, x.as_ref(), &opts).unwrap(), &[0.0], &[0.5]);
    let fitted = FittedModel {
        data: &data,
        x: x.as_ref(),
        fit: &model,
        options: &opts,
    };
    let b = basehaz(&fitted, &BaselineOptions::default()).unwrap();
    let s = &b.strata[0];
    assert_eq!(s.time, [1.0, 2.0]);
    assert_eq!(s.n_events, [1.0, 1.0]);
    assert_eq!(s.n_risk, [3.0, 2.0]);
    assert!((s.hazard[0] - 1.0 / 3.0).abs() < 4.0 * f64::EPSILON);
    assert!((s.cumulative_hazard[1] - 5.0 / 6.0).abs() < 4.0 * f64::EPSILON);
    assert!((s.cumulative_variance[1] - (1.0 / 9.0 + 0.25)).abs() < 4.0 * f64::EPSILON);
    // The means are 0 for the covariate 0, 1, 2 with weights 1: mean 1.
    assert_eq!(b.centre, [1.0]);
    // Expected number of events of the three rows: H(stop) times risk 1.
    let e = predict(
        &fitted,
        PredictType::Expected,
        None,
        &PredictOptions::default(),
    )
    .unwrap();
    assert!((e.fit[0] - 1.0 / 3.0).abs() < 4.0 * f64::EPSILON);
    assert!((e.fit[1] - 5.0 / 6.0).abs() < 4.0 * f64::EPSILON);
    assert!((e.fit[2] - 5.0 / 6.0).abs() < 4.0 * f64::EPSILON);
    // The standard error of the survival at time 2 for x = the mean: with
    // beta fixed at 0 the covariate part is g = H z - sum of xbar, where
    // xbar(s) = h(s) times the mean of z over the risk set.
    let mut new = NewData::new(x.as_ref());
    new.stratum = None;
    let curves = survfit(&fitted, &new, &SurvfitOptions::default()).unwrap();
    let c = &curves.curves[1];
    assert!((c.survival[1] - (-5.0f64 / 6.0).exp()).abs() < 4.0 * f64::EPSILON);
    // z = 0 for the middle row. xbar(1) = (1/3) (mean of z over rows 0,1,2 =
    // 0), xbar(2) = (1/2) (mean over rows 1,2 = 0.5).
    let g: f64 = 0.0 - (0.0 + 0.5 * 0.5);
    let want = (1.0 / 9.0 + 0.25 + 0.5 * g * g).sqrt();
    assert!((c.se.as_ref().unwrap()[1] - want).abs() < 8.0 * f64::EPSILON);
}

type Simulated = (Vec<f64>, Vec<f64>, Vec<i32>, Vec<f64>, Vec<u32>, Mat<f64>);

fn simulated() -> Simulated {
    let n = 120usize;
    let mut stop = Vec::new();
    let mut start = Vec::new();
    let mut status = Vec::new();
    let mut weight = Vec::new();
    let mut stratum = Vec::new();
    let mut state = 12345u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as f64 / f64::from(1u32 << 31)
    };
    let mut xs = Vec::new();
    for i in 0..n {
        let t = (next() * 20.0).ceil() + 1.0;
        stop.push(t);
        start.push((next() * (t - 1.0)).floor());
        status.push(i32::from(next() > 0.3));
        weight.push(1.0 + (next() * 3.0).floor());
        stratum.push((i % 2) as u32);
        xs.push([next() - 0.5, f64::from(u8::from(next() > 0.5))]);
    }
    let x = Mat::from_fn(n, 2, |i, j| xs[i][j]);
    (stop, start, status, weight, stratum, x)
}

#[test]
fn rows_of_weight_zero_change_no_baseline_bit() {
    let (stop, start, status, weight, stratum, x) = simulated();
    let n = stop.len();
    let beta = [0.4, -0.3];
    let variance = [0.02, 0.0, 0.0, 0.03];
    let opts = CoxOptions::default();
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .weight(&weight)
        .stratum(&stratum)
        .build()
        .unwrap();
    let model = with_coefficients(fit(&data, x.as_ref(), &opts).unwrap(), &beta, &variance);
    // The same rows followed by rows of weight zero, which carry events too.
    let extra = 17;
    let mut stop2 = stop.clone();
    let mut start2 = start.clone();
    let mut status2 = status.clone();
    let mut weight2 = weight.clone();
    let mut stratum2 = stratum.clone();
    for i in 0..extra {
        stop2.push(stop[i] + 0.5);
        start2.push(start[i]);
        status2.push(1);
        weight2.push(0.0);
        stratum2.push(stratum[i]);
    }
    let x2 = Mat::from_fn(n + extra, 2, |i, j| x[(i % n, j)]);
    let data2 = SurvData::builder(&stop2, &status2)
        .start(&start2)
        .weight(&weight2)
        .stratum(&stratum2)
        .build()
        .unwrap();
    let model2 = with_coefficients(fit(&data2, x2.as_ref(), &opts).unwrap(), &beta, &variance);
    let given = BaselineOptions {
        centring: Centring::Given(&[0.1, 0.2]),
        ..BaselineOptions::default()
    };
    let b1 = basehaz(
        &FittedModel {
            data: &data,
            x: x.as_ref(),
            fit: &model,
            options: &opts,
        },
        &given,
    )
    .unwrap();
    let f2 = FittedModel {
        data: &data2,
        x: x2.as_ref(),
        fit: &model2,
        options: &opts,
    };
    let b2 = basehaz(&f2, &given).unwrap();
    for (s1, s2) in b1.strata.iter().zip(&b2.strata) {
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&s1.time), bits(&s2.time));
        assert_eq!(bits(&s1.cumulative_hazard), bits(&s2.cumulative_hazard));
        assert_eq!(bits(&s1.cumulative_variance), bits(&s2.cumulative_variance));
    }
    assert_eq!(b2.n_used, n);
    // A row of weight zero still has a prediction.
    let p = predict(&f2, PredictType::Risk, None, &PredictOptions::default()).unwrap();
    assert_eq!(p.fit.len(), n + extra);
}

#[test]
fn curves_do_not_depend_on_the_means() {
    let (stop, start, status, weight, stratum, x) = simulated();
    let opts = CoxOptions::default();
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .weight(&weight)
        .stratum(&stratum)
        .build()
        .unwrap();
    let model = with_coefficients(
        fit(&data, x.as_ref(), &opts).unwrap(),
        &[0.4, -0.3],
        &[0.02, 0.001, 0.001, 0.03],
    );
    let fitted = FittedModel {
        data: &data,
        x: x.as_ref(),
        fit: &model,
        options: &opts,
    };
    let new_x = Mat::from_fn(2, 2, |i, j| 0.5 * (i + j) as f64);
    let strata = [0u32, 1];
    let mut new = NewData::new(new_x.as_ref());
    new.stratum = Some(&strata);
    let reference = survfit(&fitted, &new, &SurvfitOptions::default()).unwrap();
    // An error of the baseline is at most the sums of its sizes times eps, and
    // the means move the risk scores by exp of a shift of at most 2 |beta|.
    let eps = f64::EPSILON;
    let bound = 8.0 * eps * (25.0 + 2.0 + 4.0 + 4.0);
    for centring in [
        Centring::Weighted,
        Centring::Given(&[0.0, 0.0]),
        Centring::Given(&[3.0, -2.0]),
    ] {
        let other = survfit(
            &fitted,
            &new,
            &SurvfitOptions {
                centring,
                ..SurvfitOptions::default()
            },
        )
        .unwrap();
        for (a, b) in reference.curves.iter().zip(&other.curves) {
            for (u, v) in a.cumulative_hazard.iter().zip(&b.cumulative_hazard) {
                assert!((u - v).abs() <= bound * u.abs(), "{u} {v}");
            }
            let (sa, sb) = (a.se.as_ref().unwrap(), b.se.as_ref().unwrap());
            for (u, v) in sa.iter().zip(sb) {
                assert!((u - v).abs() <= bound * 10.0 * u.abs(), "{u} {v}");
            }
        }
    }
    // Efron and Breslow give different hazards with ties.
    let breslow = survfit(
        &fitted,
        &new,
        &SurvfitOptions {
            hazard: Some(HazardType::Breslow),
            ..SurvfitOptions::default()
        },
    )
    .unwrap();
    assert!(
        breslow.curves[0]
            .cumulative_hazard
            .iter()
            .zip(&reference.curves[0].cumulative_hazard)
            .any(|(a, b)| (a - b).abs() > 1e-6)
    );
}

#[test]
fn refusals() {
    let (stop, start, status, weight, stratum, x) = simulated();
    let opts = CoxOptions::default();
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .weight(&weight)
        .stratum(&stratum)
        .build()
        .unwrap();
    let model = with_coefficients(
        fit(&data, x.as_ref(), &opts).unwrap(),
        &[0.4, -0.3],
        &[0.02, 0.0, 0.0, 0.03],
    );
    let fitted = FittedModel {
        data: &data,
        x: x.as_ref(),
        fit: &model,
        options: &opts,
    };
    let new_x = Mat::from_fn(1, 2, |_, j| j as f64);
    let strata = [0u32];
    let strata_unknown = [9u32];
    let t0 = [0.5];
    let t1 = [2.0];
    let expect_err = |r: Result<_, _>, what: &str| {
        assert!(
            matches!(
                r,
                Err(epistat_survival::Error::Input(_) | epistat_survival::Error::Invalid(_))
            ),
            "{what} should be refused"
        );
    };
    let plain = NewData::new(new_x.as_ref());
    expect_err(
        survfit(&fitted, &plain, &SurvfitOptions::default()).map(|_| ()),
        "no stratum",
    );
    let mut new = plain;
    new.stratum = Some(&strata_unknown);
    expect_err(
        survfit(&fitted, &new, &SurvfitOptions::default()).map(|_| ()),
        "unknown stratum",
    );
    new.stratum = Some(&strata);
    expect_err(
        survfit(
            &fitted,
            &new,
            &SurvfitOptions {
                variance: Variance::Robust,
                ..SurvfitOptions::default()
            },
        )
        .map(|_| ()),
        "robust without clusters",
    );
    // Expected needs a stop, and a start with entry times.
    let opts_e = PredictOptions::default();
    expect_err(
        predict(&fitted, PredictType::Expected, Some(&new), &opts_e).map(|_| ()),
        "no stop",
    );
    new.stop = Some(&t1);
    expect_err(
        predict(&fitted, PredictType::Expected, Some(&new), &opts_e).map(|_| ()),
        "no start with entry times",
    );
    new.start = Some(&t0);
    assert!(predict(&fitted, PredictType::Expected, Some(&new), &opts_e).is_ok());
    new.start = Some(&t1);
    expect_err(
        predict(&fitted, PredictType::Expected, Some(&new), &opts_e).map(|_| ()),
        "start not before stop",
    );
    // A strata reference needs the stratum of a new row.
    let mut bare = NewData::new(new_x.as_ref());
    bare.stratum = None;
    expect_err(
        predict(
            &fitted,
            PredictType::LinearPredictor,
            Some(&bare),
            &PredictOptions {
                reference: Some(Reference::Strata),
                ..PredictOptions::default()
            },
        )
        .map(|_| ()),
        "stratum of a new row",
    );
    expect_err(
        predict_terms(&fitted, &[&[0, 2]], None, &opts_e).map(|_| ()),
        "term column out of range",
    );
    let wide = Mat::from_fn(1, 3, |_, _| 0.0);
    let mut too_wide = NewData::new(wide.as_ref());
    too_wide.stratum = Some(&strata);
    expect_err(
        survfit(&fitted, &too_wide, &SurvfitOptions::default()).map(|_| ()),
        "columns",
    );
    // A fit without an estimate has nothing to predict from.
    let mut empty = model.clone();
    empty.coefficients = vec![None, None];
    expect_err(
        basehaz(
            &FittedModel {
                data: &data,
                x: x.as_ref(),
                fit: &empty,
                options: &opts,
            },
            &BaselineOptions::default(),
        )
        .map(|_| ()),
        "no estimate",
    );
    // Limits come only with a standard error.
    let none = survfit(
        &fitted,
        &new,
        &SurvfitOptions {
            se_fit: false,
            ..SurvfitOptions::default()
        },
    )
    .unwrap();
    assert!(none.curves[0].se.is_none() && none.curves[0].lower.is_none());
    let nolimits = survfit(
        &fitted,
        &new,
        &SurvfitOptions {
            conf_type: ConfType::None,
            ..SurvfitOptions::default()
        },
    )
    .unwrap();
    assert!(nolimits.curves[0].se.is_some() && nolimits.curves[0].upper.is_none());
}

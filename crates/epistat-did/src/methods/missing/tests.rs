use epistat_linalg::Mat;

use super::*;

/// Two treated units and two controls, by hand. Treated 0 has `Y0`, treated
/// 1 does not; control 2 has `Y0`, control 3 does not. The values are
/// chosen so that every intermediate is exact in binary floating point.
#[test]
fn a_hand_computed_example() {
    let treated = [true, true, false, false];
    let pre = [Some(1.0), None, Some(2.0), None];
    let post = [4.0, 5.0, 3.0, 2.5];
    let data = MissingData::builder(&treated, &pre, &post).build().unwrap();
    let p = MissingPredictions {
        propensity: &[0.6, 0.6, 0.5, 0.25],
        gamma_treated: &[0.5, 0.5, 0.5, 0.5],
        gamma_control: &[0.5, 0.5, 0.5, 0.5],
        pre_treated: &[1.5, 2.0, 0.0, 0.0],
        post_control: &[3.0, 3.5, 2.5, 2.0],
        pre_control: &[1.0, 1.25, 1.5, 1.75],
        nested_control: None,
    };
    let config = MissingOptions::default();
    let got = from_predictions(&data, &p, &config).unwrap();
    // Treated: Y1 - mu01 - R (Y0 - mu01) / g1 - mu10 + mu00.
    let t0 = 4.0 - 1.5 - (1.0 - 1.5) / 0.5 - 3.0 + 1.0;
    let t1 = 5.0 - 2.0 - 3.5 + 1.25;
    // Controls: -pi / (1 - pi) (Y1 - mu10 - R (Y0 - mu00) / g0).
    let c2 = -(0.5 / 0.5) * (3.0 - 2.5 - (2.0 - 1.5) / 0.5);
    let c3 = -(0.25 / 0.75) * (2.5 - 2.0);
    // c3 is the one inexact term: 1/3 times 0.5, then sums of four terms.
    let att = (t0 + t1 + c2 + c3) / 2.0;
    assert!((got.estimate.estimate - att).abs() <= 8.0 * f64::EPSILON);
    let phi = [2.0 * (t0 - att), 2.0 * (t1 - att), 2.0 * c2, 2.0 * c3];
    for (g, e) in got.influence.column(0).iter().zip(phi) {
        assert!((g - e).abs() <= 32.0 * f64::EPSILON);
    }
    assert_eq!(got.observed_pre_treated, 1);
    assert_eq!(got.observed_pre_control, 1);
    assert_eq!(got.propensity_range, (0.25, 0.5));
    assert_eq!(got.gamma_treated_range, Some((0.5, 0.5)));
    assert_eq!(got.gamma_control_range, Some((0.5, 0.5)));

    // Under MarXY1 the nested regression replaces mu00 in the treated term
    // and enters the controls' term with mu00.
    let eta = [0.5, 0.75, 1.0, 1.25];
    let q = MissingPredictions {
        nested_control: Some(&eta),
        ..p
    };
    let marxy1 = MissingOptions {
        assumption: MissingAssumption::MarXY1,
        ..config
    };
    let got = from_predictions(&data, &q, &marxy1).unwrap();
    let t0 = 4.0 - 1.5 - (1.0 - 1.5) / 0.5 - 3.0 + 0.5;
    let t1 = 5.0 - 2.0 - 3.5 + 0.75;
    let c2 = -(0.5 / 0.5) * (3.0 - 1.5 - (2.0 - 1.5) / 0.5 - 2.5 + 1.0);
    let c3 = -(0.25 / 0.75) * (2.5 - 1.75 - 2.0 + 1.25);
    let att = (t0 + t1 + c2 + c3) / 2.0;
    assert!((got.estimate.estimate - att).abs() <= 8.0 * f64::EPSILON);
}

/// With a binary covariate the logistic models are saturated, so the fitted
/// propensity is the share treated in each group and `gamma` the share with
/// `Y0` in each group and arm. Newton's method stops after a step of at most
/// `1e-10`, which leaves an error of the order of its square; what remains
/// is the rounding of the iterations, a few units in the last place of
/// probabilities of order 1, here allowed 64.
#[test]
fn saturated_logistic_fits_are_the_group_shares() {
    let n = 40;
    let group: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let treated: Vec<bool> = (0..n)
        .map(|i| (i % 5) < 2 + usize::from(group[i]))
        .collect();
    let pre: Vec<Option<f64>> = (0..n)
        .map(|i| ((i % 7) < 3 + 2 * usize::from(group[i])).then_some(i as f64 * 0.1))
        .collect();
    let post: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
    let x = Mat::from_fn(n, 1, |i, _| f64::from(u8::from(group[i])));
    let data = MissingData::builder(&treated, &pre, &post)
        .covariates(x.as_ref())
        .build()
        .unwrap();
    let got = estimate(&data, &MissingOptions::default()).unwrap();
    let share = |rows: &dyn Fn(usize) -> bool, hit: &dyn Fn(usize) -> bool| {
        let total = (0..n).filter(|&i| rows(i)).count() as f64;
        (0..n).filter(|&i| rows(i) && hit(i)).count() as f64 / total
    };
    for i in 0..n {
        let g = group[i];
        let pi = share(&|j| group[j] == g, &|j| treated[j]);
        let g1 = share(&|j| group[j] == g && treated[j], &|j| pre[j].is_some());
        let g0 = share(&|j| group[j] == g && !treated[j], &|j| pre[j].is_some());
        assert!((got.nuisance.propensity[i] - pi).abs() <= 64.0 * f64::EPSILON);
        assert!((got.nuisance.gamma_treated[i] - g1).abs() <= 64.0 * f64::EPSILON);
        assert!((got.nuisance.gamma_control[i] - g0).abs() <= 64.0 * f64::EPSILON);
    }
}

/// Every column of the data and of the predictions is checked where it
/// enters. An empty column is shorter than the treatment indicator, and the
/// units have no order.
#[test]
fn bad_columns_are_input_errors() {
    use epistat_testkit::{BadInput, assert_rejects_bad_input_with_codes};

    let treated = [true, true, false, false, true, false];
    let pre = [Some(1.0), None, Some(2.0), None, Some(0.5), Some(1.5)];
    let post = [4.0, 5.0, 3.0, 2.5, 4.5, 2.0];
    let half = [0.5; 6];
    let level = [1.0, 1.5, 1.25, 1.75, 0.75, 1.0];
    let p = MissingPredictions {
        propensity: &half,
        gamma_treated: &half,
        gamma_control: &half,
        pre_treated: &level,
        post_control: &level,
        pre_control: &level,
        nested_control: None,
    };
    let options = MissingOptions::default();
    let code = |bad: BadInput| match bad {
        BadInput::Empty => "input.length",
        other => other.code(),
    };
    let run = |post: &[f64], p: &MissingPredictions<'_>| {
        let data = MissingData::builder(&treated, &pre, post).build()?;
        from_predictions(&data, p, &options).map(|fit| fit.estimate.estimate)
    };
    assert!(run(&post, &p).is_ok());
    assert_rejects_bad_input_with_codes("missing outcome_post", code, |bad| {
        (bad != BadInput::Unsorted).then(|| run(&bad.column(&post), &p))
    });
    assert_rejects_bad_input_with_codes("missing predictions", code, |bad| {
        (bad != BadInput::Unsorted).then(|| {
            let column = bad.column(&level);
            run(
                &post,
                &MissingPredictions {
                    post_control: &column,
                    ..p
                },
            )
        })
    });
    assert_rejects_bad_input_with_codes("missing covariates", code, |bad| {
        (bad != BadInput::Unsorted).then(|| {
            let column = bad.column(&level);
            let x = Mat::from_fn(column.len(), 1, |i, _| column[i]);
            let data = MissingData::builder(&treated, &pre, &post)
                .covariates(x.as_ref())
                .build()?;
            estimate(&data, &options).map(|fit| fit.estimate.estimate)
        })
    });
    let invalid = MissingOptions {
        gamma_floor: 1.0,
        ..options.clone()
    };
    let data = MissingData::builder(&treated, &pre, &post).build().unwrap();
    epistat_testkit::assert_input_error_code("gamma_floor", "input.domain", || {
        from_predictions(&data, &p, &invalid).map(|fit| fit.estimate.estimate)
    });
}

/// The deprecated entry points give the bits of the new ones, with the
/// clusters given as a `Clusters` numbered by first appearance.
#[test]
#[expect(deprecated, reason = "the 0.1 names stay one release")]
fn the_deprecated_entry_points_give_the_same_bits() {
    let n = 60;
    let treated: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
    let pre: Vec<Option<f64>> = (0..n)
        .map(|i| (i % 4 != 1).then_some((i as f64 * 0.21).cos()))
        .collect();
    let post: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.37).sin() + f64::from(u8::from(treated[i])))
        .collect();
    let x = Mat::from_fn(n, 2, |i, j| ((i * (j + 2)) as f64 * 0.13).sin());
    // Codes in first-seen order, so that both numberings agree.
    let codes: Vec<u32> = (0..n).map(|i| u32::try_from(i / 3).unwrap()).collect();
    let clusters = Clusters::from_labels_first_seen(&codes).unwrap();
    let old = MissingPreData {
        treated: &treated,
        outcome_pre: &pre,
        outcome_post: &post,
        clusters: Some(&clusters),
    };
    let data = MissingData::builder(&treated, &pre, &post)
        .covariates(x.as_ref())
        .clusters(&codes)
        .build()
        .unwrap();
    let folds = MissingFolds::CrossFit { folds: 3, seed: 11 };
    let options = MissingOptions {
        assumption: MissingAssumption::MarXY1,
        models: NuisanceModels {
            nested: Some(vec![1]),
            ..NuisanceModels::default()
        },
        folds,
        ..MissingOptions::default()
    };
    let new = estimate(&data, &options).unwrap();
    let all = [0, 1];
    let deprecated = att_missing_pre(
        &old,
        x.as_ref(),
        MissingAssumption::MarXY1,
        &MissingModels {
            propensity: &all,
            missingness: &all,
            outcome: &all,
            nested: &[1],
        },
        folds,
        &MissingOptions::default(),
    )
    .unwrap();
    assert_eq!(new, deprecated);
    assert_eq!(
        cross_fit_folds(&data, 3, 11).unwrap(),
        missing_cross_fit_folds(n, Some(&clusters), 3, 11).unwrap()
    );
    let n_ = &new.nuisance;
    let predictions = MissingPredictions {
        propensity: &n_.propensity,
        gamma_treated: &n_.gamma_treated,
        gamma_control: &n_.gamma_control,
        pre_treated: &n_.pre_treated,
        post_control: &n_.post_control,
        pre_control: &n_.pre_control,
        nested_control: n_.nested_control.as_deref(),
    };
    let unclustered = MissingData::builder(&treated, &pre, &post).build().unwrap();
    let old_unclustered = MissingPreData {
        clusters: None,
        ..old
    };
    assert_eq!(
        from_predictions(&unclustered, &predictions, &options).unwrap(),
        att_missing_pre_from_predictions(
            &old_unclustered,
            MissingAssumption::MarXY1,
            &predictions,
            &MissingOptions::default(),
        )
        .unwrap()
    );
}

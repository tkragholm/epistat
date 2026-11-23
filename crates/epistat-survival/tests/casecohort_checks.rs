//! Refusals and edge cases of the case-cohort analysis.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::casecohort::{CaseCohortMethod, CaseCohortOptions, fit};
use epistat_survival::cox::Ties;

/// Ten rows: five cases (rows 0 to 4, of which 0 and 1 are in the subcohort)
/// and five subcohort non-cases.
struct Rows {
    stop: Vec<f64>,
    status: Vec<i32>,
    sub: Vec<bool>,
    x: Mat<f64>,
    stratum: Vec<u32>,
}

fn rows() -> Rows {
    let stop = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let status = vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
    let sub = vec![
        true, true, false, false, false, true, true, true, true, true,
    ];
    let a = [0.3, -1.0, 0.7, 1.2, -0.4, 0.1, 0.9, -0.8, 0.5, -0.2];
    let b = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    Rows {
        stop,
        status,
        sub,
        x: Mat::from_fn(10, 2, |i, j| if j == 0 { a[i] } else { b[i] }),
        stratum: vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    }
}

fn message(r: &Rows, method: CaseCohortMethod, sizes: &[f64], with_stratum: bool) -> String {
    let data = SurvData::builder(&r.stop, &r.status).build().unwrap();
    let options = CaseCohortOptions::new(method, sizes);
    let stratum = with_stratum.then_some(r.stratum.as_slice());
    fit(&data, r.x.as_ref(), &r.sub, stratum, &options)
        .unwrap_err()
        .to_string()
}

#[test]
fn refuses_a_censored_row_outside_the_subcohort() {
    let mut r = rows();
    r.sub[9] = false;
    let text = message(&r, CaseCohortMethod::SelfPrentice, &[100.0], false);
    assert!(text.contains("outside the subcohort"), "{text}");
}

#[test]
fn refuses_robust_outside_lin_ying_and_stratum_outside_borgan() {
    let r = rows();
    let data = SurvData::builder(&r.stop, &r.status).build().unwrap();
    let mut options = CaseCohortOptions::new(CaseCohortMethod::Prentice, &[100.0]);
    options.robust = true;
    assert!(fit(&data, r.x.as_ref(), &r.sub, None, &options).is_err());
    let text = message(&r, CaseCohortMethod::Prentice, &[100.0], true);
    assert!(text.contains("take no stratum"), "{text}");
    let text = message(&r, CaseCohortMethod::BorganI, &[50.0, 50.0], false);
    assert!(text.contains("require"), "{text}");
}

#[test]
fn refuses_wrong_cohort_sizes() {
    let r = rows();
    let text = message(&r, CaseCohortMethod::LinYing, &[9.0], false);
    assert!(text.contains("smaller than the number of rows"), "{text}");
    let text = message(&r, CaseCohortMethod::LinYing, &[100.0, 50.0], false);
    assert!(text.contains("one value"), "{text}");
    let text = message(&r, CaseCohortMethod::BorganI, &[50.0], true);
    assert!(text.contains("1 values for 2 strata"), "{text}");
    let text = message(&r, CaseCohortMethod::BorganI, &[50.0, 3.0], true);
    assert!(text.contains("cohort size"), "{text}");
}

#[test]
fn refuses_exact_ties_and_repeated_ids() {
    let r = rows();
    let data = SurvData::builder(&r.stop, &r.status).build().unwrap();
    let mut options = CaseCohortOptions::new(CaseCohortMethod::LinYing, &[100.0]);
    options.ties = Ties::Exact;
    assert!(fit(&data, r.x.as_ref(), &r.sub, None, &options).is_err());
    let ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9];
    let data = SurvData::builder(&r.stop, &r.status)
        .id(&ids)
        .build()
        .unwrap();
    let options = CaseCohortOptions::new(CaseCohortMethod::LinYing, &[100.0]);
    let text = fit(&data, r.x.as_ref(), &r.sub, None, &options)
        .unwrap_err()
        .to_string();
    assert!(text.contains("more than one row per id"), "{text}");
}

#[test]
fn prentice_needs_two_failure_times() {
    let mut r = rows();
    r.status = vec![1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
    r.stop = vec![5.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    r.sub = vec![true, true, false, true, true, true, true, true, true, true];
    let text = message(&r, CaseCohortMethod::Prentice, &[100.0], false);
    assert!(text.contains("two distinct failure times"), "{text}");
    // Self-Prentice does not use the shift and has an estimate.
    let data = SurvData::builder(&r.stop, &r.status).build().unwrap();
    let options = CaseCohortOptions::new(CaseCohortMethod::SelfPrentice, &[100.0]);
    let model = fit(&data, r.x.as_ref(), &r.sub, None, &options).unwrap();
    assert!(model.coefficients.iter().all(Option::is_some));
}

#[test]
fn a_fully_sampled_single_member_stratum_adds_nothing() {
    // Stratum 1 holds one row, its whole cohort sampled: weight 1, no
    // sampling variance from it. Stratum 0 holds the rest.
    let r = rows();
    let stratum: Vec<u32> = (0..10).map(|i| u32::from(i == 4)).collect();
    let mut sub = r.sub.clone();
    sub[4] = true;
    let data = SurvData::builder(&r.stop, &r.status).build().unwrap();
    let options = CaseCohortOptions::new(CaseCohortMethod::BorganI, &[40.0, 1.0]);
    let model = fit(&data, r.x.as_ref(), &sub, Some(&stratum), &options).unwrap();
    assert!(
        model
            .covariance
            .iter()
            .all(|v| v.is_some_and(f64::is_finite))
    );
    // The same row with a cohort size of 2 has one member and a weight of 2.
    let options = CaseCohortOptions::new(CaseCohortMethod::BorganI, &[40.0, 2.0]);
    let text = fit(&data, r.x.as_ref(), &sub, Some(&stratum), &options)
        .unwrap_err()
        .to_string();
    assert!(text.contains("one member"), "{text}");
}

#[test]
fn an_aliased_column_is_none() {
    let r = rows();
    let x = Mat::from_fn(10, 3, |i, j| {
        if j == 2 {
            r.x[(i, 0)] * 2.0
        } else {
            r.x[(i, j)]
        }
    });
    let data = SurvData::builder(&r.stop, &r.status).build().unwrap();
    let options = CaseCohortOptions::new(CaseCohortMethod::LinYing, &[100.0]);
    let model = fit(&data, x.as_ref(), &r.sub, None, &options).unwrap();
    assert!(model.coefficients[2].is_none());
    assert!(model.covariance[2].is_none() && model.covariance[8].is_none());
    assert!(model.covariance[0].is_some());
}

#[test]
fn counts_and_size() {
    let r = rows();
    let data = SurvData::builder(&r.stop, &r.status).build().unwrap();
    let options = CaseCohortOptions::new(CaseCohortMethod::LinYing, &[100.0]);
    let model = fit(&data, r.x.as_ref(), &r.sub, None, &options).unwrap();
    assert_eq!(
        (model.n_subcohort, model.n_cases, model.n_cases_outside),
        (7, 5, 3)
    );
    assert!((model.n_cohort - 100.0).abs() < f64::EPSILON);
}

#[test]
fn a_case_without_a_subcohort_member_at_risk_adds_nothing() {
    // The case at time 20 is outside the subcohort and every subcohort member
    // has left by then, so the subcohort gives it no risk set. Self-Prentice
    // has no term for it, which is the fit without that row.
    let r = rows();
    let mut stop = r.stop.clone();
    let mut status = r.status.clone();
    let mut sub = r.sub.clone();
    stop.push(20.0);
    status.push(1);
    sub.push(false);
    let x = Mat::from_fn(
        11,
        2,
        |i, j| if i < 10 { r.x[(i, j)] } else { [0.4, 1.0][j] },
    );
    let with = SurvData::builder(&stop, &status).build().unwrap();
    let without = SurvData::builder(&r.stop, &r.status).build().unwrap();
    let options = CaseCohortOptions::new(CaseCohortMethod::SelfPrentice, &[100.0]);
    let a = fit(&with, x.as_ref(), &sub, None, &options).unwrap();
    let b = fit(&without, r.x.as_ref(), &r.sub, None, &options).unwrap();
    for (u, v) in a.coefficients.iter().zip(&b.coefficients) {
        assert!((u.unwrap() - v.unwrap()).abs() < 1e-12);
    }
    for (u, v) in a.covariance.iter().zip(&b.covariance) {
        assert!((u.unwrap() - v.unwrap()).abs() < 1e-12 * v.unwrap().abs().max(1.0));
    }
}

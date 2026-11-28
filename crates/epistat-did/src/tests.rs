use super::*;
use epistat_core::{Bootstrap, InputError, Level, Seed};
use epistat_linalg::{Mat, MatRef};

use crate::error::InternalDidError as PortedDidError;
use crate::estimators::common::linalg::ridge_solve_lu;
use crate::estimators::common::testing::{
    horvitz_thompson, normalize_weights, stabilize, trim, xt_w_b, xt_w_x,
};
use crate::estimators::outcome::linear::LinearOutcome;
use crate::estimators::outcome::model::OutcomeModel;
use crate::estimators::propensity::common::logistic_scores;
use crate::estimators::propensity::irls::irls as irls_fit;
use crate::methods::standard::summarize_two_by_two;
use crate::types::DidCell;
use two_period::{Estimator, TwoPeriodOptions};

fn two_by_two(
    treated: &[bool],
    post: &[bool],
    y: &[f64],
    w: Option<&[f64]>,
) -> Result<epistat_core::Estimate, Error> {
    let builder = TwoPeriodData::repeated_cross_section(treated, post, y);
    let builder = match w {
        Some(w) => builder.weights(w),
        None => builder,
    };
    two_period::two_by_two(&builder.build()?, Level::default())
}

#[test]
fn estimates_att_for_simple_2x2_panel() {
    let treated = [true, true, true, true, false, false, false, false];
    let post = [false, false, true, true, false, false, true, true];
    let y = [10.0, 10.0, 18.0, 18.0, 7.0, 7.0, 12.0, 12.0];
    let result = two_by_two(&treated, &post, &y, None).expect("estimate");
    assert!((result.estimate - 3.0).abs() < 1e-12);
}

#[test]
fn validate_two_by_two_inputs() {
    let error = two_by_two(
        &[true, true, false],
        &[false, true, false],
        &[10.0, 11.0, 9.0],
        None,
    )
    .expect_err("must fail");
    assert_eq!(
        error,
        Error::EmptyCell {
            treated: false,
            post: true
        }
    );
    let treated = [true, true, false, false];
    let post = [false, true, false, true];
    let error = two_by_two(
        &treated,
        &post,
        &[10.0, 11.0, 9.0, 9.0],
        Some(&[0.0, 1.0, 1.0, 1.0]),
    )
    .expect_err("must fail");
    assert!(matches!(
        error,
        Error::Input(InputError::Domain {
            what: "weights",
            ..
        })
    ));
    let error =
        two_by_two(&treated, &post, &[10.0, f64::NAN, 9.0, 9.0], None).expect_err("must fail");
    assert_eq!(
        error,
        Error::Input(InputError::NotFinite {
            what: "outcome",
            row: 1,
            column: None
        })
    );
}

#[test]
fn summarizes_two_by_two_cells() {
    let rows = [
        (true, false, 10.0, 1.0),
        (true, true, 14.0, 1.0),
        (false, false, 8.0, 1.0),
        (false, true, 10.0, 1.0),
    ];
    let summary = summarize_two_by_two(rows.into_iter()).expect("summary");
    assert_eq!(summary.total_observations(), 4);
    assert_eq!(summary.cell(DidCell::TreatedPre).observations, 1);
    assert!((summary.cell(DidCell::TreatedPost).mean_outcome - 14.0).abs() < 1e-12);
    assert!((summary.cell(DidCell::ControlPre).mean_outcome - 8.0).abs() < 1e-12);
    assert!((summary.cell(DidCell::ControlPost).mean_outcome - 10.0).abs() < 1e-12);
}

fn dr_panel(
    treated: &[bool],
    change: &[f64],
    x: Option<&[f64]>,
    options: &TwoPeriodOptions,
) -> Result<two_period::TwoPeriod, Error> {
    let builder = TwoPeriodData::panel(treated, change);
    let builder = match x {
        Some(x) => builder.covariates(MatRef::from_column_major_slice(x, treated.len(), 1)),
        None => builder,
    };
    two_period::estimate(&builder.build()?, options)
}

fn doubly_robust() -> TwoPeriodOptions {
    TwoPeriodOptions {
        estimator: Estimator::DoublyRobust,
        ..TwoPeriodOptions::default()
    }
}

#[test]
fn drdid_estimates_att_with_covariates() {
    let treated = [true, true, true, false, false, false];
    let change = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0];
    let x = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let options = TwoPeriodOptions {
        bootstrap: Some(Bootstrap::new(199, Seed(42))),
        ..doubly_robust()
    };
    let fit = dr_panel(&treated, &change, Some(&x), &options).expect("estimate drdid");
    assert_eq!(fit.n_treated, 3);
    assert_eq!(fit.n_control, 3);
    assert!((fit.estimate.estimate - 3.0).abs() < 0.35);
    assert!(fit.estimate.se.is_some());
    assert!(fit.estimate.lower().unwrap() <= fit.estimate.upper().unwrap());
}

#[test]
fn drdid_rejects_missing_treated_or_control_groups() {
    let err = dr_panel(&[false, false], &[4.0, 5.0], None, &doubly_robust()).expect_err("fail");
    assert_eq!(err, Error::NoTreated);
    let err = dr_panel(&[true, true], &[7.0, 8.0], None, &doubly_robust()).expect_err("fail");
    assert_eq!(err, Error::NoComparison);
}

#[test]
fn drdid_rejects_invalid_options_and_values() {
    let options = TwoPeriodOptions {
        nuisance: two_period::NuisanceOptions {
            propensity_clip: 0.5,
            ..two_period::NuisanceOptions::default()
        },
        ..doubly_robust()
    };
    let err = dr_panel(&[true, false], &[9.0, 6.0], None, &options).expect_err("fail");
    assert!(matches!(
        err,
        Error::Input(InputError::Domain {
            what: "propensity_clip",
            ..
        })
    ));
    let err = TwoPeriodData::panel(&[true, false], &[9.0, 6.0])
        .weights(&[1.0, 0.0])
        .build()
        .expect_err("fail");
    assert!(matches!(
        err,
        Error::Input(InputError::Domain {
            what: "weights",
            ..
        })
    ));
    let err = dr_panel(
        &[true, false],
        &[f64::INFINITY, 6.0],
        None,
        &doubly_robust(),
    )
    .expect_err("fail");
    assert_eq!(
        err,
        Error::Input(InputError::NotFinite {
            what: "change",
            row: 0,
            column: None
        })
    );
    let err = dr_panel(
        &[true, false],
        &[9.0, 6.0],
        Some(&[f64::NAN, 0.0]),
        &doubly_robust(),
    )
    .expect_err("fail");
    assert_eq!(
        err,
        Error::Input(InputError::NotFinite {
            what: "covariates",
            row: 0,
            column: None
        })
    );
}

fn make_logit_design() -> Mat<f64> {
    let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
    Mat::from_fn(x.len(), 2, |row, col| if col == 0 { 1.0 } else { x[row] })
}

fn make_logit_target() -> Mat<f64> {
    let d = [0.0, 0.0, 0.0, 1.0, 1.0];
    Mat::from_fn(d.len(), 1, |row, _| d[row])
}

#[test]
fn irls_converges_for_simple_logistic_problem() {
    let design = make_logit_design();
    let target = make_logit_target();

    let beta = irls_fit(design.as_ref(), target.as_ref(), 200, 1e-8).expect("irls should converge");
    assert_eq!(beta.nrows(), 2);
    assert_eq!(beta.ncols(), 1);
    assert!(beta.col_as_slice(0).iter().all(|v| v.is_finite()));

    let ps = logistic_scores(design.as_ref(), beta.as_ref());
    assert_eq!(ps.len(), design.nrows());
    assert!(ps.windows(2).all(|w| w[0] <= w[1]));
}

#[test]
fn irls_reports_non_convergence_when_no_iterations_allowed() {
    let design = make_logit_design();
    let target = make_logit_target();
    let err = irls_fit(design.as_ref(), target.as_ref(), 0, 1e-8).expect_err("must fail");
    assert_eq!(
        err,
        PortedDidError::Convergence("IRLS did not converge".to_string())
    );
}

#[test]
fn ported_error_display_is_stable() {
    let estimation = PortedDidError::Estimation("bad matrix".to_string());
    let convergence = PortedDidError::Convergence("max iter".to_string());
    assert_eq!(format!("{estimation}"), "Estimation failed: bad matrix");
    assert_eq!(format!("{convergence}"), "Convergence error: max iter");
}

#[test]
fn weights_trim_stabilize_and_normalize_cover_edge_cases() {
    assert_eq!(trim(&[0.0, 0.5, 1.0], 0.1), vec![0.1, 0.5, 0.9]);

    assert_eq!(stabilize(&[]).len(), 0);
    assert_eq!(stabilize(&[1.0, -1.0]), vec![1.0, -1.0]);
    let stabilized = stabilize(&[1.0, 3.0]);
    assert!((stabilized[0] - 0.5).abs() < 1e-12);
    assert!((stabilized[1] - 1.5).abs() < 1e-12);

    let empty = Mat::<f64>::zeros(0, 1);
    assert_eq!(normalize_weights(&empty).len(), 0);

    let zero_sum = Mat::from_fn(2, 1, |row, _| if row == 0 { 1.0 } else { -1.0 });
    assert_eq!(normalize_weights(&zero_sum), vec![1.0, 1.0]);

    let base = Mat::from_fn(2, 1, |row, _| if row == 0 { 2.0 } else { 6.0 });
    let norm = normalize_weights(&base);
    assert!((norm[0] - 0.5).abs() < 1e-12);
    assert!((norm[1] - 1.5).abs() < 1e-12);
}

#[test]
fn horvitz_thompson_handles_regular_and_invalid_inputs() {
    let ht = horvitz_thompson(&[2.0, 4.0], &[1.0, 0.0], &[0.5, 0.5]).expect("ht");
    assert!((ht + 2.0).abs() < 1e-12);

    let clamped = horvitz_thompson(&[1.0, 1.0], &[1.0, 0.0], &[0.0, 1.0]).expect("clamped");
    assert!(clamped.is_finite());

    assert!(horvitz_thompson(&[], &[], &[]).is_err());
    assert!(horvitz_thompson(&[1.0], &[1.0, 0.0], &[0.5]).is_err());
    assert!(horvitz_thompson(&[1.0], &[1.0], &[f64::NAN]).is_err());
}

#[test]
fn linear_outcome_fit_and_predict_cover_weighted_and_unweighted_paths() {
    let design = Mat::from_fn(
        3,
        2,
        |row, col| {
            if col == 0 { 1.0 } else { [0.0, 1.0, 2.0][row] }
        },
    );
    let target = Mat::from_fn(3, 1, |row, _| [1.0, 3.0, 5.0][row]);
    let model = LinearOutcome::default();

    let beta_unweighted = model
        .fit(design.as_ref(), target.as_ref(), None)
        .expect("fit");
    assert!((beta_unweighted.get(0, 0) - 1.0).abs() < 1e-6);
    assert!((beta_unweighted.get(1, 0) - 2.0).abs() < 1e-6);

    let beta_weighted = model
        .fit(design.as_ref(), target.as_ref(), Some(&[1.0, 2.0, 1.0]))
        .expect("fit");
    assert!((beta_weighted.get(0, 0) - 1.0).abs() < 1e-6);
    assert!((beta_weighted.get(1, 0) - 2.0).abs() < 1e-6);

    let preds = model.predict(design.as_ref(), beta_unweighted.as_ref());
    assert_eq!(preds.len(), 3);
    assert!((preds[0] - 1.0).abs() < 1e-6);
    assert!((preds[1] - 3.0).abs() < 1e-6);
    assert!((preds[2] - 5.0).abs() < 1e-6);
}

#[test]
fn linalg_helpers_cover_matrix_products_and_ridge_solver() {
    let x_t_w = Mat::from_fn(2, 2, |row, col| match (row, col) {
        (0, 0) => 1.0,
        (0, 1) => 2.0,
        (1, 0) => 3.0,
        (1, 1) => 4.0,
        _ => 0.0,
    });
    let x = Mat::from_fn(2, 2, |row, col| match (row, col) {
        (0, 0) => 5.0,
        (0, 1) => 6.0,
        (1, 0) => 7.0,
        (1, 1) => 8.0,
        _ => 0.0,
    });
    let b = Mat::from_fn(2, 1, |row, _| if row == 0 { 4.0 } else { 8.0 });

    let prod_x = xt_w_x(&x_t_w, &x);
    assert!((prod_x.get(0, 0) - 19.0).abs() < 1e-12);
    assert!((prod_x.get(0, 1) - 22.0).abs() < 1e-12);
    assert!((prod_x.get(1, 0) - 43.0).abs() < 1e-12);
    assert!((prod_x.get(1, 1) - 50.0).abs() < 1e-12);

    let prod_b = xt_w_b(&x_t_w, &b);
    assert!((prod_b.get(0, 0) - 20.0).abs() < 1e-12);
    assert!((prod_b.get(1, 0) - 44.0).abs() < 1e-12);

    let a = Mat::from_fn(2, 2, |row, col| match (row, col) {
        (0, 0) => 2.0,
        (1, 1) => 4.0,
        _ => 0.0,
    });
    let sol = ridge_solve_lu(a, &b, 2.0).expect("finite solution");
    let singular = Mat::from_fn(2, 2, |_, _| 0.0);
    assert!(ridge_solve_lu(singular, &b, 0.0).is_err());
    assert!((sol.get(0, 0) - 1.0).abs() < 1e-12);
    assert!((sol.get(1, 0) - (8.0 / 6.0)).abs() < 1e-12);
}

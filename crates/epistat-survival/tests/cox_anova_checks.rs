//! What `cox::anova` accepts and refuses, and the cases R has no counterpart for.

use epistat_survival::SurvData;
use epistat_survival::cox::anova::{anova, anova_terms};
use epistat_survival::cox::{CoxModelFit, CoxOptions, fit};
use faer::Mat;

fn data() -> (Vec<f64>, Vec<i32>, Mat<f64>) {
    let stop: Vec<f64> = (0..40).map(|i| 1.0 + f64::from(i) * 0.37).collect();
    let status: Vec<i32> = (0..40).map(|i| i32::from(i % 3 != 0)).collect();
    let x = Mat::from_fn(40, 3, |i, j| match j {
        0 => ((i * 7) % 11) as f64,
        1 => ((i * 5) % 4) as f64,
        _ => ((i * 3) % 7) as f64,
    });
    (stop, status, x)
}

fn prefix(stop: &[f64], status: &[i32], x: &Mat<f64>, k: usize) -> CoxModelFit {
    let data = SurvData::builder(stop, status).build().unwrap();
    let sub = Mat::from_fn(x.nrows(), k, |i, j| x[(i, j)]);
    fit(&data, sub.as_ref(), &CoxOptions::default()).unwrap()
}

#[test]
fn refuses_what_it_can_see_is_wrong() {
    let (stop, status, x) = data();
    let (f1, f2) = (prefix(&stop, &status, &x, 1), prefix(&stop, &status, &x, 2));
    assert!(anova(&[&f1]).is_err(), "one fit");
    // The larger model first: the ranks fall.
    assert!(anova(&[&f2, &f1]).is_err());
    // The same model twice: no degrees of freedom.
    assert!(anova(&[&f1, &f1]).is_err());
    // Another number of rows.
    let (stop2, status2) = (&stop[..30], &status[..30]);
    let x2 = Mat::from_fn(30, 2, |i, j| x[(i, j)]);
    let other = prefix(stop2, status2, &x2, 2);
    assert!(anova(&[&f1, &other]).is_err());
    // A fit that did not converge.
    let mut unconverged = f2.clone();
    unconverged.converged = false;
    assert!(anova(&[&f1, &unconverged]).is_err());
    // A larger model with a lower log likelihood is not nested.
    let mut lower = f2.clone();
    lower.log_partial_likelihood = f1.log_partial_likelihood.map(|l| l - 1.0);
    assert!(anova(&[&f1, &lower]).is_err());
    // A fit without a log likelihood.
    let mut none = f2.clone();
    none.log_partial_likelihood = None;
    assert!(anova(&[&f1, &none]).is_err());
    // A valid pair is accepted and tests twice the difference.
    let table = anova(&[&f1, &f2]).unwrap();
    let want = 2.0 * (f2.log_partial_likelihood.unwrap() - f1.log_partial_likelihood.unwrap());
    assert!((table.rows[1].chisq.unwrap() - want).abs() <= 1e-12 * want.abs().max(1.0));
    assert_eq!(table.rows[1].df, Some(1));
}

#[test]
fn terms_refuses_and_handles_aliased_groups() {
    let (stop, status, x) = data();
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let opts = CoxOptions::default();
    let g0: &[usize] = &[0];
    let g1: &[usize] = &[1];
    assert!(anova_terms(&data, x.as_ref(), &[], &opts).is_err());
    assert!(anova_terms(&data, x.as_ref(), &[g0, &[]], &opts).is_err());
    assert!(anova_terms(&data, x.as_ref(), &[g0, &[3]], &opts).is_err());
    assert!(anova_terms(&data, x.as_ref(), &[g0, &[0, 1]], &opts).is_err());
    let init = [0.0];
    let with_init = CoxOptions {
        init: Some(&init),
        ..CoxOptions::default()
    };
    assert!(anova_terms(&data, x.as_ref(), &[g0], &with_init).is_err());

    // A column that repeats an earlier one adds no degrees of freedom and has
    // no p-value.
    let xx = Mat::from_fn(40, 3, |i, j| if j == 2 { x[(i, 0)] } else { x[(i, j)] });
    let table = anova_terms(&data, xx.as_ref(), &[g0, g1, &[2]], &opts).unwrap();
    assert_eq!(table.rows.len(), 4);
    let last = table.rows[3];
    assert_eq!(last.df, Some(0));
    assert_eq!(last.p_value, None);
    assert_eq!(last.chisq, Some(0.0));
    assert_eq!(table.rows[0].df, None);
}

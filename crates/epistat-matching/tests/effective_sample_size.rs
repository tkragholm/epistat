//! Effective sample size: the value does not depend on the scale of the weights,
//! and a weight that is negative or not finite is an error.

use epistat_core::{Classify, ErrorKind};
use epistat_matching::{
    BalanceRecord, BalanceReportOptions, BalanceWeights, CovariateValue, Estimand, balance_summary,
};
use epistat_stats::weighted::effective_sample_size;
use std::collections::HashMap;

fn day() -> chrono::NaiveDate {
    chrono::NaiveDate::from_ymd_opt(2000, 1, 1).unwrap()
}

fn rec(id: &str, x: f64) -> BalanceRecord {
    let mut r = BalanceRecord::new(id, day());
    r.covariates.insert("x".into(), CovariateValue::Numeric(x));
    r
}

/// Equal weights of any size give ESS = n, from 1e-300 to 1e300 where the
/// squares underflow or overflow.
#[test]
fn ess_of_equal_weights_is_n_at_any_scale() {
    for scale in [1e-300, 1e-160, 1e-9, 1.0, 1e9, 1e160, 1e300] {
        let ess = effective_sample_size(&[scale; 100]).unwrap().unwrap();
        assert!((ess - 100.0).abs() < 1e-9, "scale {scale}: {ess}");
    }
}

/// A weight of 0 is a unit outside the sample, and with no positive weight there
/// is no size.
#[test]
fn ess_is_none_without_a_positive_weight() {
    assert_eq!(effective_sample_size(&[]), Ok(None));
    assert_eq!(effective_sample_size(&[0.0, 0.0]), Ok(None));
    let ess = effective_sample_size(&[0.0, 2.0, 2.0]).unwrap().unwrap();
    assert!((ess - 2.0).abs() < 1e-12);
}

#[test]
fn ess_refuses_a_negative_or_non_finite_weight() {
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        let err = effective_sample_size(&[1.0, bad]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Input);
    }
}

/// The sample size table does not depend on the scale of the weights.
#[test]
fn sample_size_table_ess_is_scale_free() {
    let cases: Vec<_> = (0..4)
        .map(|i| rec(&format!("c{i}"), f64::from(i)))
        .collect();
    let controls: Vec<_> = (0..4)
        .map(|i| rec(&format!("k{i}"), f64::from(i) + 0.5))
        .collect();
    let sampling: HashMap<String, f64> = cases
        .iter()
        .chain(&controls)
        .map(|r| (r.core.id.clone(), 1e-9))
        .collect();
    let w =
        BalanceWeights::unmatched(&cases, &controls, Estimand::Att).with_sampling_weights(sampling);
    let s = balance_summary(&cases, &controls, &w, &BalanceReportOptions::default()).unwrap();
    assert_eq!(s.sample_sizes.cases.all_ess.map(f64::round), Some(4.0));
}

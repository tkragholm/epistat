//! `standardise::attributable::af_from_standardised` on constructed
//! standardised risks: the arithmetic, the scales of the limits and the results
//! that are not numbers.

use epistat_glm::attributable::IntervalScale;
use epistat_survival::standardise::Standardised;
use epistat_survival::standardise::attributable::af_from_standardised;

/// Two levels at three times: the unexposed risks, then the observed.
fn risks(cov: bool) -> Standardised {
    let risk = vec![
        Some(0.02),
        Some(0.04),
        None,
        Some(0.05),
        Some(0.10),
        Some(0.20),
    ];
    // Per time the 2 x 2 covariance, unexposed first.
    let block = |a: f64, b: f64, c: f64| vec![Some(a), Some(c), Some(c), Some(b)];
    let covariance = cov.then(|| {
        let mut v = block(1e-5, 4e-5, 1.5e-5);
        v.extend(block(2e-5, 8e-5, 3e-5));
        v.extend(vec![None; 4]);
        v
    });
    Standardised {
        n_levels: 2,
        n_times: 3,
        n_rows: 100,
        n_used: 100,
        risk,
        std_err: None,
        lower: None,
        upper: None,
        covariance,
        conf_level: 0.95,
    }
}

#[test]
fn the_fraction_is_one_minus_the_ratio_with_the_delta_method_variance() {
    let s = risks(true);
    let times = [1.0, 2.0, 3.0];
    let c = af_from_standardised(&s, &times, 0, 1, 0.95, IntervalScale::Identity).unwrap();
    assert!((c.af[0].unwrap() - 0.6).abs() < 1e-15);
    assert!((c.af[1].unwrap() - 0.6).abs() < 1e-15);
    assert!(c.af[2].is_none() && c.se[2].is_none() && c.lower[2].is_none());
    // Var(F0/F) = v0/F^2 - 2 F0 c / F^3 + F0^2 vF / F^4 at the first time.
    let (f0, f1) = (0.02_f64, 0.05_f64);
    let v = 1e-5 / (f1 * f1) - 2.0 * f0 * 1.5e-5 / (f1 * f1 * f1) + f0 * f0 * 4e-5 / f1.powi(4);
    assert!((c.se[0].unwrap().powi(2) - v).abs() < 1e-12);
    let z = 1.959_963_984_540_054;
    assert!((c.lower[0].unwrap() - (0.6 - z * v.sqrt())).abs() < 1e-9);
    assert!((c.upper[0].unwrap() - (0.6 + z * v.sqrt())).abs() < 1e-9);
    assert_eq!(c.times, times);
    assert!((c.ratio[0].unwrap() - 0.4).abs() < 1e-15);
}

#[test]
fn the_log_scale_keeps_the_upper_limit_below_one() {
    let s = risks(true);
    let c = af_from_standardised(
        &s,
        &[1.0, 2.0, 3.0],
        0,
        1,
        0.999_999,
        IntervalScale::LogComplement,
    )
    .unwrap();
    for k in 0..2 {
        assert!(c.upper[k].unwrap() < 1.0);
        assert!(c.lower[k].unwrap() < c.af[k].unwrap());
    }
    let r = c.ratio[0].unwrap();
    let half = 4.891_638_475_698_485 * c.se[0].unwrap() / r;
    assert!((c.lower[0].unwrap() - (1.0 - r * half.exp())).abs() < 1e-6);
}

#[test]
fn no_covariance_gives_no_standard_error_and_zero_unexposed_risk_gives_no_log_limits() {
    let s = risks(false);
    let c =
        af_from_standardised(&s, &[1.0, 2.0, 3.0], 0, 1, 0.95, IntervalScale::Identity).unwrap();
    assert!(c.af[0].is_some() && c.se[0].is_none() && c.lower[0].is_none());
    let mut s = risks(true);
    s.risk[0] = Some(0.0);
    let c = af_from_standardised(
        &s,
        &[1.0, 2.0, 3.0],
        0,
        1,
        0.95,
        IntervalScale::LogComplement,
    )
    .unwrap();
    assert!((c.af[0].unwrap() - 1.0).abs() < 1e-15);
    assert!(c.se[0].is_some() && c.lower[0].is_none() && c.upper[0].is_none());
    // Zero observed risk: no ratio.
    let mut s = risks(true);
    s.risk[3] = Some(0.0);
    let c =
        af_from_standardised(&s, &[1.0, 2.0, 3.0], 0, 1, 0.95, IntervalScale::Identity).unwrap();
    assert!(c.af[0].is_none());
}

#[test]
fn a_protective_exposure_has_a_negative_fraction() {
    let mut s = risks(true);
    s.risk.swap(0, 3);
    let c =
        af_from_standardised(&s, &[1.0, 2.0, 3.0], 0, 1, 0.95, IntervalScale::Identity).unwrap();
    assert!(c.af[0].unwrap() < 0.0);
}

#[test]
fn unusable_input_is_refused() {
    let s = risks(true);
    let t = [1.0, 2.0, 3.0];
    let id = IntervalScale::Identity;
    assert!(af_from_standardised(&s, &t[..2], 0, 1, 0.95, id).is_err());
    assert!(af_from_standardised(&s, &t, 0, 0, 0.95, id).is_err());
    assert!(af_from_standardised(&s, &t, 0, 2, 0.95, id).is_err());
    for level in [0.0, 1.0, -0.1] {
        assert!(af_from_standardised(&s, &t, 0, 1, level, id).is_err());
    }
}

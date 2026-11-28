//! Properties and refusals of `rmst::rmst2` that no R run is needed for.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the weights are whole numbers"
)]

use epistat_survival::SurvData;
use epistat_survival::rmst::{RmstOptions, Tau, rmst2};

fn two_arms() -> (Vec<f64>, Vec<i32>, Vec<u32>) {
    let stop = vec![1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 3.0, 6.0, 7.0];
    let status = vec![1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1];
    let arm = vec![4, 4, 4, 4, 4, 4, 9, 9, 9, 9, 9, 9];
    (stop, status, arm)
}

#[test]
fn a_horizon_of_zero_leaves_every_ratio_undefined() {
    let (stop, status, arm) = two_arms();
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let r = rmst2(&data, &arm, &RmstOptions::new(Tau::Fixed(0.0))).unwrap();
    assert_eq!(r.group0.rmst.to_bits(), 0.0f64.to_bits());
    assert_eq!(r.difference.estimate, Some(0.0));
    // 0 / 0: no statistic and so no p-value.
    assert_eq!(r.difference.p_value, None);
    for c in [r.ratio, r.loss_ratio] {
        assert_eq!(c.estimate, None);
        assert_eq!(c.lower, None);
        assert_eq!(c.p_value, None);
    }
}

#[test]
fn whole_number_weights_act_as_counts() {
    let (stop, status, arm) = two_arms();
    let weight: Vec<f64> = (0..stop.len()).map(|i| 1.0 + (i % 3) as f64).collect();
    let weighted = SurvData::builder(&stop, &status)
        .weight(&weight)
        .build()
        .unwrap();
    let (mut s2, mut st2, mut a2) = (vec![], vec![], vec![]);
    for i in 0..stop.len() {
        for _ in 0..weight[i] as usize {
            s2.push(stop[i]);
            st2.push(status[i]);
            a2.push(arm[i]);
        }
    }
    let repeated = SurvData::builder(&s2, &st2).build().unwrap();
    let opts = RmstOptions::new(Tau::Fixed(3.5));
    let (a, b) = (
        rmst2(&weighted, &arm, &opts).unwrap(),
        rmst2(&repeated, &a2, &opts).unwrap(),
    );
    for (x, y) in [
        (a.group0.rmst, b.group0.rmst),
        (a.group1.rmst, b.group1.rmst),
        (a.group0.se, b.group0.se),
        (a.group1.se, b.group1.se),
        (
            a.difference.estimate.unwrap(),
            b.difference.estimate.unwrap(),
        ),
    ] {
        assert!((x - y).abs() <= 1e-13 * y.abs().max(1.0), "{x} against {y}");
    }
}

#[test]
fn the_rows_of_weight_zero_are_not_used() {
    let (stop, status, arm) = two_arms();
    let mut weight = vec![1.0; stop.len()];
    weight[5] = 0.0;
    let data = SurvData::builder(&stop, &status)
        .weight(&weight)
        .build()
        .unwrap();
    let r = rmst2(&data, &arm, &RmstOptions::new(Tau::Fixed(3.0))).unwrap();
    assert_eq!((r.n_rows, r.n_used), (12, 11));
    assert_eq!(r.group0.n, 5);
}

#[test]
fn refusals() {
    let (stop, status, arm) = two_arms();
    let ok = SurvData::builder(&stop, &status).build().unwrap();
    // Both arms end in an event, at 5 and at 7, so both curves are determined
    // beyond their last time and the horizon is limited by the later of them.
    let r = rmst2(&ok, &arm, &RmstOptions::new(Tau::Largest)).unwrap();
    assert_eq!(r.tau.to_bits(), 7.0f64.to_bits());
    assert!(r.group0.ends_in_event && r.group1.ends_in_event);
    assert!(rmst2(&ok, &arm, &RmstOptions::new(Tau::Fixed(7.5))).is_err());
    assert!(rmst2(&ok, &arm, &RmstOptions::new(Tau::Fixed(f64::NAN))).is_err());
    assert!(rmst2(&ok, &arm, &RmstOptions::new(Tau::Fixed(-1.0))).is_err());
    // One group, three groups, a group with only rows of weight 0, and two
    // event codes.
    let one = SurvData::builder(&stop, &status).build().unwrap();
    assert!(rmst2(&one, &[0u32; 12], &RmstOptions::new(Tau::Largest)).is_err());
    let three: Vec<u32> = (0..12).map(|i| i % 3).collect();
    let many = SurvData::builder(&stop, &status).build().unwrap();
    assert!(rmst2(&many, &three, &RmstOptions::new(Tau::Largest)).is_err());
    let weight: Vec<f64> = arm
        .iter()
        .map(|&a| if a == 9 { 0.0 } else { 1.0 })
        .collect();
    let empty = SurvData::builder(&stop, &status)
        .weight(&weight)
        .build()
        .unwrap();
    assert!(rmst2(&empty, &arm, &RmstOptions::new(Tau::Largest)).is_err());
    let causes = vec![1, 2, 0, 1, 2, 1, 1, 0, 2, 1, 0, 1];
    let competing = SurvData::builder(&stop, &causes)
        .n_codes(2)
        .build()
        .unwrap();
    assert!(rmst2(&competing, &arm, &RmstOptions::new(Tau::Largest)).is_err());
}

/// A curve that ends in a censoring is determined only to its last time.
#[test]
fn a_curve_ending_in_censoring_limits_the_horizon() {
    let stop = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 5.0, 6.0];
    let status = vec![1, 1, 1, 0, 1, 0, 1, 1];
    let arm = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let r = rmst2(&data, &arm, &RmstOptions::new(Tau::Largest)).unwrap();
    // Arm 0 is censored at 4, arm 1 ends in an event at 6.
    assert_eq!(r.tau.to_bits(), 4.0f64.to_bits());
    assert!(!r.group0.ends_in_event && r.group1.ends_in_event);
    assert!(rmst2(&data, &arm, &RmstOptions::new(Tau::Fixed(4.5))).is_err());
}

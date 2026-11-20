//! Edge cases of `survdiff`: a hand-worked example, undefined results, rows of
//! weight 0, group codes that are not dense, and refused input.

use epistat_survival::SurvData;
use epistat_survival::survdiff::{Options, one_sample, test};

const U: f64 = f64::EPSILON;

#[test]
fn a_worked_example_is_exact_to_rounding() {
    // Events at 1, 2 and 3. Group A at 1 and 3, group B at 2.
    // Time 1: n = 3, A 2, B 1: E_A += 2/3, V_AA += 2/9.
    // Time 2: n = 2, A 1, B 1: E_A += 1/2, V_AA += 1/4.
    // Time 3: n = 1, A 1: E_A += 1, no variance.
    // O_A = 2, E_A = 13/6, V_AA = 17/36, statistic (1/6)^2 / (17/36) = 1/17.
    let time = [1.0, 2.0, 3.0];
    let status = [1, 1, 1];
    let group = [10, 20, 10];
    let data = SurvData::builder(&time, &status).build().unwrap();
    let got = test(&data, &group, &Options::default()).unwrap();
    assert_eq!(got.groups, vec![10, 20]);
    assert_eq!(got.n, vec![2, 1]);
    let close = |a: f64, b: f64| assert!((a - b).abs() <= 8.0 * U * b.abs(), "{a} vs {b}");
    close(got.observed[0], 2.0);
    close(got.expected[0], 13.0 / 6.0);
    close(got.expected[1], 5.0 / 6.0);
    close(got.variance[0], 17.0 / 36.0);
    close(got.variance[3], 17.0 / 36.0);
    close(got.variance[1], -17.0 / 36.0);
    close(got.statistic.unwrap(), 1.0 / 17.0);
    assert_eq!(got.df, Some(1));
    assert!(got.strata.is_none());
    assert_eq!(got.n_strata, 1);
    assert_eq!(got.n_event_times, 3);
}

#[test]
fn one_group_or_no_event_leaves_the_statistic_missing() {
    let time = [1.0, 2.0, 3.0];
    let data = SurvData::builder(&time, &[1, 0, 1]).build().unwrap();
    let one = test(&data, &[4, 4, 4], &Options::default()).unwrap();
    assert_eq!(one.groups, vec![4]);
    assert!(one.statistic.is_none() && one.df.is_none() && one.p_value.is_none());
    assert_eq!(one.observed, vec![2.0]);
    let data = SurvData::builder(&time, &[0, 0, 0]).build().unwrap();
    let none = test(&data, &[1, 2, 1], &Options::default()).unwrap();
    assert!(none.statistic.is_none() && none.p_value.is_none());
    assert_eq!(none.observed, vec![0.0, 0.0]);
    assert_eq!(none.expected, vec![0.0, 0.0]);
}

#[test]
fn rows_of_weight_zero_are_left_out_and_labels_follow_the_codes() {
    let time = [1.0, 2.0, 2.5, 3.0, 4.0, 5.0];
    let status = [1, 1, 1, 0, 1, 1];
    let group = [30, 5, 5, 30, 99, 30];
    let stratum = [7, 7, 2, 2, 7, 2];
    let weight = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
    let full = SurvData::builder(&time, &status)
        .weight(&weight)
        .stratum(&stratum)
        .build()
        .unwrap();
    let got = test(&full, &group, &Options::default()).unwrap();
    // Row 2 (group 5, stratum 2) and nothing else is left out.
    let keep = [0, 1, 3, 4, 5];
    let pick = |v: &[f64]| keep.iter().map(|&i| v[i]).collect::<Vec<_>>();
    let t = pick(&time);
    let s: Vec<i32> = keep.iter().map(|&i| status[i]).collect();
    let g: Vec<u32> = keep.iter().map(|&i| group[i]).collect();
    let st: Vec<u32> = keep.iter().map(|&i| stratum[i]).collect();
    let reduced = SurvData::builder(&t, &s).stratum(&st).build().unwrap();
    let want = test(&reduced, &g, &Options::default()).unwrap();
    assert_eq!(got.groups, vec![5, 30, 99]);
    assert_eq!(got.strata, Some(vec![2, 7]));
    assert_eq!(got.n_rows, 6);
    assert_eq!(got.n_used, 5);
    assert_eq!(got.observed, want.observed);
    assert_eq!(got.expected, want.expected);
    assert_eq!(got.variance, want.variance);
    assert_eq!(got.statistic, want.statistic);
}

#[test]
fn refused_input() {
    let time = [1.0, 2.0, 3.0];
    let status = [1, 1, 0];
    let group = [1, 2, 1];
    let opts = Options::default();
    let ok = SurvData::builder(&time, &status).build().unwrap();
    assert!(test(&ok, &group[..2], &opts).is_err());
    let nan = Options {
        rho: f64::NAN,
        ..opts
    };
    assert!(test(&ok, &group, &nan).is_err());
    let start = [0.0, 0.0, 0.0];
    let counting = SurvData::builder(&time, &status)
        .start(&start)
        .build()
        .unwrap();
    assert!(test(&counting, &group, &opts).is_err());
    let competing = SurvData::builder(&time, &[1, 2, 0])
        .n_codes(2)
        .build()
        .unwrap();
    assert!(test(&competing, &group, &opts).is_err());
    let w = [1.0, 2.0, 1.0];
    let weighted = SurvData::builder(&time, &status)
        .weight(&w)
        .build()
        .unwrap();
    assert!(test(&weighted, &group, &opts).is_err());
    let w0 = [0.0, 0.0, 0.0];
    let empty = SurvData::builder(&time, &status)
        .weight(&w0)
        .build()
        .unwrap();
    assert!(test(&empty, &group, &opts).is_err());
    assert!(one_sample(&ok, &[0.5, 0.5], &opts).is_err());
    assert!(one_sample(&ok, &[0.5, 0.0, 0.5], &opts).is_err());
    assert!(one_sample(&ok, &[0.5, 1.5, 0.5], &opts).is_err());
}

#[test]
fn one_sample_with_survival_one_has_no_statistic() {
    let time = [1.0, 2.0];
    let data = SurvData::builder(&time, &[0, 0]).build().unwrap();
    let got = one_sample(&data, &[1.0, 1.0], &Options::default()).unwrap();
    assert!(got.expected.abs() < f64::MIN_POSITIVE);
    assert!(got.statistic.is_none() && got.p_value.is_none());
}

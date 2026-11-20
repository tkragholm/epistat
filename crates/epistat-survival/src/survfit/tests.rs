#![expect(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    reason = "times tie when equal, and the sizes are a few hundred at most"
)]

use super::jackknife::{Rows, sum_of_squares};
use super::*;
use epistat_rng::Stream;

fn order(key: &[f64]) -> Vec<u32> {
    let mut index: Vec<u32> = (0..key.len() as u32).collect();
    index.sort_by(|&a, &b| key[a as usize].total_cmp(&key[b as usize]).then(a.cmp(&b)));
    index
}

/// The sum over clusters of `phi_k(J)^2` by its definition, one cluster and one
/// event time at a time.
fn naive(rows: &Rows<'_>, times: &[f64], p: &[f64], q: &[f64]) -> Vec<f64> {
    let n = rows.stop.len();
    (0..times.len())
        .map(|j| {
            let mut phi = vec![0.0; rows.n_clusters];
            for (i, &t) in times.iter().enumerate().take(j + 1) {
                for r in 0..n {
                    let start = rows.start.map_or(f64::NEG_INFINITY, |s| s[r]);
                    let k = rows.cluster[r] as usize;
                    if start < t && rows.stop[r] >= t {
                        phi[k] -= q[i] * rows.weight[r];
                    }
                    if rows.event[r] && rows.stop[r] == t {
                        phi[k] += p[i] * rows.weight[r];
                    }
                }
            }
            phi.iter().map(|v| v * v).sum()
        })
        .collect()
}

#[test]
fn the_running_sums_equal_the_sum_over_clusters() {
    let mut rng = Stream::seeded(5);
    for delayed in [false, true] {
        let n = 120;
        let mut start = Vec::new();
        let mut stop = Vec::new();
        let mut event = Vec::new();
        let mut weight = Vec::new();
        let mut cluster = Vec::new();
        for _ in 0..n {
            let s = (rng.uniform() * 6.0).floor();
            start.push(s);
            stop.push(s + 1.0 + (rng.uniform() * 8.0).floor());
            event.push(rng.below(3) == 0);
            weight.push(0.5 + (rng.uniform() * 8.0).floor() / 4.0);
            cluster.push(rng.below(17) as u32);
        }
        let by_start = order(&start);
        let by_stop = order(&stop);
        let rows = Rows {
            start: delayed.then_some(start.as_slice()),
            stop: &stop,
            event: &event,
            weight: &weight,
            cluster: &cluster,
            n_clusters: 17,
            by_start: if delayed { &by_start } else { &[] },
            by_stop: &by_stop,
        };
        let mut times: Vec<f64> = (0..n).filter(|&i| event[i]).map(|i| stop[i]).collect();
        times.sort_by(f64::total_cmp);
        times.dedup();
        let p: Vec<f64> = (0..times.len()).map(|i| 0.1 + 0.05 * i as f64).collect();
        let q: Vec<f64> = (0..times.len())
            .map(|i| 0.02 + 0.01 * (i % 3) as f64)
            .collect();
        let fast = sum_of_squares(&rows, &times, &p, &q);
        let slow = naive(&rows, &times, &p, &q);
        for (j, (a, b)) in fast.iter().zip(&slow).enumerate() {
            // Both sum a few hundred terms of size at most 1.
            assert!(
                (a - b).abs() <= 1e-12 * (1.0 + b),
                "time {j}: {a} against {b}"
            );
        }
    }
}

fn data_two_groups() -> (Vec<f64>, Vec<i32>, Vec<u32>) {
    (
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0],
        vec![1, 0, 1, 0, 1, 0, 0],
        vec![0, 0, 0, 0, 0, 9, 9],
    )
}

#[test]
fn a_curve_that_stays_above_the_level_has_no_quantile() {
    let (stop, status, stratum) = data_two_groups();
    let data = SurvData::builder(&stop, &status)
        .stratum(&stratum)
        .build()
        .unwrap();
    let fit = estimate(&data, &SurvfitOptions::default()).unwrap();
    assert_eq!(fit.groups.len(), 2);
    assert_eq!(fit.groups[1].group, 9);
    let censored = fit.group(9).unwrap().quantiles(&[0.0, 0.5]).unwrap();
    // As in R, a curve that never leaves 1 has no quantile even at 0.
    assert_eq!(censored[0].time, None);
    assert_eq!(censored[1].time, None);
    assert_eq!(censored[1].lower, None);
}

#[test]
fn input_that_survfit_cannot_use_is_refused() {
    let (stop, status, _) = data_two_groups();
    let clusters = [1u32, 1, 2, 2, 3, 3, 4];
    let data = SurvData::builder(&stop, &status)
        .cluster(&clusters)
        .build()
        .unwrap();
    let robust_false = SurvfitOptions {
        robust: Some(false),
        ..SurvfitOptions::default()
    };
    assert!(estimate(&data, &robust_false).is_err());
    let plain = SurvData::builder(&stop, &status).build().unwrap();
    for level in [0.0, 1.0, f64::NAN] {
        let opts = SurvfitOptions {
            conf_level: level,
            ..SurvfitOptions::default()
        };
        assert!(estimate(&plain, &opts).is_err());
    }
    let none = SurvfitOptions {
        conf_type: ConfType::None,
        conf_level: 2.0,
        ..SurvfitOptions::default()
    };
    assert_eq!(
        estimate(&plain, &none).unwrap().groups[0].lower,
        Vec::<Option<f64>>::new()
    );
    let start = [0.0; 7];
    let counting = SurvData::builder(&stop, &status)
        .start(&start)
        .build()
        .unwrap();
    let robust_true = SurvfitOptions {
        robust: Some(true),
        ..SurvfitOptions::default()
    };
    assert!(estimate(&counting, &robust_true).is_err());
    let competing = SurvData::builder(&stop, &status)
        .n_codes(2)
        .build()
        .unwrap();
    assert!(estimate(&competing, &SurvfitOptions::default()).is_err());
    let zero = [0.0; 7];
    let nothing = SurvData::builder(&stop, &status)
        .weight(&zero)
        .build()
        .unwrap();
    assert!(estimate(&nothing, &SurvfitOptions::default()).is_err());
    let curve = &estimate(&plain, &SurvfitOptions::default()).unwrap().groups[0];
    assert!(curve.quantiles(&[1.5]).is_err());
    assert!(curve.quantiles(&[f64::NAN]).is_err());
    assert!(curve.rmean(-1.0).is_err());
    assert!(curve.rmean(f64::INFINITY).is_err());
}

#[test]
fn rows_of_weight_zero_change_nothing() {
    let (stop, status, _) = data_two_groups();
    let weight = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    let all = SurvData::builder(&stop, &status)
        .weight(&weight)
        .build()
        .unwrap();
    let used = SurvData::builder(&stop[..5], &status[..5]).build().unwrap();
    let a = estimate(&all, &SurvfitOptions::default()).unwrap();
    let b = estimate(&used, &SurvfitOptions::default()).unwrap();
    assert_eq!((a.n_rows, a.n_used), (7, 5));
    assert_eq!(a.groups, b.groups);
}

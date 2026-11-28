//! Hand-checkable cases and refusals of `metrics`.
//!
//! Without censoring every weight is 1, so the Brier score is the mean squared
//! difference between the risk and the indicator of the event, the area is the
//! probability that a case outranks a control with ties counting one half, and
//! the standard error is that of a mean. The weights of `rttright` reproduce
//! the Kaplan-Meier estimate. Each of these is computed in the test from its
//! definition. The sums have at most `n` terms, so the tolerance is
//! `4 n eps` times the largest term, which is at most 1.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::metrics::brier::{self, BrierOptions};
use epistat_survival::metrics::calibration::{self, CalibrationOptions, Groups};
use epistat_survival::metrics::rttright::{self, RttrightOptions};
use epistat_survival::metrics::score::{
    Predicted, ScoreCensoring, ScoreOptions, SuppliedCensoring, auc, brier as score_brier, ipa,
    null_brier,
};
use epistat_survival::survfit::{SurvfitOptions, estimate as km};

const EPS: f64 = f64::EPSILON;

/// Eleven rows, all with the event, with a tie in the time and in the risk.
fn complete() -> (Vec<f64>, Vec<i32>, Vec<f64>) {
    let time = vec![1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let status = vec![1; 11];
    let risk = vec![0.9, 0.2, 0.2, 0.6, 0.1, 0.5, 0.8, 0.3, 0.4, 0.7, 0.35];
    (time, status, risk)
}

#[test]
fn without_censoring_the_metrics_are_the_plain_ones() {
    let (time, status, risk) = complete();
    let n = time.len();
    let data = SurvData::builder(&time, &status).build().unwrap();
    let times = [2.0, 5.0, 8.0];
    let m = Mat::from_fn(n, 3, |i, j| (risk[i] + 0.01 * j as f64).min(1.0));
    let opts = ScoreOptions::default();
    let b = score_brier(
        &data,
        m.as_ref(),
        &times,
        &ScoreCensoring::KaplanMeier,
        &opts,
    )
    .unwrap();
    let a = auc(
        &data,
        m.as_ref(),
        &times,
        &ScoreCensoring::KaplanMeier,
        &opts,
    )
    .unwrap();
    let tol = 4.0 * n as f64 * EPS;
    for (j, &t) in times.iter().enumerate() {
        let r: Vec<f64> = (0..n)
            .map(|i| {
                let p = m[(i, j)];
                if time[i] <= t {
                    (1.0 - p) * (1.0 - p)
                } else {
                    p * p
                }
            })
            .collect();
        let mean = r.iter().sum::<f64>() / n as f64;
        assert!((b.brier[j] - mean).abs() <= tol);
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n as f64 - 1.0);
        let se = (var / n as f64).sqrt();
        assert!((b.se[j].unwrap() - se).abs() <= tol);

        let (mut num, mut pairs) = (0.0, 0.0);
        for i in 0..n {
            for k in 0..n {
                if time[i] <= t && time[k] > t {
                    pairs += 1.0;
                    num += match m[(i, j)].total_cmp(&m[(k, j)]) {
                        std::cmp::Ordering::Greater => 1.0,
                        std::cmp::Ordering::Equal => 0.5,
                        std::cmp::Ordering::Less => 0.0,
                    };
                }
            }
        }
        assert!((a.auc[j].unwrap() - num / pairs).abs() <= tol);
        assert_eq!(a.n_cases[j] + a.n_survivors[j], n);
    }
}

#[test]
fn an_area_without_cases_or_controls_is_missing() {
    let (time, status, risk) = complete();
    let data = SurvData::builder(&time, &status).build().unwrap();
    // At the last time every row is a case and none is a control.
    let m = Mat::from_fn(time.len(), 1, |i, _| risk[i]);
    let a = auc(
        &data,
        m.as_ref(),
        &[10.0],
        &ScoreCensoring::KaplanMeier,
        &ScoreOptions::default(),
    )
    .unwrap();
    assert_eq!(a.auc, vec![None]);
    assert_eq!(a.se, vec![None]);
    // Before the first time every row is a control and none is a case.
    let a = auc(
        &data,
        m.as_ref(),
        &[0.5],
        &ScoreCensoring::KaplanMeier,
        &ScoreOptions::default(),
    )
    .unwrap();
    assert_eq!(a.auc, vec![None]);
}

#[test]
fn rows_of_weight_zero_are_left_out_and_the_influence_has_one_row_per_used_row() {
    let (time, status, risk) = complete();
    let n = time.len();
    let weight: Vec<f64> = (0..n).map(|i| if i % 4 == 1 { 0.0 } else { 1.0 }).collect();
    let keep: Vec<usize> = (0..n).filter(|&i| weight[i] > 0.0).collect();
    let t2: Vec<f64> = keep.iter().map(|&i| time[i]).collect();
    let s2: Vec<i32> = keep.iter().map(|&i| status[i]).collect();
    let weighted = SurvData::builder(&time, &status)
        .weight(&weight)
        .build()
        .unwrap();
    let reduced = SurvData::builder(&t2, &s2).build().unwrap();
    let m = Mat::from_fn(n, 1, |i, _| risk[i]);
    let m2 = Mat::from_fn(keep.len(), 1, |i, _| risk[keep[i]]);
    let opts = ScoreOptions {
        keep_influence: true,
        ..ScoreOptions::default()
    };
    let a = score_brier(
        &weighted,
        m.as_ref(),
        &[5.0],
        &ScoreCensoring::KaplanMeier,
        &opts,
    )
    .unwrap();
    let b = score_brier(
        &reduced,
        m2.as_ref(),
        &[5.0],
        &ScoreCensoring::KaplanMeier,
        &opts,
    )
    .unwrap();
    assert_eq!(a.brier, b.brier);
    assert_eq!(a.se, b.se);
    assert_eq!((a.n_rows, a.n_used), (n, keep.len()));
    let influence = a.influence.as_ref().unwrap();
    assert_eq!(influence.n_units(), keep.len());
    // The same values, with the rows of the data as units.
    let reduced = b.influence.as_ref().unwrap();
    assert_eq!(influence.column(0), reduced.column(0));
    let kept: Vec<u32> = keep.iter().map(|&i| u32::try_from(i).unwrap()).collect();
    assert_eq!(influence.rows(), Some(kept.as_slice()));
}

#[test]
fn event_free_probabilities_are_the_complement_of_the_risk() {
    let (time, status, risk) = complete();
    let n = time.len();
    let data = SurvData::builder(&time, &status).build().unwrap();
    let m = Mat::from_fn(n, 1, |i, _| risk[i]);
    let free = Mat::from_fn(n, 1, |i, _| 1.0 - risk[i]);
    let a = score_brier(
        &data,
        m.as_ref(),
        &[5.0],
        &ScoreCensoring::KaplanMeier,
        &ScoreOptions::default(),
    )
    .unwrap();
    let opts = ScoreOptions {
        predicted: Predicted::EventFreeProbability,
        ..ScoreOptions::default()
    };
    let b = score_brier(
        &data,
        free.as_ref(),
        &[5.0],
        &ScoreCensoring::KaplanMeier,
        &opts,
    )
    .unwrap();
    assert!((a.brier[0] - b.brier[0]).abs() <= 4.0 * n as f64 * EPS);
}

#[test]
fn the_null_model_and_the_ipa_of_the_null_model() {
    let (time, status, _) = complete();
    let n = time.len();
    let data = SurvData::builder(&time, &status).build().unwrap();
    let opts = ScoreOptions {
        keep_influence: true,
        ..ScoreOptions::default()
    };
    let z = null_brier(&data, &[2.0, 5.0], &ScoreCensoring::KaplanMeier, &opts).unwrap();
    // Everyone is predicted the proportion with the event by t.
    for (j, &t) in [2.0, 5.0].iter().enumerate() {
        let p = time.iter().filter(|&&u| u <= t).count() as f64 / n as f64;
        let b = p * (1.0 - p);
        assert!((z.brier[j] - b).abs() <= 4.0 * n as f64 * EPS);
    }
    // The null model against itself has index 0 and standard error 0.
    let p = ipa(&z, &z).unwrap();
    for j in 0..2 {
        assert!(p.ipa[j].unwrap().abs() <= 4.0 * EPS);
        assert!(p.se[j].unwrap().abs() <= 4.0 * n as f64 * EPS);
    }
}

#[test]
fn score_refuses_what_it_cannot_score() {
    let (time, status, risk) = complete();
    let n = time.len();
    let data = SurvData::builder(&time, &status).build().unwrap();
    let m = Mat::from_fn(n, 2, |i, _| risk[i]);
    let opts = ScoreOptions::default();
    let msg = |times: &[f64]| {
        score_brier(
            &data,
            m.as_ref(),
            times,
            &ScoreCensoring::KaplanMeier,
            &opts,
        )
        .unwrap_err()
        .to_string()
    };
    assert!(msg(&[5.0, 2.0]).contains("is not sorted"));
    assert!(msg(&[5.0, 5.0]).contains("repeats at row 1"));
    assert!(msg(&[5.0, 11.0]).contains("after the last observed time"));
    assert!(msg(&[5.0, f64::NAN]).contains("is not finite"));
    assert!(msg(&[]).contains("times is empty"));
    assert!(msg(&[5.0]).contains("is 11 by 2, but there are 11 rows and 1 times"));

    let bad = Mat::from_fn(n, 2, |i, _| risk[i] * 2.0);
    let e = score_brier(
        &data,
        bad.as_ref(),
        &[2.0, 5.0],
        &ScoreCensoring::KaplanMeier,
        &opts,
    )
    .unwrap_err();
    assert!(e.to_string().contains("not a probability"));
    // A marker for the area may be any number.
    assert!(
        auc(
            &data,
            bad.as_ref(),
            &[2.0, 5.0],
            &ScoreCensoring::KaplanMeier,
            &opts
        )
        .is_ok()
    );

    let start: Vec<f64> = vec![0.0; n];
    let delayed = SurvData::builder(&time, &status)
        .start(&start)
        .build()
        .unwrap();
    let e = score_brier(
        &delayed,
        m.as_ref(),
        &[2.0, 5.0],
        &ScoreCensoring::KaplanMeier,
        &opts,
    )
    .unwrap_err();
    assert!(e.to_string().contains("delayed entry"));

    let weight: Vec<f64> = vec![2.0; n];
    let weighted = SurvData::builder(&time, &status)
        .weight(&weight)
        .build()
        .unwrap();
    let e = score_brier(
        &weighted,
        m.as_ref(),
        &[2.0, 5.0],
        &ScoreCensoring::KaplanMeier,
        &opts,
    )
    .unwrap_err();
    assert!(e.to_string().contains("weights other than 0 and 1"));

    let e = score_brier(
        &data,
        m.as_ref(),
        &[2.0, 5.0],
        &ScoreCensoring::KaplanMeier,
        &ScoreOptions {
            cause: 2,
            ..opts.clone()
        },
    )
    .unwrap_err();
    assert!(e.to_string().contains("cause 2 is outside 1..=1"));

    let competing = SurvData::builder(&time, &status)
        .n_codes(2)
        .build()
        .unwrap();
    let free = ScoreOptions {
        predicted: Predicted::EventFreeProbability,
        ..opts.clone()
    };
    let e = score_brier(
        &competing,
        m.as_ref(),
        &[2.0, 5.0],
        &ScoreCensoring::KaplanMeier,
        &free,
    )
    .unwrap_err();
    assert!(
        e.to_string()
            .contains("event-free probabilities are for data with one event code")
    );

    let own = vec![0.5; n];
    let at = Mat::from_fn(n, 2, |_, _| 0.0);
    let supplied = ScoreCensoring::Supplied(SuppliedCensoring {
        before_own_time: &own,
        at_times: at.as_ref(),
    });
    let e = score_brier(&data, m.as_ref(), &[2.0, 5.0], &supplied, &opts).unwrap_err();
    assert!(
        e.to_string()
            .contains("censoring probabilities must be in (0, 1]")
    );

    let a = score_brier(
        &data,
        m.as_ref(),
        &[2.0, 5.0],
        &ScoreCensoring::KaplanMeier,
        &opts,
    )
    .unwrap();
    let z = null_brier(&data, &[2.0, 6.0], &ScoreCensoring::KaplanMeier, &opts).unwrap();
    assert!(
        ipa(&a, &z)
            .unwrap_err()
            .to_string()
            .contains("same data and times")
    );
}

#[test]
fn a_single_row_has_no_standard_error() {
    let time = [3.0];
    let status = [1];
    let data = SurvData::builder(&time, &status).build().unwrap();
    let m = Mat::from_fn(1, 1, |_, _| 0.4);
    let b = score_brier(
        &data,
        m.as_ref(),
        &[3.0],
        &ScoreCensoring::KaplanMeier,
        &ScoreOptions::default(),
    )
    .unwrap();
    assert!((b.brier[0] - 0.36).abs() <= 4.0 * EPS);
    assert_eq!(b.se, vec![None]);
}

#[test]
fn calibration_by_hand_and_its_refusals() {
    let (time, status, risk) = complete();
    let n = time.len();
    let data = SurvData::builder(&time, &status).build().unwrap();
    // Breaks at 0.3: the low group is risks up to 0.3, the high one the rest.
    let got = calibration::estimate(
        &data,
        &risk,
        5.0,
        &CalibrationOptions::new(Groups::Breaks(vec![0.0, 0.3, 1.0])),
    )
    .unwrap();
    assert_eq!(got.groups.len(), 2);
    let low: Vec<usize> = (0..n).filter(|&i| risk[i] <= 0.3).collect();
    let high: Vec<usize> = (0..n).filter(|&i| risk[i] > 0.3).collect();
    for (g, rows) in got.groups.iter().zip([&low, &high]) {
        assert_eq!(g.n, rows.len());
        let mean = rows.iter().map(|&i| risk[i]).sum::<f64>() / rows.len() as f64;
        let observed = rows.iter().filter(|&&i| time[i] <= 5.0).count() as f64 / rows.len() as f64;
        assert!((g.predicted - mean).abs() <= 4.0 * EPS);
        assert!((g.observed.unwrap() - observed).abs() <= 4.0 * n as f64 * EPS);
    }
    // A time after the last time of the group has no observed risk.
    let late = calibration::estimate(
        &data,
        &risk,
        10.5,
        &CalibrationOptions::new(Groups::Breaks(vec![0.0, 0.3, 1.0])),
    )
    .unwrap();
    assert!(late.groups.iter().all(|g| g.observed.is_none()));

    let o = |g| CalibrationOptions::new(g);
    let e = calibration::estimate(&data, &risk, 5.0, &o(Groups::Quantiles(0))).unwrap_err();
    assert!(e.to_string().contains("at least 1"));
    let e =
        calibration::estimate(&data, &risk, 5.0, &o(Groups::Breaks(vec![0.2, 1.0]))).unwrap_err();
    assert!(e.to_string().contains("cover every predicted risk"));
    let e = calibration::estimate(&data, &risk[..3], 5.0, &o(Groups::Quantiles(2))).unwrap_err();
    assert!(e.to_string().contains("risk has length 3, expected 11"));
    let mut off = risk.clone();
    off[4] = 1.5;
    let e = calibration::estimate(&data, &off, 5.0, &o(Groups::Quantiles(2))).unwrap_err();
    assert!(e.to_string().contains("not a probability"));
    let e = calibration::estimate(&data, &risk, f64::NAN, &o(Groups::Quantiles(2))).unwrap_err();
    assert!(e.to_string().contains("time is NaN"));
}

#[test]
fn rttright_weights_reproduce_kaplan_meier() {
    // Tied times, censorings among the deaths and case weights.
    let n = 300;
    let mut rng = epistat_rng::Stream::seeded(7);
    let time: Vec<f64> = (0..n).map(|_| 1.0 + rng.below(25) as f64).collect();
    let status: Vec<i32> = (0..n).map(|_| i32::from(rng.below(2) == 0)).collect();
    let weight: Vec<f64> = (0..n).map(|_| 0.5 + rng.uniform() * 2.0).collect();
    let data = SurvData::builder(&time, &status)
        .weight(&weight)
        .build()
        .unwrap();
    let times = [3.0, 8.0, 12.0, 20.0, 26.0];
    let w = rttright::estimate(&data, Some(&times), &RttrightOptions::default()).unwrap();
    let curve = km(&data, &SurvfitOptions::default()).unwrap();
    let g = &curve.groups[0];
    let bound = 8.0 * n as f64 * EPS;
    for (j, &t) in times.iter().enumerate() {
        // The weighted share with the event by t is one minus the curve at t.
        let share: f64 = (0..n)
            .filter(|&i| time[i] <= t && status[i] == 1)
            .map(|i| w.weights[(i, j)])
            .sum();
        let k = g.time.partition_point(|&u| u <= t);
        let surv = if k == 0 { 1.0 } else { g.surv[k - 1] };
        assert!(
            (share - (1.0 - surv)).abs() <= bound,
            "t = {t}: {share} against {}",
            1.0 - surv
        );
        // The weights add to 1 while someone is still at risk at t: a censored
        // row's weight went to the rows to its right. After the last time it
        // went nowhere, and the total is the share that has the event.
        let total: f64 = (0..n).map(|i| w.weights[(i, j)]).sum();
        let still_at_risk = time.iter().any(|&u| u >= t);
        let expected = if still_at_risk { 1.0 } else { share };
        assert!((total - expected).abs() <= bound, "t = {t}: total {total}");
    }
}

#[test]
fn rttright_and_brier_refuse_what_they_do_not_support() {
    let (time, status, risk) = complete();
    let n = time.len();
    let start: Vec<f64> = vec![0.0; n];
    let delayed = SurvData::builder(&time, &status)
        .start(&start)
        .build()
        .unwrap();
    let e = rttright::estimate(&delayed, None, &RttrightOptions::default()).unwrap_err();
    assert!(e.to_string().contains("delayed entry"));
    let data = SurvData::builder(&time, &status).build().unwrap();
    let e = rttright::estimate(&data, Some(&[f64::NAN]), &RttrightOptions::default()).unwrap_err();
    assert!(e.to_string().contains("not finite"));
    let e = rttright::estimate(&data, Some(&[]), &RttrightOptions::default()).unwrap_err();
    assert!(e.to_string().contains("times is empty"));

    let m = Mat::from_fn(n, 1, |i, _| risk[i]);
    let e = brier::estimate(&delayed, m.as_ref(), &[5.0], &BrierOptions::default()).unwrap_err();
    assert!(e.to_string().contains("delayed entry"));
    let competing = SurvData::builder(&time, &status)
        .n_codes(2)
        .build()
        .unwrap();
    let e = brier::estimate(&competing, m.as_ref(), &[5.0], &BrierOptions::default()).unwrap_err();
    assert!(e.to_string().contains("event codes"));
    let e = brier::estimate(&data, m.as_ref(), &[5.0, 6.0], &BrierOptions::default()).unwrap_err();
    assert!(
        e.to_string()
            .contains("is 11 by 1, but there are 11 rows and 2 times")
    );
    let bad = Mat::from_fn(n, 1, |_, _| 2.0);
    let e = brier::estimate(&data, bad.as_ref(), &[5.0], &BrierOptions::default()).unwrap_err();
    assert!(e.to_string().contains("not a probability"));
    let one_time = [4.0; 3];
    let one_status = [1, 0, 1];
    let single = SurvData::builder(&one_time, &one_status).build().unwrap();
    let m3 = Mat::from_fn(3, 1, |_, _| 0.5);
    let e = brier::estimate(&single, m3.as_ref(), &[4.0], &BrierOptions::default()).unwrap_err();
    assert!(e.to_string().contains("no gap between them"));
}

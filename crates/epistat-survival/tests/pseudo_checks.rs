//! Properties and refusals of `pseudo::pseudo` and the regression on its values
//! that no R run is needed for.
//!
//! The mean of the infinitesimal pseudo-values of a group is its estimate. Scaling
//! every case weight by the same factor leaves a Kaplan-Meier or
//! Aalen-Johansen estimate unchanged, so the weighted influences, whose sum is
//! the derivative along that scaling, add to zero. The tolerance is that
//! of a sum of `n` values each carrying the error of a sum of at most
//! `m + rows` terms, `4 (n + 7 m + 20) eps` times the largest pseudo-value
//! deviation plus one, times the scale of the target.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "counts and grid values are far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::pseudo::{GeeOptions, Link, Method, PseudoOptions, Target, pseudo};
use faer::Mat;

const EPS: f64 = f64::EPSILON;

struct Sample {
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    weight: Vec<f64>,
    stratum: Vec<u32>,
    id: Vec<u32>,
}

fn sample(n: usize, seed: u64) -> Sample {
    let mut rng = Stream::seeded(seed);
    let mut s = Sample {
        start: vec![],
        stop: vec![],
        status: vec![],
        weight: vec![],
        stratum: vec![],
        id: vec![],
    };
    for row in 0..n {
        let start = (rng.uniform() * 30.0).floor() / 10.0;
        s.start.push(start);
        s.stop
            .push(start + 0.1 + (rng.exp1() * 30.0).floor() / 10.0);
        s.status.push(match rng.below(10) {
            0..=2 => 1,
            3..=4 => 2,
            _ => 0,
        });
        s.weight.push(0.25 * (1 + rng.below(8)) as f64);
        s.stratum.push(rng.below(3) as u32 + 1);
        s.id.push((row / 2) as u32);
    }
    for row in (1..n).step_by(2) {
        s.stratum[row] = s.stratum[row - 1];
    }
    s
}

const TARGETS: [Target; 5] = [
    Target::Survival,
    Target::CumulativeIncidence { cause: 1 },
    Target::CumulativeIncidence { cause: 2 },
    Target::RestrictedMean,
    Target::RestrictedTimeLost { cause: 1 },
];

#[test]
fn the_infinitesimal_pseudo_values_average_to_the_estimate() {
    let s = sample(600, 7);
    let times = [0.5, 2.0, 4.0, 8.0];
    // The number of exit times of a group is at most that of the sample.
    let mut distinct = s.stop.clone();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    let m = distinct.len() as f64;
    for with_start in [false, true] {
        let mut builder = SurvData::builder(&s.stop, &s.status)
            .n_codes(2)
            .weight(&s.weight)
            .stratum(&s.stratum)
            .id(&s.id);
        if with_start {
            builder = builder.start(&s.start);
        }
        let data = builder.build().unwrap();
        for target in TARGETS {
            let p = pseudo(&data, &times, &PseudoOptions::new(target)).unwrap();
            let k = times.len();
            let mut start = 0;
            for g in &p.groups {
                let n = g.n_subjects;
                for (j, &t) in times.iter().enumerate() {
                    let mean: f64 =
                        (0..n).map(|c| p.values[(start + c) * k + j]).sum::<f64>() / n as f64;
                    let spread = (0..n)
                        .map(|c| (p.values[(start + c) * k + j] - g.estimate[j]).abs())
                        .fold(0.0f64, f64::max);
                    let scale = if matches!(
                        target,
                        Target::RestrictedMean | Target::RestrictedTimeLost { .. }
                    ) {
                        t.max(1.0)
                    } else {
                        1.0
                    };
                    let tolerance =
                        4.0 * (n as f64 + 7.0 * m + 20.0) * EPS * (spread + 1.0) * scale;
                    assert!(
                        (mean - g.estimate[j]).abs() <= tolerance,
                        "{target:?} group {} time {t}: mean {mean}, estimate {}, tolerance {tolerance:e}",
                        g.group,
                        g.estimate[j]
                    );
                }
                start += n;
            }
        }
    }
}

/// Without censoring the survival probability is the proportion still alive,
/// and the pseudo-observation of both jackknives is the indicator of being alive,
/// `n theta - (n - 1) theta(-i) = 1(T_i > t)` exactly, and `theta + n U_i` the
/// same with `U_i = (1(T_i > t) - theta) / n`. Both are sums of at most
/// `n + m` terms, so `64 (n + m) eps` bounds the difference.
#[test]
fn without_censoring_both_jackknives_give_the_indicator() {
    let mut rng = Stream::seeded(11);
    let n = 400;
    let stop: Vec<f64> = (0..n)
        .map(|_| 0.1 + (rng.exp1() * 30.0).floor() / 10.0)
        .collect();
    let status = vec![1; n];
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let times = [1.0, 2.0, 3.5];
    let tolerance = 64.0 * (2.0 * n as f64) * EPS;
    for method in [Method::InfinitesimalJackknife, Method::Jackknife] {
        let mut opts = PseudoOptions::new(Target::Survival);
        opts.method = method;
        let p = pseudo(&data, &times, &opts).unwrap();
        for (i, &t_i) in stop.iter().enumerate() {
            for (j, &t) in times.iter().enumerate() {
                let want = f64::from(t_i > t);
                let got = p.values[i * times.len() + j];
                assert!(
                    (got - want).abs() <= tolerance,
                    "{method:?} subject {i} time {t}: {got}"
                );
            }
        }
    }
}

#[test]
fn refusals() {
    let stop = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0];
    let status = [1, 2, 0, 1, 0, 2];
    let data = SurvData::builder(&stop, &status)
        .n_codes(2)
        .build()
        .unwrap();
    let ok = PseudoOptions::new(Target::Survival);
    assert!(pseudo(&data, &[1.0, 2.0], &ok).is_ok());
    assert!(pseudo(&data, &[], &ok).is_err());
    assert!(pseudo(&data, &[2.0, 1.0], &ok).is_err());
    assert!(pseudo(&data, &[1.0, 1.0], &ok).is_err());
    assert!(pseudo(&data, &[f64::NAN], &ok).is_err());
    assert!(pseudo(&data, &[f64::INFINITY], &ok).is_err());
    for cause in [0, 3] {
        for target in [
            Target::CumulativeIncidence { cause },
            Target::RestrictedTimeLost { cause },
        ] {
            assert!(pseudo(&data, &[2.0], &PseudoOptions::new(target)).is_err());
        }
    }
    // A restricted mean before the origin of the curves.
    assert!(pseudo(&data, &[-1.0], &PseudoOptions::new(Target::RestrictedMean)).is_err());
    assert!(pseudo(&data, &[-1.0], &ok).is_ok());
    // One subject cannot be left out.
    let one = SurvData::builder(&stop[..1], &status[..1]).build().unwrap();
    let mut jack = PseudoOptions::new(Target::Survival);
    jack.method = Method::Jackknife;
    assert!(pseudo(&one, &[1.0], &jack).is_err());
    assert!(pseudo(&one, &[1.0], &ok).is_ok());
    // A subject with rows in two strata.
    let id = [0, 0, 1, 1, 2, 2];
    let stratum = [1, 2, 1, 1, 2, 2];
    let crossing = SurvData::builder(&stop, &status)
        .n_codes(2)
        .id(&id)
        .stratum(&stratum)
        .build()
        .unwrap();
    assert!(pseudo(&crossing, &[2.0], &ok).is_err());
    // No row of positive weight.
    let weight = [0.0; 6];
    let none = SurvData::builder(&stop, &status)
        .n_codes(2)
        .weight(&weight)
        .build()
        .unwrap();
    assert!(pseudo(&none, &[2.0], &ok).is_err());
    // The memory bound on the influence of the time lost.
    let lost = |limit| {
        let mut o = PseudoOptions::new(Target::RestrictedTimeLost { cause: 1 });
        o.max_influence_values = limit;
        pseudo(&data, &[4.0], &o)
    };
    assert!(lost(None).is_ok());
    // 6 subjects, 4 exit times up to the horizon and 3 states.
    assert!(lost(Some(72)).is_ok());
    assert!(lost(Some(71)).is_err());
}

#[test]
fn times_beyond_the_last_exit_time_are_flagged_and_hold_the_last_value() {
    let stop = [1.0, 2.0, 3.0, 4.0];
    let status = [1, 1, 0, 1];
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let p = pseudo(
        &data,
        &[3.0, 4.0, 5.0, 9.0],
        &PseudoOptions::new(Target::Survival),
    )
    .unwrap();
    let g = &p.groups[0];
    assert_eq!(g.beyond, [false, false, true, true]);
    assert_eq!(g.estimate[2].to_bits(), g.estimate[1].to_bits());
    assert_eq!(g.estimate[3].to_bits(), g.estimate[1].to_bits());
    // The area keeps growing at the last value, which is 0 after the last event.
    let area = pseudo(
        &data,
        &[4.0, 9.0],
        &PseudoOptions::new(Target::RestrictedMean),
    )
    .unwrap();
    assert_eq!(
        area.groups[0].estimate[0].to_bits(),
        area.groups[0].estimate[1].to_bits()
    );
}

#[test]
fn a_binary_covariate_regression_is_the_difference_of_the_means() {
    // With the identity link and an intercept and a binary covariate, the fit
    // is each group's mean of the pseudo-values.
    let s = sample(500, 23);
    let data = SurvData::builder(&s.stop, &s.status)
        .n_codes(2)
        .id(&s.id)
        .build()
        .unwrap();
    let p = pseudo(
        &data,
        &[3.0],
        &PseudoOptions::new(Target::CumulativeIncidence { cause: 1 }),
    )
    .unwrap();
    let n = p.n_subjects();
    let x = Mat::from_fn(n, 1, |i, _| f64::from(p.subject[i] % 3 == 0));
    let fit = p
        .regress(Some(x.as_ref()), false, &GeeOptions::new(Link::Identity))
        .unwrap();
    let mean = |flag: bool| {
        let v: Vec<f64> = (0..n)
            .filter(|&i| (p.subject[i] % 3 == 0) == flag)
            .map(|i| p.values[i])
            .collect();
        v.iter().sum::<f64>() / v.len() as f64
    };
    assert!((fit.coefficients[0] - mean(false)).abs() < 1e-12);
    assert!((fit.coefficients[0] + fit.coefficients[1] - mean(true)).abs() < 1e-12);
    assert_eq!((fit.n_obs, fit.n_clusters), (n, n));
    // The standard error of a difference of two means with the sandwich is
    // the square root of the sum of the within-group variances over the sizes,
    // not corrected for degrees of freedom.
    let var = |flag: bool| {
        let m = mean(flag);
        let v: Vec<f64> = (0..n)
            .filter(|&i| (p.subject[i] % 3 == 0) == flag)
            .map(|i| (p.values[i] - m).powi(2))
            .collect();
        v.iter().sum::<f64>() / (v.len() as f64).powi(2)
    };
    let se = fit.standard_errors();
    assert!((se[1] - (var(true) + var(false)).sqrt()).abs() < 1e-12);
}

#[test]
fn the_design_has_an_effect_for_each_time_after_the_first() {
    let s = sample(200, 31);
    let data = SurvData::builder(&s.stop, &s.status)
        .n_codes(2)
        .build()
        .unwrap();
    let p = pseudo(
        &data,
        &[1.0, 2.0, 3.0],
        &PseudoOptions::new(Target::Survival),
    )
    .unwrap();
    let st = p.stacked(None, true).unwrap();
    assert_eq!((st.x.nrows(), st.x.ncols()), (600, 3));
    assert_eq!(st.cluster[..4], [0, 0, 0, 1]);
    assert_eq!((st.x[(0, 1)], st.x[(1, 1)], st.x[(2, 1)]), (0.0, 1.0, 0.0));
    assert_eq!((st.x[(0, 2)], st.x[(1, 2)], st.x[(2, 2)]), (0.0, 0.0, 1.0));
    let plain = p.stacked(None, false).unwrap();
    assert_eq!(plain.x.ncols(), 1);
    // Wrong covariate count.
    let bad = Mat::<f64>::zeros(5, 1);
    assert!(p.stacked(Some(bad.as_ref()), true).is_err());
}

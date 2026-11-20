//! Exact ties checked against enumeration of the subsets of the risk sets, the
//! cases `coxph` refuses, and `clogit`'s input rules.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::clogit::{ClogitMethod, ClogitOptions, clogit};
use epistat_survival::cox::{CoxOptions, Ties, fit, score_residuals};

/// The log partial likelihood at `beta` by listing every subset of the risk
/// set at each event time, which shares no code with the recursion.
fn enumerated_loglik(
    start: &[f64],
    stop: &[f64],
    event: &[bool],
    stratum: &[u32],
    x: &Mat<f64>,
    beta: &[f64],
) -> f64 {
    let eta = |i: usize| (0..beta.len()).map(|j| x[(i, j)] * beta[j]).sum::<f64>();
    let mut total = 0.0;
    let mut strata: Vec<u32> = stratum.to_vec();
    strata.sort_unstable();
    strata.dedup();
    for s in strata {
        let rows: Vec<usize> = (0..stop.len()).filter(|&i| stratum[i] == s).collect();
        let mut times: Vec<f64> = rows
            .iter()
            .filter(|&&i| event[i])
            .map(|&i| stop[i])
            .collect();
        times.sort_by(f64::total_cmp);
        times.dedup();
        for t in times {
            let dying: Vec<usize> = rows
                .iter()
                .copied()
                .filter(|&i| event[i] && stop[i].total_cmp(&t).is_eq())
                .collect();
            let risk: Vec<usize> = rows
                .iter()
                .copied()
                .filter(|&i| start[i] < t && stop[i] >= t)
                .collect();
            let d = dying.len();
            let mut d0 = 0.0;
            for mask in 0u32..(1 << risk.len()) {
                if mask.count_ones() as usize == d {
                    d0 += (0..risk.len())
                        .filter(|k| mask >> k & 1 == 1)
                        .map(|k| eta(risk[k]))
                        .sum::<f64>()
                        .exp();
                }
            }
            total += dying.iter().map(|&i| eta(i)).sum::<f64>() - d0.ln();
        }
    }
    total
}

#[test]
fn loglik_equals_the_enumeration() {
    // Tied times in three strata, with entry times, a time with three events
    // and times with two.
    let stop = [
        2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0, 2.0, 2.0, 3.0, 4.0, 4.0, 6.0, 2.0, 3.0, 3.0, 5.0,
        5.0, 6.0,
    ];
    let start = [
        0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 4.0,
    ];
    let event = [
        true, true, true, true, true, false, true, true, true, true, false, true, true, false,
        true, true, false, true, true, true,
    ];
    let stratum = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2];
    let x = Mat::from_fn(20, 2, |i, j| {
        let v = (i as f64 * 0.7 + j as f64 * 1.3).sin();
        if j == 0 { v } else { v * 0.5 + (i % 3) as f64 }
    });
    let status: Vec<i32> = event.iter().map(|&e| i32::from(e)).collect();
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .stratum(&stratum)
        .build()
        .unwrap();
    for beta in [[0.0, 0.0], [0.4, -0.3], [-1.2, 0.8]] {
        let options = CoxOptions {
            ties: Ties::Exact,
            init: Some(&beta),
            iter_max: 0,
            ..CoxOptions::default()
        };
        let model = fit(&data, x.as_ref(), &options).unwrap();
        let expected = enumerated_loglik(&start, &stop, &event, &stratum, &x, &beta);
        let got = model.log_partial_likelihood.unwrap();
        // About 2^10 subsets are added at the largest risk set.
        assert!(
            (got - expected).abs() <= 1024.0 * f64::EPSILON * expected.abs().max(1.0),
            "{beta:?}: {got} {expected}"
        );
    }
}

#[test]
fn one_event_per_time_is_the_breslow_fit() {
    let stop: Vec<f64> = (1..=12).map(f64::from).collect();
    let status = [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1];
    let x = Mat::from_fn(12, 1, |i, _| ((i * 7 % 5) as f64) - 1.5);
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let run = |ties| {
        fit(
            &data,
            x.as_ref(),
            &CoxOptions {
                ties,
                ..CoxOptions::default()
            },
        )
        .unwrap()
    };
    let exact = run(Ties::Exact);
    let breslow = run(Ties::Breslow);
    assert_eq!(exact.coefficients, breslow.coefficients);
    assert_eq!(exact.log_partial_likelihood, breslow.log_partial_likelihood);
}

#[test]
fn what_coxph_refuses_is_an_error() {
    let stop = [1.0, 1.0, 2.0, 3.0];
    let status = [1, 1, 1, 0];
    let weight = [1.0, 2.0, 1.0, 1.0];
    let cluster = [0, 0, 1, 1];
    let x = Mat::from_fn(4, 1, |i, _| i as f64);
    let exact = CoxOptions {
        ties: Ties::Exact,
        ..CoxOptions::default()
    };
    let weighted = SurvData::builder(&stop, &status)
        .weight(&weight)
        .build()
        .unwrap();
    assert!(fit(&weighted, x.as_ref(), &exact).is_err());
    let clustered = SurvData::builder(&stop, &status)
        .cluster(&cluster)
        .build()
        .unwrap();
    assert!(fit(&clustered, x.as_ref(), &exact).is_err());
    let plain = SurvData::builder(&stop, &status).build().unwrap();
    assert!(score_residuals(&plain, x.as_ref(), &[Some(0.1)], &exact).is_err());
    // A weight of 0 drops the row; weights of 1 are fine.
    let zero = [1.0, 1.0, 1.0, 0.0];
    let dropped = SurvData::builder(&stop, &status)
        .weight(&zero)
        .build()
        .unwrap();
    let model = fit(&dropped, x.as_ref(), &exact).unwrap();
    assert_eq!((model.n_rows, model.n_used), (4, 3));
}

#[test]
fn clogit_rules() {
    let case = [true, false, false, true, false];
    let set = [1, 1, 1, 2, 2];
    let x = Mat::from_fn(5, 1, |i, _| [0.5, -0.2, 1.1, 0.9, 0.1][i]);
    let model = clogit(&case, &set, x.as_ref(), &ClogitOptions::default()).unwrap();
    assert_eq!((model.n_rows, model.n_events), (5, 2.0));
    assert!(clogit(&case[..4], &set, x.as_ref(), &ClogitOptions::default()).is_err());
    let weight = [1.0, 1.0, 1.0, 2.0, 1.0];
    let weighted = ClogitOptions {
        weight: Some(&weight),
        ..ClogitOptions::default()
    };
    assert!(clogit(&case, &set, x.as_ref(), &weighted).is_err());
    let approximate = ClogitOptions {
        method: ClogitMethod::Approximate,
        ..weighted
    };
    assert!(clogit(&case, &set, x.as_ref(), &approximate).is_ok());
    let cluster = [0, 0, 0, 1, 1];
    let clustered = ClogitOptions {
        cluster: Some(&cluster),
        ..ClogitOptions::default()
    };
    assert!(clogit(&case, &set, x.as_ref(), &clustered).is_err());
    // No case at all gives no estimate, as in the Cox fit.
    let none = clogit(&[false; 5], &set, x.as_ref(), &ClogitOptions::default()).unwrap();
    assert!(none.coefficients[0].is_none());
}

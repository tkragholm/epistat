//! Checks of `relsurv` that need no R reference: the interval estimators
//! converge to the exact ones as the width goes to 0, an interval without
//! events sits at the boundary of the excess hazard model, the flexible model
//! with a zero expected hazard is the ordinary one, and invalid input is an
//! error.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::flexpar::{self, FlexParOptions, Scale};
use epistat_survival::relsurv::{
    ExcessOptions, ExpectedHazard, Grid, Method, NetOptions, excess_hazard, net_survival,
};

mod relsurv_sim;
use relsurv_sim::{cohort, table};

/// The interval estimate converges to the exact one as the width goes to 0
/// when the grid comes to resolve the entry and exit times, here all multiples
/// of 5 days. With exits inside an interval its person-time denominator is
/// smaller than the risk set at the exit, so its limit is the estimator with
/// person-time denominators, which differs from the exact one by about the
/// number of deaths over the square of the number at risk.
#[test]
fn interval_estimate_converges_to_the_exact_one() {
    let t = table();
    let mut c = cohort(600, 3, true);
    for i in 0..c.time.len() {
        c.time[i] = (c.time[i] / 5.0).ceil() * 5.0;
        c.start[i] = (c.start[i] / 5.0).floor() * 5.0;
    }
    let data = c.data(false);
    let expected = c.expected(&t, &data);
    for method in [Method::PoharPerme, Method::EdererII] {
        let exact = net_survival(
            &data,
            &expected,
            &NetOptions {
                method,
                ..NetOptions::default()
            },
        )
        .unwrap()
        .remove(0);
        let target = *exact.cumulative_excess_hazard.last().unwrap();
        let errors: Vec<f64> = [80.0, 40.0, 20.0, 10.0, 5.0]
            .iter()
            .map(|&w| {
                let g = net_survival(
                    &data,
                    &expected,
                    &NetOptions {
                        method,
                        grid: Grid::Width(w),
                        ..NetOptions::default()
                    },
                )
                .unwrap()
                .remove(0);
                assert_eq!(g.times.last(), exact.times.last());
                (g.cumulative_excess_hazard.last().unwrap() - target).abs()
            })
            .collect();
        println!("{method:?}: errors at the last time {errors:?}");
        assert!(
            errors.windows(2).all(|e| e[1] < e[0]),
            "{method:?} {errors:?}"
        );
        // At 5 days Ederer II is exact to rounding. The Pohar Perme weights at
        // the middle of each interval differ from those at its end by a factor
        // within `lambda_max w / 2` of 1, so each increment moves by at most
        // that share of its observed and expected parts.
        let bound = match method {
            Method::EdererII | Method::EdererI | Method::Hakulinen => 1e-12,
            Method::PoharPerme => {
                let observed: f64 = exact
                    .events
                    .iter()
                    .zip(&exact.n_at_risk)
                    .map(|(d, n)| d / n)
                    .sum();
                let lambda_max = t.rates().iter().copied().fold(0.0, f64::max);
                let oldest = (0..c.time.len())
                    .map(|i| c.age0[i] + c.time[i])
                    .fold(0.0, f64::max);
                let lambda_old = (-10.5 + 0.09 * (oldest / 365.0).ceil() + 0.35).exp() / 365.0;
                lambda_max.min(lambda_old) * 2.5 * (observed + (observed - target).abs())
            }
        };
        assert!(errors[4] < bound, "{method:?} {errors:?}");
    }
}

#[test]
fn an_interval_without_events_is_at_the_boundary() {
    let t = table();
    let mut c = cohort(800, 5, false);
    // No deaths in (100, 300]: those deaths become censorings.
    for i in 0..c.time.len() {
        if c.status[i] == 1 && c.time[i] > 100.0 && c.time[i] <= 300.0 {
            c.status[i] = 0;
        }
    }
    let data = c.data(false);
    let expected = c.expected(&t, &data);
    let cuts = [0.0, 100.0, 300.0, 700.0, 1500.0, 2700.0];
    for aggregate in [false, true] {
        let opts = ExcessOptions {
            aggregate,
            ..ExcessOptions::new(&cuts)
        };
        let fit = excess_hazard(&data, &[&c.x2], &expected, &opts).unwrap();
        assert!(fit.converged);
        assert_eq!(fit.at_bound, [false, true, false, false, false, false]);
        assert_eq!(fit.coefficients[1], Some(f64::NEG_INFINITY));
        assert!(
            fit.coefficients
                .iter()
                .enumerate()
                .all(|(k, b)| k == 1 || b.unwrap().is_finite())
        );
        let cov = fit.covariance.as_ref().unwrap();
        let d = cuts.len();
        for k in 0..d {
            assert!(cov.get(1, k).is_none() && cov.get(k, 1).is_none());
        }
        assert!(cov.get(0, 0).unwrap() > 0.0);
        assert_eq!(fit.events[1].to_bits(), 0.0_f64.to_bits());
        assert!(fit.expected_events[1] > 0.0);
    }
}

#[test]
fn zero_expected_hazard_gives_the_ordinary_flexible_model() {
    let c = cohort(500, 7, true);
    let time: Vec<f64> = c.time.iter().map(|v| v / 365.25).collect();
    let start: Vec<f64> = c.start.iter().map(|v| v / 365.25).collect();
    let data = SurvData::builder(&time, &c.status)
        .start(&start)
        .build()
        .unwrap();
    let x = Mat::from_fn(
        c.time.len(),
        2,
        |i, j| if j == 0 { c.x1[i] } else { c.x2[i] },
    );
    let zero = vec![0.0; c.time.len()];
    let plain = flexpar::fit(&data, x.as_ref(), &FlexParOptions::new(Scale::Hazard, 3)).unwrap();
    let opts = FlexParOptions {
        bhazard: Some(&zero),
        ..FlexParOptions::new(Scale::Hazard, 3)
    };
    let excess = flexpar::fit(&data, x.as_ref(), &opts).unwrap();
    // Debug prints each float so that it reads back to the same bits.
    assert_eq!(format!("{plain:?}"), format!("{excess:?}"));
    let odds = FlexParOptions {
        bhazard: Some(&zero),
        ..FlexParOptions::new(Scale::Odds, 3)
    };
    assert!(flexpar::fit(&data, x.as_ref(), &odds).is_err());
    let negative = vec![-1e-3; c.time.len()];
    let bad = FlexParOptions {
        bhazard: Some(&negative),
        ..FlexParOptions::new(Scale::Hazard, 3)
    };
    assert!(flexpar::fit(&data, x.as_ref(), &bad).is_err());
}

#[test]
fn rows_of_weight_zero_are_left_out() {
    let t = table();
    let c = cohort(400, 9, true);
    let keep: Vec<usize> = (0..c.time.len()).filter(|i| i % 3 != 1).collect();
    let weight: Vec<f64> = (0..c.time.len())
        .map(|i| if i % 3 == 1 { 0.0 } else { 1.0 })
        .collect();
    let weighted = SurvData::builder(&c.time, &c.status)
        .start(&c.start)
        .weight(&weight)
        .build()
        .unwrap();
    let pick = |v: &[f64]| keep.iter().map(|&i| v[i]).collect::<Vec<f64>>();
    let (time, start, age0, date0) = (pick(&c.time), pick(&c.start), pick(&c.age0), pick(&c.date0));
    let status: Vec<i32> = keep.iter().map(|&i| c.status[i]).collect();
    let sex: Vec<u32> = keep.iter().map(|&i| c.sex[i]).collect();
    let x2 = pick(&c.x2);
    let subset = SurvData::builder(&time, &status)
        .start(&start)
        .build()
        .unwrap();
    let e_w = ExpectedHazard::from_table(&t, &weighted, &[&c.age0, &c.date0], &[&c.sex]).unwrap();
    let e_s = ExpectedHazard::from_table(&t, &subset, &[&age0, &date0], &[&sex]).unwrap();
    for grid in [Grid::EventTimes, Grid::Width(30.0)] {
        let opts = NetOptions {
            grid,
            ..NetOptions::default()
        };
        assert_eq!(
            net_survival(&weighted, &e_w, &opts).unwrap(),
            net_survival(&subset, &e_s, &opts).unwrap()
        );
    }
    let cuts = [0.0, 365.0, 1000.0, 2700.0];
    let a = excess_hazard(&weighted, &[&c.x2], &e_w, &ExcessOptions::new(&cuts)).unwrap();
    let b = excess_hazard(&subset, &[&x2], &e_s, &ExcessOptions::new(&cuts)).unwrap();
    assert_eq!(a, b);
}

#[test]
fn invalid_input_is_an_error() {
    let t = table();
    let c = cohort(200, 11, false);
    let data = c.data(false);
    let expected = c.expected(&t, &data);
    let bad_width = NetOptions {
        grid: Grid::Width(0.0),
        ..NetOptions::default()
    };
    assert!(net_survival(&data, &expected, &bad_width).is_err());
    // Expected hazards of other rows.
    let other = cohort(150, 12, false);
    let other_data = other.data(false);
    let other_expected = other.expected(&t, &other_data);
    assert!(net_survival(&data, &other_expected, &NetOptions::default()).is_err());
    // Weights other than 0 and 1.
    let w = vec![2.0; c.time.len()];
    let weighted = SurvData::builder(&c.time, &c.status)
        .weight(&w)
        .build()
        .unwrap();
    assert!(net_survival(&weighted, &expected, &NetOptions::default()).is_err());
    // Follow-up off the table: a date before its first cut.
    let early: Vec<f64> = c.date0.iter().map(|d| d - 20.0 * 365.0).collect();
    assert!(ExpectedHazard::from_table(&t, &data, &[&c.age0, &early], &[&c.sex]).is_err());
    // Cut points that do not start at 0 or do not increase, a collinear
    // covariate.
    for cuts in [&[10.0, 100.0][..], &[0.0, 100.0, 100.0], &[0.0]] {
        assert!(excess_hazard(&data, &[], &expected, &ExcessOptions::new(cuts)).is_err());
    }
    let twice: Vec<f64> = c.x1.iter().map(|v| 2.0 * v).collect();
    assert!(
        excess_hazard(
            &data,
            &[&c.x1, &twice],
            &expected,
            &ExcessOptions::new(&[0.0, 500.0, 3000.0])
        )
        .is_err()
    );
}

#[test]
fn a_net_survival_curve_is_read_at_the_last_time_before() {
    let curve = epistat_survival::relsurv::NetSurvival {
        stratum: 0,
        times: vec![1.0, 2.5, 4.0],
        cumulative_excess_hazard: vec![0.1, 0.2, 0.3],
        variance: vec![0.01, 0.02, 0.03],
        survival: vec![0.9, 0.8, 0.7],
        lower: vec![Some(0.85), None, Some(0.6)],
        upper: vec![Some(0.95), None, Some(0.8)],
        n_at_risk: vec![10.0, 8.0, 5.0],
        events: vec![1.0, 1.0, 1.0],
    };
    assert!(curve.at(0.5).is_none());
    let at = curve.at(3.0).unwrap();
    assert_eq!((at.time, at.survival, at.lower), (2.5, 0.8, None));
    assert_eq!(curve.at(9.0).unwrap().survival, 0.7);
    assert_eq!(curve.position_at(1.0), Some(0));
}

//! Checks of the expected survival, Ederer I and Hakulinen, the crude
//! probabilities and the test of equal net survival that need no R reference:
//! identical bits on 1, 2, 3, 4 and 8 threads, the conditional curve against
//! the expected part of Ederer II with delayed entry, and the refusals.

use epistat_survival::SurvData;
use epistat_survival::ratetable::{RateDim, RateTable};
use epistat_survival::relsurv::{
    ExpectedHazard, ExpectedMethod, Grid, Method, NetOptions, crude_mortality, expected_survival,
    net_survival, net_survival_test,
};
use epistat_testkit::{POOLS, assert_same_on_pools};

mod relsurv_sim;
use relsurv_sim::{cohort, table};

#[test]
fn the_new_estimators_on_threads() {
    let t = table();
    let c = cohort(3000, 7, false);
    let data = c.data(true);
    let expected = c.expected(&t, &data);
    let potential: Vec<f64> = c
        .time
        .iter()
        .zip(&c.status)
        .map(|(&time, &s)| if s == 1 { time + 500.0 } else { time })
        .collect();
    let times = [0.0, 100.0, 365.25, 1000.0, 2500.0];
    assert_same_on_pools(&POOLS, || {
        let curves = expected_survival(&data, &expected, &times, ExpectedMethod::EdererI).unwrap();
        curves
            .iter()
            .flat_map(|c| c.survival.iter().map(|v| v.unwrap_or(f64::NAN)))
            .collect::<Vec<f64>>()
    });
    for (method, potential) in [
        (Method::EdererI, None),
        (Method::Hakulinen, Some(&potential[..])),
    ] {
        for grid in [Grid::EventTimes, Grid::Width(30.4375)] {
            let opts = NetOptions {
                method,
                grid,
                potential,
                ..NetOptions::default()
            };
            assert_same_on_pools(&POOLS, || {
                net_survival(&data, &expected, &opts)
                    .unwrap()
                    .iter()
                    .flat_map(|c| c.cumulative_excess_hazard.clone())
                    .collect::<Vec<f64>>()
            });
        }
    }
    let late = cohort(3000, 8, true);
    let late_data = late.data(true);
    let late_expected = late.expected(&t, &late_data);
    assert_same_on_pools(&POOLS, || {
        crude_mortality(&late_data, &late_expected, 0.95)
            .unwrap()
            .iter()
            .flat_map(|c| [c.disease.clone(), c.disease_variance.clone()].concat())
            .collect::<Vec<f64>>()
    });
    let plain = late.data(false);
    let plain_expected = late.expected(&t, &plain);
    assert_same_on_pools(&POOLS, || {
        let r = net_survival_test(&plain, &late.group, &plain_expected, Some(1.0)).unwrap();
        [vec![r.statistic], r.z, r.covariance].concat()
    });
}

#[test]
fn the_conditional_curve_is_the_expected_part_of_ederer_two() {
    let t = table();
    let c = cohort(2000, 11, true);
    let data = c.data(true);
    let expected = c.expected(&t, &data);
    let e2 = net_survival(
        &data,
        &expected,
        &NetOptions {
            method: Method::EdererII,
            ..NetOptions::default()
        },
    )
    .unwrap();
    for curve in &e2 {
        let conditional =
            expected_survival(&data, &expected, &curve.times, ExpectedMethod::Conditional).unwrap();
        let mine = &conditional[curve.stratum];
        let mut observed = 0.0;
        for k in 0..curve.times.len() {
            observed += curve.events[k] / curve.n_at_risk[k];
            let lambda_expected = observed - curve.cumulative_excess_hazard[k];
            let got = -mine.survival[k].unwrap().ln();
            // Both are sums of the same increments over at most a few
            // thousand steps, grouped differently, so they agree to a few
            // thousand eps of the observed hazard.
            assert!(
                (got - lambda_expected).abs() <= 1e-12 * observed.max(1.0),
                "{got} {lambda_expected}"
            );
            assert_eq!(mine.n[k] as f64, curve.n_at_risk[k]);
        }
    }
}

#[test]
fn refusals() {
    let t = table();
    let c = cohort(400, 3, true);
    let data = c.data(false);
    let expected = c.expected(&t, &data);
    let hakulinen = NetOptions {
        method: Method::Hakulinen,
        ..NetOptions::default()
    };
    assert!(net_survival(&data, &expected, &hakulinen).is_err());
    let potential = c.time.clone();
    let pohar = NetOptions {
        potential: Some(&potential),
        ..NetOptions::default()
    };
    assert!(net_survival(&data, &expected, &pohar).is_err());
    // Delayed entry has no Ederer I.
    let ederer = NetOptions {
        method: Method::EdererI,
        ..NetOptions::default()
    };
    assert!(net_survival(&data, &expected, &ederer).is_err());
    assert!(expected_survival(&data, &expected, &[1.0, 0.5], ExpectedMethod::Conditional).is_err());

    // A table that ends: following a row beyond its follow-up leaves it.
    let short = RateTable::new(
        vec![RateDim::Time {
            cuts: vec![0.0, 10.0],
        }],
        vec![0.1],
        1.0,
    )
    .unwrap();
    let stop = [4.0, 6.0];
    let status = [1, 0];
    let d = SurvData::builder(&stop, &status).build().unwrap();
    let h = ExpectedHazard::from_table(&short, &d, &[&[0.0, 0.0]], &[]).unwrap();
    assert_eq!(h.cumulative(0, 12.0), None);
    assert_eq!(h.hazard(0, 12.0), None);
    assert!((h.cumulative(0, 8.0).unwrap() - 0.8).abs() < 1e-15);
    assert_eq!(h.hazard(0, 8.0), Some(0.1));
    assert!(expected_survival(&d, &h, &[12.0], ExpectedMethod::EdererI).is_err());
    let open = ExpectedHazard::from_table(&short.extend_ends(), &d, &[&[0.0, 0.0]], &[]).unwrap();
    assert!((open.cumulative(0, 12.0).unwrap() - 1.2).abs() < 1e-15);
    let curve = expected_survival(&d, &open, &[12.0], ExpectedMethod::EdererI).unwrap();
    assert!((curve[0].survival[0].unwrap() - (-1.2_f64).exp()).abs() < 1e-15);
}

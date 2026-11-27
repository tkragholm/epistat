//! `relsurv::net_survival`, `excess_hazard` and the flexible excess hazard
//! model of `flexpar` give identical bits in rayon pools of 1, 2, 3, 4 and 8
//! threads, for cohorts that span several blocks.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::flexpar::{self, FlexParOptions, Scale};
use epistat_survival::relsurv::{
    ExcessFit, ExcessOptions, Grid, Method, NetOptions, NetSurvival, excess_hazard, net_survival,
};
use epistat_testkit::{POOLS, assert_same_on_pools};

mod relsurv_sim;
use relsurv_sim::{cohort, table};

fn flat_net(curves: &[NetSurvival]) -> Vec<f64> {
    let mut out = Vec::new();
    for c in curves {
        out.extend(&c.times);
        out.extend(&c.cumulative_excess_hazard);
        out.extend(&c.variance);
        out.extend(c.lower.iter().map(|v| v.unwrap_or(f64::NAN)));
        out.extend(&c.n_at_risk);
    }
    out
}

fn flat_excess(f: &ExcessFit) -> Vec<f64> {
    let mut out: Vec<f64> = f
        .coefficients
        .iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect();
    out.extend(
        f.covariance
            .iter()
            .flat_map(epistat_survival::Covariance::as_slice)
            .map(|v| v.unwrap_or(f64::NAN)),
    );
    out.extend([f.log_likelihood, f.deviance]);
    out.extend(&f.expected_events);
    out
}

#[test]
fn net_survival_on_threads() {
    let t = table();
    let c = cohort(5000, 21, true);
    let data = c.data(true);
    let expected = c.expected(&t, &data);
    for method in [Method::PoharPerme, Method::EdererII] {
        for grid in [Grid::Width(30.4375), Grid::EventTimes] {
            let opts = NetOptions {
                method,
                grid,
                ..NetOptions::default()
            };
            assert_same_on_pools(&POOLS, || {
                flat_net(&net_survival(&data, &expected, &opts).unwrap())
            });
        }
    }
}

#[test]
fn excess_hazard_on_threads() {
    let t = table();
    let c = cohort(5000, 22, true);
    let data = c.data(false);
    let expected = c.expected(&t, &data);
    let cuts = [0.0, 90.0, 365.0, 730.0, 1460.0, 2700.0];
    for aggregate in [false, true] {
        let opts = ExcessOptions {
            aggregate,
            ..ExcessOptions::new(&cuts)
        };
        let x: Vec<&[f64]> = if aggregate {
            vec![&c.x2]
        } else {
            vec![&c.x1, &c.x2]
        };
        assert_same_on_pools(&POOLS, || {
            flat_excess(&excess_hazard(&data, &x, &expected, &opts).unwrap())
        });
    }
}

#[test]
fn flexible_excess_model_on_threads() {
    let t = table();
    let c = cohort(5000, 23, true);
    let days = c.data(false);
    let expected = c.expected(&t, &days);
    let n = c.time.len();
    let bh: Vec<f64> = (0..n)
        .map(|i| expected.hazard(i, c.time[i]).unwrap() * 365.25)
        .collect();
    let time: Vec<f64> = c.time.iter().map(|v| v / 365.25).collect();
    let start: Vec<f64> = c.start.iter().map(|v| v / 365.25).collect();
    let data = SurvData::builder(&time, &c.status)
        .start(&start)
        .build()
        .unwrap();
    let x = Mat::from_fn(n, 2, |i, j| if j == 0 { c.x1[i] } else { c.x2[i] });
    let opts = FlexParOptions {
        bhazard: Some(&bh),
        ..FlexParOptions::new(Scale::Hazard, 3)
    };
    assert_same_on_pools(&POOLS, || {
        let f = flexpar::fit(&data, x.as_ref(), &opts).unwrap();
        let mut out = f.gamma.clone();
        out.extend(f.beta.iter().map(|b| b.unwrap_or(f64::NAN)));
        out.extend(
            f.covariance
                .iter()
                .flat_map(epistat_survival::Covariance::as_slice)
                .map(|v| v.unwrap_or(f64::NAN)),
        );
        out.push(f.log_likelihood);
        out
    });
}

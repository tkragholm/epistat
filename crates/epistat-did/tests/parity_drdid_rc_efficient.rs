//! Parity of the locally efficient repeated-cross-section estimator against
//! `DRDID::drdid_rc` 1.3.0, which is what `did::att_gt(est_method = "dr")` calls
//! on repeated cross sections.

use epistat_testkit::load_json;

use epistat_did::TwoPeriodData;
use epistat_did::two_period::{self, Estimator, NuisanceOptions, TwoPeriod, TwoPeriodOptions};
use epistat_linalg::MatRef;
use serde::Deserialize;

#[derive(Deserialize)]
struct Cell {
    group: i32,
    time: i32,
    att: f64,
    se: f64,
    inffunc: Vec<f64>,
    y: Vec<f64>,
    post: Vec<f64>,
    d: Vec<f64>,
    lpop: Vec<f64>,
}

/// The cell's fit by `estimator`; the estimator adds the intercept, and R
/// was given `cbind(1, lpop)` to match.
fn fit(cell: &Cell, estimator: Estimator, ridge: f64) -> TwoPeriod {
    let treated: Vec<bool> = cell.d.iter().map(|&v| v > 0.5).collect();
    let post: Vec<bool> = cell.post.iter().map(|&v| v > 0.5).collect();
    let data = TwoPeriodData::repeated_cross_section(&treated, &post, &cell.y)
        .covariates(MatRef::from_column_major_slice(&cell.lpop, cell.y.len(), 1))
        .build()
        .expect("data");
    let options = TwoPeriodOptions {
        estimator,
        nuisance: NuisanceOptions {
            ridge,
            ..NuisanceOptions::default()
        },
        ..TwoPeriodOptions::default()
    };
    two_period::estimate(&data, &options)
        .unwrap_or_else(|e| panic!("cell ({}, {}): {e}", cell.group, cell.time))
}

#[test]
fn locally_efficient_rc_matches_drdid_rc() {
    let cells: Vec<Cell> = load_json("tests/data/drdid_rc_efficient_ref.json");
    assert_eq!(cells.len(), 12);

    let mut worst_att = 0.0_f64;
    let mut worst_se = 0.0_f64;
    let mut worst_influence = 0.0_f64;

    for cell in &cells {
        let fit = fit(cell, Estimator::DoublyRobust, 0.0);
        worst_att = worst_att.max((fit.estimate.estimate - cell.att).abs());
        worst_se = worst_se.max((fit.estimate.se.expect("se") - cell.se).abs());
        assert_eq!(fit.influence.n_units(), cell.inffunc.len());
        for (ours, theirs) in fit.influence.column(0).iter().zip(&cell.inffunc) {
            worst_influence = worst_influence.max((ours - theirs).abs());
        }
    }

    println!(
        "worst vs DRDID::drdid_rc -- att {worst_att:e}, se {worst_se:e}, \
         influence {worst_influence:e}"
    );
    assert!(worst_att < 1e-9, "worst att deviation {worst_att:e}");
    assert!(worst_se < 1e-9, "worst se deviation {worst_se:e}");
    assert!(
        worst_influence < 1e-7,
        "worst influence deviation {worst_influence:e}"
    );
}

/// The two RC estimators agree on the point estimate and disagree on the
/// standard error, which is the whole reason both exist.
#[test]
fn the_two_rc_estimators_differ_only_in_precision() {
    let cells: Vec<Cell> = load_json("tests/data/drdid_rc_efficient_ref.json");

    let mut differed = 0_usize;
    for cell in &cells {
        let ridge = NuisanceOptions::default().ridge;
        let efficient = fit(cell, Estimator::DoublyRobust, ridge);
        let traditional = fit(cell, Estimator::TraditionalDoublyRobust, ridge);
        let (efficient_se, traditional_se) = (
            efficient.estimate.se.expect("se"),
            traditional.estimate.se.expect("se"),
        );

        assert!(
            (efficient.estimate.estimate - traditional.estimate.estimate).abs() < 1e-8,
            "({}, {}) point estimates should agree: {} vs {}",
            cell.group,
            cell.time,
            efficient.estimate.estimate,
            traditional.estimate.estimate
        );
        if (efficient_se - traditional_se).abs() / traditional_se > 1e-3 {
            differed += 1;
        }
    }
    assert!(
        differed >= cells.len() / 2,
        "the two estimators gave near-identical standard errors in all but {differed} cells; \
         one of them may have been routed to the other"
    );
}

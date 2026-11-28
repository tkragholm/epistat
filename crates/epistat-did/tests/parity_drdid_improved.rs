//! Parity of the improved doubly robust estimators against
//! `DRDID::drdid_imp_rc` and `DRDID::drdid_imp_panel` 1.3.0, which fit the
//! propensity score by Sant'Anna and Zhao's calibration (`pscore.cal`).
//! Fixtures from `tests/data/generate_drdid_improved_reference.R`.

use epistat_testkit::load_json;

use epistat_did::TwoPeriodData;
use epistat_did::two_period::{self, Estimator, NuisanceOptions, TwoPeriodOptions};
use epistat_linalg::MatRef;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    rc: Vec<RcCase>,
    panel: Vec<PanelCase>,
}

#[derive(Deserialize)]
struct RcData {
    y: Vec<f64>,
    post: Vec<f64>,
    d: Vec<f64>,
    x: Vec<Vec<f64>>,
    weights: Vec<f64>,
}

#[derive(Deserialize)]
struct RcCase {
    name: String,
    att: f64,
    se: f64,
    inffunc: Vec<f64>,
    data: RcData,
}

#[derive(Deserialize)]
struct PanelData {
    y1: Vec<f64>,
    y0: Vec<f64>,
    d: Vec<f64>,
    x: Vec<Vec<f64>>,
    weights: Vec<f64>,
}

#[derive(Deserialize)]
struct PanelCase {
    name: String,
    att: f64,
    se: f64,
    inffunc: Vec<f64>,
    data: PanelData,
}

fn fixture() -> Fixture {
    load_json("tests/data/drdid_improved_ref.json")
}

fn worst(ours: &[f64], theirs: &[f64]) -> f64 {
    assert_eq!(ours.len(), theirs.len());
    ours.iter()
        .zip(theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn options() -> TwoPeriodOptions {
    TwoPeriodOptions {
        estimator: Estimator::Improved,
        nuisance: NuisanceOptions {
            ridge: 0.0,
            ..NuisanceOptions::default()
        },
        ..TwoPeriodOptions::default()
    }
}

/// Rows of covariates as one column-major buffer.
fn column_major(x: &[Vec<f64>]) -> (Vec<f64>, usize) {
    let k = x.first().map_or(0, Vec::len);
    (
        (0..k)
            .flat_map(|j| x.iter().map(move |row| row[j]))
            .collect(),
        k,
    )
}

#[test]
fn improved_rc_matches_drdid_imp_rc() {
    for case in fixture().rc {
        let d = &case.data;
        let treated: Vec<bool> = d.d.iter().map(|&v| v > 0.5).collect();
        let post: Vec<bool> = d.post.iter().map(|&v| v > 0.5).collect();
        let (x, k) = column_major(&d.x);
        let data = TwoPeriodData::repeated_cross_section(&treated, &post, &d.y)
            .weights(&d.weights)
            .covariates(MatRef::from_column_major_slice(&x, d.y.len(), k))
            .build()
            .expect("data");
        let fit = two_period::estimate(&data, &options())
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let fit_att = fit.estimate.estimate;
        let fit_se = fit.estimate.se.expect("se");
        let (att, se) = ((fit_att - case.att).abs(), (fit_se - case.se).abs());
        let influence = worst(fit.influence.column(0), &case.inffunc);
        println!(
            "{}: att {att:e}, se {se:e}, influence {influence:e}",
            case.name
        );
        assert!(att < 1e-8 && se < 1e-8 && influence < 1e-6, "{}", case.name);
    }
}

#[test]
fn improved_panel_matches_drdid_imp_panel() {
    for case in fixture().panel {
        let d = &case.data;
        let treated: Vec<bool> = d.d.iter().map(|&v| v > 0.5).collect();
        let change: Vec<f64> = d.y1.iter().zip(&d.y0).map(|(a, b)| a - b).collect();
        let (x, k) = column_major(&d.x);
        let data = TwoPeriodData::panel(&treated, &change)
            .weights(&d.weights)
            .covariates(MatRef::from_column_major_slice(&x, change.len(), k))
            .build()
            .expect("data");
        let fit = two_period::estimate(&data, &options())
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let fit_att = fit.estimate.estimate;
        let fit_se = fit.estimate.se.expect("se");
        let (att, se) = ((fit_att - case.att).abs(), (fit_se - case.se).abs());
        let influence = worst(fit.influence.column(0), &case.inffunc);
        println!(
            "{}: att {att:e}, se {se:e}, influence {influence:e}",
            case.name
        );
        assert!(att < 1e-8 && se < 1e-8 && influence < 1e-6, "{}", case.name);
    }
}

/// The improved estimators' interval is the normal one at the level asked
/// for, `att +- z se` with `z` the normal quantile, as the other two-period
/// estimators form it, with and without clusters. The tolerance is a few
/// rounding errors of `att + z se`: `4 eps max(|att|, z se)`.
#[test]
fn improved_interval_follows_the_level() {
    let level = epistat_core::Level::confidence(0.9).unwrap();
    let z = epistat_stats::dist::normal::quantile(level.upper_probability());
    let check = |fit: &two_period::TwoPeriod| {
        let estimate = &fit.estimate;
        let interval = estimate.interval.expect("interval");
        assert_eq!(interval.level, level);
        let (att, se) = (estimate.estimate, estimate.se.expect("se"));
        let tolerance = 4.0 * f64::EPSILON * att.abs().max(z * se);
        assert!((interval.lower.unwrap() - (att - z * se)).abs() <= tolerance);
        assert!((interval.upper.unwrap() - (att + z * se)).abs() <= tolerance);
    };
    let fixture = fixture();
    let options = TwoPeriodOptions { level, ..options() };
    let d = &fixture.rc[0].data;
    let treated: Vec<bool> = d.d.iter().map(|&v| v > 0.5).collect();
    let post: Vec<bool> = d.post.iter().map(|&v| v > 0.5).collect();
    let (x, k) = column_major(&d.x);
    let n = d.y.len();
    let clusters: Vec<u32> = (0..n).map(|i| u32::try_from(i % 7).unwrap()).collect();
    for clustered in [false, true] {
        let builder = TwoPeriodData::repeated_cross_section(&treated, &post, &d.y)
            .weights(&d.weights)
            .covariates(MatRef::from_column_major_slice(&x, n, k));
        let builder = if clustered {
            builder.clusters(&clusters)
        } else {
            builder
        };
        check(&two_period::estimate(&builder.build().unwrap(), &options).unwrap());
    }
    let d = &fixture.panel[0].data;
    let treated: Vec<bool> = d.d.iter().map(|&v| v > 0.5).collect();
    let change: Vec<f64> = d.y1.iter().zip(&d.y0).map(|(a, b)| a - b).collect();
    let (x, k) = column_major(&d.x);
    let data = TwoPeriodData::panel(&treated, &change)
        .weights(&d.weights)
        .covariates(MatRef::from_column_major_slice(&x, change.len(), k))
        .build()
        .unwrap();
    check(&two_period::estimate(&data, &options).unwrap());
}

//! Parity of the two-period IPW and outcome-regression estimators with
//! `DRDID` 1.3.0 (`std_ipw_did_panel`, `reg_did_panel`, `std_ipw_did_rc`,
//! `reg_did_rc`), on data with two covariates and sampling weights written by
//! `tests/data/generate_ipw_or_reference.R`.
//!
//! Tolerances. Both sides solve the same equations in double precision: the
//! logistic score equations to convergence and the least-squares normal
//! equations exactly. The ridge of the outcome regressions is set to 0, as R
//! has none. What remains is rounding in sums over at most 600 rows of terms
//! of order one, about `600 * 2.2e-16 = 1.3e-13` relative, amplified by the
//! condition number of the 3 x 3 Gram matrices, which is below 100 here, so
//! the ATT and the standard error agree to about 1e-11. `1e-10` leaves a
//! factor of ten for the logistic fits, which stop at a relative deviance
//! change of 1e-8, after which a Newton step changes the coefficients by
//! less than the square of the previous step. Each influence value is a
//! per-row quantity of order `n` times smaller than the sum the ATT is, and
//! the same bound applies to it relative to its own size; it is checked
//! against `1e-9`.

use epistat_core::Bootstrap;
use epistat_did::TwoPeriodData;
use epistat_did::two_period::{self, Estimator, NuisanceOptions, TwoPeriod, TwoPeriodOptions};
use epistat_linalg::MatRef;
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fit {
    att: f64,
    se: f64,
    inffunc: Vec<f64>,
}

#[derive(Deserialize)]
struct Panel {
    d: Vec<f64>,
    y0: Vec<f64>,
    y1: Vec<f64>,
    x1: Vec<f64>,
    x2: Vec<f64>,
    w: Vec<f64>,
    ipw: Fit,
    reg: Fit,
}

#[derive(Deserialize)]
struct Repeated {
    d: Vec<f64>,
    post: Vec<f64>,
    y: Vec<f64>,
    x1: Vec<f64>,
    x2: Vec<f64>,
    w: Vec<f64>,
    ipw: Fit,
    reg: Fit,
    dr1: Fit,
}

#[derive(Deserialize)]
struct Reference {
    panel: Panel,
    rc: Repeated,
}

const ESTIMATE_TOLERANCE: f64 = 1e-10;
const INFLUENCE_TOLERANCE: f64 = 1e-9;

fn options(estimator: Estimator) -> TwoPeriodOptions {
    TwoPeriodOptions {
        estimator,
        nuisance: NuisanceOptions {
            ridge: 0.0,
            ..NuisanceOptions::default()
        },
        ..TwoPeriodOptions::default()
    }
}

fn check(label: &str, fit: &TwoPeriod, reference: &Fit) {
    let (att, se) = (fit.estimate.estimate, fit.estimate.se.expect("se"));
    assert!(
        (att - reference.att).abs() < ESTIMATE_TOLERANCE,
        "{label} att {att} vs R {}",
        reference.att
    );
    assert!(
        (se - reference.se).abs() < ESTIMATE_TOLERANCE,
        "{label} se {se} vs R {}",
        reference.se
    );
    assert_eq!(fit.influence.n_units(), reference.inffunc.len());
    let worst = fit
        .influence
        .column(0)
        .iter()
        .zip(&reference.inffunc)
        .map(|(ours, theirs)| (ours - theirs).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        worst < INFLUENCE_TOLERANCE,
        "{label} influence off by {worst:e}"
    );
}

/// The panel with the covariates `x1` and `x2`; the estimator adds the
/// intercept.
fn panel(panel: &Panel, options: &TwoPeriodOptions) -> TwoPeriod {
    let treated: Vec<bool> = panel.d.iter().map(|&v| v > 0.5).collect();
    let change: Vec<f64> = panel.y1.iter().zip(&panel.y0).map(|(a, b)| a - b).collect();
    let x: Vec<f64> = panel.x1.iter().chain(&panel.x2).copied().collect();
    let data = TwoPeriodData::panel(&treated, &change)
        .weights(&panel.w)
        .covariates(MatRef::from_column_major_slice(&x, change.len(), 2))
        .build()
        .expect("data");
    two_period::estimate(&data, options).expect("fit")
}

fn repeated(rc: &Repeated, options: &TwoPeriodOptions) -> TwoPeriod {
    let treated: Vec<bool> = rc.d.iter().map(|&v| v > 0.5).collect();
    let post: Vec<bool> = rc.post.iter().map(|&v| v > 0.5).collect();
    let x: Vec<f64> = rc.x1.iter().chain(&rc.x2).copied().collect();
    let data = TwoPeriodData::repeated_cross_section(&treated, &post, &rc.y)
        .weights(&rc.w)
        .covariates(MatRef::from_column_major_slice(&x, rc.y.len(), 2))
        .build()
        .expect("data");
    two_period::estimate(&data, options).expect("fit")
}

#[test]
fn panel_ipw_matches_std_ipw_did_panel() {
    let reference: Reference = load_json("tests/data/ipw_or_two_period_ref.json");
    let fit = panel(
        &reference.panel,
        &options(Estimator::InverseProbabilityWeighting),
    );
    check("panel ipw", &fit, &reference.panel.ipw);
}

#[test]
fn panel_or_matches_reg_did_panel() {
    let reference: Reference = load_json("tests/data/ipw_or_two_period_ref.json");
    let fit = panel(&reference.panel, &options(Estimator::OutcomeRegression));
    check("panel reg", &fit, &reference.panel.reg);
}

#[test]
fn repeated_ipw_matches_std_ipw_did_rc() {
    let reference: Reference = load_json("tests/data/ipw_or_two_period_ref.json");
    let fit = repeated(
        &reference.rc,
        &options(Estimator::InverseProbabilityWeighting),
    );
    check("rc ipw", &fit, &reference.rc.ipw);
}

#[test]
fn repeated_or_matches_reg_did_rc() {
    let reference: Reference = load_json("tests/data/ipw_or_two_period_ref.json");
    let fit = repeated(&reference.rc, &options(Estimator::OutcomeRegression));
    check("rc reg", &fit, &reference.rc.reg);
}

/// The interval is `att +- z se` unless a bootstrap is asked for.
#[test]
fn the_interval_is_analytic_by_default() {
    let reference: Reference = load_json("tests/data/ipw_or_two_period_ref.json");
    let fit = panel(
        &reference.panel,
        &options(Estimator::InverseProbabilityWeighting),
    );
    let (att, se) = (fit.estimate.estimate, fit.estimate.se.expect("se"));
    let z = 1.959_963_984_540_054;
    assert!((fit.estimate.lower().expect("lower") - (att - z * se)).abs() < 1e-12);
    assert!((fit.estimate.upper().expect("upper") - (att + z * se)).abs() < 1e-12);
    let boot = panel(
        &reference.panel,
        &TwoPeriodOptions {
            bootstrap: Some(Bootstrap::new(199, 17_431)),
            ..options(Estimator::InverseProbabilityWeighting)
        },
    );
    assert_eq!(boot.estimate.estimate.to_bits(), att.to_bits());
    assert_ne!(boot.estimate.lower(), fit.estimate.lower());
}

/// The traditional doubly robust estimator with sampling weights,
/// `DRDID::drdid_rc1`.
#[test]
fn repeated_dr_matches_drdid_rc1_with_weights() {
    let reference: Reference = load_json("tests/data/ipw_or_two_period_ref.json");
    let fit = repeated(&reference.rc, &options(Estimator::TraditionalDoublyRobust));
    check("rc dr1", &fit, &reference.rc.dr1);
}

/// The legacy computation is still reached, and is a different estimator on
/// weighted data: its propensity score ignores the weights and its influence
/// function lacks the nuisance terms.
#[test]
fn the_legacy_repeated_dr_is_kept_as_an_option() {
    let reference: Reference = load_json("tests/data/ipw_or_two_period_ref.json");
    let mut legacy = options(Estimator::TraditionalDoublyRobust);
    legacy.nuisance.legacy_repeated_cross_section = true;
    let fit = repeated(&reference.rc, &legacy);
    assert!((fit.estimate.estimate - reference.rc.dr1.att).abs() > 1e-6);
}

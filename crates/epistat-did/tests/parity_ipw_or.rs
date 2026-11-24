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

use epistat_did::{
    DrDidConfig, DrDidObservation, DrDidRepeatedObservation, estimate_drdid_repeated_cross_section,
    estimate_ipw_did_panel, estimate_ipw_did_repeated_cross_section, estimate_or_did_panel,
    estimate_or_did_repeated_cross_section,
};
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

fn config() -> DrDidConfig {
    DrDidConfig::builder().ridge(0.0).build()
}

fn check(label: &str, fit: &epistat_did::DrDidEstimate, reference: &Fit) {
    assert!(
        (fit.att - reference.att).abs() < ESTIMATE_TOLERANCE,
        "{label} att {} vs R {}",
        fit.att,
        reference.att
    );
    assert!(
        (fit.se - reference.se).abs() < ESTIMATE_TOLERANCE,
        "{label} se {} vs R {}",
        fit.se,
        reference.se
    );
    assert_eq!(fit.influence_function.len(), reference.inffunc.len());
    let worst = fit
        .influence_function
        .iter()
        .zip(&reference.inffunc)
        .map(|(ours, theirs)| (ours - theirs).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        worst < INFLUENCE_TOLERANCE,
        "{label} influence off by {worst:e}"
    );
}

fn panel_rows(panel: &Panel) -> Vec<DrDidObservation> {
    (0..panel.d.len())
        .map(|i| DrDidObservation {
            treated: panel.d[i] > 0.5,
            delta_outcome: panel.y1[i] - panel.y0[i],
            weight: panel.w[i],
            covariates: vec![1.0, panel.x1[i], panel.x2[i]],
        })
        .collect()
}

fn repeated_rows(rc: &Repeated) -> Vec<DrDidRepeatedObservation> {
    (0..rc.d.len())
        .map(|i| DrDidRepeatedObservation {
            treated: rc.d[i] > 0.5,
            post_period: rc.post[i] > 0.5,
            outcome: rc.y[i],
            weight: rc.w[i],
            covariates: vec![rc.x1[i], rc.x2[i]],
        })
        .collect()
}

#[test]
fn panel_ipw_matches_std_ipw_did_panel() {
    let reference: Reference = load_json("tests/data/ipw_or_two_period_ref.json");
    let fit = estimate_ipw_did_panel(&panel_rows(&reference.panel), config()).expect("fit");
    check("panel ipw", &fit, &reference.panel.ipw);
}

#[test]
fn panel_or_matches_reg_did_panel() {
    let reference: Reference = load_json("tests/data/ipw_or_two_period_ref.json");
    let fit = estimate_or_did_panel(&panel_rows(&reference.panel), config()).expect("fit");
    check("panel reg", &fit, &reference.panel.reg);
}

#[test]
fn repeated_ipw_matches_std_ipw_did_rc() {
    let reference: Reference = load_json("tests/data/ipw_or_two_period_ref.json");
    let fit = estimate_ipw_did_repeated_cross_section(&repeated_rows(&reference.rc), config())
        .expect("fit");
    check("rc ipw", &fit, &reference.rc.ipw);
}

#[test]
fn repeated_or_matches_reg_did_rc() {
    let reference: Reference = load_json("tests/data/ipw_or_two_period_ref.json");
    let fit = estimate_or_did_repeated_cross_section(&repeated_rows(&reference.rc), config())
        .expect("fit");
    check("rc reg", &fit, &reference.rc.reg);
}

/// The interval is `att +- z se` unless a bootstrap is asked for.
#[test]
fn the_interval_is_analytic_by_default() {
    let reference: Reference = load_json("tests/data/ipw_or_two_period_ref.json");
    let fit = estimate_ipw_did_panel(&panel_rows(&reference.panel), config()).expect("fit");
    let z = 1.959_963_984_540_054;
    assert!((fit.ci_low - (fit.att - z * fit.se)).abs() < 1e-12);
    assert!((fit.ci_high - (fit.att + z * fit.se)).abs() < 1e-12);
    let boot = estimate_ipw_did_panel(
        &panel_rows(&reference.panel),
        DrDidConfig::builder()
            .ridge(0.0)
            .bootstrap_reps(199)
            .build(),
    )
    .expect("fit");
    assert_eq!(boot.att.to_bits(), fit.att.to_bits());
    assert_ne!(boot.ci_low.to_bits(), fit.ci_low.to_bits());
}

/// The traditional doubly robust estimator with sampling weights,
/// `DRDID::drdid_rc1`.
#[test]
fn repeated_dr_matches_drdid_rc1_with_weights() {
    let reference: Reference = load_json("tests/data/ipw_or_two_period_ref.json");
    let fit = estimate_drdid_repeated_cross_section(&repeated_rows(&reference.rc), config())
        .expect("fit");
    check("rc dr1", &fit, &reference.rc.dr1);
}

/// The legacy computation is still reached, and is a different estimator on
/// weighted data: its propensity score ignores the weights and its influence
/// function lacks the nuisance terms.
#[test]
fn the_legacy_repeated_dr_is_kept_as_an_option() {
    let reference: Reference = load_json("tests/data/ipw_or_two_period_ref.json");
    let rows = repeated_rows(&reference.rc);
    let legacy = estimate_drdid_repeated_cross_section(
        &rows,
        DrDidConfig::builder()
            .ridge(0.0)
            .legacy_repeated_cross_section(true)
            .build(),
    )
    .expect("fit");
    assert!((legacy.att - reference.rc.dr1.att).abs() > 1e-6);
}

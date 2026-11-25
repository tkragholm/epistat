//! The Goodman-Bacon decomposition checked against `bacondecomp::bacon`
//! 0.1.1 and the static TWFE coefficient of `fixest::feols` 0.14.2
//! (`tests/data/generate_bacon_reference.R`).

use epistat_did::{BaconKind, BaconPanel, bacon_decomposition};
use epistat_linalg::Mat;
use epistat_testkit::{assert_same_on_threads, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Table {
    treated: Vec<i32>,
    untreated: Vec<Option<i32>>,
    estimate: Vec<f64>,
    weight: Vec<f64>,
    #[serde(rename = "type")]
    kind: Vec<String>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    unit: Vec<u32>,
    time: Vec<i32>,
    treated: Vec<u8>,
    outcome: Vec<f64>,
    x1: Vec<f64>,
    x2: Vec<f64>,
    plain: Table,
    controlled: Table,
    omega: f64,
    beta_within: f64,
    twfe: f64,
    twfe_controlled: f64,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/bacon_ref.json"
    ))
}

const EPS: f64 = f64::EPSILON;

fn label(kind: BaconKind) -> &'static str {
    match kind {
        BaconKind::TreatedVsUntreated => "Treated vs Untreated",
        BaconKind::EarlierVsLaterTreated => "Earlier vs Later Treated",
        BaconKind::LaterVsEarlierTreated => "Later vs Earlier Treated",
        BaconKind::LaterVsAlwaysTreated => "Later vs Always Treated",
        BaconKind::BothTreated => "Both Treated",
    }
}

fn treated(case: &Case) -> Vec<bool> {
    case.treated.iter().map(|&t| t == 1).collect()
}

fn controls(case: &Case) -> Mat<f64> {
    Mat::from_fn(case.unit.len(), 2, |i, j| {
        if j == 0 { case.x1[i] } else { case.x2[i] }
    })
}

/// The rounding allowed to a quantity that passes through `fits` least-squares
/// fits with unit and period dummies: each, by Householder QR in R, carries an
/// error of order n m eps relative to its inputs (Higham, Accuracy and
/// Stability, thm 20.3), with n rows and m dummies; the projections here are
/// exact after one sweep on a balanced panel and add only their rounding.
fn rounding(case: &Case, fits: f64) -> f64 {
    let units = case.unit.iter().max().unwrap() + 1;
    let periods = case
        .time
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    fits * (case.unit.len() * (units as usize + periods)) as f64 * EPS
}

#[test]
fn the_decomposition_matches_bacondecomp() {
    for case in &reference().cases {
        let d = treated(case);
        let x = controls(case);
        for (with_controls, table) in [(false, &case.plain), (true, &case.controlled)] {
            let panel = BaconPanel {
                unit: &case.unit,
                time: &case.time,
                treated: &d,
                outcome: &case.outcome,
                controls: with_controls.then(|| x.as_ref()),
            };
            let out = bacon_decomposition(&panel).unwrap();
            assert_eq!(out.comparisons.len(), table.treated.len(), "{}", case.name);
            // Plain: one fit per estimate; with covariates, four chained fits
            // (the treatment on the covariates, D~ on the group means, B_D
            // and B_b). Weights without covariates are closed forms of
            // counts.
            let tol = rounding(case, if with_controls { 4.0 } else { 1.0 });
            let tol_weight = if with_controls { tol } else { 64.0 * EPS };
            for (k, c) in out.comparisons.iter().enumerate() {
                let tag = format!(
                    "{} {with_controls} {} {:?}",
                    case.name, c.treated, c.untreated
                );
                assert_eq!(c.treated, table.treated[k], "{tag}");
                assert_eq!(c.untreated, table.untreated[k], "{tag}");
                assert_eq!(label(c.kind), table.kind[k], "{tag}");
                let Some(e) = c.estimate else {
                    panic!("{tag}: {c:?}")
                };
                assert!(
                    (e - table.estimate[k]).abs() <= tol * table.estimate[k].abs().max(1.0),
                    "{tag} estimate {e} {}",
                    table.estimate[k]
                );
                assert!(
                    (c.weight - table.weight[k]).abs() <= tol_weight,
                    "{tag} weight {} {}",
                    c.weight,
                    table.weight[k]
                );
            }
            // The weighted mean of the comparisons is the TWFE coefficient
            // (theorem 1 of Goodman-Bacon), and with covariates
            // Omega beta_w + (1 - Omega) times it; both hold in the
            // references to rounding.
            let between: f64 = out
                .comparisons
                .iter()
                .map(|c| c.weight * c.estimate.unwrap())
                .sum();
            match out.within {
                None => {
                    assert!(!with_controls);
                    assert!(
                        (between - case.twfe).abs() <= tol * case.twfe.abs().max(1.0),
                        "{}",
                        case.name
                    );
                }
                Some(w) => {
                    assert!((w.omega - case.omega).abs() <= tol, "{} omega", case.name);
                    assert!(
                        (w.estimate - case.beta_within).abs()
                            <= tol * case.beta_within.abs().max(1.0),
                        "{} beta_w",
                        case.name
                    );
                    let total = w.omega * w.estimate + (1.0 - w.omega) * between;
                    assert!(
                        (total - case.twfe_controlled).abs()
                            <= tol * case.twfe_controlled.abs().max(1.0),
                        "{} total {total} {}",
                        case.name,
                        case.twfe_controlled
                    );
                }
            }
        }
    }
}

#[test]
fn an_unbalanced_panel_or_a_treatment_that_stops_is_an_error() {
    let reference = reference();
    let case = &reference.cases[0];
    let d = treated(case);
    let n = case.unit.len() - 1;
    let short = BaconPanel {
        unit: &case.unit[..n],
        time: &case.time[..n],
        treated: &d[..n],
        outcome: &case.outcome[..n],
        controls: None,
    };
    assert!(bacon_decomposition(&short).is_err());
    let mut stops = d.clone();
    let row = (0..stops.len())
        .find(|&i| {
            stops[i] && i + 1 < stops.len() && stops[i + 1] && case.unit[i] == case.unit[i + 1]
        })
        .unwrap();
    stops[row + 1] = false;
    let panel = BaconPanel {
        unit: &case.unit,
        time: &case.time,
        treated: &stops,
        outcome: &case.outcome,
        controls: None,
    };
    assert!(bacon_decomposition(&panel).is_err());
}

#[test]
fn the_decomposition_does_not_depend_on_the_number_of_threads() {
    let reference = reference();
    let case = &reference.cases[0];
    let d = treated(case);
    let x = controls(case);
    assert_same_on_threads(|| {
        let panel = BaconPanel {
            unit: &case.unit,
            time: &case.time,
            treated: &d,
            outcome: &case.outcome,
            controls: Some(x.as_ref()),
        };
        let out = bacon_decomposition(&panel).unwrap();
        out.comparisons
            .iter()
            .flat_map(|c| [c.estimate.unwrap(), c.weight])
            .collect()
    });
}

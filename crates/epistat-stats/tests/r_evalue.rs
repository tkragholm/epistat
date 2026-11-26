//! E-values of a risk difference and for selection, against `EValue` 4.1.4
//! (`tests/data/evalue.R`).
//!
//! * `evalues.RD`: the E-value of the estimate comes from the same quadratic
//!   and square root, so it agrees to 1e-12 relative. For the lower limit,
//!   `evalues.RD` takes the first point of a grid of step `1e-4` in the bias
//!   factor `B`, and ours is the crossing itself, so R's `B` lies in `[ours,
//!   ours + 1e-4]`. R reports the E-value `E`, and `B` is recovered from it as
//!   the bounding factor of `(E, E)`, `E^2 / (2 E - 1)`, to a few units in the
//!   last place, allowed as 1e-12.
//! * `svalues.RR`: a square root or two of the same ratio, to 1e-14 relative.

use epistat_stats::bias::Table2x2;
use epistat_stats::evalue::{SelectionBias, bounding_factor, evalues_rd, selection_evalues_rr};
use epistat_testkit::load_json;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct Reference {
    versions: HashMap<String, String>,
    rd: Vec<Rd>,
    selection: Vec<Sv>,
}

#[derive(Deserialize)]
struct Rd {
    table: [f64; 4],
    #[serde(rename = "true")]
    true_value: f64,
    alpha: f64,
    grid: f64,
    estimate: f64,
    lower: f64,
}

#[derive(Deserialize)]
struct Sv {
    assumption: String,
    est: f64,
    lo: Option<f64>,
    hi: Option<f64>,
    #[serde(rename = "true")]
    true_value: f64,
    point: Option<f64>,
    lower: Option<f64>,
    upper: Option<f64>,
}

fn reference() -> Reference {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/evalue.json"
    ));
    assert!(r.versions["r"].starts_with("R version 4.6"));
    assert_eq!(r.versions["evalue"], "4.1.4");
    r
}

#[test]
fn risk_difference_evalues_agree_with_evalues_rd() {
    let r = reference();
    for case in &r.rd {
        let [a, b, c, d] = case.table;
        let e = evalues_rd(&Table2x2 { a, b, c, d }, case.true_value, case.alpha).unwrap();
        assert!(
            (e.estimate - case.estimate).abs() <= 1e-12 * case.estimate,
            "{:?}: {} against {}",
            case.table,
            e.estimate,
            case.estimate
        );
        if case.lower.to_bits() == 1.0_f64.to_bits() {
            assert_eq!(e.lower, 1.0, "{:?}", case.table);
            assert_eq!(e.bias_factor_lower, 1.0);
            continue;
        }
        let from_r = bounding_factor(case.lower, case.lower).unwrap();
        assert!(
            e.bias_factor_lower - 1e-12 <= from_r && from_r <= e.bias_factor_lower + case.grid,
            "{:?}: {} against {from_r}",
            case.table,
            e.bias_factor_lower
        );
        // The E-value is the threshold of the bias factor.
        let b = e.bias_factor_lower;
        assert!((e.lower - (b + (b * (b - 1.0)).sqrt())).abs() <= 1e-14 * e.lower);
    }
}

#[test]
fn selection_evalues_agree_with_svalues_rr() {
    let r = reference();
    let mut seen = HashMap::new();
    for case in &r.selection {
        let assumption = match case.assumption.as_str() {
            "general" => SelectionBias::General,
            "selected" => SelectionBias::SelectedPopulation,
            "increased" => SelectionBias::IncreasedRisk,
            "decreased" => SelectionBias::DecreasedRisk,
            "s_eq_u" => SelectionBias::SelectionIsTheFactor,
            "s_eq_u_increased" => SelectionBias::SelectionIsTheFactorIncreasedRisk,
            "s_eq_u_decreased" => SelectionBias::SelectionIsTheFactorDecreasedRisk,
            other => panic!("{other}"),
        };
        let e =
            selection_evalues_rr(case.est, case.lo, case.hi, case.true_value, assumption).unwrap();
        let same = |got: Option<f64>, want: Option<f64>, what: &str| match (got, want) {
            (Some(g), Some(w)) => assert!(
                (g - w).abs() <= 1e-14 * w,
                "{} {} {what}: {g} against {w}",
                case.assumption,
                case.est
            ),
            (None, None) => {}
            _ => panic!(
                "{} {} {what}: {got:?} against {want:?}",
                case.assumption, case.est
            ),
        };
        same(Some(e.estimate), case.point, "estimate");
        same(e.lower, case.lower, "lower");
        same(e.upper, case.upper, "upper");
        *seen.entry(case.assumption.clone()).or_insert(0) += 1;
    }
    assert_eq!(seen.len(), 7);
}

//! Pair matching under the ATC checked against R 4.6's `MatchIt` 4.8.2 and
//! cobalt 5.1.0.
//!
//! The reference is `tests/data/matchit_atc.json`, written by
//! `tests/data/matchit_atc.R`: nearest neighbour matching 1:2 with replacement
//! on `lalonde`, and nearest 1:1 and 1:2 and optimal 1:2 without replacement
//! on all its treated units and 80 controls, every run with
//! `estimand = "ATC"` on a propensity score given as a numeric distance.
//! Every control gets its full ratio, so the realised estimand is the ATC.
//!
//! Under the ATC the control group is focal, so the controls are passed to
//! [`run_spec`] as the cases, as the pair matchers read their first slice.
//! What the tests check, on every run:
//!
//! * The matched sets: each focal control's treated units, in order of
//!   assignment for the nearest method and as a set for the optimal one,
//!   equal `match.matrix`. The optimal run is given a matrix of absolute
//!   score differences with noise below 1e-4, whose optimum is unique (the
//!   script checks that a further perturbation keeps R's sets), since
//!   distances on a line have many optima.
//! * The weights of [`MatchWeightMethod::for_estimand`] for the outcome's
//!   realised estimand equal `matchit(normalize = FALSE)`'s, and their
//!   normalised form `matchit()`'s: every focal control weighs 1 and a treated
//!   unit the sum of `1/m` over the controls it serves.
//! * The standardised mean differences of [`balance_report`] before and
//!   after matching equal those of `summary.matchit()` and of `bal.tab()`,
//!   treated minus control over the standard deviation of the controls
//!   before matching. The report states the difference of its first slice
//!   minus its second, so under the ATC, whose first slice is the controls,
//!   it is R's difference with the sign changed.
//!
//! The weights are sums of a few reciprocals of 1 and 2 and are compared to
//! 1e-12. The differences are means and standard deviations of up to 614
//! values and are compared to 1e-10 relative.

#[path = "common/r_reference.rs"]
mod r_reference;

use epistat_core::Estimand;
use epistat_matching::spec::{DistanceSpec, MatchMethod};
use epistat_matching::{
    BalanceRecord, BalanceReportOptions, CovariateValue, MatchSpec, MatchWeightMethod, SpecInput,
    SpecOutput, balance_report, run_spec,
};
use epistat_testkit::Tolerance;
use r_reference::{Versions, assert_r_version, covariate_unit, reference};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    covariates: Vec<String>,
    data: Data,
    matrix: Matrix,
    runs: Vec<Run>,
}

/// The optimal run's distances, one row per treated unit.
#[derive(Deserialize)]
struct Matrix {
    treated: Vec<String>,
    controls: Vec<String>,
    values: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct Data {
    id: Vec<String>,
    ps: Vec<f64>,
    #[serde(flatten)]
    columns: HashMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct Run {
    name: String,
    method: String,
    ratio: usize,
    replace: bool,
    ids: Vec<String>,
    treat: Vec<u8>,
    sets: Vec<Set>,
    weights_raw: Vec<f64>,
    weights: Vec<f64>,
    summary_smd_all: Vec<f64>,
    summary_smd_matched: Vec<f64>,
    bal_tab_smd_un: Vec<f64>,
    bal_tab_smd_adj: Vec<f64>,
}

#[derive(Deserialize)]
struct Set {
    focal: String,
    matched: Matched,
}

/// R writes a vector of one element as a bare string.
#[derive(Deserialize)]
#[serde(untagged)]
enum Matched {
    Many(Vec<String>),
    One(String),
}

impl Matched {
    fn to_vec(&self) -> Vec<String> {
        match self {
            Self::Many(ids) => ids.clone(),
            Self::One(id) => vec![id.clone()],
        }
    }
}

/// The covariates and the score of every unit of `lalonde`, by id.
struct Units {
    records: HashMap<String, BalanceRecord>,
    scores: BTreeMap<String, f64>,
}

fn units(reference: &Reference) -> Units {
    let data = &reference.data;
    let column = |name: &str| -> Vec<f64> {
        serde_json::from_value(data.columns[name].clone()).expect("a numeric column")
    };
    let columns: Vec<(String, Vec<f64>)> = reference
        .covariates
        .iter()
        .map(|name| (name.clone(), column(name)))
        .collect();
    let records = data
        .id
        .iter()
        .enumerate()
        .map(|(row, id)| {
            let covariates = columns
                .iter()
                .map(|(name, values)| (name.clone(), CovariateValue::Numeric(values[row])));
            (id.clone(), covariate_unit(id.clone(), covariates))
        })
        .collect();
    let scores = data
        .id
        .iter()
        .cloned()
        .zip(data.ps.iter().copied())
        .collect();
    Units { records, scores }
}

/// The spec of a run: the score as the distance of the nearest runs, and the
/// matrix, transposed to one row per focal control, for the optimal run.
fn spec(run: &Run, units: &Units, matrix: &Matrix) -> MatchSpec {
    let (method, distance) = match run.method.as_str() {
        "nearest" => (
            MatchMethod::Nearest,
            DistanceSpec::Scores {
                scores: run
                    .ids
                    .iter()
                    .map(|id| (id.clone(), units.scores[id]))
                    .collect(),
                caliper: None,
            },
        ),
        "optimal" => (
            MatchMethod::Optimal,
            DistanceSpec::Matrix {
                case_ids: matrix.controls.clone(),
                control_ids: matrix.treated.clone(),
                values: (0..matrix.controls.len())
                    .map(|c| matrix.values.iter().map(|row| Some(row[c])).collect())
                    .collect(),
                caliper: None,
            },
        ),
        other => panic!("method {other}"),
    };
    MatchSpec {
        method,
        distance,
        ratio: run.ratio,
        replace: run.replace,
        estimand: Estimand::Atc,
        seed: Some(20_261_128),
        ..MatchSpec::default()
    }
}

#[track_caller]
fn assert_close(label: &str, got: f64, expected: f64, tolerance: Tolerance) {
    assert!(
        tolerance.accepts(got, expected),
        "{label}: {got} against R {expected}"
    );
}

#[test]
fn atc_pairs_weights_and_balance_agree_with_matchit_and_cobalt() {
    let reference: Reference = reference("matchit_atc.json");
    assert_r_version(&reference.versions["r"]);
    assert_eq!(reference.versions["matchit"], "4.8.2");
    assert_eq!(reference.versions["cobalt"], "5.1.0");
    let units = units(&reference);
    assert_eq!(reference.runs.len(), 4);

    for run in &reference.runs {
        let label = run.name.as_str();
        // The focal group first, as the pair matchers read it.
        let mut controls = Vec::new();
        let mut treated = Vec::new();
        for (id, treat) in run.ids.iter().zip(&run.treat) {
            let record = units.records[id].clone();
            if *treat == 0 {
                controls.push(record);
            } else {
                treated.push(record);
            }
        }
        let SpecOutput::Pairs { outcome, .. } = run_spec(
            &spec(run, &units, &reference.matrix),
            SpecInput::Standard {
                cases: &controls,
                controls: &treated,
            },
        )
        .unwrap_or_else(|error| panic!("{label}: {error}")) else {
            panic!("{label}: expected pairs");
        };
        assert_eq!(outcome.diagnostics.realized_estimand, Estimand::Atc);

        // The matched sets.
        let mut ours: BTreeMap<&str, Vec<String>> = BTreeMap::new();
        for pair in &outcome.pairs {
            ours.entry(pair.case_id.as_str())
                .or_default()
                .push(pair.control_id.clone());
        }
        assert_eq!(ours.len(), run.sets.len(), "{label}: matched controls");
        for set in &run.sets {
            let mut expected = set.matched.to_vec();
            let mut got = ours[set.focal.as_str()].clone();
            if run.method == "optimal" {
                expected.sort();
                got.sort();
            }
            assert_eq!(got, expected, "{label}: the set of {}", set.focal);
        }

        // The weights.
        let method = MatchWeightMethod::for_estimand(outcome.diagnostics.realized_estimand)
            .expect("a matched estimand");
        let raw = outcome.match_weights(method);
        let normalized = raw.normalized();
        let exact = Tolerance::exact().abs(1e-12);
        for (row, (id, treat)) in run.ids.iter().zip(&run.treat).enumerate() {
            let weight_of = |set: &epistat_matching::UnitWeightSet| {
                if *treat == 0 {
                    set.case_weight(id)
                } else {
                    set.control_weight(id)
                }
                .unwrap_or(0.0)
            };
            assert_close(
                &format!("{label}: raw weight of {id}"),
                weight_of(&raw),
                run.weights_raw[row],
                exact,
            );
            assert_close(
                &format!("{label}: weight of {id}"),
                weight_of(&normalized),
                run.weights[row],
                exact,
            );
        }

        // The standardised mean differences, first slice minus second.
        let report = balance_report(
            &controls,
            &treated,
            &outcome,
            &BalanceReportOptions::default(),
        )
        .expect("a balance report");
        let close = Tolerance::exact().rel(1e-10);
        for (k, name) in reference.covariates.iter().enumerate() {
            let row = report
                .numeric
                .iter()
                .find(|row| &row.name == name)
                .unwrap_or_else(|| panic!("{label}: no row for {name}"));
            let pre = -row.smd_pre.expect("a difference before matching");
            let post = -row.smd_post.expect("a difference after matching");
            assert_close(
                &format!("{label}: {name} before"),
                pre,
                run.summary_smd_all[k],
                close,
            );
            assert_close(
                &format!("{label}: {name} after"),
                post,
                run.summary_smd_matched[k],
                close,
            );
            assert_close(
                &format!("{label}: {name} before, bal.tab"),
                pre,
                run.bal_tab_smd_un[k],
                close,
            );
            assert_close(
                &format!("{label}: {name} after, bal.tab"),
                post,
                run.bal_tab_smd_adj[k],
                close,
            );
        }
    }
}

//! The efficient ATT(g,t) without covariates and its aggregations, checked
//! against the estimator written out in R from its formulas and against
//! `did::att_gt` (did 2.5.1). See `data/generate_efficient_reference.R`.
//!
//! # The tolerances
//!
//! Every quantity is a sum of at most `N` terms, so by the bound for recursive
//! summation its rounding error is at most `N eps` times the sum of the
//! magnitudes of its terms. Solving for the baseline weights adds `kappa eps`,
//! with `kappa` the condition number of `Omega` that R records. Both sides
//! round, so values agree within `2 (N + kappa) eps` times the magnitude of
//! their terms: the largest outcome change `|Y_t - Y_b|` for estimates, and the
//! largest influence value for influence values and standard errors. The
//! reference is written with 17 significant digits, which read back exactly.
//!
//! `did` fits an intercept-only propensity score and outcome regression. Their
//! estimation terms cancel in the estimate and in the influence function, so
//! `did`'s numbers differ from the weighted means by rounding alone, and the
//! same bound applies.

use epistat_did::{
    AttGtAggregatedEstimate, AttGtAggregationConfig, AttGtAggregationWeighting, AttGtObservation,
    EfficientDidConfig, EfficientDidEstimate, aggregate_efficient_att_gt_by_calendar_time,
    aggregate_efficient_att_gt_by_cohort, aggregate_efficient_att_gt_event_time_with_influence,
    aggregate_efficient_att_gt_overall, estimate_att_gt_efficient,
};
use epistat_testkit::load_json;
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;

#[derive(Deserialize)]
struct Rows {
    id: Vec<i64>,
    g: Vec<i32>,
    t: Vec<i32>,
    y: Vec<f64>,
    w: Vec<f64>,
}

#[derive(Deserialize)]
struct Cell {
    group: i32,
    time: i32,
    att: f64,
    se: f64,
    treated_n: usize,
    control_n: usize,
    total_weight: f64,
    #[serde(deserialize_with = "one_or_many_i32")]
    baselines: Vec<i32>,
    #[serde(deserialize_with = "one_or_many_f64")]
    baseline_att: Vec<f64>,
    #[serde(deserialize_with = "one_or_many_f64")]
    baseline_weight: Vec<f64>,
    kappa: f64,
    influence: Vec<f64>,
}

#[derive(Deserialize)]
struct Aggregate {
    kind: String,
    weighting: String,
    key: i32,
    estimate: f64,
    se: f64,
    se_naive: f64,
    components: usize,
    influence: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct DidCell {
    group: i32,
    time: i32,
    att: f64,
    se: f64,
    influence: Vec<f64>,
}

#[derive(Deserialize)]
struct Did {
    cells: Vec<DidCell>,
    cohort2_equal_estimate: f64,
    cohort2_equal_se: f64,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    rows: Rows,
    ids: Vec<i64>,
    cells: Vec<Cell>,
    aggregates: Vec<Aggregate>,
    did: Option<Did>,
}

#[derive(Deserialize)]
struct Versions {
    did: String,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
}

/// `jsonlite` with `auto_unbox` writes a vector of length one as a scalar.
#[derive(Deserialize)]
#[serde(untagged)]
enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

fn one_or_many_f64<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
    Ok(match OneOrMany::<f64>::deserialize(d)? {
        OneOrMany::One(x) => vec![x],
        OneOrMany::Many(v) => v,
    })
}

fn one_or_many_i32<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<i32>, D::Error> {
    Ok(match OneOrMany::<i32>::deserialize(d)? {
        OneOrMany::One(x) => vec![x],
        OneOrMany::Many(v) => v,
    })
}

fn load() -> Reference {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/efficient_ref.json"
    ));
    assert_eq!(reference.versions.did, "2.5.1");
    reference
}

fn observations(rows: &Rows) -> Vec<AttGtObservation> {
    (0..rows.id.len())
        .map(|i| AttGtObservation {
            unit_id: Some(rows.id[i]),
            first_treated_time: (rows.g[i] != 0).then_some(rows.g[i]),
            time: rows.t[i],
            outcome: rows.y[i],
            weight: rows.w[i],
        })
        .collect()
}

/// The largest `|Y_t - Y_b|` is at most twice the largest `|Y|`.
fn largest_change(rows: &Rows) -> f64 {
    2.0 * rows.y.iter().fold(0.0_f64, |m, y| m.max(y.abs()))
}

fn largest(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn assert_within(label: &str, got: f64, want: f64, bound: f64) {
    assert!(
        (got - want).abs() <= bound,
        "{label}: {got} against {want}, off by {:e} with bound {bound:e}",
        (got - want).abs()
    );
}

fn find(estimates: &[EfficientDidEstimate], group: i32, time: i32) -> &EfficientDidEstimate {
    estimates
        .iter()
        .find(|e| e.group == group && e.time == time)
        .unwrap_or_else(|| panic!("no cell ({group}, {time})"))
}

fn weighting(name: &str) -> AttGtAggregationWeighting {
    match name {
        "equal" => AttGtAggregationWeighting::Equal,
        "treated_count" => AttGtAggregationWeighting::ByTreatedCount,
        "total_weight" => AttGtAggregationWeighting::ByTotalWeight,
        other => panic!("unknown weighting {other}"),
    }
}

fn ours_aggregate(
    estimates: &[EfficientDidEstimate],
    kind: &str,
    key: i32,
    config: AttGtAggregationConfig,
) -> (AttGtAggregatedEstimate, Option<Vec<f64>>) {
    match kind {
        "event_time" => {
            let out = aggregate_efficient_att_gt_event_time_with_influence(estimates, config)
                .expect("event time");
            let index = out
                .estimates
                .iter()
                .position(|e| e.event_time == key)
                .expect("event time present");
            (
                out.estimates[index].summary,
                Some(out.influence_functions[index].clone()),
            )
        }
        "cohort" => (
            aggregate_efficient_att_gt_by_cohort(estimates, config)
                .expect("cohort")
                .into_iter()
                .find(|e| e.group == key)
                .expect("cohort present")
                .summary,
            None,
        ),
        "calendar" => (
            aggregate_efficient_att_gt_by_calendar_time(estimates, config)
                .expect("calendar")
                .into_iter()
                .find(|e| e.time == key)
                .expect("calendar present")
                .summary,
            None,
        ),
        "overall" => (
            aggregate_efficient_att_gt_overall(estimates, config).expect("overall"),
            None,
        ),
        other => panic!("unknown aggregation {other}"),
    }
}

#[test]
fn efficient_cells_and_aggregates_match_r() {
    let reference = load();
    for case in &reference.cases {
        let rows = observations(&case.rows);
        let estimates =
            estimate_att_gt_efficient(&rows, EfficientDidConfig::default()).expect("estimates");
        assert_eq!(estimates.len(), case.cells.len(), "{}", case.name);
        let n = case.ids.len();
        let n_f = n as f64;
        let change = largest_change(&case.rows);
        let kappa = case.cells.iter().fold(1.0_f64, |m, c| m.max(c.kappa));
        let rounding = 2.0 * (n_f + kappa) * EPS;

        let mut largest_psi = 0.0_f64;
        for cell in &case.cells {
            let ours = find(&estimates, cell.group, cell.time);
            let label = format!("{} ({}, {})", case.name, cell.group, cell.time);
            assert_eq!(ours.influence_function.len(), n, "{label}");
            assert_eq!(ours.treated_n, cell.treated_n, "{label}");
            assert_eq!(ours.control_n, cell.control_n, "{label}");
            assert_within(
                &format!("{label} total weight"),
                ours.total_weight,
                cell.total_weight,
                2.0 * n_f * EPS * cell.total_weight,
            );
            let psi = largest(&cell.influence);
            largest_psi = largest_psi.max(psi);
            assert_within(
                &format!("{label} att"),
                ours.att,
                cell.att,
                rounding * change,
            );
            assert_within(
                &format!("{label} se"),
                ours.se,
                cell.se,
                rounding * psi / n_f.sqrt(),
            );
            let baselines = ours
                .baseline_weights
                .iter()
                .map(|b| b.baseline_time)
                .collect::<Vec<_>>();
            assert_eq!(baselines, cell.baselines, "{label}");
            for (j, baseline) in ours.baseline_weights.iter().enumerate() {
                assert_within(
                    &format!("{label} theta_{}", baseline.baseline_time),
                    baseline.att,
                    cell.baseline_att[j],
                    rounding * change,
                );
                assert_within(
                    &format!("{label} a_{}", baseline.baseline_time),
                    baseline.weight,
                    cell.baseline_weight[j],
                    rounding,
                );
            }
            for (i, (got, want)) in ours
                .influence_function
                .iter()
                .zip(&cell.influence)
                .enumerate()
            {
                assert_within(&format!("{label} psi[{i}]"), *got, *want, rounding * psi);
            }
        }

        for aggregate in &case.aggregates {
            let config = AttGtAggregationConfig {
                weighting: weighting(&aggregate.weighting),
                ..AttGtAggregationConfig::default()
            };
            let (summary, influence) =
                ours_aggregate(&estimates, &aggregate.kind, aggregate.key, config);
            let label = format!(
                "{} {} {} {}",
                case.name, aggregate.kind, aggregate.weighting, aggregate.key
            );
            assert_eq!(summary.components, aggregate.components, "{label}");
            assert_within(
                &format!("{label} estimate"),
                summary.estimate,
                aggregate.estimate,
                rounding * change,
            );
            assert_within(
                &format!("{label} se"),
                summary.se,
                aggregate.se,
                rounding * largest_psi / n_f.sqrt(),
            );
            if let (Some(ours), Some(want)) = (influence, &aggregate.influence) {
                for (i, (got, want)) in ours.iter().zip(want).enumerate() {
                    assert_within(
                        &format!("{label} psi[{i}]"),
                        *got,
                        *want,
                        rounding * largest_psi,
                    );
                }
            }
        }

        // The cohort summaries combine cells that share the cohort's baselines,
        // and the variance without their covariance is far from the variance.
        let cohort_two = case
            .aggregates
            .iter()
            .find(|a| a.kind == "cohort" && a.key == 2 && a.weighting == "equal")
            .expect("cohort 2");
        assert!(
            cohort_two.se > 1.5 * cohort_two.se_naive,
            "{}: {} against {}",
            case.name,
            cohort_two.se,
            cohort_two.se_naive
        );

        if let Some(did) = &case.did {
            for cell in &did.cells {
                let ours = find(&estimates, cell.group, cell.time);
                let label = format!("{} did ({}, {})", case.name, cell.group, cell.time);
                let at_previous = ours
                    .baseline_weights
                    .iter()
                    .find(|b| b.baseline_time == cell.group - 1)
                    .expect("baseline g - 1");
                assert_within(&label, at_previous.att, cell.att, rounding * change);
                if ours.baseline_weights.len() == 1 {
                    let psi = largest(&cell.influence);
                    assert_within(
                        &format!("{label} se"),
                        ours.se,
                        cell.se,
                        rounding * psi / n_f.sqrt(),
                    );
                    for (i, (got, want)) in ours
                        .influence_function
                        .iter()
                        .zip(&cell.influence)
                        .enumerate()
                    {
                        assert_within(&format!("{label} psi[{i}]"), *got, *want, rounding * psi);
                    }
                }
            }
            let config = AttGtAggregationConfig {
                weighting: AttGtAggregationWeighting::Equal,
                ..AttGtAggregationConfig::default()
            };
            let cohort = aggregate_efficient_att_gt_by_cohort(&estimates, config)
                .expect("cohort")
                .into_iter()
                .find(|e| e.group == 2)
                .expect("cohort 2")
                .summary;
            assert_within(
                "did cohort 2 estimate",
                cohort.estimate,
                did.cohort2_equal_estimate,
                rounding * change,
            );
            assert_within(
                "did cohort 2 se",
                cohort.se,
                did.cohort2_equal_se,
                rounding * largest_psi / n_f.sqrt(),
            );
        }
    }
}

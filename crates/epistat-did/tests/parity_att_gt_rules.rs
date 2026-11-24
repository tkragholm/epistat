//! Parity of the ATT(g,t) routes with `did::att_gt` 2.5.1 for inverse
//! probability weighting and outcome regression, on panels and repeated
//! cross-sections, and for `att_gt`'s pre-processing: periods with gaps,
//! cohorts treated at the start or after the end, no never-treated units,
//! anticipation, unbalanced panels and time-varying covariates and weights.
//! The references are written by `tests/data/generate_att_gt_rules_reference.R`
//! (`att_gt_rules_ref.json`) and `generate_cs_panel_reference.R` and
//! `generate_reference_data.R` (the varying-base `mpdta` fixtures).
//!
//! Tolerances. The cells are the two-period estimators of
//! `tests/parity_ipw_or.rs`, whose estimates and standard errors agree with
//! `DRDID` to rounding, about 1e-11, on samples of this size; the ridge of
//! the outcome regressions is 0 here, as R has none. The doubly robust panel
//! cells keep the crate's default ridge, `1e-8` on Gram matrices whose
//! smallest eigenvalue is of the order of the cell size (at least 20), so it
//! moves the coefficients by at most `1e-8 / 20 = 5e-10` relative; the
//! estimate, a mean of predictions, moves by as much times the outcome scale,
//! about 1, and `1e-9` covers it. A cell's influence vector is scaled by
//! `n / n_cell`, at most 6 here, and the per-unit values are of order 10, so
//! the same relative error gives an absolute `1e-11 * 6 * 10`; `1e-8` is used
//! for every route, which also covers the doubly robust ridge.

use epistat_did::{
    AggteConfig, AggteType, AttGtConfig, AttGtDrConfig, AttGtDrObservation, AttGtInfluenceOutput,
    BasePeriod, ComparisonGroup, DrDidConfig, InferenceConfig, UnbalancedPanel, aggregate_att_gt,
    estimate_att_gt_dr_efficient_with_influence, estimate_att_gt_dr_panel_with_influence,
    estimate_att_gt_ipw_panel_with_influence, estimate_att_gt_ipw_with_influence,
    estimate_att_gt_or_panel_with_influence, estimate_att_gt_or_with_influence,
};
use epistat_testkit::load_json;
use serde::Deserialize;

const ESTIMATE_TOLERANCE: f64 = 1e-9;
const INFLUENCE_TOLERANCE: f64 = 1e-8;

#[derive(Deserialize)]
struct Row {
    id: Option<i64>,
    period: i32,
    g: i32,
    y: f64,
    x1: f64,
    x2: f64,
    w: f64,
}

#[derive(Deserialize)]
struct Run {
    panel: bool,
    est_method: String,
    control_group: String,
    base_period: String,
    anticipation: i32,
    allow_unbalanced_panel: bool,
    group: Vec<i32>,
    t: Vec<i32>,
    att: Vec<Option<f64>>,
    se: Vec<Option<f64>>,
    n: usize,
    inffunc: Option<Vec<Vec<f64>>>,
    ids: Vec<i64>,
    aggregates: Option<Aggregates>,
}

#[derive(Deserialize)]
struct Aggregate {
    overall_att: f64,
    overall_se: f64,
    egt: Vec<i32>,
    att_egt: Vec<f64>,
    se_egt: Vec<Option<f64>>,
}

#[derive(Deserialize)]
struct Aggregates {
    simple: Aggregate,
    group: Aggregate,
    dynamic: Aggregate,
}

#[derive(Deserialize)]
struct GapCounts {
    treated_first: usize,
    treated_second: usize,
    after_last: usize,
}

#[derive(Deserialize)]
struct UnbalancedCounts {
    missing_units: usize,
}

#[derive(Deserialize)]
struct MpdtaRun {
    group: Vec<i32>,
    t: Vec<i32>,
    att: Vec<Option<f64>>,
    se: Vec<Option<f64>>,
    n: usize,
}

#[derive(Deserialize)]
struct Reference {
    balanced: Vec<Row>,
    methods_panel: Vec<Run>,
    repeated: Vec<Row>,
    methods_rc: Vec<Run>,
    gaps: Vec<Row>,
    gaps_counts: GapCounts,
    gaps_runs: Vec<Run>,
    late: Vec<Row>,
    late_runs: Vec<Run>,
    unbalanced: Vec<Row>,
    unbalanced_counts: UnbalancedCounts,
    unbalanced_runs: Vec<Run>,
    varying: Vec<Row>,
    varying_runs: Vec<Run>,
    mpdta_anticipation: Vec<MpdtaRun>,
}

fn reference() -> Reference {
    load_json("tests/data/att_gt_rules_ref.json")
}

fn observations(rows: &[Row], panel: bool) -> Vec<AttGtDrObservation> {
    rows.iter()
        .map(|row| AttGtDrObservation {
            unit_id: if panel { row.id } else { None },
            // `did` codes never treated as 0.
            first_treated_time: (row.g != 0).then_some(row.g),
            time: row.period,
            outcome: row.y,
            weight: row.w,
            covariates: vec![row.x1, row.x2],
            comparison_cohort: None,
        })
        .collect()
}

fn config(run: &Run, ridge: f64) -> AttGtDrConfig {
    AttGtDrConfig {
        att_gt: AttGtConfig {
            comparison_group: match run.control_group.as_str() {
                "nevertreated" => ComparisonGroup::NeverTreated,
                _ => ComparisonGroup::NotYetTreated,
            },
            base_period: match run.base_period.as_str() {
                "varying" => BasePeriod::Varying,
                _ => BasePeriod::Universal,
            },
            anticipation_periods: run.anticipation,
            unbalanced_panel: if run.allow_unbalanced_panel {
                UnbalancedPanel::RepeatedCrossSection
            } else {
                UnbalancedPanel::Balance
            },
            ..AttGtConfig::default()
        },
        drdid: DrDidConfig::builder().ridge(ridge).build(),
    }
}

fn estimate(rows: &[Row], run: &Run) -> AttGtInfluenceOutput {
    let observations = observations(rows, run.panel);
    let ridge = if run.est_method == "dr" && run.panel {
        DrDidConfig::default().ridge
    } else {
        0.0
    };
    let config = config(run, ridge);
    match (run.panel, run.est_method.as_str()) {
        (true, "dr") => estimate_att_gt_dr_panel_with_influence(&observations, config),
        (true, "ipw") => estimate_att_gt_ipw_panel_with_influence(&observations, config),
        (true, "reg") => estimate_att_gt_or_panel_with_influence(&observations, config),
        (false, "dr") => estimate_att_gt_dr_efficient_with_influence(&observations, config),
        (false, "ipw") => estimate_att_gt_ipw_with_influence(&observations, config),
        (false, "reg") => estimate_att_gt_or_with_influence(&observations, config),
        other => panic!("no route for {other:?}"),
    }
    .unwrap_or_else(|error| panic!("{} {}: {error}", run.est_method, run.base_period))
}

/// Every cell R reports with a standard error, in R's order, against ours;
/// the influence vectors joined on the ids. R's universal-base reference cell
/// has att 0 and no standard error, and is not a cell here.
fn check(label: &str, out: &AttGtInfluenceOutput, run: &Run) {
    let expected: Vec<usize> = (0..run.group.len())
        .filter(|&i| run.se[i].is_some())
        .collect();
    let cells: Vec<(i32, i32)> = out.estimates.iter().map(|e| (e.group, e.time)).collect();
    let want: Vec<(i32, i32)> = expected.iter().map(|&i| (run.group[i], run.t[i])).collect();
    assert_eq!(cells, want, "{label}: cells");
    assert!(!cells.is_empty(), "{label}: no cells");
    for (ours, &i) in out.estimates.iter().zip(&expected) {
        let (att, se) = (run.att[i].expect("att"), run.se[i].expect("se"));
        assert!(
            (ours.att - att).abs() < ESTIMATE_TOLERANCE,
            "{label} att({},{}) {} vs R {att}",
            ours.group,
            ours.time,
            ours.att
        );
        assert!(
            (ours.se - se).abs() < ESTIMATE_TOLERANCE,
            "{label} se({},{}) {} vs R {se}",
            ours.group,
            ours.time,
            ours.se
        );
    }
    assert_eq!(out.sample.ids.len(), run.n, "{label}: n");
    let r_ids: Vec<i64> = run
        .ids
        .iter()
        // R numbers repeated cross-section rows from 1.
        .map(|&id| if run.panel { id } else { id - 1 })
        .collect();
    let mut sorted = r_ids.clone();
    sorted.sort_unstable();
    let mut ours_sorted = out.sample.ids.clone();
    ours_sorted.sort_unstable();
    assert_eq!(ours_sorted, sorted, "{label}: ids");
    let Some(inffunc) = &run.inffunc else { return };
    let mut worst = 0.0_f64;
    for (r_row, id) in r_ids.iter().enumerate() {
        let position = out
            .sample
            .ids
            .binary_search(id)
            .unwrap_or_else(|_| panic!("{label}: id {id} missing"));
        for (cell, &i) in expected.iter().enumerate() {
            let deviation = (out.influence_functions[cell][position] - inffunc[r_row][i]).abs();
            worst = worst.max(deviation);
        }
    }
    assert!(
        worst < INFLUENCE_TOLERANCE,
        "{label}: influence off by {worst:e}"
    );
}

#[test]
fn panel_ipw_and_or_cells_match_att_gt() {
    let reference = reference();
    for (index, run) in reference.methods_panel.iter().enumerate() {
        let out = estimate(&reference.balanced, run);
        check(&format!("panel {index} {}", run.est_method), &out, run);
    }
}

#[test]
fn repeated_cross_section_cells_match_att_gt() {
    let reference = reference();
    for (index, run) in reference.methods_rc.iter().enumerate() {
        let out = estimate(&reference.repeated, run);
        check(&format!("rc {index} {}", run.est_method), &out, run);
    }
}

/// Periods 1, 2, 4, 5 and 7, so the base periods skip the gaps; cohort 1 is
/// treated in the first period and dropped; cohort 9 is after the last
/// period and recoded as never treated; with one period of anticipation,
/// cohort 2 is dropped as well.
#[test]
fn gaps_and_cohorts_at_the_edges_are_handled_as_att_gt() {
    let reference = reference();
    for (index, run) in reference.gaps_runs.iter().enumerate() {
        let out = estimate(&reference.gaps, run);
        check(&format!("gaps {index}"), &out, run);
        let dropped = reference.gaps_counts.treated_first
            + if run.anticipation > 0 {
                reference.gaps_counts.treated_second
            } else {
                0
            };
        assert_eq!(out.sample.dropped_treated_at_start, dropped);
        assert_eq!(
            out.sample.recoded_never_treated,
            reference.gaps_counts.after_last
        );
        assert_eq!(out.sample.periods, vec![1, 2, 4, 5, 7]);
    }
}

/// `aggte(max_e = 1)` on the gapped panel. `did` counts `max_e` in steps of
/// its list of periods and cohorts for the simple and group summaries, and in
/// calendar periods for the dynamic one; the cells and the sample's unit panel
/// of the pre-processed fit give the same summaries. R's universal-base path
/// also carries the base cells, att 0 without a standard error, which are not
/// cells here.
#[test]
fn aggregates_with_max_e_on_gapped_periods_match_aggte() {
    let reference = reference();
    for (index, run) in reference.gaps_runs.iter().enumerate() {
        let Some(aggregates) = &run.aggregates else {
            continue;
        };
        let out = estimate(&reference.gaps, run);
        let panel = out.sample.unit_panel();
        for (aggregation, want) in [
            (AggteType::Simple, &aggregates.simple),
            (AggteType::Group, &aggregates.group),
            (AggteType::Dynamic, &aggregates.dynamic),
        ] {
            let got = aggregate_att_gt(
                &out.estimates,
                &out.influence_functions,
                &panel,
                AggteConfig {
                    aggregation,
                    balance_e: None,
                    min_e: None,
                    max_e: Some(1),
                    confidence_level: InferenceConfig::default(),
                    bootstrap: None,
                },
            )
            .expect("aggte");
            let label = format!("gaps {index} {aggregation:?}");
            assert!(
                (got.overall_att - want.overall_att).abs() < ESTIMATE_TOLERANCE,
                "{label}: att {} vs R {}",
                got.overall_att,
                want.overall_att
            );
            assert!(
                (got.overall_se - want.overall_se).abs() < ESTIMATE_TOLERANCE,
                "{label}: se {} vs R {}",
                got.overall_se,
                want.overall_se
            );
            let points: Vec<usize> = (0..want.egt.len())
                .filter(|&i| want.se_egt[i].is_some())
                .collect();
            let keys: Vec<i32> = got.by_key.iter().map(|point| point.key).collect();
            if aggregation != AggteType::Simple {
                assert_eq!(
                    keys,
                    points.iter().map(|&i| want.egt[i]).collect::<Vec<_>>(),
                    "{label}: keys"
                );
            }
            for (point, &i) in got.by_key.iter().zip(&points) {
                assert!(
                    (point.att - want.att_egt[i]).abs() < ESTIMATE_TOLERANCE,
                    "{label}"
                );
                let se = want.se_egt[i].expect("se");
                assert!((point.se - se).abs() < ESTIMATE_TOLERANCE, "{label}");
            }
        }
    }
}

/// Cohorts 3, 4 and 5 and no never-treated unit: the periods from 5 on (from
/// 4 on with anticipation) are dropped and cohort 5 is the comparison group.
#[test]
fn without_never_treated_units_the_last_cohort_is_the_comparison() {
    let reference = reference();
    for (index, run) in reference.late_runs.iter().enumerate() {
        let out = estimate(&reference.late, run);
        check(&format!("late {index}"), &out, run);
        assert_eq!(out.sample.last_cohort_as_comparison, Some(5));
        assert_eq!(out.sample.cohorts, vec![3, 4]);
        let last = if run.anticipation > 0 { 3 } else { 4 };
        assert_eq!(out.sample.periods, (1..=last).collect::<Vec<_>>());
        let never = out.sample.groups.iter().filter(|g| g.is_none()).count();
        if run.control_group == "nevertreated" {
            assert!(never > 0, "cohort 5 is recoded as never treated");
        } else {
            assert_eq!(never, 0);
        }
    }
}

/// Thirty units each miss one period. Balanced by dropping them, as R's
/// default; or kept, each cell estimated as repeated cross-sections with the
/// unit's rows summed in the influence function.
#[test]
fn unbalanced_panels_are_balanced_or_estimated_as_repeated_cross_sections() {
    let reference = reference();
    for (index, run) in reference.unbalanced_runs.iter().enumerate() {
        let out = estimate(&reference.unbalanced, run);
        check(&format!("unbalanced {index} {}", run.est_method), &out, run);
        if run.allow_unbalanced_panel {
            assert!(out.sample.unbalanced_as_repeated_cross_section);
            assert_eq!(out.sample.dropped_unbalanced, 0);
        } else {
            assert_eq!(
                out.sample.dropped_unbalanced,
                reference.unbalanced_counts.missing_units
            );
        }
    }
}

/// A covariate and a weight that change over time. `did` reads both from
/// the earlier of a cell's two periods: the base period, except in a
/// pre-treatment cell under a universal base, whose base is the later one.
#[test]
fn time_varying_covariates_and_weights_come_from_the_earlier_period() {
    let reference = reference();
    for (index, run) in reference.varying_runs.iter().enumerate() {
        let out = estimate(&reference.varying, run);
        check(&format!("varying {index}"), &out, run);
    }
}

/// The same balanced panel under [`UnbalancedPanel::PerCell`] and the
/// default: nothing to drop, so the same numbers to the bit.
#[test]
fn a_balanced_panel_gives_the_same_bits_under_every_unbalanced_rule() {
    let reference = reference();
    let run = &reference.methods_panel[0];
    let observations = observations(&reference.balanced, true);
    let base = config(run, 0.0);
    let fits: Vec<AttGtInfluenceOutput> = [
        UnbalancedPanel::Balance,
        UnbalancedPanel::RepeatedCrossSection,
        UnbalancedPanel::PerCell,
    ]
    .into_iter()
    .map(|unbalanced_panel| {
        let config = AttGtDrConfig {
            att_gt: AttGtConfig {
                unbalanced_panel,
                ..base.att_gt
            },
            ..base
        };
        estimate_att_gt_ipw_panel_with_influence(&observations, config).expect("fit")
    })
    .collect();
    assert_eq!(fits[0], fits[1]);
    assert_eq!(fits[0], fits[2]);
}

#[derive(Deserialize)]
struct MpdtaRow {
    year: i32,
    countyreal: i64,
    lemp: f64,
    #[serde(rename = "first.treat")]
    first_treat: i32,
    lpop: f64,
}

#[derive(Deserialize)]
struct MpdtaData {
    data_subset: Vec<MpdtaRow>,
}

fn mpdta() -> Vec<AttGtDrObservation> {
    let data: MpdtaData = load_json("tests/data/did_attgt_dr_ref.json");
    data.data_subset
        .iter()
        .map(|row| AttGtDrObservation {
            unit_id: Some(row.countyreal),
            first_treated_time: (row.first_treat != 0).then_some(row.first_treat),
            time: row.year,
            outcome: row.lemp,
            weight: 1.0,
            covariates: vec![row.lpop],
            comparison_cohort: None,
        })
        .collect()
}

fn mpdta_config(
    comparison_group: ComparisonGroup,
    base_period: BasePeriod,
    anticipation_periods: i32,
) -> AttGtDrConfig {
    AttGtDrConfig {
        att_gt: AttGtConfig {
            comparison_group,
            base_period,
            anticipation_periods,
            ..AttGtConfig::default()
        },
        drdid: DrDidConfig::default(),
    }
}

fn check_mpdta(label: &str, out: &AttGtInfluenceOutput, run: &MpdtaRun) {
    let expected: Vec<usize> = (0..run.group.len())
        .filter(|&i| run.se[i].is_some())
        .collect();
    let cells: Vec<(i32, i32)> = out.estimates.iter().map(|e| (e.group, e.time)).collect();
    let want: Vec<(i32, i32)> = expected.iter().map(|&i| (run.group[i], run.t[i])).collect();
    assert_eq!(cells, want, "{label}: cells");
    for (ours, &i) in out.estimates.iter().zip(&expected) {
        assert!(
            (ours.att - run.att[i].expect("att")).abs() < ESTIMATE_TOLERANCE,
            "{label}"
        );
        assert!(
            (ours.se - run.se[i].expect("se")).abs() < ESTIMATE_TOLERANCE,
            "{label}"
        );
    }
    assert_eq!(out.sample.ids.len(), run.n, "{label}: n");
}

/// One period of anticipation on `mpdta`: the 2004 cohort, treated within
/// one period of 2003, is dropped (R: "Dropped 20 units"), and the bases move
/// back a period.
#[test]
fn anticipation_matches_att_gt_on_mpdta() {
    let reference = reference();
    let rows = mpdta();
    let runs = [
        (ComparisonGroup::NeverTreated, BasePeriod::Varying),
        (ComparisonGroup::NotYetTreated, BasePeriod::Universal),
    ];
    for ((comparison, base), run) in runs.into_iter().zip(&reference.mpdta_anticipation) {
        let out = estimate_att_gt_dr_panel_with_influence(&rows, mpdta_config(comparison, base, 1))
            .expect("fit");
        check_mpdta("mpdta anticipation", &out, run);
        assert_eq!(out.sample.dropped_treated_at_start, 20);
    }
}

#[derive(Deserialize)]
struct VaryingRef {
    group: Vec<i32>,
    t: Vec<i32>,
    att: Vec<f64>,
    se: Vec<f64>,
    n_units: usize,
    inffunc: Vec<Vec<f64>>,
    inffunc_unit_order: Vec<i64>,
}

#[derive(Deserialize)]
struct DefaultRef {
    group: Vec<i32>,
    t: Vec<i32>,
    att: Vec<f64>,
    se: Vec<f64>,
}

/// `att_gt`'s default base period on `mpdta`, every cell, from the two
/// fixtures that hold it: `cs_panel_dr_varying_ref.json` with the influence
/// matrix, and `did_attgt_dr_ref.json`, written with 15 significant digits,
/// so its tolerance adds `1e-14` relative on values below 1.
#[test]
fn the_varying_base_matches_att_gt_on_mpdta() {
    let rows = mpdta();
    let out = estimate_att_gt_dr_panel_with_influence(
        &rows,
        mpdta_config(ComparisonGroup::NeverTreated, BasePeriod::Varying, 0),
    )
    .expect("fit");

    let varying: VaryingRef = load_json("tests/data/cs_panel_dr_varying_ref.json");
    let cells: Vec<(i32, i32)> = out.estimates.iter().map(|e| (e.group, e.time)).collect();
    let want: Vec<(i32, i32)> = varying
        .group
        .iter()
        .copied()
        .zip(varying.t.iter().copied())
        .collect();
    assert_eq!(cells, want);
    for (ours, (att, se)) in out
        .estimates
        .iter()
        .zip(varying.att.iter().zip(&varying.se))
    {
        assert!((ours.att - att).abs() < ESTIMATE_TOLERANCE);
        assert!((ours.se - se).abs() < ESTIMATE_TOLERANCE);
    }
    assert_eq!(out.sample.ids.len(), varying.n_units);
    let mut worst = 0.0_f64;
    for (r_row, id) in varying.inffunc_unit_order.iter().enumerate() {
        let position = out.sample.ids.binary_search(id).expect("unit");
        for (cell, psi) in out.influence_functions.iter().enumerate() {
            worst = worst.max((psi[position] - varying.inffunc[r_row][cell]).abs());
        }
    }
    assert!(worst < INFLUENCE_TOLERANCE, "influence off by {worst:e}");

    let default: DefaultRef = load_json("tests/data/did_attgt_dr_ref.json");
    let want: Vec<(i32, i32)> = default
        .group
        .iter()
        .copied()
        .zip(default.t.iter().copied())
        .collect();
    assert_eq!(cells, want);
    for (ours, (att, se)) in out
        .estimates
        .iter()
        .zip(default.att.iter().zip(&default.se))
    {
        assert!((ours.att - att).abs() < ESTIMATE_TOLERANCE + 1e-14);
        assert!((ours.se - se).abs() < ESTIMATE_TOLERANCE + 1e-14);
    }
}

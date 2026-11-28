#![expect(
    dead_code,
    reason = "the fixtures carry fields that only some tests read"
)]

use epistat_did::att_gt;
use epistat_did::did_cc::{self, Assumption, DidCcOptions};
use epistat_did::triple_did::{self, TripleDidData, TripleDidOptions};
use epistat_did::two_period::{self, TwoPeriodOptions};
use epistat_did::{StaggeredPanel, TwoPeriodData};
use epistat_linalg::{Mat, MatRef};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct DrDidRef {
    att: f64,
    se: f64,
    y1: Vec<f64>,
    y0: Vec<f64>,
    treat: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    influence: Vec<f64>,
}

#[test]
fn parity_drdid_panel_lalonde() {
    let reference: DrDidRef = load_json("tests/data/drdid_panel_ref.json");

    let n = reference.treat.len();
    let treated: Vec<bool> = reference.treat.iter().map(|&t| t == 1).collect();
    let change: Vec<f64> = reference
        .y1
        .iter()
        .zip(&reference.y0)
        .map(|(post, pre)| post - pre)
        .collect();
    // The reference's design carries the intercept in its first column; the
    // estimator adds its own.
    assert!(
        reference
            .covariates
            .iter()
            .all(|row| row[0].to_bits() == 1.0_f64.to_bits())
    );
    let k = reference.covariates[0].len() - 1;
    let x: Vec<f64> = (1..=k)
        .flat_map(|j| reference.covariates.iter().map(move |row| row[j]))
        .collect();
    let data = TwoPeriodData::panel(&treated, &change)
        .covariates(MatRef::from_column_major_slice(&x, n, k))
        .build()
        .expect("data");
    let fit = two_period::estimate(
        &data,
        &TwoPeriodOptions {
            estimator: two_period::Estimator::DoublyRobust,
            ..TwoPeriodOptions::default()
        },
    )
    .expect("estimate");
    let (att, se) = (fit.estimate.estimate, fit.estimate.se.expect("se"));
    let worst_influence = fit
        .influence
        .column(0)
        .iter()
        .zip(&reference.influence)
        .map(|(ours, theirs)| (ours - theirs).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "DRDID panel: att {:e}, se {:e}, influence {worst_influence:e}",
        (att - reference.att).abs(),
        (se - reference.se).abs()
    );
    // The outcomes are earnings in dollars, so the ATT is about 2,300, its
    // standard error about 1,850 and the influence values up to about 4e5.
    // Both sides stop the propensity score's iteration at a convergence
    // tolerance, which leaves differences of about 1e-10 relative; the bounds
    // allow a hundred times that on each scale.
    assert!((att - reference.att).abs() < 1e-6);
    assert!((se - reference.se).abs() < 1e-6);
    assert!(worst_influence < 1e-3);
}

#[derive(Deserialize)]
struct MpdtaRow {
    year: i32,
    #[serde(rename = "first.treat")]
    first_treat: i32,
    lemp: f64,
    countyreal: i64,
}

#[derive(Deserialize)]
struct DidRef {
    group: Vec<i32>,
    t: Vec<i32>,
    att: Vec<f64>,
    se: Vec<f64>,
    data_subset: Vec<MpdtaRow>,
}

#[test]
fn parity_did_attgt_mpdta() {
    let reference: DidRef = load_json("tests/data/did_attgt_ref.json");

    // `did::att_gt` without covariates: the doubly robust panel estimator on
    // an intercept, the unconditional difference in differences.
    let mut rows = reference.data_subset;
    rows.sort_by_key(|row| (row.countyreal, row.year));
    let unit: Vec<u32> = rows
        .iter()
        .map(|row| u32::try_from(row.countyreal).expect("a county number"))
        .collect();
    let period: Vec<i64> = rows.iter().map(|row| i64::from(row.year)).collect();
    let cohort: Vec<Option<i64>> = rows
        .iter()
        .map(|row| (row.first_treat != 0).then_some(i64::from(row.first_treat)))
        .collect();
    let outcome: Vec<f64> = rows.iter().map(|row| row.lemp).collect();
    let panel = StaggeredPanel::builder(&unit, &period, &cohort, &outcome)
        .build()
        .expect("panel");
    let fit = att_gt::estimate(&panel, &att_gt::AttGtOptions::default()).expect("estimate");
    let results: Vec<(i64, i64, f64, f64)> = fit
        .cells()
        .iter()
        .filter_map(|cell| {
            let estimate = cell.estimate?;
            Some((cell.group, cell.time, estimate.estimate, estimate.se?))
        })
        .collect();

    // Every cell R reports must be among ours and agree; a cell of ours that
    // R does not report fails as well.
    let mut matched = 0;
    for &(group, time, att, se) in &results {
        let idx = reference
            .group
            .iter()
            .zip(reference.t.iter())
            .position(|(&g, &t)| i64::from(g) == group && i64::from(t) == time)
            .unwrap_or_else(|| panic!("cell ({group}, {time}) is not in R's"));
        assert!((att - reference.att[idx]).abs() < 1e-8);
        assert!((se - reference.se[idx]).abs() < 1e-8);
        matched += 1;
    }
    assert_eq!(matched, reference.group.len(), "R's cells without a match");
}

#[derive(Deserialize)]
struct TripleDidDataRow {
    id: i32,
    year: i32,
    partition: i32,
    x1: f64,
    x2: f64,
    treat: i32,
    outcome: f64,
}

#[derive(Deserialize)]
struct TripleDidRef {
    att: f64,
    se: f64,
    data: Vec<TripleDidDataRow>,
}

#[test]
fn parity_triple_did_panel() {
    let reference: TripleDidRef = load_json("tests/data/triple_did_ref.json");

    // The outcome changes of the two-period panel.
    let (mut group, mut partition, mut change, mut x1, mut x2) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for pair in reference.data.chunks(2) {
        let (row_pre, row_post) = (&pair[0], &pair[1]);
        assert_eq!(row_pre.id, row_post.id);
        group.push(row_post.treat == 1);
        partition.push(row_post.partition == 1);
        change.push(row_post.outcome - row_pre.outcome);
        x1.push(row_post.x1);
        x2.push(row_post.x2);
    }
    let x: Vec<f64> = x1.into_iter().chain(x2).collect();
    let data = TripleDidData::builder(&group, &partition, &change)
        .covariates(MatRef::from_column_major_slice(&x, change.len(), 2))
        .build()
        .expect("data");
    let result = triple_did::estimate(&data, &TripleDidOptions::default()).expect("ddd");
    let att = result.estimate.estimate;
    println!(
        "R Triple-DiD ATT: {}, Rust Triple-DiD ATT: {att}",
        reference.att
    );
    assert!((att - reference.att).abs() < 1e-6);
}

#[derive(Deserialize)]
struct WaldRef {
    estimates: Vec<f64>,
    vcov: Vec<Vec<f64>>,
    statistic: f64,
    p_value: f64,
}

#[derive(Deserialize)]
struct DidCcScaffoldRow {
    treated: bool,
    post_period: bool,
    outcome: f64,
    covariates: Vec<f64>,
}

#[derive(Deserialize)]
struct DidCcScaffoldDesign {
    rows: Vec<DidCcScaffoldRow>,
    robust: DidCcEstimateRef,
    stationary: DidCcEstimateRef,
    hausman: DidCcHausmanRef,
}

#[derive(Deserialize)]
struct DidCcEstimateRef {
    att: f64,
    se: f64,
}

#[derive(Deserialize)]
struct DidCcHausmanRef {
    difference: f64,
    difference_se: f64,
    statistic: f64,
    p_value: f64,
}

#[derive(Deserialize)]
struct DidCcScaffoldFixture {
    aligned_constant_effect: DidCcScaffoldDesign,
    composition_shift: DidCcScaffoldDesign,
}

#[test]
fn parity_wald_test() {
    let reference: WaldRef = load_json("tests/data/wald_test_ref.json");

    let k = reference.estimates.len();
    let vcov = Mat::from_fn(k, k, |i, j| reference.vcov[i][j]);
    let test = epistat_inference::wald::wald(&reference.estimates, vcov.as_ref()).expect("wald");
    assert!((test.statistic.unwrap() - reference.statistic).abs() < 1e-10);
    assert!((test.p_value.unwrap() - reference.p_value).abs() < 1e-10);
}

#[test]
fn did_cc_scaffold_fixture_schema_loads() {
    let fixture: DidCcScaffoldFixture = load_json("tests/data/did_cc_ref_scaffold.json");

    assert!(!fixture.aligned_constant_effect.rows.is_empty());
    assert!(!fixture.composition_shift.rows.is_empty());
    assert!(
        fixture
            .aligned_constant_effect
            .rows
            .iter()
            .all(|row| row.outcome.is_finite() && !row.covariates.is_empty())
    );
    assert!(
        fixture
            .composition_shift
            .rows
            .iter()
            .any(|row| row.treated && row.post_period)
    );
}

#[test]
fn parity_did_cc_authored_reference_fixture() {
    let fixture: DidCcScaffoldFixture = load_json("tests/data/did_cc_ref_scaffold.json");

    for design in [&fixture.aligned_constant_effect, &fixture.composition_shift] {
        let rows = &design.rows;
        let treated: Vec<bool> = rows.iter().map(|row| row.treated).collect();
        let post: Vec<bool> = rows.iter().map(|row| row.post_period).collect();
        let outcome: Vec<f64> = rows.iter().map(|row| row.outcome).collect();
        let k = rows[0].covariates.len();
        let x: Vec<f64> = (0..k)
            .flat_map(|j| rows.iter().map(move |row| row.covariates[j]))
            .collect();
        let data = TwoPeriodData::repeated_cross_section(&treated, &post, &outcome)
            .covariates(MatRef::from_column_major_slice(&x, rows.len(), k))
            .build()
            .expect("data");
        let estimate = |assumption| {
            did_cc::estimate(
                &data,
                &DidCcOptions {
                    assumption,
                    ..DidCcOptions::default()
                },
            )
            .expect("did_cc")
            .estimate
        };
        let robust = estimate(Assumption::Robust);
        let stationary = estimate(Assumption::Stationary);
        let hausman = did_cc::test(&data, &DidCcOptions::default()).expect("hausman did_cc");

        assert!((robust.estimate - design.robust.att).abs() < 1e-6);
        assert!((robust.se.expect("se") - design.robust.se).abs() < 1e-6);
        assert!((stationary.estimate - design.stationary.att).abs() < 1e-6);
        assert!((stationary.se.expect("se") - design.stationary.se).abs() < 1e-6);
        assert!((hausman.difference - design.hausman.difference).abs() < 1e-6);
        assert!((hausman.se - design.hausman.difference_se).abs() < 1e-6);
        assert!((hausman.statistic - design.hausman.statistic).abs() < 1e-6);
        assert!((hausman.p_value - design.hausman.p_value).abs() < 1e-6);
    }
}

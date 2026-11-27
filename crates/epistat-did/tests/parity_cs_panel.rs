//! Parity of the panel `ATT(g,t)` route against `did` 2.5.1 and `DRDID` 1.3.0.
//!
//! Fixtures are written by `tests/data/generate_cs_panel_reference.R`, which records
//! every `att_gt` default it overrides. Three of those defaults are the usual
//! source of a spurious failure here and are worth naming again:
//! `base_period = "varying"`, `bstrap = TRUE` (so R's headline SEs are bootstrap,
//! not analytic) and `cband = TRUE` (so its intervals are simultaneous, not
//! pointwise). The fixtures set all three explicitly.

use epistat_testkit::load_json;

#[path = "common/att_gt_rows.rs"]
mod rows;

use epistat_core::{Bootstrap, InputError, Level};
use epistat_did::att_gt::{
    AggteOptions, AggteType, AggteWeights, AttGt, AttGtOptions, BasePeriod, ComparisonGroup,
    Design, Estimator, NuisanceOptions,
};
use epistat_did::{DrDidConfig, DrDidObservation, Error, estimate_drdid_panel};
use rows::{Cell, Obs, Summary};
use serde::Deserialize;

#[derive(Deserialize)]
struct DataRow {
    year: i32,
    countyreal: i64,
    lemp: f64,
    #[serde(rename = "first.treat")]
    first_treat: i32,
    lpop: f64,
}

#[derive(Deserialize)]
struct CsPanelRef {
    group: Vec<i32>,
    t: Vec<i32>,
    att: Vec<f64>,
    /// `null` at the (g, g-1) anchor cell that a universal base period emits.
    se: Vec<Option<f64>>,
    n_units: usize,
    inffunc: Vec<Vec<f64>>,
    /// The ids that `inffunc`'s ROWS correspond to. `did` reorders units by
    /// cohort during pre-processing, so this is not the sorted id list.
    inffunc_unit_order: Vec<i64>,
    data_subset: Vec<DataRow>,
}

#[derive(Deserialize)]
struct DrDidCellRef {
    att: f64,
    se: f64,
    inffunc: Vec<f64>,
    d: Vec<f64>,
    lemp_base: Vec<f64>,
    lemp_post: Vec<f64>,
    lpop: Vec<f64>,
}

fn observations(rows: &[DataRow]) -> Vec<Obs> {
    rows.iter()
        .map(|row| Obs {
            covariates: vec![row.lpop],
            // `did` encodes never-treated as gname == 0, not as a missing value.
            ..Obs::new(
                Some(row.countyreal),
                (row.first_treat != 0).then_some(row.first_treat),
                row.year,
                row.lemp,
            )
        })
        .collect()
}

fn universal_config(comparison_group: ComparisonGroup) -> AttGtOptions {
    AttGtOptions {
        comparison_group,
        base_period: BasePeriod::Universal,
        ..AttGtOptions::default()
    }
}

/// A fit with its cells and influence vectors read out.
struct Out {
    fit: AttGt,
    estimates: Vec<Cell>,
    influence_functions: Vec<Vec<f64>>,
}

fn fit(rows: &[Obs], options: &AttGtOptions) -> Out {
    let fit = rows::fit(rows, options).expect("ATT(g,t)");
    Out {
        estimates: rows::cells(&fit),
        influence_functions: rows::influence(&fit),
        fit,
    }
}

/// `rows` with each unit's cluster from `cluster_of_unit`, aligned to
/// `unit_order`.
fn clustered(rows: &[Obs], unit_order: &[i64], cluster_of_unit: &[i64]) -> Vec<Obs> {
    let of: std::collections::BTreeMap<i64, u32> = unit_order
        .iter()
        .zip(cluster_of_unit)
        .map(|(&id, &c)| (id, u32::try_from(c).expect("a cluster code")))
        .collect();
    rows.iter()
        .map(|row| Obs {
            cluster: Some(of[&row.unit.expect("unit")]),
            ..row.clone()
        })
        .collect()
}

/// The pair estimator alone, on one `(g, t)` cell lifted out of `mpdta` by hand.
///
/// Checked first because if this fails nothing above it can pass, and the failure
/// is then in `drdid_panel` rather than in the `ATT(g,t)` loop around it.
#[test]
fn drdid_panel_matches_r_on_one_lifted_cell() {
    let fixture: DrDidCellRef = load_json("tests/data/drdid_panel_cell_ref.json");

    let rows = (0..fixture.d.len())
        .map(|i| DrDidObservation {
            treated: fixture.d[i] > 0.5,
            delta_outcome: fixture.lemp_post[i] - fixture.lemp_base[i],
            weight: 1.0,
            // estimate_drdid_panel takes a finished design matrix, so the
            // intercept is explicit here. R was given cbind(1, lpop) to match.
            covariates: vec![1.0, fixture.lpop[i]],
        })
        .collect::<Vec<_>>();

    let fit = estimate_drdid_panel(&rows, DrDidConfig::builder().build()).expect("panel fit");

    assert!(
        (fit.att - fixture.att).abs() < 1e-10,
        "att {} vs R {}",
        fit.att,
        fixture.att
    );
    assert!(
        (fit.se - fixture.se).abs() < 1e-10,
        "se {} vs R {}",
        fit.se,
        fixture.se
    );
    assert_eq!(fit.influence_function.len(), fixture.inffunc.len());
    for (ours, theirs) in fit.influence_function.iter().zip(&fixture.inffunc) {
        assert!((ours - theirs).abs() < 1e-9, "influence {ours} vs {theirs}");
    }
}

/// The full grid, universal base period, never-treated comparison group.
#[test]
fn panel_att_gt_matches_r_universal_never_treated() {
    let fixture: CsPanelRef = load_json("tests/data/cs_panel_dr_universal_ref.json");

    let out = fit(
        &observations(&fixture.data_subset),
        &universal_config(ComparisonGroup::NeverTreated),
    );

    // R emits the anchor cell (g, g-1) with att = 0 and se = NA under a universal
    // base period. We skip it rather than emit a degenerate estimate, so the
    // comparison is over R's non-anchor cells only.
    let expected = (0..fixture.att.len())
        .filter(|&i| fixture.se[i].is_some())
        .map(|i| {
            (
                fixture.group[i],
                fixture.t[i],
                fixture.att[i],
                fixture.se[i].unwrap_or_default(),
            )
        })
        .collect::<Vec<_>>();

    assert_eq!(
        out.estimates.len(),
        expected.len(),
        "cell count: rust {:?} vs R {:?}",
        out.estimates
            .iter()
            .map(|e| (e.group, e.time))
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|(g, t, _, _)| (*g, *t))
            .collect::<Vec<_>>()
    );

    for (ours, (group, time, att, se)) in out.estimates.iter().zip(&expected) {
        assert_eq!((ours.group, ours.time), (*group, *time));
        assert!(
            (ours.att - att).abs() < 1e-10,
            "att({group},{time}) {} vs R {att}",
            ours.att
        );
        assert!(
            (ours.se - se).abs() < 1e-10,
            "se({group},{time}) {} vs R {se}",
            ours.se
        );
    }

    // The influence matrix is the thing everything downstream is a function of,
    // so it is compared element for element rather than trusted because the SEs
    // agreed. R's inffunc is n_units x n_cells in the fixture's row-major form.
    assert_eq!(fixture.inffunc_unit_order.len(), fixture.n_units);
    for (cell, ours) in out.influence_functions.iter().enumerate() {
        assert_eq!(
            ours.len(),
            fixture.n_units,
            "influence length for cell {cell}"
        );
    }

    // Our rows are indexed by position in the sorted distinct unit ids; R's are
    // indexed by its own cohort-major order. Join on the id rather than assuming
    // either. `our_position` is the same map `unit_universe` builds internally.
    let mut sorted_ids = fixture
        .data_subset
        .iter()
        .map(|row| row.countyreal)
        .collect::<Vec<_>>();
    sorted_ids.sort_unstable();
    sorted_ids.dedup();
    assert_eq!(sorted_ids.len(), fixture.n_units);

    let r_columns = (0..fixture.att.len())
        .filter(|&i| fixture.se[i].is_some())
        .collect::<Vec<_>>();
    // The influence function is held to a looser bound than the ATT and the SE,
    // deliberately. It is a PER-UNIT quantity that depends directly on the fitted
    // propensity score, so it inherits the full difference between R's logistic
    // solver and ours. The ATT and SE are averages over hundreds of units, where
    // those differences cancel to below 1e-10. Asserting one worst case rather
    // than 6000 individually so that the reported number is the real bound.
    let mut worst = 0.0_f64;
    let mut worst_at = (0_i64, 0_usize);
    for (ours, &r_col) in out.influence_functions.iter().zip(&r_columns) {
        for (r_row, id) in fixture.inffunc_unit_order.iter().enumerate() {
            let our_position = sorted_ids
                .binary_search(id)
                .unwrap_or_else(|_| panic!("unit {id} missing from the sorted universe"));
            let deviation = (ours[our_position] - fixture.inffunc[r_row][r_col]).abs();
            if deviation > worst {
                worst = deviation;
                worst_at = (*id, r_col);
            }
        }
    }
    assert!(
        worst < 1e-7,
        "worst influence deviation {worst:e} at unit {}, cell {}",
        worst_at.0,
        worst_at.1
    );
    println!("worst influence deviation vs did 2.5.1: {worst:e}");
}

/// The not-yet-treated comparison group, which is the switch that measures
/// comparator contamination.
#[test]
fn panel_att_gt_matches_r_universal_not_yet_treated() {
    let fixture: CsPanelRef = load_json("tests/data/cs_panel_dr_notyet_ref.json");
    let base_fixture: CsPanelRef = load_json("tests/data/cs_panel_dr_universal_ref.json");

    let out = fit(
        &observations(&base_fixture.data_subset),
        &universal_config(ComparisonGroup::NotYetTreated),
    );

    let expected = (0..fixture.att.len())
        .filter(|&i| fixture.se[i].is_some())
        .map(|i| {
            (
                fixture.group[i],
                fixture.t[i],
                fixture.att[i],
                fixture.se[i].unwrap_or_default(),
            )
        })
        .collect::<Vec<_>>();

    assert_eq!(out.estimates.len(), expected.len());
    for (ours, (group, time, att, se)) in out.estimates.iter().zip(&expected) {
        assert_eq!((ours.group, ours.time), (*group, *time));
        assert!(
            (ours.att - att).abs() < 1e-10,
            "att({group},{time}) {} vs R {att}",
            ours.att
        );
        assert!(
            (ours.se - se).abs() < 1e-10,
            "se({group},{time}) {} vs R {se}",
            ours.se
        );
    }
}

/// A unit-id-free input must fail loudly rather than quietly produce the
/// repeated-cross-section answer under a panel name.
// ---------------------------------------------------------------------------
// Aggregation parity against did::aggte
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct AggRef {
    overall_att: f64,
    overall_se: f64,
    egt: Option<Vec<i32>>,
    att_egt: Option<Vec<f64>>,
    se_egt: Option<Vec<Option<f64>>>,
}

#[derive(Deserialize)]
struct Aggregations {
    simple: AggRef,
    dynamic: AggRef,
    group: AggRef,
    calendar: AggRef,
    dynamic_balanced: AggRef,
}

#[derive(Deserialize)]
struct CsPanelWithAgg {
    data_subset: Vec<DataRow>,
    aggregations: Aggregations,
}

/// The fit of the universal-base fixture, which is the input every
/// aggregation below is computed from.
fn universal_run() -> (Out, Aggregations) {
    let fixture: CsPanelWithAgg = load_json("tests/data/cs_panel_dr_universal_ref.json");
    let out = fit(
        &observations(&fixture.data_subset),
        &universal_config(ComparisonGroup::NeverTreated),
    );
    (out, fixture.aggregations)
}

fn check_overall(label: &str, got: &Summary, want: &AggRef) {
    assert!(
        (got.overall_att - want.overall_att).abs() < 1e-9,
        "{label} overall att {} vs R {}",
        got.overall_att,
        want.overall_att
    );
    assert!(
        (got.overall_se - want.overall_se).abs() < 1e-9,
        "{label} overall se {} vs R {}",
        got.overall_se,
        want.overall_se
    );
}

/// Compare the path against R's, minus the anchor.
///
/// R's dynamic path carries the base event time (`e = -1`) as a point with
/// `att = 0` and `se = NA`. We do not emit it, and that is deliberate rather
/// than an omission: it is not an estimate, it is the normalisation, and a row
/// with zero variance would make the influence covariance singular. That matrix
/// is exactly what `EventTimeResult::to_honest_input` feeds to the HonestDiD
/// FLCI solve, which needs to invert it. The anchor belongs on a plot, not in an
/// estimate vector, so R's `NA`-se entries are dropped here.
fn check_path(label: &str, got: &Summary, want: &AggRef) {
    let keys = want.egt.as_ref().expect("egt");
    let atts = want.att_egt.as_ref().expect("att_egt");
    let ses = want.se_egt.as_ref().expect("se_egt");
    let expected = (0..keys.len())
        .filter(|&i| ses[i].is_some())
        .map(|i| (keys[i], atts[i], ses[i].unwrap_or_default()))
        .collect::<Vec<_>>();

    assert_eq!(got.by_key.len(), expected.len(), "{label} path length");
    for (point, (key, att, se)) in got.by_key.iter().zip(&expected) {
        assert_eq!(point.key, *key, "{label} key");
        assert!(
            (point.att - att).abs() < 1e-9,
            "{label} att at {key}: {} vs R {att}",
            point.att
        );
        assert!(
            (point.se - se).abs() < 1e-9,
            "{label} se at {key}: {} vs R {se}",
            point.se
        );
    }
}

fn run(out: &Out, aggregation: AggteType, balance_e: Option<i64>) -> Summary {
    rows::summary(
        &out.fit
            .aggregate(&AggteOptions {
                aggregation,
                balance_e,
                ..AggteOptions::default()
            })
            .expect("aggregate"),
    )
}

#[test]
fn aggte_simple_matches_r() {
    let (out, want) = universal_run();
    let got = run(&out, AggteType::Simple, None);
    check_overall("simple", &got, &want.simple);
    assert!(got.by_key.is_empty(), "simple has no path");
}

#[test]
fn aggte_dynamic_matches_r() {
    let (out, want) = universal_run();
    let got = run(&out, AggteType::Dynamic, None);
    check_overall("dynamic", &got, &want.dynamic);
    check_path("dynamic", &got, &want.dynamic);
}

#[test]
fn aggte_group_matches_r() {
    let (out, want) = universal_run();
    let got = run(&out, AggteType::Group, None);
    check_overall("group", &got, &want.group);
    check_path("group", &got, &want.group);
}

#[test]
fn aggte_calendar_matches_r() {
    let (out, want) = universal_run();
    let got = run(&out, AggteType::Calendar, None);
    check_overall("calendar", &got, &want.calendar);
    check_path("calendar", &got, &want.calendar);
}

/// The one that matters most for Study 1: a dynamic path on a fixed cohort
/// composition. With `balance_e = 1` on mpdta, cohort 2007 is dropped (it is
/// observed for 0 post periods, not 1) and the path is cut at event time 1.
#[test]
fn aggte_dynamic_balanced_matches_r() {
    let (out, want) = universal_run();
    let got = run(&out, AggteType::Dynamic, Some(1));
    check_overall("dynamic_balanced", &got, &want.dynamic_balanced);
    check_path("dynamic_balanced", &got, &want.dynamic_balanced);

    // The point of balancing: the unbalanced path runs further and is made of a
    // different set of cohorts, so the two must not coincide.
    let unbalanced = run(&out, AggteType::Dynamic, None);
    assert!(unbalanced.by_key.len() > got.by_key.len());
    assert!((unbalanced.overall_att - got.overall_att).abs() > 1e-6);
}

// ---------------------------------------------------------------------------
// Multiplier bootstrap
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct MbootRef {
    se: Vec<Option<f64>>,
    crit_val: f64,
}

/// The bootstrap cannot be reproduced across languages: R's RNG stream is not
/// ours, so the draws differ even with matched weights. What is checkable, and
/// what actually matters, is that the two agree to within Monte Carlo error.
/// Both sides use 20,000 iterations to make that error small enough for the
/// comparison to mean something.
///
/// The critical value is the load-bearing number: it is what makes the bands
/// simultaneous rather than pointwise, and it is the one the manuscript's
/// figures depend on.
#[test]
fn mboot_bands_agree_with_did_within_simulation_error() {
    let fixture: MbootRef = load_json("tests/data/cs_panel_mboot_ref.json");
    let (out, _) = universal_run();

    let bands = out
        .fit
        .bands(Level::default(), Bootstrap::new(20_000, 20_260_819))
        .expect("mboot bands")
        .cells;

    // The implied critical value, recovered from any band: (high - att) / se.
    let implied = (bands[0].upper.unwrap() - bands[0].estimate) / bands[0].se.unwrap();
    assert!(
        (implied - fixture.crit_val).abs() < 0.05,
        "critical value {implied} vs did {}",
        fixture.crit_val
    );

    // Simultaneous, so it must exceed the pointwise normal quantile. If this
    // ever fails the bands are not doing the job their name claims.
    assert!(
        implied > 1.96,
        "a simultaneous critical value below the pointwise one: {implied}"
    );

    // Bootstrap standard errors, cell by cell, skipping R's NA anchors.
    let expected = fixture
        .se
        .iter()
        .filter_map(|value| *value)
        .collect::<Vec<_>>();
    assert_eq!(bands.len(), expected.len());
    for (band, want) in bands.iter().zip(&expected) {
        let se = band.se.expect("se");
        let relative = (se - want).abs() / want;
        assert!(
            relative < 0.05,
            "bootstrap se {se} vs did {want} for ({}, {})",
            band.group,
            band.time
        );
    }
}

/// The correlation-aware aggregated variance, pinned.
///
/// `aggregate_att_gt_event_time_with_influence` used to compute
/// `sum(w^2 * se^2)`, the variance of a weighted sum of INDEPENDENT components.
/// The ATT(g,t) cells share units, so that understated the standard error. This
/// asserts the two formulas actually disagree on real data, so a regression back
/// to the cheap one cannot pass silently.
#[test]
fn aggregated_variance_uses_the_influence_functions_not_the_diagonal() {
    let (out, _) = universal_run();
    let estimates = &out.estimates;
    let aggregated = rows::summary(
        &out.fit
            .aggregate(&AggteOptions {
                aggregation: AggteType::Dynamic,
                weights: AggteWeights::TreatedCount,
                ..AggteOptions::default()
            })
            .expect("aggregate"),
    );

    // Rebuild the naive number for the event time with the most components.
    let busiest = aggregated
        .by_key
        .iter()
        .enumerate()
        .max_by_key(|(_, e)| e.components)
        .map(|(index, _)| index)
        .expect("some event time");
    let event_time = aggregated.by_key[busiest].components;
    assert!(event_time > 1, "need a multi-component bucket to compare");

    let ours = aggregated.by_key[busiest].se;
    assert!(ours.is_finite() && ours > 0.0);

    let naive = {
        let key = aggregated.by_key[busiest].key;
        let members = estimates
            .iter()
            .filter(|e| e.event_time == key)
            .collect::<Vec<_>>();
        let total: f64 = members
            .iter()
            .map(|e| f64::from(u32::try_from(e.treated_n).expect("a small count")))
            .sum();
        members
            .iter()
            .map(|e| {
                let w = f64::from(u32::try_from(e.treated_n).expect("a small count")) / total;
                w * w * e.se * e.se
            })
            .sum::<f64>()
            .sqrt()
    };
    assert!(
        (ours - naive).abs() > 1e-6,
        "influence-based se {ours} is indistinguishable from the independence \
         approximation {naive}; the fix may have been reverted"
    );
}

// ---------------------------------------------------------------------------
// Clustering
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ClusteredRef {
    n_states: usize,
    se: Vec<Option<f64>>,
    crit_val: f64,
    unit_order: Vec<i64>,
    cluster_of_unit: Vec<i64>,
}

/// Clustered bands against `did` with `clustervars`.
///
/// This is the shape Study 1 has: two parents of one child are two units sharing
/// every household shock, so the family is the cluster and the unit is not.
/// mpdta stands in for it with counties clustered into states.
#[test]
fn clustered_mboot_bands_agree_with_did() {
    let fixture: ClusteredRef = load_json("tests/data/cs_panel_mboot_clustered_ref.json");
    let base: CsPanelWithAgg = load_json("tests/data/cs_panel_dr_universal_ref.json");
    let rows = observations(&base.data_subset);
    let plain = fit(&rows, &universal_config(ComparisonGroup::NeverTreated));
    let out = fit(
        &clustered(&rows, &fixture.unit_order, &fixture.cluster_of_unit),
        &universal_config(ComparisonGroup::NeverTreated),
    );
    // `unit_order` is the sorted id list, which is exactly our unit index.
    assert_eq!(fixture.unit_order.len(), out.influence_functions[0].len());
    let draws = Bootstrap::new(20_000, 20_260_819);

    let bands = out
        .fit
        .bands(Level::default(), draws)
        .expect("clustered mboot bands")
        .cells;

    let implied = (bands[0].upper.unwrap() - bands[0].estimate) / bands[0].se.unwrap();
    // Looser than the unclustered case on purpose: the bootstrap now draws 29
    // weights per replication rather than 500, so the Monte Carlo error of a
    // maximum over twelve statistics is correspondingly larger.
    assert!(
        (implied - fixture.crit_val).abs() < 0.15,
        "clustered critical value {implied} vs did {}",
        fixture.crit_val
    );

    let expected = fixture
        .se
        .iter()
        .filter_map(|value| *value)
        .collect::<Vec<_>>();
    assert_eq!(bands.len(), expected.len());
    for (band, want) in bands.iter().zip(&expected) {
        let se = band.se.expect("se");
        let relative = (se - want).abs() / want;
        assert!(
            relative < 0.10,
            "clustered se {se} vs did {want} for ({}, {})",
            band.group,
            band.time
        );
    }

    // Clustering must actually change something. With 29 clusters standing in for
    // 500 units the standard errors move substantially, and a test that passed
    // whether or not the labels were used would be worthless.
    let unclustered = plain
        .fit
        .bands(Level::default(), draws)
        .expect("unclustered bands")
        .cells;
    let moved = bands
        .iter()
        .zip(&unclustered)
        .filter(|(a, b)| (a.se.unwrap() - b.se.unwrap()).abs() / b.se.unwrap() > 0.1)
        .count();
    assert!(
        moved >= bands.len() / 2,
        "clustering barely moved the standard errors: {moved} of {} cells",
        bands.len()
    );
    assert_eq!(
        fixture.n_states,
        fixture
            .cluster_of_unit
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    );
}

/// The identity that makes clustering safe to switch on: one cluster per unit
/// must reproduce the unclustered answer exactly, not approximately.
///
/// Without this, turning clustering on for a design that has none would move
/// every number for no reason, and there would be no way to tell that from a
/// real clustering effect.
#[test]
fn clustering_by_unit_reproduces_the_unclustered_standard_error() {
    let (out, _) = universal_run();
    let base: CsPanelWithAgg = load_json("tests/data/cs_panel_dr_universal_ref.json");
    let rows = observations(&base.data_subset);
    let ids = rows::ids(&out.fit);
    let singleton: Vec<i64> = (0..ids.len())
        .map(|i| i64::try_from(i).expect("a small index"))
        .collect();
    let by_unit = fit(
        &clustered(&rows, &ids, &singleton),
        &universal_config(ComparisonGroup::NeverTreated),
    );

    let options = AggteOptions {
        aggregation: AggteType::Dynamic,
        ..AggteOptions::default()
    };
    let plain = rows::summary(&out.fit.aggregate(&options).expect("unclustered"));
    let clustered = rows::summary(
        &by_unit
            .fit
            .aggregate(&options)
            .expect("singleton-clustered"),
    );

    assert!((plain.overall_se - clustered.overall_se).abs() < 1e-12);
    for (a, b) in plain.by_key.iter().zip(&clustered.by_key) {
        assert!(
            (a.se - b.se).abs() < 1e-12,
            "event time {}: {} vs {}",
            a.key,
            a.se,
            b.se
        );
    }

    let draws = Bootstrap::new(2_000, 4_242);
    let plain_bands = out.fit.bands(Level::default(), draws).expect("plain");
    let singleton_bands = by_unit
        .fit
        .bands(Level::default(), draws)
        .expect("singleton");
    for (a, b) in plain_bands.cells.iter().zip(&singleton_bands.cells) {
        let (a, b) = (a.se.unwrap(), b.se.unwrap());
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }
}

#[test]
fn weights_flow_through_estimation_and_aggregation() {
    #[derive(Deserialize)]
    struct WeightedRef {
        att: Vec<f64>,
        se: Vec<Option<f64>>,
        unit_order: Vec<i64>,
        weight_of_unit: Vec<f64>,
        dynamic: AggRef,
    }

    let fixture: WeightedRef = load_json("tests/data/cs_panel_weighted_ref.json");
    let base_fixture: CsPanelWithAgg = load_json("tests/data/cs_panel_dr_universal_ref.json");

    let by_unit = fixture
        .unit_order
        .iter()
        .zip(&fixture.weight_of_unit)
        .map(|(id, weight)| (*id, *weight))
        .collect::<std::collections::BTreeMap<i64, f64>>();

    let rows = base_fixture
        .data_subset
        .iter()
        .map(|row| Obs {
            weight: by_unit[&row.countyreal],
            covariates: vec![row.lpop],
            ..Obs::new(
                Some(row.countyreal),
                (row.first_treat != 0).then_some(row.first_treat),
                row.year,
                row.lemp,
            )
        })
        .collect::<Vec<_>>();

    let out = fit(&rows, &universal_config(ComparisonGroup::NeverTreated));

    let expected = (0..fixture.att.len())
        .filter(|&i| fixture.se[i].is_some())
        .map(|i| fixture.att[i])
        .collect::<Vec<_>>();
    assert_eq!(out.estimates.len(), expected.len());
    for (ours, want) in out.estimates.iter().zip(&expected) {
        assert!(
            (ours.att - want).abs() < 1e-9,
            "weighted att({}, {}) {} vs R {want}",
            ours.group,
            ours.time,
            ours.att
        );
    }

    let dynamic = run(&out, AggteType::Dynamic, None);
    check_overall("weighted dynamic", &dynamic, &fixture.dynamic);
    check_path("weighted dynamic", &dynamic, &fixture.dynamic);

    // The weights must matter. If they were being dropped somewhere this would
    // agree with the unweighted run and every assertion above would still pass
    // only because R was also given weights it ignored.
    let unweighted = universal_run().0.estimates;
    let moved = out
        .estimates
        .iter()
        .zip(&unweighted)
        .filter(|(a, b)| (a.att - b.att).abs() > 1e-6)
        .count();
    assert_eq!(
        moved,
        out.estimates.len(),
        "weighting left some cells untouched"
    );
}

/// Analytic clustered per-cell standard errors against `did`.
///
/// `did` reports these under `bstrap = FALSE` with `clustervars`, so the per-cell
/// standard error has a reference and does not have to be left to the bootstrap.
#[test]
fn clustered_per_cell_standard_errors_match_did() {
    #[derive(Deserialize)]
    struct AnalyticClusteredRef {
        se: Vec<Option<f64>>,
        unit_order: Vec<i64>,
        cluster_of_unit: Vec<i64>,
    }

    let fixture: AnalyticClusteredRef =
        load_json("tests/data/cs_panel_clustered_analytic_ref.json");
    let (out, _) = universal_run();
    let estimates = &out.estimates;
    assert_eq!(fixture.unit_order.len(), out.influence_functions[0].len());
    let base: CsPanelWithAgg = load_json("tests/data/cs_panel_dr_universal_ref.json");
    let clustered = fit(
        &clustered(
            &observations(&base.data_subset),
            &fixture.unit_order,
            &fixture.cluster_of_unit,
        ),
        &universal_config(ComparisonGroup::NeverTreated),
    )
    .estimates;

    let expected = fixture
        .se
        .iter()
        .filter_map(|value| *value)
        .collect::<Vec<_>>();
    assert_eq!(clustered.len(), expected.len());
    for (ours, want) in clustered.iter().zip(&expected) {
        assert!(
            (ours.se - want).abs() < 1e-10,
            "clustered se({}, {}) {} vs did {want}",
            ours.group,
            ours.time,
            ours.se
        );
    }

    // It has to actually change the answer, in both directions. On this data the
    // first cell falls from 0.0221 to 0.0102 and another rises from 0.0312 to
    // 0.0489, so a rule of thumb like "clustering inflates standard errors" would
    // be wrong here.
    let up = clustered
        .iter()
        .zip(estimates)
        .filter(|(a, b)| a.se > b.se * 1.05)
        .count();
    let down = clustered
        .iter()
        .zip(estimates)
        .filter(|(a, b)| a.se < b.se * 0.95)
        .count();
    assert!(up > 0 && down > 0, "moved up {up}, down {down}");
}

/// With one cluster per unit, the clustered per-cell standard error must return
/// the pair estimator's own number.
///
/// This is what makes the function safe to apply unconditionally: on a design
/// with no clustering it is a no-op rather than a small unexplained shift.
#[test]
fn clustered_per_cell_reduces_to_the_pair_estimator() {
    let (out, _) = universal_run();
    let base: CsPanelWithAgg = load_json("tests/data/cs_panel_dr_universal_ref.json");
    let rows = observations(&base.data_subset);
    let ids = rows::ids(&out.fit);
    let singleton: Vec<i64> = (0..ids.len())
        .map(|i| i64::try_from(i).expect("a small index"))
        .collect();
    let same = fit(
        &clustered(&rows, &ids, &singleton),
        &universal_config(ComparisonGroup::NeverTreated),
    )
    .estimates;

    for (ours, original) in same.iter().zip(&out.estimates) {
        assert!(
            (ours.se - original.se).abs() < 1e-12,
            "({}, {}): {} vs pair estimator {}",
            ours.group,
            ours.time,
            ours.se,
            original.se
        );
    }
}

/// `min_e` and `max_e` against `did`, including their effect on the overall.
#[test]
fn aggte_dynamic_trimming_matches_r() {
    #[derive(Deserialize)]
    struct TrimmedRef {
        min_e: i32,
        max_e: i32,
        overall_att: f64,
        overall_se: f64,
        egt: Option<Vec<i32>>,
        att_egt: Option<Vec<f64>>,
        se_egt: Option<Vec<Option<f64>>>,
    }

    let fixtures: Vec<TrimmedRef> = load_json("tests/data/cs_panel_trimmed_ref.json");
    let (out, _) = universal_run();
    assert_eq!(fixtures.len(), 3);

    for want in &fixtures {
        let got = rows::summary(
            &out.fit
                .aggregate(&AggteOptions {
                    aggregation: AggteType::Dynamic,
                    min_e: Some(i64::from(want.min_e)),
                    max_e: Some(i64::from(want.max_e)),
                    ..AggteOptions::default()
                })
                .expect("trimmed dynamic"),
        );

        let label = format!("min_e={} max_e={}", want.min_e, want.max_e);
        let as_agg = AggRef {
            overall_att: want.overall_att,
            overall_se: want.overall_se,
            egt: want.egt.clone(),
            att_egt: want.att_egt.clone(),
            se_egt: want.se_egt.clone(),
        };
        check_overall(&label, &got, &as_agg);
        check_path(&label, &got, &as_agg);

        // Trimming must change the overall, or the bounds are being ignored.
        if want.max_e < 3 {
            let untrimmed = run(&out, AggteType::Dynamic, None);
            assert!(
                (untrimmed.overall_att - got.overall_att).abs() > 1e-9,
                "{label} left the overall unchanged"
            );
        }
    }
}

/// The repeated-cross-section route.
///
/// `did_attgt_dr_ref.json` is `att_gt`'s default `panel = TRUE` output, a
/// panel estimate, so this test does not compare against it.
///
/// # These are two different estimators, not two implementations of one
///
/// `did` calls `DRDID::drdid_rc` for `est_method = "dr"`, the LOCALLY EFFICIENT
/// repeated-cross-section estimator. This crate's RC route implements the
/// TRADITIONAL one, `DRDID::drdid_rc1`. On mpdta the point estimates agree to
/// 1e-9 and the standard errors differ by about 2.8%: for cell (2004, 2004),
/// 0.0922804 traditional against 0.0897358 locally efficient.
///
/// So the reference here is `drdid_rc1`, cell by cell. Comparing against `did`'s
/// `panel = FALSE` output would be comparing against a different estimator and
/// the 2.8% gap would look like a bug.
#[test]
fn rc_route_matches_the_traditional_estimator_and_aggregates() {
    #[derive(Deserialize)]
    struct RcCell {
        group: i32,
        time: i32,
        att: f64,
        se_traditional: f64,
        se_locally_efficient: f64,
    }
    #[derive(Deserialize)]
    struct RcRef {
        n_rows: usize,
        dynamic: AggRef,
    }

    let cells: Vec<RcCell> = load_json("tests/data/drdid_rc1_cells_ref.json");
    let did_rc: RcRef = load_json("tests/data/cs_rc_ref.json");
    let base: CsPanelWithAgg = load_json("tests/data/cs_panel_dr_universal_ref.json");

    let rows = observations(&base.data_subset);
    assert_eq!(rows.len(), did_rc.n_rows);
    let out = fit(
        &rows,
        &AttGtOptions {
            estimator: Estimator::TraditionalDoublyRobust,
            design: Design::RepeatedCrossSection,
            ..universal_config(ComparisonGroup::NeverTreated)
        },
    );
    assert_eq!(out.estimates.len(), cells.len());

    let mut worst_se = 0.0_f64;
    for (ours, want) in out.estimates.iter().zip(&cells) {
        assert_eq!((ours.group, ours.time), (want.group, want.time));
        assert!(
            (ours.att - want.att).abs() < 1e-9,
            "rc att({}, {}) {} vs drdid_rc1 {}",
            ours.group,
            ours.time,
            ours.att,
            want.att
        );
        // Loose by the standards of the panel route's 1e-10. Our traditional RC
        // standard error tracks drdid_rc1 to within 9e-4 relative but is not
        // identical to it. The crate's 1e-8 ridge is NOT the cause: disabling it
        // leaves the worst deviation at 8.5e-4 unchanged. Nor is it a general
        // weakness of this crate's repeated-cross-section code, since
        // `estimate_drdid_repeated_efficient` matches DRDID::drdid_rc to 1.5e-11
        // on the point estimate and 3.4e-12 on the standard error. Something in
        // the TRADITIONAL variant specifically is slightly off, and since `did`
        // calls the locally efficient one, it is bounded here rather than chased.
        let relative = (ours.se - want.se_traditional).abs() / want.se_traditional;
        worst_se = worst_se.max(relative);
        assert!(
            relative < 1e-3,
            "rc se({}, {}) {} vs drdid_rc1 {} (relative {relative:e})",
            ours.group,
            ours.time,
            ours.se,
            want.se_traditional
        );
        // And it is the traditional estimator, not the locally efficient one.
        assert!(
            (ours.se - want.se_traditional).abs() < (ours.se - want.se_locally_efficient).abs(),
            "rc se({}, {}) {} sits closer to drdid_rc {} than to drdid_rc1 {}; the \
             route may have been switched to the locally efficient estimator",
            ours.group,
            ours.time,
            ours.se,
            want.se_locally_efficient,
            want.se_traditional
        );
    }

    println!("worst rc se deviation vs DRDID::drdid_rc1: {worst_se:e}");

    // The point of this test for item 2 of the plan: the aggregation now works on
    // the RC route, given a ROW-indexed panel. The point estimates match did's
    // aggte; the standard errors cannot, because did aggregated a different
    // estimator's influence functions.
    let dynamic = run(&out, AggteType::Dynamic, None);
    let keys = did_rc.dynamic.egt.as_ref().expect("egt");
    let atts = did_rc.dynamic.att_egt.as_ref().expect("att");
    let ses = did_rc.dynamic.se_egt.as_ref().expect("se");
    let expected = (0..keys.len())
        .filter(|&i| ses[i].is_some())
        .map(|i| (keys[i], atts[i]))
        .collect::<Vec<_>>();
    assert_eq!(dynamic.by_key.len(), expected.len());
    for (point, (key, att)) in dynamic.by_key.iter().zip(&expected) {
        assert_eq!(point.key, *key);
        assert!(
            (point.att - att).abs() < 1e-9,
            "rc dynamic att at {key}: {} vs R {att}",
            point.att
        );
    }
    assert!(
        (dynamic.overall_att - did_rc.dynamic.overall_att).abs() < 1e-9,
        "rc dynamic overall {} vs R {}",
        dynamic.overall_att,
        did_rc.dynamic.overall_att
    );

    // What the panel route buys, as a number rather than a claim.
    let panel_estimates = universal_run().0.estimates;
    for (rc, pnl) in out.estimates.iter().zip(&panel_estimates) {
        assert!((rc.att - pnl.att).abs() < 1e-9);
        assert!(
            rc.se > pnl.se * 2.0,
            "discarding the pairing should inflate the se: {} vs {}",
            rc.se,
            pnl.se
        );
    }
}

/// A repeated panel row is an error, not a frequency weight.
///
/// Expressing "this unit counts four times" by repeating its rows works on the
/// repeated-cross-section route, where a row is an observation. On the panel
/// route the copies collapse into one unit and the intended weight silently
/// vanishes. That is a wrong answer with no symptom, so it is refused.
#[test]
fn a_repeated_panel_row_is_refused() {
    let fixture: CsPanelWithAgg = load_json("tests/data/cs_panel_dr_universal_ref.json");
    let mut rows = observations(&fixture.data_subset);
    rows.push(rows[0].clone());

    let err = rows::fit(&rows, &universal_config(ComparisonGroup::NeverTreated))
        .expect_err("a duplicated panel row must be refused");
    assert!(
        matches!(err, Error::Input(InputError::Duplicate { .. })),
        "got {err:?}"
    );
}

/// The locally efficient RC route against `did`'s `panel = FALSE` output.
///
/// This is what `cs_rc_ref.json` was generated from, and it is the comparison
/// the traditional route cannot pass: `did` calls `DRDID::drdid_rc`, and only
/// this route does.
#[test]
fn efficient_rc_route_matches_did_panel_false() {
    #[derive(Deserialize)]
    struct RcRef {
        att: Vec<f64>,
        se: Vec<Option<f64>>,
        dynamic: AggRef,
    }

    let did_rc: RcRef = load_json("tests/data/cs_rc_ref.json");
    let base: CsPanelWithAgg = load_json("tests/data/cs_panel_dr_universal_ref.json");
    let rows = observations(&base.data_subset);

    // Ridge off, because R fits these regressions unregularised. The crate's
    // 1e-8 default moves the influence function by about 2e-7 while cancelling
    // in the point estimate, which is enough to matter at this tolerance and not
    // enough to matter in use.
    let out = fit(
        &rows,
        &AttGtOptions {
            design: Design::RepeatedCrossSection,
            nuisance: NuisanceOptions {
                ridge: 0.0,
                ..NuisanceOptions::default()
            },
            ..universal_config(ComparisonGroup::NeverTreated)
        },
    );

    let expected = (0..did_rc.att.len())
        .filter(|&i| did_rc.se[i].is_some())
        .map(|i| (did_rc.att[i], did_rc.se[i].unwrap_or_default()))
        .collect::<Vec<_>>();
    assert_eq!(out.estimates.len(), expected.len());
    for (ours, (att, se)) in out.estimates.iter().zip(&expected) {
        assert!(
            (ours.att - att).abs() < 1e-9,
            "att({}, {}) {} vs did {att}",
            ours.group,
            ours.time,
            ours.att
        );
        assert!(
            (ours.se - se).abs() < 1e-9,
            "se({}, {}) {} vs did {se}",
            ours.group,
            ours.time,
            ours.se
        );
    }

    // And the aggregation, which the traditional route could only match on point
    // estimates because did had aggregated a different estimator's influence.
    let dynamic = run(&out, AggteType::Dynamic, None);
    check_overall("efficient rc dynamic", &dynamic, &did_rc.dynamic);
    check_path("efficient rc dynamic", &dynamic, &did_rc.dynamic);
}

/// The fixed-weight aggregation understates, and by how much.
///
/// `aggregate_att_gt_event_time_with_influence` combines the cell influence
/// vectors with FIXED weights. Under `ByTreatedCount` those weights are the
/// empirical cohort shares, which are estimated from the sample, so the
/// standard error omits `did:::wif` and comes out too small. `aggregate_att_gt`
/// carries the term and matches `did`.
///
/// This is not a bug report against either function: the fixed-weight path is
/// the right thing for `Equal` and `ByTotalWeight`, which are not cohort shares
/// and which `did` has no counterpart for. It is a guard, so that a caller who
/// wires `ByTreatedCount` into an event study learns the cost from a test
/// rather than from a referee. `register-studies` had it wired for months.
///
/// The measured shape is the useful part: **the point estimates agree exactly,
/// and the standard errors agree only where a single cohort contributes.**
/// Wherever more than one does, the fixed-weight one is strictly smaller. That
/// is why the defect survived: on a fixture with little cohort overlap most
/// cells look right.
#[test]
fn the_fixed_weight_event_time_aggregation_understates_multi_cohort_cells() {
    let (out, aggregations) = universal_run();
    let r = &aggregations.dynamic;

    let fixed = rows::summary(
        &out.fit
            .aggregate(&AggteOptions {
                aggregation: AggteType::Dynamic,
                weights: AggteWeights::TreatedCount,
                ..AggteOptions::default()
            })
            .expect("fixed-weight aggregation"),
    );
    let ported = run(&out, AggteType::Dynamic, None);

    let egt = r.egt.clone().expect("egt");
    let att = r.att_egt.clone().expect("att");
    let se = r.se_egt.clone().expect("se");

    let mut multi_cohort_cells = 0;
    for (index, event_time) in egt.iter().enumerate() {
        let Some(cell) = fixed.by_key.iter().find(|row| row.key == *event_time) else {
            continue;
        };
        let Some(want_se) = se[index] else { continue };

        // Both paths reproduce did::aggte's point estimate.
        assert!(
            (cell.att - att[index]).abs() < 1e-9,
            "fixed-weight att {} vs did {} at e={event_time}",
            cell.att,
            att[index]
        );
        let ported_cell = ported
            .by_key
            .iter()
            .find(|row| row.key == *event_time)
            .expect("ported cell");
        assert!((ported_cell.att - att[index]).abs() < 1e-9);
        assert!((ported_cell.se - want_se).abs() / want_se < 1e-6);

        if cell.components == 1 {
            assert!(
                (cell.se - want_se).abs() / want_se < 1e-9,
                "one cohort contributes, so there is nothing estimated to correct \
                 for and the two must agree at e={event_time}"
            );
        } else {
            multi_cohort_cells += 1;
            assert!(
                cell.se < want_se,
                "the fixed-weight se {} is not below did's {want_se} at e={event_time}",
                cell.se
            );
        }
    }
    assert!(
        multi_cohort_cells >= 3,
        "the fixture must exercise several overlapping cohorts, saw {multi_cohort_cells}"
    );
}

#[derive(Deserialize)]
struct CsPanelCellsAndAgg {
    group: Vec<i32>,
    t: Vec<i32>,
    att: Vec<f64>,
    inffunc: Vec<Vec<f64>>,
    inffunc_unit_order: Vec<i64>,
    data_subset: Vec<DataRow>,
    aggregations: Aggregations,
}

/// `aggregate_att_gt_by_cohort`, `aggregate_att_gt_by_calendar_time` and
/// `aggregate_att_gt_overall` under treated-count weighting against
/// `did::aggte` 2.5.1 (`type = "group"`, `"calendar"` and `"simple"`).
///
/// # The tolerance
///
/// It is propagated from the cells, whose largest differences from `did` are
/// measured here: `da` in the estimates and `dpsi` in the influence values. An
/// aggregate is `sum_c w_c ATT_c` with non-negative weights summing to one, so
/// its estimate is within `da`. Its influence value adds `sum_c wif_ic ATT_c`,
/// the estimated shares' influence, and since `|ind_ig - p_g| <= w_max`, both
/// terms of `did:::wif` sum over the cells to at most `k w_max / P`, with `P`
/// the sum of the cells' shares, which is at least `k p_min`. So an influence
/// value is within `dpsi + 2 (w_max / p_min) da`, and the standard error, a
/// root mean square over `n` units, within that over `sqrt(n)`. Both sides also
/// round sums of `n + k` terms, `2 (n + k) eps` of the magnitudes, and the
/// fixture carries 15 decimal places, `5e-16` each.
#[test]
fn cohort_calendar_and_overall_match_aggte() {
    let fixture: CsPanelCellsAndAgg = load_json("tests/data/cs_panel_dr_universal_ref.json");
    let rows = observations(&fixture.data_subset);
    let out = fit(&rows, &universal_config(ComparisonGroup::NeverTreated));
    let sample = out.fit.sample();

    let mut sorted_ids = fixture
        .data_subset
        .iter()
        .map(|row| row.countyreal)
        .collect::<Vec<_>>();
    sorted_ids.sort_unstable();
    sorted_ids.dedup();
    let n = sorted_ids.len();
    let mut da = 0.0_f64;
    let mut dpsi = 0.0_f64;
    let mut largest_att = 0.0_f64;
    let mut largest_psi = 0.0_f64;
    for (estimate, ours) in out.estimates.iter().zip(&out.influence_functions) {
        let column = (0..fixture.att.len())
            .find(|&i| fixture.group[i] == estimate.group && fixture.t[i] == estimate.time)
            .expect("cell in the fixture");
        da = da.max((estimate.att - fixture.att[column]).abs());
        largest_att = largest_att.max(estimate.att.abs());
        for (r_row, id) in fixture.inffunc_unit_order.iter().enumerate() {
            let position = sorted_ids.binary_search(id).expect("unit");
            dpsi = dpsi.max((ours[position] - fixture.inffunc[r_row][column]).abs());
            largest_psi = largest_psi.max(ours[position].abs());
        }
    }
    let shares = out
        .estimates
        .iter()
        .map(|e| {
            let members = sample
                .groups
                .iter()
                .zip(&sample.weights)
                .filter(|(g, _)| **g == Some(i64::from(e.group)))
                .map(|(_, w)| *w)
                .sum::<f64>();
            members / n as f64
        })
        .collect::<Vec<_>>();
    let p_min = shares.iter().copied().fold(f64::INFINITY, f64::min);
    let w_max = sample.weights.iter().copied().fold(0.0_f64, f64::max);
    let k = out.estimates.len() as f64;
    let rounding = 2.0 * (n as f64 + k) * f64::EPSILON;
    let att_bound = da + rounding * largest_att + 5e-16;
    let se_bound =
        (dpsi + 2.0 * (w_max / p_min) * da + rounding * largest_psi) / (n as f64).sqrt() + 5e-16;
    println!("cells: da {da:e}, dpsi {dpsi:e}; bounds: att {att_bound:e}, se {se_bound:e}");

    let check = |label: &str, (estimate, se_ours): (f64, f64), att: f64, se: f64| {
        assert!(
            (estimate - att).abs() <= att_bound,
            "{label} att {estimate} vs did {att}, bound {att_bound:e}"
        );
        assert!(
            (se_ours - se).abs() <= se_bound,
            "{label} se {se_ours} vs did {se}, bound {se_bound:e}"
        );
    };
    let path = |want: &AggRef| {
        let keys = want.egt.clone().expect("egt");
        let atts = want.att_egt.clone().expect("att");
        let ses = want.se_egt.clone().expect("se");
        (0..keys.len())
            .filter_map(|i| ses[i].map(|se| (keys[i], atts[i], se)))
            .collect::<Vec<_>>()
    };

    let cohort = run(&out, AggteType::Group, None);
    let want = path(&fixture.aggregations.group);
    assert_eq!(cohort.by_key.len(), want.len());
    for (got, (group, att, se)) in cohort.by_key.iter().zip(&want) {
        assert_eq!(got.key, *group);
        check(&format!("cohort {group}"), (got.att, got.se), *att, *se);
    }

    let calendar = run(&out, AggteType::Calendar, None);
    let want = path(&fixture.aggregations.calendar);
    assert_eq!(calendar.by_key.len(), want.len());
    for (got, (time, att, se)) in calendar.by_key.iter().zip(&want) {
        assert_eq!(got.key, *time);
        check(&format!("calendar {time}"), (got.att, got.se), *att, *se);
    }

    let overall = run(&out, AggteType::Simple, None);
    check(
        "overall",
        (overall.overall_att, overall.overall_se),
        fixture.aggregations.simple.overall_att,
        fixture.aggregations.simple.overall_se,
    );
    let post = out.estimates.iter().filter(|e| e.time >= e.group).count();
    assert!(post < out.estimates.len());
    assert_eq!(overall.overall_components, post);
}

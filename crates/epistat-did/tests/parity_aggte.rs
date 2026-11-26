//! Aggregation of `ATT(g,t)` against `did::aggte` 2.5.1: `max_e` for the simple
//! and group summaries, clustered aggregates, the multiplier bootstrap and the
//! Wald pre-test.
//!
//! The fixture is written by `tests/data/generate_aggte_reference.R`. Every
//! scenario carries the cells `att_gt` returned, with their influence functions
//! in the order of the sorted unit ids, so the aggregation is the only thing
//! compared. The influence functions of the aggregates are compared too: they
//! are deterministic, and the bootstrap is a function of them.

use epistat_did::{
    AggteBootstrap, AggteConfig, AggteError, AggteResult, AggteType, AttGtEstimate, PretestError,
    UnitPanel, aggregate_att_gt, pretrend_wald_test,
};
use epistat_inference::Clusters;
use epistat_rng::Stream;
use epistat_testkit::{assert_same_on_threads, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Pretest {
    #[serde(rename = "W")]
    w: f64,
    #[serde(rename = "Wpval")]
    p_value: f64,
}

#[derive(Deserialize)]
struct Cells {
    group: Vec<i32>,
    t: Vec<i32>,
    att: Vec<f64>,
    /// One row per unit, one column per cell.
    inffunc: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct Scenario {
    #[serde(flatten)]
    cells: Cells,
    #[serde(rename = "W")]
    w: f64,
    #[serde(rename = "Wpval")]
    p_value: f64,
    clustered_pretest: Pretest,
}

#[derive(Deserialize)]
struct Agg {
    overall_att: f64,
    overall_se: f64,
    inf_overall: Vec<f64>,
    egt: Option<Vec<i32>>,
    att_egt: Option<Vec<f64>>,
    se_egt: Option<Vec<f64>>,
    /// One row per unit, one column per path point.
    inf_path: Option<Vec<Vec<f64>>>,
}

#[derive(Deserialize)]
struct MaxE {
    #[serde(rename = "max_e")]
    bound: i32,
    simple: Agg,
    group: Agg,
    dynamic_balanced: Agg,
    simple_min_e: Agg,
    group_balanced: Agg,
}

#[derive(Deserialize)]
struct Clustered {
    simple: Agg,
    dynamic: Agg,
    group: Agg,
    calendar: Agg,
}

#[derive(Deserialize)]
struct BootAgg {
    #[serde(flatten)]
    agg: Agg,
    crit_val: Option<f64>,
    analytic_overall_se: f64,
    overall_se_rel_mc: f64,
    se_rel_mc: Option<Vec<f64>>,
    crit_mc_se: Option<f64>,
}

#[derive(Deserialize)]
struct BootUnclustered {
    simple: BootAgg,
    dynamic: BootAgg,
    group: BootAgg,
    calendar: BootAgg,
}

#[derive(Deserialize)]
struct BootClustered {
    dynamic: BootAgg,
}

#[derive(Deserialize)]
struct Bootstrap {
    biters: usize,
    alp: f64,
    unclustered: BootUnclustered,
    clustered: BootClustered,
}

#[derive(Deserialize)]
struct GapMaxE {
    max_e: i32,
    simple: Agg,
    group: Agg,
}

#[derive(Deserialize)]
struct Gap {
    unit_group: Vec<f64>,
    cells: Cells,
    max_e: Vec<GapMaxE>,
}

#[derive(Deserialize)]
struct Reference {
    unit_group: Vec<f64>,
    unit_state: Vec<f64>,
    universal: Scenario,
    varying: Scenario,
    max_e: Vec<MaxE>,
    clustered: Clustered,
    bootstrap: Bootstrap,
    gap: Gap,
}

fn reference() -> Reference {
    load_json("tests/data/aggte_ref.json")
}

fn estimates(cells: &Cells) -> Vec<AttGtEstimate> {
    (0..cells.group.len())
        .map(|k| AttGtEstimate {
            group: cells.group[k],
            time: cells.t[k],
            event_time: cells.t[k] - cells.group[k],
            att: cells.att[k],
            ..AttGtEstimate::default()
        })
        .collect()
}

/// One vector per cell (or path point), from the fixture's unit-by-column rows.
fn columns(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    (0..rows[0].len())
        .map(|j| rows.iter().map(|row| row[j]).collect())
        .collect()
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "cohorts are small whole numbers written as doubles by R"
)]
fn panel(unit_group: &[f64]) -> UnitPanel {
    UnitPanel::unweighted(
        unit_group
            .iter()
            .map(|&g| (g != 0.0).then_some(g as i32))
            .collect(),
    )
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "state codes are small whole numbers written as doubles by R"
)]
fn clustered_panel(r: &Reference) -> UnitPanel {
    panel(&r.unit_group).clustered_by(r.unit_state.iter().map(|&s| s as i64).collect())
}

fn config(aggregation: AggteType) -> AggteConfig {
    AggteConfig {
        aggregation,
        ..AggteConfig::default()
    }
}

fn close(got: f64, want: f64, tolerance: f64, label: &str) {
    assert!(
        (got - want).abs() <= tolerance * want.abs().max(1.0),
        "{label}: {got} vs R {want}"
    );
}

fn assert_vectors_close(got: &[f64], want: &[f64], tolerance: f64, label: &str) {
    assert_eq!(got.len(), want.len(), "{label} length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        close(*g, *w, tolerance, &format!("{label}[{i}]"));
    }
}

/// Estimates, standard errors and influence functions of the overall effect and
/// the path, against R's.
fn check(label: &str, got: &AggteResult, want: &Agg) {
    close(
        got.overall_att,
        want.overall_att,
        1e-9,
        &format!("{label} att"),
    );
    close(
        got.overall_se,
        want.overall_se,
        1e-9,
        &format!("{label} se"),
    );
    assert_vectors_close(
        &got.overall_influence,
        &want.inf_overall,
        1e-9,
        &format!("{label} overall influence"),
    );
    let Some(keys) = &want.egt else {
        assert!(got.by_key.is_empty(), "{label} has no path");
        return;
    };
    let atts = want.att_egt.as_ref().expect("att_egt");
    let ses = want.se_egt.as_ref().expect("se_egt");
    let influence = columns(want.inf_path.as_ref().expect("inf_path"));
    assert_eq!(got.by_key.len(), keys.len(), "{label} path length");
    for (j, point) in got.by_key.iter().enumerate() {
        assert_eq!(point.key, keys[j], "{label} key");
        close(
            point.att,
            atts[j],
            1e-9,
            &format!("{label} att at {}", keys[j]),
        );
        close(
            point.se,
            ses[j],
            1e-9,
            &format!("{label} se at {}", keys[j]),
        );
        assert_vectors_close(
            &point.influence,
            &influence[j],
            1e-9,
            &format!("{label} influence at {}", keys[j]),
        );
    }
}

#[test]
fn max_e_applies_to_simple_and_group_as_in_did() {
    let r = reference();
    let cells = estimates(&r.universal.cells);
    let influence = columns(&r.universal.cells.inffunc);
    let p = panel(&r.unit_group);
    for case in &r.max_e {
        let with = |aggregation, balance_e, min_e| {
            aggregate_att_gt(
                &cells,
                &influence,
                &p,
                AggteConfig {
                    aggregation,
                    balance_e,
                    min_e,
                    max_e: Some(case.bound),
                    ..AggteConfig::default()
                },
            )
            .expect("aggregate")
        };
        let label = |name: &str| format!("{name} max_e={}", case.bound);
        check(
            &label("simple"),
            &with(AggteType::Simple, None, None),
            &case.simple,
        );
        check(
            &label("group"),
            &with(AggteType::Group, None, None),
            &case.group,
        );
        check(
            &label("dynamic balanced"),
            &with(AggteType::Dynamic, Some(1), None),
            &case.dynamic_balanced,
        );
        // min_e and balance_e do not apply to simple and group.
        check(
            &label("simple min_e"),
            &with(AggteType::Simple, None, Some(1)),
            &case.simple_min_e,
        );
        check(
            &label("group balanced"),
            &with(AggteType::Group, Some(1), None),
            &case.group_balanced,
        );
    }
    // The bound bites: max_e = 0 leaves one cell per cohort, so it differs from
    // the unrestricted summary.
    let free = aggregate_att_gt(&cells, &influence, &p, config(AggteType::Simple)).unwrap();
    assert!((free.overall_att - r.max_e[0].simple.overall_att).abs() > 1e-4);
}

/// With a missing period did counts `max_e` for simple and group in steps of the
/// sorted period list, so one step can span two years.
#[test]
fn max_e_counts_periods_in_the_list_of_periods_as_in_did() {
    let r = reference();
    let cells = estimates(&r.gap.cells);
    let influence = columns(&r.gap.cells.inffunc);
    let p = panel(&r.gap.unit_group);
    for case in &r.gap.max_e {
        for (aggregation, want) in [
            (AggteType::Simple, &case.simple),
            (AggteType::Group, &case.group),
        ] {
            let got = aggregate_att_gt(
                &cells,
                &influence,
                &p,
                AggteConfig {
                    aggregation,
                    max_e: Some(case.max_e),
                    ..AggteConfig::default()
                },
            )
            .expect("aggregate");
            check(
                &format!("gap {aggregation:?} max_e={}", case.max_e),
                &got,
                want,
            );
        }
    }
}

#[test]
fn a_max_e_below_zero_leaves_no_post_treatment_cell() {
    let r = reference();
    let cells = estimates(&r.universal.cells);
    let influence = columns(&r.universal.cells.inffunc);
    let got = aggregate_att_gt(
        &cells,
        &influence,
        &panel(&r.unit_group),
        AggteConfig {
            aggregation: AggteType::Simple,
            max_e: Some(-1),
            ..AggteConfig::default()
        },
    );
    assert_eq!(got.unwrap_err(), AggteError::NoPostTreatmentCells);
}

#[test]
fn clustered_aggregates_match_did() {
    let r = reference();
    let cells = estimates(&r.universal.cells);
    let influence = columns(&r.universal.cells.inffunc);
    let clustered = clustered_panel(&r);
    let plain = panel(&r.unit_group);
    for (aggregation, want) in [
        (AggteType::Simple, &r.clustered.simple),
        (AggteType::Dynamic, &r.clustered.dynamic),
        (AggteType::Group, &r.clustered.group),
        (AggteType::Calendar, &r.clustered.calendar),
    ] {
        let got = aggregate_att_gt(&cells, &influence, &clustered, config(aggregation))
            .expect("aggregate");
        check(&format!("clustered {aggregation:?}"), &got, want);
        // The clusters change the standard error by more than the tolerance.
        let unclustered =
            aggregate_att_gt(&cells, &influence, &plain, config(aggregation)).unwrap();
        assert!(
            (unclustered.overall_se - got.overall_se).abs() > 1e-4 * got.overall_se,
            "{aggregation:?}: clustering did not move the standard error"
        );
    }
}

#[test]
fn the_pre_test_matches_did() {
    let r = reference();
    for (name, scenario) in [("universal", &r.universal), ("varying", &r.varying)] {
        let cells = estimates(&scenario.cells);
        let influence = columns(&scenario.cells.inffunc);
        let got = pretrend_wald_test(&cells, &influence, &panel(&r.unit_group)).expect("test");
        close(got.statistic, scenario.w, 1e-8, &format!("{name} W"));
        // did rounds the p-value to five decimals.
        assert!(
            (got.p_value - scenario.p_value).abs() <= 5e-6,
            "{name} p {} vs R {}",
            got.p_value,
            scenario.p_value
        );
        let pre = cells.iter().filter(|c| c.time < c.group).count();
        assert_eq!(got.df, pre, "{name} degrees of freedom");

        let clustered =
            pretrend_wald_test(&cells, &influence, &clustered_panel(&r)).expect("clustered test");
        close(
            clustered.statistic,
            scenario.clustered_pretest.w,
            1e-8,
            &format!("{name} clustered W"),
        );
        assert!(
            (clustered.p_value - scenario.clustered_pretest.p_value).abs() <= 5e-6,
            "{name} clustered p"
        );
        assert!((clustered.statistic - got.statistic).abs() > 1e-3);
    }
}

#[test]
fn the_pre_test_refuses_what_has_nothing_to_test() {
    let r = reference();
    let cells = estimates(&r.universal.cells);
    let influence = columns(&r.universal.cells.inffunc);
    let post = (0..cells.len())
        .filter(|&i| cells[i].time >= cells[i].group)
        .collect::<Vec<_>>();
    let only_post = post.iter().map(|&i| cells[i]).collect::<Vec<_>>();
    let only_post_influence = post
        .iter()
        .map(|&i| influence[i].clone())
        .collect::<Vec<_>>();
    assert_eq!(
        pretrend_wald_test(&only_post, &only_post_influence, &panel(&r.unit_group)),
        Err(PretestError::NoPreTreatmentCells)
    );

    // Two pre-treatment cells with the same influence function: singular.
    let pre = cells.iter().position(|c| c.time < c.group).unwrap();
    let doubled = [cells[pre], cells[pre]];
    let doubled_influence = [influence[pre].clone(), influence[pre].clone()];
    assert_eq!(
        pretrend_wald_test(&doubled, &doubled_influence, &panel(&r.unit_group)),
        Err(PretestError::SingularCovariance)
    );
}

fn bootstrap(
    replicates: usize,
    seed: u64,
    standard_errors: bool,
    uniform_band: bool,
) -> AggteBootstrap {
    AggteBootstrap {
        replicates,
        seed,
        standard_errors,
        uniform_band,
    }
}

fn boot_config(aggregation: AggteType, replicates: usize, seed: u64) -> AggteConfig {
    AggteConfig {
        aggregation,
        bootstrap: Some(bootstrap(replicates, seed, true, true)),
        ..AggteConfig::default()
    }
}

/// Bootstrap standard errors and the uniform critical value against `did`'s.
///
/// The draws differ, so the two agree to Monte Carlo error only. For each
/// estimate the error of a standard error taken as `(q75 - q25) / 1.349` of `B`
/// draws is known from the draws, and R's fixture carries it as a fraction of
/// the standard error (`se_rel_mc`, from the quartile densities). The critical
/// value is a quantile of the maximum with error `sqrt(p (1 - p) / B) / f`,
/// also in the fixture. Both sides use `B = 100000`, so the difference has
/// standard deviation `sqrt(2)` times the one-sided error, and the bound is four
/// of those.
fn check_bootstrap(label: &str, got: &AggteResult, want: &BootAgg, biters: usize) {
    let four = 4.0 * 2.0_f64.sqrt();
    assert!(
        got.uniform_band || want.crit_val.is_none(),
        "{label} uniform band"
    );
    close(
        got.overall_att,
        want.agg.overall_att,
        1e-9,
        &format!("{label} overall att"),
    );
    let bound = four * want.overall_se_rel_mc * want.agg.overall_se;
    assert!(
        (got.overall_se - want.agg.overall_se).abs() <= bound,
        "{label} overall se {} vs R {} (bound {bound})",
        got.overall_se,
        want.agg.overall_se
    );
    // The analytic standard error is not what is replaced: the two differ.
    assert!((got.overall_se - want.analytic_overall_se).abs() > 0.0);
    let Some(rel) = &want.se_rel_mc else {
        return;
    };
    let ses = want.agg.se_egt.as_ref().expect("se_egt");
    for (j, point) in got.by_key.iter().enumerate() {
        let bound = four * rel[j] * ses[j];
        assert!(
            (point.se - ses[j]).abs() <= bound,
            "{label} se at {}: {} vs R {} (bound {bound})",
            point.key,
            point.se,
            ses[j]
        );
        // The interval is the uniform one.
        let margin = got.critical_value * point.se;
        assert!((point.ci_high - point.att - margin).abs() <= 1e-12);
        assert!((point.att - point.ci_low - margin).abs() <= 1e-12);
    }
    let crit = want.crit_val.expect("crit");
    let bound = four * want.crit_mc_se.expect("crit mc se");
    assert!(
        (got.critical_value - crit).abs() <= bound,
        "{label} critical value {} vs R {crit} (bound {bound}, B = {biters})",
        got.critical_value
    );
}

#[test]
fn bootstrap_aggregates_agree_with_did_within_monte_carlo_error() {
    let r = reference();
    let cells = estimates(&r.universal.cells);
    let influence = columns(&r.universal.cells.inffunc);
    let p = panel(&r.unit_group);
    let b = r.bootstrap.biters;
    for (aggregation, want) in [
        (AggteType::Simple, &r.bootstrap.unclustered.simple),
        (AggteType::Dynamic, &r.bootstrap.unclustered.dynamic),
        (AggteType::Group, &r.bootstrap.unclustered.group),
        (AggteType::Calendar, &r.bootstrap.unclustered.calendar),
    ] {
        let got = aggregate_att_gt(
            &cells,
            &influence,
            &p,
            boot_config(aggregation, b, 20_261_124),
        )
        .expect("aggregate");
        check_bootstrap(&format!("{aggregation:?}"), &got, want, b);
        // The influence functions that go into the bootstrap are R's.
        let exact = aggregate_att_gt(&cells, &influence, &p, config(aggregation)).unwrap();
        for (point, analytic) in got.by_key.iter().zip(&exact.by_key) {
            assert_eq!(point.influence, analytic.influence);
        }
        assert_eq!(got.overall_influence, exact.overall_influence);
    }
    let clustered = aggregate_att_gt(
        &cells,
        &influence,
        &clustered_panel(&r),
        boot_config(AggteType::Dynamic, b, 20_261_124),
    )
    .expect("clustered");
    check_bootstrap(
        "clustered dynamic",
        &clustered,
        &r.bootstrap.clustered.dynamic,
        b,
    );
}

#[test]
fn bootstrap_results_do_not_depend_on_the_thread_count() {
    let r = reference();
    let cells = estimates(&r.universal.cells);
    let influence = columns(&r.universal.cells.inffunc);
    let p = clustered_panel(&r);
    assert_same_on_threads(|| {
        let got = aggregate_att_gt(
            &cells,
            &influence,
            &p,
            boot_config(AggteType::Dynamic, 3000, 7),
        )
        .unwrap();
        let mut out = vec![got.critical_value, got.overall_se];
        out.extend(got.by_key.iter().map(|point| point.se));
        out
    });
}

/// The uniform critical value covers the whole path with the stated
/// probability under fresh multiplier draws.
///
/// The critical value is the `1 - alpha` quantile of the studentised maxima of
/// `B` draws, so the probability that a fresh draw stays below it is
/// `1 - alpha` up to the error of an empirical quantile, `sqrt(p (1 - p) / B)`.
/// The fresh draws are `M` independent ones, so the share below the critical
/// value has binomial error `sqrt(p (1 - p) / M)`. The bound is four standard
/// deviations of the difference.
#[test]
fn the_uniform_critical_value_has_the_stated_coverage_on_fresh_draws() {
    let r = reference();
    let cells = estimates(&r.universal.cells);
    let influence = columns(&r.universal.cells.inffunc);
    let (b, m) = (20_000_usize, 40_000_usize);
    for clustered in [false, true] {
        let p = if clustered {
            clustered_panel(&r)
        } else {
            panel(&r.unit_group)
        };
        let got = aggregate_att_gt(
            &cells,
            &influence,
            &p,
            boot_config(AggteType::Dynamic, b, 99),
        )
        .unwrap();
        let groups = p
            .clusters
            .as_ref()
            .map(|labels| Clusters::from_sorted_labels(labels));
        let n = p.groups.len();
        let g = groups.as_ref().map_or(n, Clusters::count);
        // Cluster sums of the path's influence functions, one row per cluster.
        let sums: Vec<Vec<f64>> = got
            .by_key
            .iter()
            .map(|point| match &groups {
                Some(c) => {
                    let mut sums = vec![0.0; c.count()];
                    for (value, &cluster) in point.influence.iter().zip(c.of()) {
                        sums[cluster] += value;
                    }
                    sums
                }
                None => point.influence.clone(),
            })
            .collect();
        // The scale each standard error was computed with: se = sigma sqrt(G) / n.
        let sigma: Vec<f64> = got
            .by_key
            .iter()
            .map(|point| point.se * n as f64 / (g as f64).sqrt())
            .collect();
        let mut signs = vec![0.0; g];
        let mut below = 0_usize;
        for draw in 0..m {
            let mut stream = Stream::seeded(1_000_003 + draw as u64);
            stream.fill_signs(&mut signs);
            let worst = sums
                .iter()
                .zip(&sigma)
                .map(|(column, sigma)| {
                    let total: f64 = column.iter().zip(&signs).map(|(a, s)| a * s).sum();
                    (total / (g as f64).sqrt()).abs() / sigma
                })
                .fold(0.0, f64::max);
            below += usize::from(worst <= got.critical_value);
        }
        let level = 1.0 - r.bootstrap.alp;
        let share = below as f64 / m as f64;
        let bound = 4.0 * (level * (1.0 - level) * (1.0 / b as f64 + 1.0 / m as f64)).sqrt();
        assert!(
            (share - level).abs() <= bound,
            "clustered={clustered}: share below the critical value {share}, level {level}, bound {bound}"
        );
        assert!(got.critical_value > epistat_stats::dist::normal::quantile(1.0 - r.bootstrap.alp / 2.0));
    }
}

#[test]
fn the_flags_choose_what_the_bootstrap_replaces() {
    let r = reference();
    let cells = estimates(&r.universal.cells);
    let influence = columns(&r.universal.cells.inffunc);
    let p = panel(&r.unit_group);
    let analytic = aggregate_att_gt(&cells, &influence, &p, config(AggteType::Dynamic)).unwrap();
    let full = aggregate_att_gt(
        &cells,
        &influence,
        &p,
        boot_config(AggteType::Dynamic, 2000, 5),
    )
    .unwrap();

    // cband only: analytic standard errors, uniform critical value.
    let band_only = aggregate_att_gt(
        &cells,
        &influence,
        &p,
        AggteConfig {
            bootstrap: Some(bootstrap(2000, 5, false, true)),
            ..config(AggteType::Dynamic)
        },
    )
    .unwrap();
    assert!(band_only.uniform_band);
    assert_eq!(
        band_only.critical_value.to_bits(),
        full.critical_value.to_bits()
    );
    assert_eq!(
        band_only.overall_se.to_bits(),
        analytic.overall_se.to_bits()
    );
    for (a, b) in band_only.by_key.iter().zip(&analytic.by_key) {
        assert_eq!(a.se.to_bits(), b.se.to_bits());
    }

    // bstrap only: bootstrap standard errors, pointwise critical value.
    let se_only = aggregate_att_gt(
        &cells,
        &influence,
        &p,
        AggteConfig {
            bootstrap: Some(bootstrap(2000, 5, true, false)),
            ..config(AggteType::Dynamic)
        },
    )
    .unwrap();
    assert!(!se_only.uniform_band);
    assert_eq!(
        se_only.critical_value.to_bits(),
        analytic.critical_value.to_bits()
    );
    assert_eq!(se_only.overall_se.to_bits(), full.overall_se.to_bits());
    assert!((se_only.by_key[0].se - analytic.by_key[0].se).abs() > 1e-12);

    for bad in [bootstrap(0, 1, true, true), bootstrap(10, 1, false, false)] {
        let refused = aggregate_att_gt(
            &cells,
            &influence,
            &p,
            AggteConfig {
                bootstrap: Some(bad),
                ..config(AggteType::Dynamic)
            },
        );
        assert_eq!(refused.unwrap_err(), AggteError::InvalidBootstrap);
    }
}

//! Aggregation of `ATT(g,t)` against `did::aggte` 2.5.1: `max_e` for the simple
//! and group summaries, clustered aggregates, the multiplier bootstrap and the
//! Wald pre-test.
//!
//! The fixture is written by `tests/data/generate_aggte_reference.R`. Every
//! scenario carries the cells `att_gt` returned, with their influence functions
//! in the order of the sorted unit ids. The cells here are this crate's fit of
//! the same `mpdta` panel, which agree with R's to `1e-10` in the estimates
//! and `1e-7` in the influence values (`parity_cs_panel.rs`). The aggregates
//! are weighted means of the cells with weights below one, plus the
//! estimated-share correction, whose factor `w / p` is below 10 here, so the
//! estimates and standard errors agree with R's to `1e-9` and the aggregate
//! influence values to `1e-6`. The influence functions of the aggregates are
//! compared too: they are deterministic, and the bootstrap is a function of
//! them.

#[path = "common/att_gt_rows.rs"]
#[expect(dead_code, reason = "the shared helpers serve several tests")]
mod rows;

use epistat_core::{Bootstrap as Draws, Classify, Seed};
use epistat_did::Error;
use epistat_did::att_gt::{
    AggteOptions, AggteType, AttGt, AttGtOptions, BasePeriod, Estimator, Inference,
};
use epistat_inference::Clusters;
use epistat_rng::Stream;
use epistat_testkit::{Tolerance, assert_same_on_threads, load_json};
use rows::{Obs, Summary};
use serde::Deserialize;

#[derive(Deserialize)]
struct Pretest {
    #[serde(rename = "W")]
    w: f64,
    #[serde(rename = "Wpval")]
    p_value: f64,
}

#[derive(Deserialize)]
struct Scenario {
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
    max_e: Vec<GapMaxE>,
}

#[derive(Deserialize)]
struct Reference {
    unit_group: Vec<f64>,
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
struct Data {
    data_subset: Vec<DataRow>,
}

/// `mpdta` with log population as the covariate, and the state, the county
/// number divided by 1000, as the cluster when asked, as the fixture's
/// `att_gt(clustervars = "state")`.
fn mpdta(clustered: bool, keep: impl Fn(&DataRow) -> bool) -> Vec<Obs> {
    let data: Data = load_json("tests/data/cs_panel_dr_universal_ref.json");
    data.data_subset
        .iter()
        .filter(|row| keep(row))
        .map(|row| Obs {
            covariates: vec![row.lpop],
            cluster: clustered.then(|| u32::try_from(row.countyreal / 1000).expect("a state")),
            ..Obs::new(
                Some(row.countyreal),
                (row.first_treat != 0).then_some(row.first_treat),
                row.year,
                row.lemp,
            )
        })
        .collect()
}

fn fit(rows: &[Obs], base_period: BasePeriod) -> AttGt {
    rows::fit(
        rows,
        &AttGtOptions {
            base_period,
            ..AttGtOptions::default()
        },
    )
    .expect("ATT(g,t)")
}

fn universal(clustered: bool) -> AttGt {
    fit(&mpdta(clustered, |_| true), BasePeriod::Universal)
}

fn config(aggregation: AggteType) -> AggteOptions {
    AggteOptions {
        aggregation,
        ..AggteOptions::default()
    }
}

fn aggregate(fit: &AttGt, options: AggteOptions) -> Summary {
    rows::summary(&fit.aggregate(&options).expect("aggregate"))
}

/// The tolerances of the module documentation.
const ESTIMATE: f64 = 1e-9;
const INFLUENCE: f64 = 1e-6;

fn close(got: f64, want: f64, tolerance: f64, label: &str) {
    assert!(
        Tolerance::exact()
            .abs(tolerance)
            .rel(tolerance)
            .accepts(got, want),
        "{label}: {got} vs R {want}"
    );
}

fn assert_vectors_close(got: &[f64], want: &[f64], tolerance: f64, label: &str) {
    assert_eq!(got.len(), want.len(), "{label} length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        close(*g, *w, tolerance, &format!("{label}[{i}]"));
    }
}

/// One vector per path point, from the fixture's unit-by-column rows.
fn columns(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    (0..rows[0].len())
        .map(|j| rows.iter().map(|row| row[j]).collect())
        .collect()
}

/// Estimates, standard errors and influence functions of the overall effect
/// and the path, against R's.
fn check(label: &str, got: &Summary, want: &Agg) {
    close(
        got.overall_att,
        want.overall_att,
        ESTIMATE,
        &format!("{label} att"),
    );
    close(
        got.overall_se,
        want.overall_se,
        ESTIMATE,
        &format!("{label} se"),
    );
    assert_vectors_close(
        &got.overall_influence,
        &want.inf_overall,
        INFLUENCE,
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
            ESTIMATE,
            &format!("{label} att at {}", keys[j]),
        );
        close(
            point.se,
            ses[j],
            ESTIMATE,
            &format!("{label} se at {}", keys[j]),
        );
        assert_vectors_close(
            &point.influence,
            &influence[j],
            INFLUENCE,
            &format!("{label} influence at {}", keys[j]),
        );
    }
}

#[test]
fn the_units_and_cohorts_are_the_fixtures() {
    let r = reference();
    let fit = universal(false);
    let groups: Vec<f64> = fit
        .sample()
        .groups
        .iter()
        .map(|g| g.map_or(0.0, |g| g as f64))
        .collect();
    assert_eq!(groups, r.unit_group);
}

#[test]
fn max_e_applies_to_simple_and_group_as_in_did() {
    let r = reference();
    let fit = universal(false);
    for case in &r.max_e {
        let with = |aggregation, balance_e, min_e| {
            aggregate(
                &fit,
                AggteOptions {
                    aggregation,
                    balance_e,
                    min_e,
                    max_e: Some(i64::from(case.bound)),
                    ..AggteOptions::default()
                },
            )
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
    let free = aggregate(&fit, config(AggteType::Simple));
    assert!((free.overall_att - r.max_e[0].simple.overall_att).abs() > 1e-4);
}

/// With a missing period did counts `max_e` for simple and group in steps of
/// the sorted period list, so one step can span two years.
#[test]
fn max_e_counts_periods_in_the_list_of_periods_as_in_did() {
    let r = reference();
    let fit = fit(&mpdta(false, |row| row.year != 2005), BasePeriod::Universal);
    let groups: Vec<f64> = fit
        .sample()
        .groups
        .iter()
        .map(|g| g.map_or(0.0, |g| g as f64))
        .collect();
    assert_eq!(groups, r.gap.unit_group);
    for case in &r.gap.max_e {
        for (aggregation, want) in [
            (AggteType::Simple, &case.simple),
            (AggteType::Group, &case.group),
        ] {
            let got = aggregate(
                &fit,
                AggteOptions {
                    aggregation,
                    max_e: Some(i64::from(case.max_e)),
                    ..AggteOptions::default()
                },
            );
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
    let got = universal(false).aggregate(&AggteOptions {
        aggregation: AggteType::Simple,
        max_e: Some(-1),
        ..AggteOptions::default()
    });
    assert_eq!(got.unwrap_err(), Error::NoPostTreatmentCells);
}

#[test]
fn clustered_aggregates_match_did() {
    let r = reference();
    let clustered = universal(true);
    let plain = universal(false);
    for (aggregation, want) in [
        (AggteType::Simple, &r.clustered.simple),
        (AggteType::Dynamic, &r.clustered.dynamic),
        (AggteType::Group, &r.clustered.group),
        (AggteType::Calendar, &r.clustered.calendar),
    ] {
        let got = aggregate(&clustered, config(aggregation));
        check(&format!("clustered {aggregation:?}"), &got, want);
        // The clusters change the standard error by more than the tolerance.
        let unclustered = aggregate(&plain, config(aggregation));
        assert!(
            (unclustered.overall_se - got.overall_se).abs() > 1e-4 * got.overall_se,
            "{aggregation:?}: clustering did not move the standard error"
        );
    }
}

/// The statistic is a quadratic form in the pre-treatment cells, whose
/// estimates agree with R's to `1e-10` and whose covariance agrees to the
/// influence values' `1e-7` relative to values of order one, so `W` agrees to
/// about `1e-6` relative.
#[test]
fn the_pre_test_matches_did() {
    let r = reference();
    for (name, scenario, base) in [
        ("universal", &r.universal, BasePeriod::Universal),
        ("varying", &r.varying, BasePeriod::Varying),
    ] {
        let fitted = fit(&mpdta(false, |_| true), base);
        let got = fitted.pretest().expect("test");
        close(got.statistic, scenario.w, 1e-6, &format!("{name} W"));
        // did rounds the p-value to five decimals.
        assert!(
            (got.p_value - scenario.p_value).abs() <= 5e-6,
            "{name} p {} vs R {}",
            got.p_value,
            scenario.p_value
        );
        let pre = fitted
            .cells()
            .iter()
            .filter(|c| c.estimate.is_some() && c.time < c.group)
            .count();
        assert_eq!(got.df, pre, "{name} degrees of freedom");

        let clustered = fit(&mpdta(true, |_| true), base)
            .pretest()
            .expect("clustered test");
        close(
            clustered.statistic,
            scenario.clustered_pretest.w,
            1e-6,
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
    // The efficient estimator has no pre-treatment cells.
    let plain: Vec<Obs> = mpdta(false, |_| true)
        .into_iter()
        .map(|row| Obs {
            covariates: Vec::new(),
            ..row
        })
        .collect();
    let efficient = rows::fit(
        &plain,
        &AttGtOptions {
            estimator: Estimator::Efficient,
            ..AttGtOptions::default()
        },
    )
    .expect("efficient");
    assert_eq!(efficient.pretest().unwrap_err(), Error::NoPreTreatmentCells);
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
fn check_bootstrap(label: &str, got: &Summary, want: &BootAgg, biters: usize) {
    let four = 4.0 * 2.0_f64.sqrt();
    assert!(
        got.uniform_band || want.crit_val.is_none(),
        "{label} uniform band"
    );
    close(
        got.overall_att,
        want.agg.overall_att,
        ESTIMATE,
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

fn boot_config(aggregation: AggteType, replicates: usize, seed: u64) -> AggteOptions {
    AggteOptions {
        aggregation,
        inference: Inference::UniformBand(Draws::new(replicates, Seed(seed))),
        ..AggteOptions::default()
    }
}

#[test]
fn bootstrap_aggregates_agree_with_did_within_monte_carlo_error() {
    let r = reference();
    let fitted = universal(false);
    let b = r.bootstrap.biters;
    for (aggregation, want) in [
        (AggteType::Simple, &r.bootstrap.unclustered.simple),
        (AggteType::Dynamic, &r.bootstrap.unclustered.dynamic),
        (AggteType::Group, &r.bootstrap.unclustered.group),
        (AggteType::Calendar, &r.bootstrap.unclustered.calendar),
    ] {
        let got = aggregate(&fitted, boot_config(aggregation, b, 20_261_124));
        check_bootstrap(&format!("{aggregation:?}"), &got, want, b);
        // The influence functions that go into the bootstrap are the
        // analytic ones.
        let exact = aggregate(&fitted, config(aggregation));
        for (point, analytic) in got.by_key.iter().zip(&exact.by_key) {
            assert_eq!(point.influence, analytic.influence);
        }
        assert_eq!(got.overall_influence, exact.overall_influence);
    }
    let clustered = aggregate(
        &universal(true),
        boot_config(AggteType::Dynamic, b, 20_261_124),
    );
    check_bootstrap(
        "clustered dynamic",
        &clustered,
        &r.bootstrap.clustered.dynamic,
        b,
    );
}

#[test]
fn bootstrap_results_do_not_depend_on_the_thread_count() {
    let fitted = universal(true);
    assert_same_on_threads(|| {
        let got = aggregate(&fitted, boot_config(AggteType::Dynamic, 3000, 7));
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
    let (b, m) = (20_000_usize, 40_000_usize);
    for clustered in [false, true] {
        let fitted = universal(clustered);
        let got = aggregate(&fitted, boot_config(AggteType::Dynamic, b, 99));
        let groups = fitted.clusters().map(Clusters::from_codes);
        let n = fitted.sample().units.len();
        let g = groups.as_ref().map_or(n, Clusters::count);
        // Cluster sums of the path's influence functions, one row per cluster.
        let sums: Vec<Vec<f64>> = got
            .by_key
            .iter()
            .map(|point| match &groups {
                Some(c) => {
                    let mut sums = vec![0.0; c.count()];
                    for (value, &cluster) in point.influence.iter().zip(c.of()) {
                        sums[cluster as usize] += value;
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
        assert!(
            got.critical_value > epistat_stats::dist::normal::quantile(1.0 - r.bootstrap.alp / 2.0)
        );
    }
}

#[test]
fn the_options_choose_what_the_bootstrap_replaces() {
    let fitted = universal(false);
    let analytic = aggregate(&fitted, config(AggteType::Dynamic));
    let full = aggregate(&fitted, boot_config(AggteType::Dynamic, 2000, 5));
    assert!(full.uniform_band);

    // The bootstrap without the band: bootstrap standard errors, pointwise
    // critical value.
    let se_only = aggregate(
        &fitted,
        AggteOptions {
            inference: Inference::Bootstrap(Draws::new(2000, Seed(5))),
            ..config(AggteType::Dynamic)
        },
    );
    assert!(!se_only.uniform_band);
    assert_eq!(
        se_only.critical_value.to_bits(),
        analytic.critical_value.to_bits()
    );
    assert_eq!(se_only.overall_se.to_bits(), full.overall_se.to_bits());
    assert!((se_only.by_key[0].se - analytic.by_key[0].se).abs() > 1e-12);

    // The bootstrap needs a replicate.
    let none = fitted.aggregate(&boot_config(AggteType::Dynamic, 0, 1));
    assert_eq!(none.unwrap_err().code(), "input.domain");
}
